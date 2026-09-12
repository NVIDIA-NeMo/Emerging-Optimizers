# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token embedding regression under a heavy-tailed Zipfian distribution.

This example benchmarks optimizers on a synthetic embedding reconstruction task where token
frequencies follow a power-law (Zipfian) distribution. It demonstrates:

1. How standard Riemannian SGD starves infrequent (long-tail) tokens of gradient signal.
2. How the $L_1 \\to \\text{RMS}$ Linear Minimization Oracle (LMO) in ``ObliqueSteepestSGD``
   restores uniform angular updates across all tokens, matching or outperforming AdamW
   with zero second-moment memory states.
3. How DeepSeek-V4.1's 2D Sinkhorn balancing compares against 1D oblique manifold optimization.
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
from statistics import fmean, pstdev
from typing import Callable, Iterator, override

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.optim.optimizer import Optimizer

from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import (
    OrthogonalizedOptimizer,
)
from emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils import (
    sinkhorn_balance,
)
from emerging_optimizers.riemannian_optimizers.normalized_optimizer import (
    ObliqueSteepestSGD,
)


class EmbeddingModel(nn.Module):
    """Embedding layer constrained to have unit RMS rows."""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)

        with torch.no_grad():
            nn.functional.normalize(
                self.emb.weight,
                p=2.0,
                dim=1,
                out=self.emb.weight,
            ).mul_(hidden_dim**0.5)

    @override
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.emb(tokens)


def make_problem(
    vocab_size: int,
    hidden_dim: int,
    batch_size: int,
    steps: int,
    zipf_exponent: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct an independent target embedding table and a deterministic token stream."""
    target_generator = torch.Generator(device=device).manual_seed(seed)
    target_embeddings = torch.randn(
        vocab_size,
        hidden_dim,
        generator=target_generator,
        device=device,
    )
    nn.functional.normalize(
        target_embeddings,
        p=2.0,
        dim=1,
        out=target_embeddings,
    ).mul_(hidden_dim**0.5)

    # Heavy-tailed Zipfian frequency distribution
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=device)
    token_probabilities = ranks.pow(-zipf_exponent)
    token_probabilities /= token_probabilities.sum()

    # Pre-sample the exact token sequence shared across all optimizers
    token_generator = torch.Generator(device=device).manual_seed(seed + 1000)
    batches = torch.multinomial(
        token_probabilities,
        num_samples=batch_size * steps,
        replacement=True,
        generator=token_generator,
    ).reshape(steps, batch_size)

    token_counts = torch.bincount(batches.flatten(), minlength=vocab_size)

    return target_embeddings, token_probabilities, batches, token_counts


def make_sinkhorn_optimizer(
    parameters: Iterator[nn.Parameter],
    steps: int,
    lr: float = 0.108,
) -> tuple[Optimizer, LRScheduler]:
    """DeepSeek-V4.1 Sinkhorn-balanced optimizer configuration."""
    optimizer: Optimizer = OrthogonalizedOptimizer(
        parameters,
        lr=lr,
        momentum=0.55,
        nesterov=True,
        weight_decay=0.01,
        weight_decay_method="decoupled",
        fp32_matmul_prec="highest",
        scaled_orthogonalize_fn=partial(
            sinkhorn_balance,
            num_normalization_steps=3,
            zero_row_threshold=0.02,
            eps=1e-20,
        ),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.01 * lr)
    return optimizer, scheduler


def make_adamw_optimizer(
    parameters: Iterator[nn.Parameter],
    steps: int,
    lr: float = 0.15,
) -> tuple[Optimizer, LRScheduler]:
    """Standard AdamW optimizer baseline."""
    optimizer: Optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=(0.8, 0.999),
        eps=1e-8,
        weight_decay=0.02,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.01 * lr)
    return optimizer, scheduler


def make_oblique_steepest_sgd_optimizer(
    parameters: Iterator[nn.Parameter],
    steps: int,
    lr: float = 0.01,
) -> tuple[Optimizer, LRScheduler]:
    """Oblique steepest descent optimizer using the $L_1 \\to \\text{RMS}$ LMO."""
    optimizer: Optimizer = ObliqueSteepestSGD(
        list(parameters),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        dim=1,
        scale_mode="unit_rms_norm",
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.1 * lr)
    return optimizer, scheduler


def train(
    model: EmbeddingModel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    batches: torch.Tensor,
    target_embeddings: torch.Tensor,
) -> float:
    """Train the model over the pre-generated batch stream asynchronously without CUDA syncs."""
    total_steps = batches.size(0)
    tail_window = min(100, total_steps)
    start_tail_step = total_steps - tail_window
    tail_losses = torch.empty(tail_window, device=batches.device, dtype=torch.float32)

    for step_idx, tokens in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)

        predictions = model(tokens)
        targets = target_embeddings[tokens]
        loss = nn.functional.mse_loss(predictions, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Asynchronous device assignment: avoids calling loss.item() inside the hot loop
        if step_idx >= start_tail_step:
            tail_losses[step_idx - start_tail_step] = loss.detach()

    return tail_losses.mean().item()


def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean of masked values safely on device without Python branching or NaNs."""
    float_mask = mask.to(dtype=values.dtype)
    count = float_mask.sum()
    masked_sum = (values * float_mask).sum()
    return torch.where(count > 0, masked_sum / count.clamp_min(1.0), torch.zeros_like(masked_sum))


@torch.no_grad()
def evaluate(
    model: EmbeddingModel,
    target_embeddings: torch.Tensor,
    token_probabilities: torch.Tensor,
    token_counts: torch.Tensor,
    vocab_size: int,
) -> dict[str, float]:
    """Evaluate full-vocabulary and tail-token reconstruction metrics using a single D2H transfer."""
    weights = model.emb.weight

    row_mse = (weights - target_embeddings).square().mean(dim=1)
    row_cosine = nn.functional.cosine_similarity(weights, target_embeddings, dim=1)
    weight_rms = weights.square().mean(dim=1).sqrt()

    token_indices = torch.arange(vocab_size, device=weights.device)
    head_mask = token_indices < (vocab_size // 10)
    tail_mask = token_indices >= (vocab_size // 2)
    seen_mask = token_counts > 0
    seen_tail_mask = seen_mask & tail_mask
    unseen_mask = ~seen_mask

    # Vectorized computations on device; handles empty masks without NaNs or syncs
    expected_mse = (row_mse * token_probabilities).sum()
    head_mse = _safe_masked_mean(row_mse, head_mask)
    tail_mse = _safe_masked_mean(row_mse, tail_mask)
    seen_tail_mse = _safe_masked_mean(row_mse, seen_tail_mask)
    seen_tail_cosine = _safe_masked_mean(row_cosine, seen_tail_mask)
    unseen_mse = _safe_masked_mean(row_mse, unseen_mask)
    mean_weight_rms = weight_rms.mean()

    metric_keys = [
        "expected_mse",
        "head_mse",
        "tail_mse",
        "seen_tail_mse",
        "seen_tail_cosine",
        "unseen_mse",
        "mean_weight_rms",
    ]
    # Single batched device-to-host transfer
    metric_values = torch.stack(
        [
            expected_mse,
            head_mse,
            tail_mse,
            seen_tail_mse,
            seen_tail_cosine,
            unseen_mse,
            mean_weight_rms,
        ]
    ).tolist()

    return dict(zip(metric_keys, metric_values, strict=True))


def run_seed(
    seed: int,
    device: torch.device,
    vocab_size: int,
    hidden_dim: int,
    batch_size: int,
    steps: int,
    zipf_exponent: float,
) -> dict[str, dict[str, float]]:
    """Run all benchmarked optimizers on an identical problem initialization."""
    target_embeddings, token_probabilities, batches, token_counts = make_problem(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        steps=steps,
        zipf_exponent=zipf_exponent,
        seed=seed,
        device=device,
    )

    torch.manual_seed(seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1)

    initial_model = EmbeddingModel(vocab_size, hidden_dim).to(device)

    optimizer_factories: dict[str, Callable[[Iterator[nn.Parameter]], tuple[Optimizer, LRScheduler]]] = {
        "Sinkhorn K=3": lambda params: make_sinkhorn_optimizer(params, steps),
        "AdamW": lambda params: make_adamw_optimizer(params, steps),
        "ObliqueSteepestSGD": lambda params: make_oblique_steepest_sgd_optimizer(params, steps),
    }

    results: dict[str, dict[str, float]] = {}

    for name, factory in optimizer_factories.items():
        model = copy.deepcopy(initial_model)
        optimizer, scheduler = factory(model.emb.parameters())

        last_100_loss = train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batches=batches,
            target_embeddings=target_embeddings,
        )

        metrics = evaluate(
            model=model,
            target_embeddings=target_embeddings,
            token_probabilities=token_probabilities,
            token_counts=token_counts,
            vocab_size=vocab_size,
        )
        metrics["last_100_training_loss"] = last_100_loss
        results[name] = metrics

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark optimizers on Zipfian token embedding regression.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmark on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Vocabulary size (number of embedding rows).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Embedding vector dimension.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size per training step.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--zipf-exponent",
        type=float,
        default=1.1,
        help="Zipf distribution power law exponent.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 77, 123],
        help="Random seeds for benchmark runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the multi-seed Zipfian embedding regression benchmark."""
    args = parse_args()
    device = torch.device(args.device)

    print(
        f"Starting Zipfian embedding benchmark on {device.type.upper()} "
        f"(Vocab: {args.vocab_size}, Dim: {args.hidden_dim}, Steps: {args.steps}, Seeds: {args.seeds})"
    )

    all_results: dict[str, list[dict[str, float]]] = {}

    for seed in args.seeds:
        print(f"--> Running seed {seed}...")
        seed_results = run_seed(
            seed=seed,
            device=device,
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            steps=args.steps,
            zipf_exponent=args.zipf_exponent,
        )

        for opt_name, metrics in seed_results.items():
            all_results.setdefault(opt_name, []).append(metrics)

    adamw_tail_mse = fmean(r["seen_tail_mse"] for r in all_results["AdamW"])

    print("\nBenchmark Summary (Aggregated across seeds)")
    print("=" * 95)
    header = f"{'Optimizer':<22} | {'Seen-Tail MSE':<20} | {'Tail MSE':<12} | {'Exp. MSE':<12} | {'vs. AdamW':<10}"
    print(header)
    print("-" * 95)

    for opt_name, results in all_results.items():
        seen_tail_vals = [r["seen_tail_mse"] for r in results]
        mean_seen_tail = fmean(seen_tail_vals)
        std_seen_tail = pstdev(seen_tail_vals) if len(seen_tail_vals) > 1 else 0.0

        mean_tail = fmean(r["tail_mse"] for r in results)
        mean_exp = fmean(r["expected_mse"] for r in results)

        rel_change = (1.0 - mean_seen_tail / adamw_tail_mse) * 100.0
        rel_str = f"{rel_change:+.1f}%" if opt_name != "AdamW" else "0.0% (ref)"

        print(
            f"{opt_name:<22} | {mean_seen_tail:.6f} ± {std_seen_tail:.6f} | "
            f"{mean_tail:<12.6f} | {mean_exp:<12.6f} | {rel_str:<10}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
