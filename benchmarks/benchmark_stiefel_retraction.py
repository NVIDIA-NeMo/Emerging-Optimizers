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
"""Benchmark Stiefel retractions in Iso: latency, throughput, and numerical precision."""

from collections.abc import Callable
from typing import Any

import torch
import triton

from emerging_optimizers import utils
from emerging_optimizers.riemannian_optimizers.retractions.stiefel import (
    cayley_retraction,
    newton_schulz_retraction,
    polar_retraction,
    qr_retraction,
)


def bench(fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> float:
    """Time fn with triton.testing.do_bench (returns execution time in ms)."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def test_numerical_accuracy(
    ref: torch.Tensor,
    qr: torch.Tensor,
    cayley: torch.Tensor,
    ns5: torch.Tensor,
    ns8: torch.Tensor,
) -> dict[str, float]:
    """Verify factor orthogonality and error vs exact polar factor in a single batched D2H sync."""
    k = ref.shape[-1]
    eye = torch.eye(k, device=ref.device, dtype=ref.dtype)

    # Orthogonality deviations on GPU: ||Q^T Q - I||_max
    orth_polar = (ref.mT @ ref - eye).abs().max()
    orth_qr = (qr.mT @ qr - eye).abs().max()
    orth_cayley = (cayley.mT @ cayley - eye).abs().max()
    orth_ns5 = (ns5.mT @ ns5 - eye).abs().max()
    orth_ns8 = (ns8.mT @ ns8 - eye).abs().max()

    # Approximation error vs analytical polar factor on GPU
    ref_f = ref.float()
    diff_ns5 = (ns5.float() - ref_f).abs()
    err_ns5_abs = diff_ns5.max()
    err_ns5_rel = (diff_ns5 / ref_f.abs().clamp_min(1e-8)).max()

    diff_ns8 = (ns8.float() - ref_f).abs()
    err_ns8_abs = diff_ns8.max()
    err_ns8_rel = (diff_ns8 / ref_f.abs().clamp_min(1e-8)).max()

    # Batch all 9 metrics into a single device-to-host synchronization
    metrics = torch.stack(
        [
            orth_polar,
            orth_qr,
            orth_cayley,
            orth_ns5,
            orth_ns8,
            err_ns5_abs,
            err_ns5_rel,
            err_ns8_abs,
            err_ns8_rel,
        ]
    ).tolist()

    return {
        "orth_polar": metrics[0],
        "orth_qr": metrics[1],
        "orth_cayley": metrics[2],
        "orth_ns5": metrics[3],
        "orth_ns8": metrics[4],
        "err_ns5_abs": metrics[5],
        "err_ns5_rel": metrics[6],
        "err_ns8_abs": metrics[7],
        "err_ns8_rel": metrics[8],
    }


def run_case(
    m: int,
    n: int,
    lr: float = 1e-3,
    device: str = "cuda",
    fp32_prec: utils.FP32MatmulPrecT = "high",
) -> None:
    """Benchmark retractions for a single parameter tensor of shape (M, N)."""
    torch.manual_seed(0)
    k = min(m, n)

    # Initialize factor matrices U (M, K) and V (N, K) on the Stiefel manifold
    raw_u = torch.randn(m, k, device=device, dtype=torch.float32)
    raw_v = torch.randn(n, k, device=device, dtype=torch.float32)
    u_init, _ = torch.linalg.qr(raw_u, mode="reduced")
    v_init, _ = torch.linalg.qr(raw_v, mode="reduced")

    # Synthetic momentum updates
    mom_u = torch.randn_like(u_init)
    mom_v = torch.randn_like(v_init)

    # --- Correctness & Numerical Precision ---
    ref_u = polar_retraction(u_init, mom_u, lr)

    qr_u = qr_retraction(u_init, mom_u, lr)
    cayley_u = cayley_retraction(u_init, mom_u, lr)
    ns5_u = newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=5)
    ns8_u = newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=8)

    acc = test_numerical_accuracy(ref_u, qr_u, cayley_u, ns5_u, ns8_u)

    # --- Latency Benchmarking (Factor U + Factor V) ---
    with utils.fp32_matmul_precision(fp32_prec):
        t_polar = bench(lambda: (polar_retraction(u_init, mom_u, lr), polar_retraction(v_init, mom_v, lr)))
        t_qr = bench(lambda: (qr_retraction(u_init, mom_u, lr), qr_retraction(v_init, mom_v, lr)))
        t_cayley = bench(
            lambda: (
                cayley_retraction(u_init, mom_u, lr),
                cayley_retraction(v_init, mom_v, lr),
            )
        )
        t_ns5 = bench(
            lambda: (
                newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=5),
                newton_schulz_retraction(v_init, mom_v, lr, num_ns_steps=5),
            )
        )
        t_ns8 = bench(
            lambda: (
                newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=8),
                newton_schulz_retraction(v_init, mom_v, lr, num_ns_steps=8),
            )
        )

    tag = f"M={m:<5d} N={n:<5d} K={k:<5d}"
    speedup_ns8_vs_polar = t_polar / t_ns8

    print(
        f"{tag} | polar(svd) {t_polar:8.3f} ms | qr {t_qr:7.3f} ms | cayley {t_cayley:7.3f} ms | "
        f"ns(steps=5) {t_ns5:7.3f} ms | ns(steps=8) {t_ns8:7.3f} ms | "
        f"ns8 speedup vs svd: {speedup_ns8_vs_polar:6.2f}x"
    )
    print(
        f"{'':>4}orth drift ||Q^T Q - I||: svd={acc['orth_polar']:.2e} | qr={acc['orth_qr']:.2e} | "
        f"cayley={acc['orth_cayley']:.2e} | ns5={acc['orth_ns5']:.2e} | ns8={acc['orth_ns8']:.2e}"
    )
    print(
        f"{'':>4}ns vs polar factor: ns5 abs={acc['err_ns5_abs']:.2e} rel={acc['err_ns5_rel']:.2e} | "
        f"ns8 abs={acc['err_ns8_abs']:.2e} rel={acc['err_ns8_rel']:.2e}\n"
    )


def main() -> None:
    """Run benchmark across square and rectangular transformer projection dimensions."""
    torch.cuda.init()
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    cases = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (4096, 2048),
        (8192, 2048),
    ]

    print("=== Iso Stiefel Retraction Benchmark Suite ===")
    for m, n in cases:
        run_case(m, n)


if __name__ == "__main__":
    main()
