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

"""Small MoE language model convergence test on C4 dataset.

Uses a small Mixtral-architecture MoE from HuggingFace transformers,
trained from scratch on streamed C4 data with optimizers from this repo.
"""

import math

import datasets
import torch
from absl import app, flags, logging
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM

from emerging_optimizers.orthogonalized_optimizers.muon import Muon
from emerging_optimizers.soap.soap import SOAP


FLAGS = flags.FLAGS

flags.DEFINE_enum("optimizer", "muon", ["muon", "soap", "adamw"], "Optimizer to use for training.")
flags.DEFINE_float("lr", 0.001, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.01, "Weight decay.")
flags.DEFINE_integer("max_steps", 1000, "Number of training steps.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_integer("seq_len", 512, "Sequence length.")
flags.DEFINE_integer("log_interval", 50, "Log every N steps.")
flags.DEFINE_integer("seed", 13, "Random seed.")
flags.DEFINE_float("loss_target", 20.0, "Loss target.")

# Model hyperparameters
flags.DEFINE_integer("d_model", 512, "Model dimension.")
flags.DEFINE_integer("n_layers", 8, "Number of transformer layers.")
flags.DEFINE_integer("n_heads", 8, "Number of attention heads.")
flags.DEFINE_integer("n_kv_heads", 4, "Number of key-value heads (GQA).")
flags.DEFINE_integer("n_experts", 8, "Number of experts per MoE layer.")
flags.DEFINE_integer("top_k", 2, "Number of active experts per token.")
flags.DEFINE_integer("d_ff", 1408, "Feed-forward intermediate dimension per expert.")


class C4Dataset(IterableDataset):
    """Streams C4 dataset from HuggingFace datasets."""

    def __init__(self, seq_len: int, tokenizer: AutoTokenizer, split: str = "train"):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset("allenai/c4", "en", split=split, streaming=True, trust_remote_code=True)

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            tokens = self.tokenizer(example["text"], add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]


def build_optimizer(
    model: MixtralForCausalLM,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build optimizer with proper param groups.

    Muon and SOAP require 2D params. Embeddings, biases, norms, and the
    LM head are handled by AdamW.
    """
    params_2d = []
    params_other = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and "embed" not in name and "lm_head" not in name:
            params_2d.append(p)
        else:
            params_other.append(p)

    if optimizer_name == "muon":
        muon_opt = Muon(params_2d, lr=lr, momentum=0.95, weight_decay=weight_decay)
        adam_opt = torch.optim.AdamW(params_other, lr=lr * 0.1, weight_decay=weight_decay)
        return _CombinedOptimizer([muon_opt, adam_opt])

    elif optimizer_name == "soap":
        soap_opt = SOAP(params_2d, lr=lr, weight_decay=weight_decay, precondition_frequency=10)
        adam_opt = torch.optim.AdamW(params_other, lr=lr, weight_decay=weight_decay)
        return _CombinedOptimizer([soap_opt, adam_opt])

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


class _CombinedOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers so they can be used as one."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        self.defaults = optimizers[0].defaults
        self.state: dict = {}  # type: ignore[assignment]
        self.param_groups: list[dict] = []  # type: ignore[assignment]
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)


def get_cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int = 100) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(argv: list[str]) -> None:
    """Main training function."""
    del argv

    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    config = MixtralConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=FLAGS.d_model,
        intermediate_size=FLAGS.d_ff,
        num_hidden_layers=FLAGS.n_layers,
        num_attention_heads=FLAGS.n_heads,
        num_key_value_heads=FLAGS.n_kv_heads,
        num_local_experts=FLAGS.n_experts,
        num_experts_per_tok=FLAGS.top_k,
        max_position_embeddings=FLAGS.seq_len,
        router_aux_loss_coef=0.01,
        output_router_logits=True,
        tie_word_embeddings=True,
    )
    model = MixtralForCausalLM(config).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    logging.info("Model: %.1fM total params", total_params / 1e6)

    optimizer = build_optimizer(model, FLAGS.optimizer, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    logging.info("Optimizer: %s, lr=%s", FLAGS.optimizer, FLAGS.lr)

    dataset = C4Dataset(seq_len=FLAGS.seq_len, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=2, pin_memory=True)

    model.train()
    data_iter = iter(dataloader)

    logging.info("Starting training...")
    for step in range(FLAGS.max_steps):
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.cuda()
        labels = labels.cuda()

        current_lr = get_cosine_lr(step, FLAGS.max_steps, FLAGS.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(input_ids=input_ids, labels=labels)

        optimizer.zero_grad()
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lm_loss = float("inf")
        if step % FLAGS.log_interval == 0 or step == FLAGS.max_steps - 1:
            lm_loss = outputs.loss.item()
            ppl = math.exp(min(lm_loss, 20.0))
            logging.info("step=%5d | loss=%.4f | ppl=%.2f | lr=%.6f", step, lm_loss, ppl, current_lr)

    if lm_loss > FLAGS.loss_target:
        raise ValueError(f"Loss target {FLAGS.loss_target} not reached. Final loss: {lm_loss:.4f}")


if __name__ == "__main__":
    app.run(train)
