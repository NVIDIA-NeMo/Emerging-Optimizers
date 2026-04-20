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
import time
from collections.abc import Iterator
from typing import Any, Callable, override
from xml.etree import ElementTree as ET

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
flags.DEFINE_float("loss_target", 8.0, "Loss target.")

# Model hyperparameters
flags.DEFINE_integer("d_model", 512, "Model dimension.")
flags.DEFINE_integer("n_layers", 8, "Number of transformer layers.")
flags.DEFINE_integer("n_heads", 8, "Number of attention heads.")
flags.DEFINE_integer("n_kv_heads", 4, "Number of key-value heads (GQA).")
flags.DEFINE_integer("n_experts", 8, "Number of experts per MoE layer.")
flags.DEFINE_integer("top_k", 2, "Number of active experts per token.")
flags.DEFINE_integer("d_ff", 1408, "Feed-forward intermediate dimension per expert.")
flags.DEFINE_string("xml_output_file", None, "Path to write JUnit XML report.")


class C4Dataset(IterableDataset):
    """Streams C4 dataset from HuggingFace datasets."""

    def __init__(self, seq_len: int, tokenizer: AutoTokenizer, split: str = "train"):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset("allenai/c4", "en", split=split, streaming=True)

    @override
    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer: list[int] = []
        for example in self.dataset:
            tokens = self.tokenizer(example["text"], add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                yield torch.tensor(chunk, dtype=torch.long)


class _CombinedOptimizer:
    """Wraps multiple optimizers so they can be used as one."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        self.param_groups: list[dict] = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None) -> None:
        for opt in self.optimizers:
            opt.step(closure)


def build_optimizer(
    model: MixtralForCausalLM,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer | _CombinedOptimizer:
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

    common_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
    }
    if optimizer_name == "muon":
        muon_opt = Muon(params_2d, **common_kwargs)
        adam_opt = torch.optim.AdamW(params_other, **common_kwargs)
        return _CombinedOptimizer([muon_opt, adam_opt])

    elif optimizer_name == "soap":
        soap_opt = SOAP(params_2d, **common_kwargs)
        adam_opt = torch.optim.AdamW(params_other, **common_kwargs)
        return _CombinedOptimizer([soap_opt, adam_opt])

    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common_kwargs)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int = 100) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(_: Any) -> None:
    """Main training function."""
    start_time = time.monotonic()
    torch.manual_seed(FLAGS.seed)
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
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size)

    model.train()
    data_iter = iter(dataloader)

    logging.info("Starting training...")
    lm_loss = float("inf")
    for step in range(FLAGS.max_steps):
        try:
            tokens = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            tokens = next(data_iter)

        tokens = tokens.cuda()

        current_lr = get_cosine_lr(step, FLAGS.max_steps, FLAGS.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(input_ids=tokens, labels=tokens)

        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % FLAGS.log_interval == 0 or step == FLAGS.max_steps - 1:
            lm_loss = outputs.loss.item()
            ppl = math.exp(min(lm_loss, 20.0))
            logging.info("step=%5d | loss=%.4f | ppl=%.2f | lr=%.6f", step, lm_loss, ppl, current_lr)

    passed = lm_loss <= FLAGS.loss_target
    failure_msg = None if passed else f"Loss target {FLAGS.loss_target} not reached. Final loss: {lm_loss:.4f}"

    if FLAGS.xml_output_file is not None:
        _write_junit_xml(
            test_name=f"convergence_{FLAGS.optimizer}",
            class_name="moe_c4_convergence",
            elapsed=time.monotonic() - start_time,
            failure_msg=failure_msg,
            output_file=FLAGS.xml_output_file,
        )

    if not passed:
        raise ValueError(failure_msg)


def _write_junit_xml(
    test_name: str,
    class_name: str,
    elapsed: float,
    failure_msg: str | None,
    output_file: str,
) -> None:
    """Write a single-testcase JUnit XML report."""
    import os

    testsuite = ET.Element(
        "testsuite",
        name=class_name,
        tests="1",
        failures="0" if failure_msg is None else "1",
        errors="0",
        time=f"{elapsed:.2f}",
    )
    testcase = ET.SubElement(
        testsuite,
        "testcase",
        name=test_name,
        classname=class_name,
        time=f"{elapsed:.2f}",
    )
    if failure_msg is not None:
        failure = ET.SubElement(testcase, "failure", message=failure_msg)
        failure.text = failure_msg

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tree = ET.ElementTree(testsuite)
    ET.indent(tree)
    tree.write(output_file, xml_declaration=True, encoding="utf-8")
    logging.info("JUnit XML report written to %s", output_file)


if __name__ == "__main__":
    app.run(train)
