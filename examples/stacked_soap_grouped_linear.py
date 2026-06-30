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
import os


os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"

import torch
import transformer_engine.pytorch as te
from absl import app, flags

from emerging_optimizers.soap.soap import StackedSoap


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_experts", 8, "Number of experts (grouped GEMMs).")
flags.DEFINE_integer("in_features", 512, "Input feature dimension per expert.")
flags.DEFINE_integer("out_features", 1024, "Output feature dimension per expert.")
flags.DEFINE_integer("tokens", 256, "Total number of tokens routed across experts.")
flags.DEFINE_integer("steps", 5, "Number of optimization steps.")
flags.DEFINE_float("lr", 1e-3, "Learning rate.")


def main(argv: list[str]) -> None:
    """Build a Transformer Engine GroupedLinear with a single 3D weight and train it with StackedSoap."""
    del argv
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA device (Transformer Engine is GPU-only).")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    grouped_linear = te.GroupedLinear(
        FLAGS.num_experts,
        FLAGS.in_features,
        FLAGS.out_features,
        bias=False,
        single_grouped_weight=True,
        params_dtype=dtype,
        device=device,
    )

    weight = grouped_linear.weight
    print(f"Single expert weight tensor: shape={tuple(weight.shape)}, dtype={weight.dtype}")

    optimizer = StackedSoap(grouped_linear.parameters(), lr=FLAGS.lr, weight_decay=0.0)

    # MoE routing: split `tokens` rows across experts; m_splits must sum to the token count.
    base = FLAGS.tokens // FLAGS.num_experts
    m_splits = [base] * FLAGS.num_experts
    m_splits[-1] += FLAGS.tokens - sum(m_splits)

    for step in range(FLAGS.steps):
        x = torch.randn(FLAGS.tokens, FLAGS.in_features, device=device, dtype=dtype, requires_grad=True)
        out = grouped_linear(x, m_splits)
        loss = out.float().square().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step}: loss={loss.item():.6f}")


if __name__ == "__main__":
    app.run(main)
