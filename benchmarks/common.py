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

"""Shared fixtures for Qwen3-30B-A3B benchmark scripts.

Qwen3-30B-A3B is an MoE model (128 experts, 8 active) with hidden_size=2048,
head_dim=128, 16 attention heads, 4 KV heads, intermediate_size=2048, 48 layers.

Linear layers per transformer block:
    Attention:
        q_proj   (2048, 2048)  x1
        k_proj   (512,  2048)  x1
        v_proj   (512,  2048)  x1
        o_proj   (2048, 2048)  x1
    MoE (per expert, 128 experts):
        gate_proj (2048, 2048) x128
        up_proj   (2048, 2048) x128
        down_proj (2048, 2048) x128
    Router:
        gate      (128,  2048) x1

By default benchmarks use a single-layer slice with all unique shapes and
correct multiplicity. Use --num_layers to scale up.
"""

from collections.abc import Iterator
from typing import TypeAlias

import torch
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_layers", 1, "Number of transformer layers to simulate.")
flags.DEFINE_integer("num_experts", 128, "Number of MoE experts per layer.")
flags.DEFINE_integer("warmup_steps", 1, "Warmup iterations before timing.")
flags.DEFINE_integer("benchmark_steps", 5, "Benchmark iterations to time.")

LinearSpec: TypeAlias = tuple[str, tuple[int, int]]
ShapeCounts: TypeAlias = dict[LinearSpec, int]

# Qwen3-30B-A3B linear layer shapes (out_features, in_features) per transformer block.
QWEN3_30B_A3B_ATTN_SHAPES: list[LinearSpec] = [
    ("q_proj", (2048, 2048)),
    ("k_proj", (512, 2048)),
    ("v_proj", (512, 2048)),
    ("o_proj", (2048, 2048)),
]

QWEN3_30B_A3B_EXPERT_SHAPES: list[LinearSpec] = [
    ("gate_proj", (2048, 2048)),
    ("up_proj", (2048, 2048)),
    ("down_proj", (2048, 2048)),
]

QWEN3_30B_A3B_ROUTER_SHAPE: LinearSpec = ("router", (128, 2048))


def iter_qwen3_30b_a3b_linear_specs(num_layers: int, num_experts: int) -> Iterator[LinearSpec]:
    """Yield Qwen3-30B-A3B linear layer names and shapes with multiplicity."""
    for _ in range(num_layers):
        yield from QWEN3_30B_A3B_ATTN_SHAPES

        for _ in range(num_experts):
            yield from QWEN3_30B_A3B_EXPERT_SHAPES

        yield QWEN3_30B_A3B_ROUTER_SHAPE


def qwen3_30b_a3b_shape_counts(num_layers: int, num_experts: int) -> ShapeCounts:
    """Return Qwen3-30B-A3B linear layer shape counts."""
    shape_counts: ShapeCounts = {}
    for spec in iter_qwen3_30b_a3b_linear_specs(num_layers, num_experts):
        shape_counts[spec] = shape_counts.get(spec, 0) + 1
    return shape_counts


def build_qwen3_30b_a3b_params(
    num_layers: int,
    num_experts: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.nn.Parameter], ShapeCounts]:
    """Create synthetic parameters matching Qwen3-30B-A3B linear layers."""
    params: list[torch.nn.Parameter] = []
    shape_counts: ShapeCounts = {}

    for spec in iter_qwen3_30b_a3b_linear_specs(num_layers, num_experts):
        name, shape = spec
        params.append(torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype)))
        shape_counts[(name, shape)] = shape_counts.get((name, shape), 0) + 1

    return params, shape_counts


def count_shape_elements(shape_counts: ShapeCounts) -> int:
    """Count total tensor elements represented by shape_counts."""
    return sum(shape[0] * shape[1] * count for (_, shape), count in shape_counts.items())


def print_shape_breakdown(shape_counts: ShapeCounts) -> None:
    """Print a compact table of layer names, shapes, and counts."""
    print("\nShape breakdown:")
    rows = [(name, str(shape), str(count)) for (name, shape), count in shape_counts.items()]
    w_name = max(len(r[0]) for r in rows)
    w_shape = max(len(r[1]) for r in rows)
    w_count = max(len(r[2]) for r in rows)
    w_name, w_shape, w_count = max(w_name, 4), max(w_shape, 5), max(w_count, 5)
    sep = f"  +-{'-' * w_name}-+-{'-' * w_shape}-+-{'-' * w_count}-+"
    print(sep)
    print(f"  | {'Name':<{w_name}} | {'Shape':<{w_shape}} | {'Count':>{w_count}} |")
    print(sep)
    for name, shape, count in rows:
        print(f"  | {name:<{w_name}} | {shape:>{w_shape}} | {count:>{w_count}} |")
    print(sep)
