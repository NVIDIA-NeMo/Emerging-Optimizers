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

"""Benchmark the SOAP optimizer using linear-layer shapes from Qwen3-30B-A3B.

Qwen3-30B-A3B is an MoE model (128 experts, 8 active) with hidden_size=2048,
head_dim=128, 16 attention heads, 4 KV heads, intermediate_size=2048, 48 layers.

Linear layers per transformer block:
    Attention:
        q_proj   (2048, 2048)  ×1
        k_proj   (512,  2048)  ×1
        v_proj   (512,  2048)  ×1
        o_proj   (2048, 2048)  ×1
    MoE (per expert, 128 experts):
        gate_proj (2048, 2048) ×128
        up_proj   (2048, 2048) ×128
        down_proj (2048, 2048) ×128
    Router:
        gate      (128,  2048) ×1

By default the benchmark uses a single-layer slice (all unique shapes with
correct multiplicity).  Use --num_layers to scale up.
"""

import time
from typing import Any

import torch
from absl import app, flags

from emerging_optimizers.soap.soap import SOAP


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_layers", 1, "Number of transformer layers to simulate.")
flags.DEFINE_integer("num_experts", 128, "Number of MoE experts per layer.")
flags.DEFINE_integer("warmup_steps", 1, "Optimizer warmup steps before timing.")
flags.DEFINE_integer("benchmark_steps", 5, "Optimizer steps to time.")
flags.DEFINE_bool("use_eigh", False, "Use eigh instead of QR iteration.")
flags.DEFINE_string("dtype", "bfloat16", "Parameter dtype: float32, bfloat16, or float16.")
flags.DEFINE_integer("num_streams", None, "Number of CUDA streams for SOAP. None means no stream_list.")
flags.DEFINE_bool("cpu_offload", False, "Round-trip optimizer state to pinned CPU memory each step.")

# Qwen3-30B-A3B linear layer shapes (out_features, in_features) per transformer block.
QWEN3_30B_A3B_ATTN_SHAPES: list[tuple[str, tuple[int, int]]] = [
    ("q_proj", (2048, 2048)),
    ("k_proj", (512, 2048)),
    ("v_proj", (512, 2048)),
    ("o_proj", (2048, 2048)),
]

QWEN3_30B_A3B_EXPERT_SHAPES: list[tuple[str, tuple[int, int]]] = [
    ("gate_proj", (2048, 2048)),
    ("up_proj", (2048, 2048)),
    ("down_proj", (2048, 2048)),
]

QWEN3_30B_A3B_ROUTER_SHAPE: tuple[str, tuple[int, int]] = ("router", (128, 2048))


def build_params(
    num_layers: int,
    num_experts: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.nn.Parameter], dict[str, int]]:
    """Create synthetic parameters matching Qwen3-30B-A3B linear layers."""
    params: list[torch.nn.Parameter] = []
    shape_counts: dict[str, int] = {}

    for _ in range(num_layers):
        # Attention projections
        for name, shape in QWEN3_30B_A3B_ATTN_SHAPES:
            p = torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype))
            params.append(p)
            shape_counts[f"{name} {shape}"] = shape_counts.get(f"{name} {shape}", 0) + 1

        # MoE expert projections
        for _ in range(num_experts):
            for name, shape in QWEN3_30B_A3B_EXPERT_SHAPES:
                p = torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype))
                params.append(p)
                shape_counts[f"{name} {shape}"] = shape_counts.get(f"{name} {shape}", 0) + 1

        # Router
        name, shape = QWEN3_30B_A3B_ROUTER_SHAPE
        p = torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype))
        params.append(p)
        shape_counts[f"{name} {shape}"] = shape_counts.get(f"{name} {shape}", 0) + 1

    return params, shape_counts


def main(_: Any) -> None:
    """Run the SOAP benchmark."""
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    device = torch.device("cuda")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[FLAGS.dtype]

    num_layers = FLAGS.num_layers
    num_experts = FLAGS.num_experts

    print("=" * 70)
    print("SOAP Optimizer Benchmark — Qwen3-30B-A3B linear layer shapes")
    print("=" * 70)

    params, shape_counts = build_params(num_layers, num_experts, device, dtype)
    for p in params:
        p.grad = torch.randn_like(p)

    total_params = sum(p.numel() for p in params)
    print("\nModel configuration:")
    print(f"  Layers           : {num_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Total parameters : {total_params:,} ({total_params * dtype.itemsize / 1e9:.2f} GB)")
    print(f"  Num tensors      : {len(params)}")
    print(f"  Param dtype      : {dtype}")
    print("\nShape breakdown:")
    rows = [(d[: d.index("(")].rstrip(), d[d.index("(") :], str(count)) for d, count in shape_counts.items()]
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

    stream_list = [torch.cuda.Stream() for _ in range(FLAGS.num_streams)] if FLAGS.num_streams else None

    cpu_states_buffer: torch.Tensor | None = None
    offload_stream: torch.cuda.Stream | None = None
    required_numel = 0
    if FLAGS.cpu_offload:
        required_numel = sum(
            2 * p.shape[0] * p.shape[1] + 2 * p.shape[0] ** 2 + 2 * p.shape[1] ** 2 for p in params if p.dim() == 2
        )
        try:
            cpu_states_buffer = torch.empty(required_numel, dtype=torch.float32, pin_memory=True)
        except torch.AcceleratorError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"\nPinning {required_numel * 4 / 1e9:.2f} GB of host memory failed ({e});")
            print("  falling back to pageable (pin_memory=False). Copies will be blocking.")
            cpu_states_buffer = torch.empty(required_numel, dtype=torch.float32, pin_memory=False)
        offload_stream = torch.cuda.Stream()

    optimizer = SOAP(
        params,
        lr=0.25,
        use_eigh=FLAGS.use_eigh,
        stream_list=stream_list,
        cpu_states_buffer=cpu_states_buffer,
    )

    print("\nSOAP settings:")
    print(f"  use_eigh    : {FLAGS.use_eigh}")
    print(f"  num_streams : {FLAGS.num_streams}")
    print(f"  cpu_offload : {FLAGS.cpu_offload}")
    if FLAGS.cpu_offload:
        print(f"  CPU buffer  : {required_numel:,} fp32 elements ({required_numel * 4 / 1e9:.2f} GB pinned)")

    print(f"\nWarming up ({FLAGS.warmup_steps} steps)...")
    for _ in range(FLAGS.warmup_steps):
        optimizer.step()
    torch.cuda.synchronize()

    benchmark_steps = FLAGS.benchmark_steps
    print(f"Benchmarking ({benchmark_steps} steps)...")

    step_times: list[float] = []
    reload_times: list[float] = []
    mem_after_offload_gb: float | None = None
    for _ in range(benchmark_steps):
        if FLAGS.cpu_offload:
            t0 = time.perf_counter()
            reload_event = optimizer.move_states_to_gpu(stream=offload_stream)
            torch.cuda.current_stream().wait_event(reload_event)
            reload_times.append((time.perf_counter() - t0) * 1000)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()

        if FLAGS.cpu_offload:
            assert offload_stream is not None  # Help mypy understand that offload_stream is a Stream after this.
            offload_stream.wait_stream(torch.cuda.current_stream())
            offload_event = optimizer.move_states_to_cpu(stream=offload_stream)
            torch.cuda.current_stream().wait_event(offload_event)

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)

        if FLAGS.cpu_offload:
            mem_after_offload_gb = torch.cuda.memory_allocated(device) / (1024**3)

    avg_ms = sum(step_times) / len(step_times)
    min_ms = min(step_times)
    max_ms = max(step_times)
    median_ms = sorted(step_times)[len(step_times) // 2]

    print(f"\n{'Results':=^70}")
    print(f"  Steps          : {benchmark_steps}")
    print(f"  Avg step time  : {avg_ms:.2f} ms")
    print(f"  Median         : {median_ms:.2f} ms")
    print(f"  Min            : {min_ms:.2f} ms")
    print(f"  Max            : {max_ms:.2f} ms")

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"  Peak GPU mem   : {peak_mem:.2f} GB")

    if FLAGS.cpu_offload:
        print(f"  Avg reload ms  : {sum(reload_times) / len(reload_times):.2f}")
        if mem_after_offload_gb is not None:
            print(f"  GPU mem after offload: {mem_after_offload_gb:.2f} GB")

    print("=" * 70)


if __name__ == "__main__":
    app.run(main)
