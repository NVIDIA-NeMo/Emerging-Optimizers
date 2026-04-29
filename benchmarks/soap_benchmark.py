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

"""Benchmark the SOAP optimizer using linear-layer shapes from Qwen3-30B-A3B."""

import time
from typing import Any

import torch
from absl import app, flags
from common import FLAGS, build_qwen3_30b_a3b_params, print_shape_breakdown

from emerging_optimizers.soap.soap import SOAP


flags.DEFINE_bool("use_eigh", False, "Use eigh instead of QR iteration.")
flags.DEFINE_string("dtype", "bfloat16", "Parameter dtype: float32, bfloat16, or float16.")
flags.DEFINE_integer("num_streams", None, "Number of CUDA streams for SOAP. None means no stream_list.")


def main(_: Any) -> None:
    """Run the SOAP benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[FLAGS.dtype]

    num_layers = FLAGS.num_layers
    num_experts = FLAGS.num_experts

    print("=" * 70)
    print("SOAP Optimizer Benchmark — Qwen3-30B-A3B linear layer shapes")
    print("=" * 70)

    params, shape_counts = build_qwen3_30b_a3b_params(num_layers, num_experts, device, dtype)
    for p in params:
        p.grad = torch.randn_like(p)

    total_params = sum(p.numel() for p in params)
    print("\nModel configuration:")
    print(f"  Layers           : {num_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Total parameters : {total_params:,} ({total_params * dtype.itemsize / 1024**3:.2f} GiB)")
    print(f"  Num tensors      : {len(params)}")
    print(f"  Param dtype      : {dtype}")
    print_shape_breakdown(shape_counts)

    stream_list = [torch.cuda.Stream() for _ in range(FLAGS.num_streams)] if FLAGS.num_streams else None

    optimizer = SOAP(
        params,
        lr=0.25,
        use_eigh=FLAGS.use_eigh,
        stream_list=stream_list,
    )

    print("\nSOAP settings:")
    print(f"  use_eigh    : {FLAGS.use_eigh}")
    print(f"  num_streams : {FLAGS.num_streams}")

    print(f"\nWarming up ({FLAGS.warmup_steps} steps)...")
    for _ in range(FLAGS.warmup_steps):
        optimizer.step()
    torch.cuda.synchronize()

    benchmark_steps = FLAGS.benchmark_steps
    print(f"Benchmarking ({benchmark_steps} steps)...")

    step_times: list[float] = []
    for _ in range(benchmark_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        step_times.append(elapsed_ms)

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

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  Peak GPU mem   : {peak_mem:.2f} GiB")
    print("=" * 70)


if __name__ == "__main__":
    app.run(main)
