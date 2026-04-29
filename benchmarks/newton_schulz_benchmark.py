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

"""Benchmark Newton-Schulz iteration on a synthetic matrix."""

import os
import time
from typing import Any

import torch
from absl import app, flags

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
from emerging_optimizers.utils import fp32_matmul_precision


FLAGS = flags.FLAGS

flags.DEFINE_integer("m", 2048, "Number of matrix rows.")
flags.DEFINE_integer("n", 2048, "Number of matrix columns.")
flags.DEFINE_integer("warmup_steps", 1, "Warmup iterations before timing.")
flags.DEFINE_integer("benchmark_steps", 5, "Benchmark iterations to time.")
flags.DEFINE_integer("ns_steps", 5, "Newton-Schulz iterations per pass.")
flags.DEFINE_enum("matmul_precision", "medium", ["highest", "high", "medium"], "Float32 matmul precision.")
flags.DEFINE_enum("device", "cuda", ["cpu", "cuda"], "Device to run on.")
flags.DEFINE_enum(
    "partition_dim",
    None,
    ["0", "1"],
    "Tensor dimension to shard. Required when WORLD_SIZE > 1, must be unset otherwise.",
)
flags.DEFINE_enum("tp_mode", "distributed", ["distributed", "duplicated"], "Tensor-parallel mode.")
flags.DEFINE_enum(
    "backend",
    "emerging",
    ["emerging", "te"],
    "Newton-Schulz implementation: 'emerging' (this repo) or 'te' (TransformerEngine cuSolverMp).",
)


def main(_: Any) -> None:
    """Run benchmark."""
    if FLAGS.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when --device=cuda.")
        device_id = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(device_id)
        device = torch.device("cuda", device_id)
    else:
        device = torch.device("cpu")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    partition_dim = int(FLAGS.partition_dim) if FLAGS.partition_dim is not None else None
    if world_size > 1:
        if partition_dim is None:
            raise ValueError(f"--partition_dim is required when WORLD_SIZE={world_size}.")
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        tp_group = torch.distributed.group.WORLD
    else:
        if partition_dim is not None:
            raise ValueError(f"--partition_dim={partition_dim} requires WORLD_SIZE > 1.")
        tp_group = None

    tp_size = world_size
    is_rank0 = tp_group is None or tp_group.rank() == 0

    matrix_shape = (FLAGS.m, FLAGS.n)
    local_shape = list(matrix_shape)
    if tp_group is not None:
        if matrix_shape[partition_dim] % tp_size != 0:
            raise ValueError(f"Shape {matrix_shape} is not divisible by tp_size={tp_size} on dim {partition_dim}.")
        local_shape[partition_dim] //= tp_size
    x = torch.randn(local_shape, device=device, dtype=torch.float32)
    # Save the initial random state so every benchmark pass starts from the same input.
    # Required for the te backend (in-place orthogonalization); harmless for emerging.
    x_init = x.clone()

    te_ctx = None
    te_newton_schulz = None
    if FLAGS.backend == "te":
        if device.type != "cuda":
            raise ValueError("--backend=te requires --device=cuda.")
        if tp_group is None:
            raise ValueError("--backend=te requires WORLD_SIZE > 1.")
        if partition_dim != 1:
            raise ValueError("--backend=te requires --partition_dim=1 (column-distributed). UNVERIFIED.")
        from transformer_engine.pytorch.newton_schulz import CusolverMpCtx
        from transformer_engine.pytorch.newton_schulz import newton_schulz as te_newton_schulz

        te_ctx = CusolverMpCtx(tp_group)

    def run() -> None:
        if te_ctx is not None:
            te_newton_schulz(x, te_ctx, num_iterations=FLAGS.ns_steps)
        else:
            newton_schulz_tp(
                x,
                steps=FLAGS.ns_steps,
                coefficient_type="quintic",
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode=FLAGS.tp_mode,
            )

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if tp_group is not None:
            torch.distributed.barrier(group=tp_group)

    if is_rank0:
        print("=" * 70)
        print("Newton-Schulz Benchmark")
        print("=" * 70)
        print(f"  Matrix shape     : {matrix_shape}")
        print(f"  Local shape      : {tuple(x.shape)}")
        print(f"  Device           : {device}")
        print(f"  ns_steps         : {FLAGS.ns_steps}")
        print(f"  matmul_precision : {FLAGS.matmul_precision}")
        print(f"  tp_size          : {tp_size}")
        print(f"  partition_dim    : {partition_dim}")
        print(f"  tp_mode          : {FLAGS.tp_mode}")
        print(f"  backend          : {FLAGS.backend}")

    step_times: list[float] = []
    with fp32_matmul_precision(FLAGS.matmul_precision), torch.no_grad():
        for _ in range(FLAGS.warmup_steps):
            x.copy_(x_init)
            run()
        sync()

        for _ in range(FLAGS.benchmark_steps):
            x.copy_(x_init)
            sync()
            t0 = time.perf_counter()
            run()
            sync()
            step_times.append((time.perf_counter() - t0) * 1000)

    if is_rank0:
        avg_ms = sum(step_times) / len(step_times)
        median_ms = sorted(step_times)[len(step_times) // 2]
        print(f"\n{'Results':=^70}")
        print(f"  Passes : {FLAGS.benchmark_steps}")
        print(f"  Avg    : {avg_ms:.2f} ms")
        print(f"  Median : {median_ms:.2f} ms")
        print(f"  Min    : {min(step_times):.2f} ms")
        print(f"  Max    : {max(step_times):.2f} ms")
        print("=" * 70)

    if te_ctx is not None:
        te_ctx.destroy()
    if tp_group is not None:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
