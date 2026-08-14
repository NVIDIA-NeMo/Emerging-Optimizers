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
"""Benchmark batched_tsyrk_ex vs torch.baddbmm/bmm: latency and precision."""

from collections.abc import Callable
from typing import Any

import torch
import triton

from emerging_optimizers.triton_kernels.batched_syrk import batched_tsyrk_ex
from emerging_optimizers.triton_kernels.syrk import tsyrk_ex_small_matrix


def max_err(x: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return (max abs err, max rel err) of x against ref."""
    ref = ref.float()
    assert ref.abs().amax() > 0.0, "reference is all zeros"
    diff = (x.float() - ref).abs()
    return diff.max().item(), (diff / ref.abs()).max().item()


def bench(fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> float:
    """Time fn with triton.testing.do_bench."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def run_case(
    B: int,
    N: int,
    K: int,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans: bool = False,
    device: str = "cuda",
) -> None:
    """Benchmark one (B, N, K) case: correctness and latency vs torch in bf16."""
    torch.manual_seed(0)
    if trans:
        a = torch.randn(B, K, N, device=device, dtype=torch.bfloat16).mT
    else:
        a = torch.randn(B, N, K, device=device, dtype=torch.bfloat16)
    c: torch.Tensor | None = None
    if beta != 0.0:
        s = torch.randn(B, N, N, device=device, dtype=torch.bfloat16)
        c = ((s + s.mT) * 0.5).contiguous()

    # --- correctness ---
    d_triton = batched_tsyrk_ex(a, c, alpha=alpha, beta=beta)
    if c is not None:
        ref = torch.baddbmm(c, a, a.mT, beta=beta, alpha=alpha)
    else:
        ref = torch.bmm(a, a.mT) if alpha == 1.0 else alpha * torch.bmm(a, a.mT)

    tri_abs, tri_rel = max_err(d_triton, ref)
    sym = (d_triton - d_triton.mT).abs().max().item()

    # --- timing ---
    if c is not None:
        c_bench = c
        t_torch = bench(lambda: torch.baddbmm(c_bench, a, a.mT, beta=beta, alpha=alpha))
    else:
        t_torch = bench(lambda: torch.bmm(a, a.mT))
    t_batched = bench(lambda: batched_tsyrk_ex(a, c, alpha=alpha, beta=beta))

    # loop over batch with the single-matrix kernel (what batching replaces)
    a2d = [a[i] for i in range(B)]
    c2d: list[torch.Tensor | None] = [c[i] for i in range(B)] if c is not None else [None] * B

    def loop_fn() -> None:
        for i in range(B):
            tsyrk_ex_small_matrix(a2d[i], c2d[i], alpha=alpha, beta=beta)

    loop_fn()  # trigger autotune outside timing
    t_loop = bench(loop_fn)

    full_tflops = 2 * B * N * N * K / 1e12
    tag = f"B={B:<3d} N={N:<5d} K={K:<5d} trans={int(trans)} beta={beta}"
    print(
        f"{tag} | baddbmm/bmm {t_torch:7.3f} ms ({full_tflops / t_torch * 1e3:6.1f} TF) | "
        f"batched_syrk {t_batched:7.3f} ms ({full_tflops / t_batched * 1e3:6.1f} TF-eq) | "
        f"loop_syrk {t_loop:7.3f} ms | speedup vs torch {t_torch / t_batched:5.2f}x, vs loop {t_loop / t_batched:5.2f}x"
    )
    print(f"{'':>4}err vs torch bf16: triton abs={tri_abs:.4f} rel={tri_rel:.2e} | triton symmetry err={sym}")


def main() -> None:
    """Run the benchmark suite."""
    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}")

    cases = [
        (8, 2048, 2048),
        (4, 4096, 4096),
        (32, 512, 2048),
        (64, 256, 1024),
        (16, 2048, 8192),
    ]
    print("\n=== alpha=1.0, beta=0.0, contiguous a ===")
    for B, N, K in cases:
        run_case(B, N, K)


if __name__ == "__main__":
    main()
