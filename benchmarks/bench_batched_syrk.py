# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# type: ignore
"""Benchmark batched_tsyrk_ex vs torch.baddbmm/bmm: latency and precision."""

import torch
import triton

from emerging_optimizers.triton_kernels.batched_syrk import batched_tsyrk_ex
from emerging_optimizers.triton_kernels.syrk import tsyrk_ex_small_matrix


def ref_fp32(a, c, alpha, beta):
    """Compute the fp32 reference of alpha * a @ a.mT + beta * c."""
    out = alpha * (a.float() @ a.mT.float())
    if beta != 0.0:
        out += beta * c.float()
    return out


def max_err(x, ref):
    """Return (max abs err, max rel err) of x against a fp32 reference."""
    diff = (x.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-6)
    return diff.max().item(), (diff / denom).max().item()


def bench(fn, warmup=25, rep=100):
    """Time fn with triton.testing.do_bench."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def run_case(B, N, K, alpha=1.0, beta=0.0, trans=False, device="cuda"):
    """Benchmark one (B, N, K) case: correctness vs fp32 and latency vs torch."""
    torch.manual_seed(0)
    if trans:
        a = torch.randn(B, K, N, device=device, dtype=torch.bfloat16).mT
    else:
        a = torch.randn(B, N, K, device=device, dtype=torch.bfloat16)
    c = None
    if beta != 0.0:
        s = torch.randn(B, N, N, device=device, dtype=torch.bfloat16)
        c = ((s + s.mT) * 0.5).contiguous()

    ref = ref_fp32(a, c, alpha, beta)

    # --- correctness ---
    d_triton = batched_tsyrk_ex(a, c, alpha=alpha, beta=beta)
    if beta != 0.0:
        d_torch = torch.baddbmm(c, a, a.mT, beta=beta, alpha=alpha)
    else:
        d_torch = torch.bmm(a, a.mT) if alpha == 1.0 else alpha * torch.bmm(a, a.mT)

    tri_abs, tri_rel = max_err(d_triton, ref)
    pt_abs, pt_rel = max_err(d_torch, ref)
    sym = (d_triton - d_triton.mT).abs().max().item()

    # --- timing ---
    if beta != 0.0:
        t_torch = bench(lambda: torch.baddbmm(c, a, a.mT, beta=beta, alpha=alpha))
    else:
        t_torch = bench(lambda: torch.bmm(a, a.mT))
    t_batched = bench(lambda: batched_tsyrk_ex(a, c, alpha=alpha, beta=beta))

    # loop over batch with the single-matrix kernel (what batching replaces)
    a2d = [a[i] for i in range(B)]
    c2d = [c[i] for i in range(B)] if c is not None else [None] * B

    def loop_fn():
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
    print(
        f"{'':>4}err vs fp32: triton abs={tri_abs:.4f} rel={tri_rel:.2e} | "
        f"torch abs={pt_abs:.4f} rel={pt_rel:.2e} | triton symmetry err={sym}"
    )


def main():
    """Run the benchmark suite."""
    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}")

    cases = [
        # Muon-style grouped Gram matrices
        (8, 512, 512),
        (8, 1024, 1024),
        (8, 2048, 2048),
        (4, 4096, 4096),
        # fat K (grad of wide layers)
        (32, 512, 2048),
        (64, 256, 1024),
        # MoE expert stacks
        (64, 1408, 2048),
        (16, 2048, 7168),
    ]
    print("\n=== alpha=1.0, beta=0.0, contiguous a ===")
    for B, N, K in cases:
        run_case(B, N, K)

    print("\n=== transposed input (a.mT contiguous) ===")
    for B, N, K in [(8, 1024, 1024), (32, 512, 2048)]:
        run_case(B, N, K, trans=True)

    print("\n=== beta=0.5 (baddbmm accumulate path) ===")
    for B, N, K in [(8, 1024, 1024), (64, 1408, 2048)]:
        run_case(B, N, K, alpha=1.0, beta=0.5)

    # skip_upper_triangle sanity
    a = torch.randn(4, 512, 768, device="cuda", dtype=torch.bfloat16)
    d = batched_tsyrk_ex(a, skip_upper_triangle=True)
    ref = torch.bmm(a.float(), a.mT.float())
    tril_err = (torch.tril(d.float()) - torch.tril(ref)).abs().max().item()
    print(f"\nskip_upper_triangle lower-tri max abs err vs fp32: {tril_err:.4f}")


if __name__ == "__main__":
    main()
