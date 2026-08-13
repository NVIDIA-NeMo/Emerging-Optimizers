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
"""Batched bf16 syrk Triton kernel, a drop-in replacement for ``torch.baddbmm(c, a, a.mT)``."""

import sys

import torch
import triton
import triton.language as tl
from absl import logging


try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    HAS_TRITON_340 = True
except ImportError:
    HAS_TRITON_340 = False


__all__ = ["batched_tsyrk_ex", "can_use_batched_tsyrk"]


# Hopper (sm90) has 228 KB of shared memory per SM. Leave headroom for the TMA barriers/descriptors.
_SM90_SMEM_BUDGET = 224 * 1024


def prune_invalid_batched_configs(configs: list[triton.Config], named_args: dict, **kwargs) -> list[triton.Config]:
    """Prune batched syrk configs that cannot run or are known to be slow on Hopper.

    Args:
        configs: List of Triton kernel configs.
        named_args: Named arguments for the kernel.
        **kwargs: Additional keyword arguments.

    Returns:
        List of valid Triton kernel configs.
    """
    N = named_args["N"]
    K = named_args["K"]

    pruned_config = []
    for c in configs:
        TILE_M = c.kwargs.get("TILE_M", 0)
        TILE_N = c.kwargs.get("TILE_N", 0)
        TILE_K = c.kwargs.get("TILE_K", 0)
        WS = c.kwargs.get("WARP_SPECIALIZE", False)

        # The triangular grid mapping assumes square tiles.
        if TILE_M != TILE_N:
            continue
        # bf16 operands, double buffered over num_stages.
        smem = (TILE_M * TILE_K + TILE_N * TILE_K) * 2 * c.num_stages
        if smem > _SM90_SMEM_BUDGET:
            continue
        if TILE_K > 64 and K < 2 * TILE_K:
            continue

        # Shortlists from an exhaustive latency sweep on H20 (sm90); see benchmarks/sweep_batched_syrk.py.
        if N < 768:
            # Small N favors 64-wide tiles, but with deep K (few large batches) 128-wide tiles win;
            # keep both and let the per-(N, K) autotune key decide.
            keep = not WS and (TILE_M, TILE_K, c.num_warps, c.num_stages) in (
                (64, 64, 4, 3),
                (64, 64, 4, 4),
                (64, 64, 8, 2),
                (64, 128, 4, 2),
                (128, 32, 4, 2),
                (128, 32, 4, 4),
                (128, 64, 4, 2),
                (128, 64, 4, 3),
            )
        elif N < 1280:
            keep = (TILE_M, TILE_K, c.num_warps, c.num_stages) in (
                (128, 32, 4, 2),
                (128, 32, 4, 3),
                (64, 64, 8, 4),
                (128, 64, 4, 2),
            )
        else:
            keep = TILE_M == 128 and TILE_K == 64 and c.num_stages in (2, 3)
        if keep:
            pruned_config.append(c)

    if len(pruned_config) == 0:
        logging.warning(
            "prune_invalid_batched_configs: all configs pruned for N=%d, K=%d, falling back to full config list.", N, K
        )
        return configs
    return pruned_config


def batched_matmul_tma_set_block_size_hook(nargs: dict) -> None:
    """Sets the block shapes for the batched tensor descriptors based on tile sizes.

    Args:
        nargs: Named arguments for the kernel.
    """
    TILE_M = nargs["TILE_M"]
    TILE_N = nargs["TILE_N"]
    TILE_K = nargs["TILE_K"]
    TRANS = nargs["TRANS"]
    nargs["a_desc"].block_shape = [1, TILE_K, TILE_M] if TRANS else [1, TILE_M, TILE_K]
    nargs["a_t_desc"].block_shape = [1, TILE_K, TILE_N] if TRANS else [1, TILE_N, TILE_K]
    if nargs["c_desc"] is not None:
        nargs["c_desc"].block_shape = [1, TILE_M, TILE_N]
    nargs["d_desc"].block_shape = [1, TILE_M, TILE_N]
    nargs["d_t_desc"].block_shape = [1, TILE_N, TILE_M]


_BATCHED_CONFIGS = [
    triton.Config(
        {"TILE_M": tm, "TILE_N": tm, "TILE_K": tk, "WARP_SPECIALIZE": ws},
        num_warps=nw,
        num_stages=ns,
        num_ctas=1,
        pre_hook=batched_matmul_tma_set_block_size_hook,
    )
    for tm in (64, 128)
    for tk in (32, 64, 128)
    for nw in (4, 8)
    for ns in (2, 3, 4)
    for ws in (False, True)
]

if "absl.testing" in sys.modules.keys():
    logging.warning("Running in absl.testing mode, disable autotune for triton.")
    _BATCHED_CONFIGS = [
        triton.Config(
            {"TILE_M": 64, "TILE_N": 64, "TILE_K": 64, "WARP_SPECIALIZE": False},
            num_warps=4,
            num_stages=2,
            num_ctas=1,
            pre_hook=batched_matmul_tma_set_block_size_hook,
        )
    ]


@triton.autotune(
    configs=_BATCHED_CONFIGS,
    key=["N", "K", "TRANS"],
    prune_configs_by={"early_config_prune": prune_invalid_batched_configs},
)
@triton.jit
def batched_syrk_kernel_bf16(
    d_desc,
    d_t_desc,
    a_desc,
    a_t_desc,
    c_desc,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    SKIP_UPPER_TRIANGLE: tl.constexpr,
    TRANS: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    # input A tensor of shape (B, N, K)
    # computes D[b] = alpha * A[b] * A[b]^T + beta * C[b] (-> produces B x N x N)
    # NOTE: If beta != 0, then each C[b] must be symmetric (i.e., C[b] == C[b]^T)

    # axis 0 walks the lower-triangular tiles of one batch element, axis 1 walks the batch. Keeping the
    # batch on its own axis means the triangular mapping math is identical to the single-matrix kernel.
    pid = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    # ======== Triangular grid mapping (see syrk.py) ========
    # Map the 1D linear pid onto lower triangular tile coordinates so no CTA is launched for the
    # upper triangle: m = floor((sqrt(8*pid + 1) - 1) / 2)
    f_m = (tl.sqrt(8.0 * pid + 1.0) - 1.0) / 2.0
    pid_m = f_m.to(tl.int32)
    pid_n = pid - (pid_m * (pid_m + 1) // 2)
    # Correction: if sqrt underestimated, pid_n will exceed pid_m
    if pid_n > pid_m:
        pid_m = pid_m + 1
        pid_n = pid - (pid_m * (pid_m + 1) // 2)
    # =======================================================

    IS_BELOW_DIAG = pid_m > pid_n

    # hints for the compiler
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_b >= 0)

    offs_row = pid_m * TILE_M
    offs_col = pid_n * TILE_N

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    num_tiles_k = tl.cdiv(K, TILE_K)
    for k in tl.range(num_tiles_k, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * TILE_K
        if TRANS:
            x = a_desc.load([pid_b, offs_k, offs_row]).reshape(TILE_K, TILE_M)
            y = a_t_desc.load([pid_b, offs_k, offs_col]).reshape(TILE_K, TILE_N)
            acc = tl.dot(x.T, y, acc=acc)
        else:
            x = a_desc.load([pid_b, offs_row, offs_k]).reshape(TILE_M, TILE_K)
            y = a_t_desc.load([pid_b, offs_col, offs_k]).reshape(TILE_N, TILE_K)
            acc = tl.dot(x, y.T, acc=acc)

    if alpha != 1.0:
        acc = alpha * acc
    if beta != 0.0:
        z = c_desc.load([pid_b, offs_row, offs_col]).reshape(TILE_M, TILE_N).to(tl.float32)
        acc = beta * z + acc

    d = acc.to(tl.bfloat16)

    d_desc.store([pid_b, offs_row, offs_col], d.reshape(1, TILE_M, TILE_N))

    # store replicated values above the diagonal. if skip_upper_triangle is True, only the lower triangle is written.
    if IS_BELOW_DIAG:
        if not SKIP_UPPER_TRIANGLE:
            d_t_desc.store([pid_b, offs_col, offs_row], d.T.reshape(1, TILE_N, TILE_M))


def can_use_batched_tsyrk(a: torch.Tensor, c: torch.Tensor | None = None) -> bool:
    """Check whether ``batched_tsyrk_ex`` can run on the given operands.

    Callers use this to decide between the Triton path and :func:`torch.baddbmm` without catching exceptions.
    The TMA descriptors require the innermost dimension of a bf16 tensor to be 16-byte aligned, i.e. a multiple
    of 8 elements.

    Args:
        a: Candidate input tensor of shape ``(B, N, K)``.
        c: Optional candidate accumulator of shape ``(B, N, N)``.

    Returns:
        True if the batched Triton syrk kernel supports these operands.
    """
    if not HAS_TRITON_340 or not a.is_cuda or a.dtype != torch.bfloat16 or a.dim() != 3:
        return False
    if not (a.is_contiguous() or a.mT.is_contiguous()):
        return False
    _, n, k = a.shape
    if n % 8 != 0 or k % 8 != 0:
        return False
    if c is not None:
        if c.shape != (a.shape[0], n, n) or c.dtype != a.dtype:
            return False
        if not (c.is_contiguous() or c.mT.is_contiguous()):
            return False
    return True


def batched_tsyrk_ex(
    a: torch.Tensor,
    c: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    skip_upper_triangle: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched bf16 syrk, a drop-in replacement for ``torch.baddbmm(c, a, a.mT, beta=beta, alpha=alpha)``.

    Computes ``d[b] = alpha * a[b] @ a[b].mT + beta * c[b]`` for each batch element. Because the result is
    symmetric, only the lower-triangular tiles are computed and the diagonal-off tiles are mirrored on store,
    which is roughly half the MMA work of a general ``baddbmm``.

    Note:
        If beta != 0, then each ``c[b]`` must be symmetric (i.e., ``c[b] == c[b].mT``).
        Only profitable for large per-matrix work (roughly N >= 1024 on H20; 1.5-1.7x at N >= 2048,
        slower than cuBLAS at N <= 512) — see the module docstring for the measured guidance.

    Args:
        a: Input tensor of shape ``(B, N, K)``, bfloat16, with ``a`` or ``a.mT`` contiguous.
        c: None or batch of symmetric tensors of shape ``(B, N, N)``.
        alpha: Scaling factor for the matrix multiplication.
        beta: Scaling factor for the matrix addition.
        skip_upper_triangle: Whether to skip writing the upper triangle of each output matrix.
        out: Optional output tensor to store the result. If None, a new tensor will be allocated.

    Returns:
        Output tensor of shape ``(B, N, N)``.
    """
    if not HAS_TRITON_340:
        raise RuntimeError("Triton version doesn't support tensor descriptor API. Minimum required version is 3.4.0.")
    if a.dtype != torch.bfloat16:
        raise TypeError("Input tensor must be bfloat16")
    if a.dim() != 3:
        raise TypeError("Input tensor must be 3D of shape (B, N, K)")
    if not (a.is_contiguous() or a.mT.is_contiguous()):
        raise TypeError("invalid input tensor layout. a or a.mT must be contiguous.")

    batch, N, K = a.shape
    if N % 8 != 0 or K % 8 != 0:
        raise RuntimeError(f"N and K must be multiples of 8 for bf16 TMA alignment, got N={N}, K={K}")
    if not ((c is None and beta == 0.0) or (c is not None and c.shape == (batch, N, N))):
        raise RuntimeError("if c is provided, c must be of shape (B, N, N)")
    if not (c is None or c.is_contiguous() or c.mT.is_contiguous()):
        raise RuntimeError("if c is provided, c or c.mT must be contiguous")

    if out is None:
        d = torch.empty((batch, N, N), device=a.device, dtype=a.dtype)
    else:
        if out.shape != (batch, N, N) or out.dtype != a.dtype or out.device != a.device:
            raise RuntimeError("out must be same shape/device/dtype as output")
        if not out.is_contiguous():
            raise RuntimeError("out must be contiguous")
        d = out

    dummy_block = [1, 1, 1]

    is_trans = not a.is_contiguous()

    if is_trans:
        # the descriptor relies on a contiguous tensor to load the data
        a = a.mT
    # With square tiles the a/a_t (and d/d_t) descriptors have identical block shapes, so one
    # TensorDescriptor serves both roles; the ~2 us per constructor matters for small shapes.
    # descriptor to load [1, TILE_M, TILE_K] from a and [1, TILE_N, TILE_K] from a.mT
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    # descriptor to store [1, TILE_M, TILE_N] to d and [1, TILE_N, TILE_M] to d.mT
    d_desc = TensorDescriptor(d, d.shape, d.stride(), dummy_block)

    if beta != 0.0:
        # c[b] is symmetric, so reading it transposed is equivalent and keeps the descriptor contiguous.
        c = c.mT if not c.is_contiguous() else c
        # descriptor to load [1, TILE_M, TILE_N] from c
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    else:
        c_desc = None

    def grid(META):
        assert META["TILE_M"] == META["TILE_N"], "batched_syrk_kernel_bf16 requires square tiles."
        num_tiles = triton.cdiv(N, META["TILE_M"])
        return (num_tiles * (num_tiles + 1) // 2, batch)

    batched_syrk_kernel_bf16[grid](
        d_desc,
        d_desc,
        a_desc,
        a_desc,
        c_desc,
        alpha,
        beta,
        skip_upper_triangle,
        is_trans,
        batch,
        N,
        K,
    )
    return d
