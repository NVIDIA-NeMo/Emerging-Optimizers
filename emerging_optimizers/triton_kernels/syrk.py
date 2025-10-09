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
import torch
import triton
import triton.language as tl


try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    raise ImportError(
        f"Triton version ({triton.__version__}) doesn't support tensor descriptor API. Minimum required version is 3.4.0."
    )

__all__ = ["ssyrk", "tsyrk_ex"]


@triton.jit
def cvt_tf32_rn(x: tl.tensor) -> tl.tensor:
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.autotune(
    configs=[
        triton.Config({"TILE_N": tn, "TILE_K": tk}, num_warps=nw, num_stages=ns)
        for tn in (64, 128)
        for tk in (16, 32, 64)
        for nw in (4, 8)
        for ns in (3, 4)
    ],
    key=["N", "K", "ALLOW_TF32"],
)
@triton.jit
def syrk_op_n_simple_kernel(
    c_ptr,
    a_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # receives tensor of shape (N, K)
    # computes A * A^T (-> produces NxN)

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    IS_BELOW_DIAG = pid_row < pid_col
    IS_ABOVE_DIAG = pid_row > pid_col

    if IS_ABOVE_DIAG:
        return

    offs_row = pid_row * TILE_N + tl.arange(0, TILE_N)
    offs_col = pid_col * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    mask_row = offs_row < N
    mask_col = offs_col < N

    a_ptrs_x = a_ptr + offs_row[:, None] * STRIDE_N + offs_k[None, :] * STRIDE_K
    a_ptrs_y = a_ptr + offs_col[None, :] * STRIDE_N + offs_k[:, None] * STRIDE_K

    acc = tl.zeros((TILE_N, TILE_N), dtype=tl.float32)

    num_tiles_k = tl.cdiv(K, TILE_K)
    for k in range(0, num_tiles_k):
        mask_k = offs_k < K - k * TILE_K
        mask_x = mask_row[:, None] & mask_k[None, :]
        mask_y = mask_col[None, :] & mask_k[:, None]
        x = tl.load(a_ptrs_x, mask=mask_x, other=0.0)
        y = tl.load(a_ptrs_y, mask=mask_y, other=0.0)

        if ALLOW_TF32 == 0:
            acc = tl.dot(x, y, acc=acc, input_precision="ieee")
        elif ALLOW_TF32 == 1:
            x = cvt_tf32_rn(x)
            y = cvt_tf32_rn(y)
            acc = tl.dot(x, y, acc=acc, input_precision="tf32")
        else:
            tl.static_assert(False, "Unsupported precision.")

        a_ptrs_x += TILE_K * STRIDE_K
        a_ptrs_y += TILE_K * STRIDE_K

    # store diagonal or below diagonal values
    c_ptrs = c_ptr + offs_row[:, None] * N + offs_col[None, :]
    mask_c = mask_row[:, None] & mask_col[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)

    # store replicated values above diagonal
    if IS_BELOW_DIAG:
        c_ptrs_diag = c_ptr + offs_col[None, :] * N + offs_row[:, None]
        tl.store(c_ptrs_diag, acc, mask=mask_c)


def ssyrk(a: torch.Tensor, trans: bool = False) -> torch.Tensor:
    """Triton implementation of BLAS ssyrk operation.

    Note:
        This function assumes row major layout of the input tensor.

    TODO(mstadler): Add support for alpha, beta and c.

    Args:
        a: Input tensor of shape (N, K) or (K, N)
        trans: Whether to compute A * A^T (trans=False) or A^T * A (trans=True)

    Returns:
        Output tensor of shape (N, N)
    """
    assert a.dim() == 2, "Input tensor must be 2D"
    N, K = a.shape
    if trans:
        raise NotImplementedError("Transpose is not supported yet.")

    STRIDE_N = a.stride(0)
    STRIDE_K = a.stride(1)

    if (fp32_matmul_prec := torch.get_float32_matmul_precision()) == "highest":
        ALLOW_TF32 = 0
    elif fp32_matmul_prec == "high":
        ALLOW_TF32 = 1
    else:
        raise ValueError(f"Unsupported precision {fp32_matmul_prec}, only 'highest' and 'high' are supported.")

    c = torch.empty((N, N), dtype=a.dtype, device=a.device)

    def grid(META):
        return (triton.cdiv(N, META["TILE_N"]), triton.cdiv(N, META["TILE_N"]))

    if not trans:
        syrk_op_n_simple_kernel[grid](c, a, N, K, STRIDE_N, STRIDE_K, ALLOW_TF32)

    return c


def prune_invalid_configs(configs: list[triton.Config], named_args: dict, **kwargs) -> list[triton.Config]:
    """Prune invalid Triton kernel configs based on input size and tile parameters.

    Args:
        configs: List of Triton kernel configs.
        named_args: Named arguments for the kernel.
        **kwargs: Additional keyword arguments.

    Returns:
        List of valid Triton kernel configs.
    """
    N = named_args["N"]

    conf = []
    for c in configs:
        TILE_M = c.kwargs.get("TILE_M", 0)
        TILE_N = c.kwargs.get("TILE_N", 0)
        TILE_K = c.kwargs.get("TILE_K", 0)

        # 5000 is an empirically determined threshold from size shmoo to select the best config
        if N >= 5000:
            if TILE_M == 128 and TILE_N == 256 and TILE_K == 64:
                conf.append(c)
        else:
            if TILE_M <= 128 and TILE_N >= TILE_M and TILE_K <= 128:
                conf.append(c)
    return conf


def matmul_tma_set_block_size_hook(nargs: dict) -> None:
    """Sets the block shapes for tensor descriptors based on tile sizes.

    Args:
        nargs: Named arguments for the kernel.
    """
    TILE_M = nargs["TILE_M"]
    TILE_N = nargs["TILE_N"]
    TILE_K = nargs["TILE_K"]
    TRANS = nargs["TRANS"]
    nargs["a_desc"].block_shape = [TILE_K, TILE_M] if TRANS else [TILE_M, TILE_K]
    nargs["a_t_desc"].block_shape = [TILE_K, TILE_N] if TRANS else [TILE_N, TILE_K]
    if nargs["c_desc"] is not None:
        nargs["c_desc"].block_shape = [TILE_M, TILE_N]
    nargs["d_desc"].block_shape = [TILE_M, TILE_N]
    nargs["d_t_desc"].block_shape = [TILE_N, TILE_M]


@triton.autotune(
    configs=[
        triton.Config(
            {"TILE_M": tm, "TILE_N": tn, "TILE_K": tk, "GROUP_SIZE_M": gm},
            num_warps=nw,
            num_stages=ns,
            num_ctas=nc,
            pre_hook=matmul_tma_set_block_size_hook,
        )
        for tm in (64, 128, 256)
        for tn in (64, 128, 256)
        for tk in (64, 128, 256)
        for gm in (2, 4, 8)
        for nw in (4, 8)
        for ns in (2, 3, 4)
        for nc in (1,)
    ],
    key=["N", "K", "TRANS", "WARP_SPECIALIZE"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def syrk_kernel_bf16(
    d_desc,
    d_t_desc,
    a_desc,
    a_t_desc,
    c_desc,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    SKIP_UPPER_TRIANGLE: tl.constexpr,
    TRANS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    # input A tensor of shape (N, K)
    # computes D = alpha * A * A^T + beta * C (-> produces NxN)
    # NOTE: If beta != 0, then C must be a symmetric matrix (i.e., C == C^T)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    IS_BELOW_DIAG = pid_m * TILE_M >= pid_n * TILE_N + TILE_N
    IS_ABOVE_DIAG = pid_m * TILE_M + TILE_M <= pid_n * TILE_N
    IS_SQUARE_TILE = TILE_M == TILE_N

    if IS_ABOVE_DIAG:
        return

    # hints for the compiler
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_row = pid_m * TILE_M
    offs_col = pid_n * TILE_N

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    num_tiles_k = tl.cdiv(K, TILE_K)
    for k in tl.range(num_tiles_k, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * TILE_K
        if TRANS:
            x = a_desc.load([offs_k, offs_row])
            y = a_t_desc.load([offs_k, offs_col])
            acc = tl.dot(x.T, y, acc=acc)
        else:
            x = a_desc.load([offs_row, offs_k])
            y = a_t_desc.load([offs_col, offs_k])
            acc = tl.dot(x, y.T, acc=acc)

    if alpha != 1.0:
        acc = alpha * acc
    if beta != 0.0:
        z = c_desc.load([offs_row, offs_col]).to(tl.float32)
        acc = beta * z + acc

    d = acc.to(tl.bfloat16)

    offs_row = pid_m * TILE_M
    offs_col = pid_n * TILE_N
    d_desc.store([offs_row, offs_col], d)

    # store replicated values above diagonal. if skip_upper_triangle is True, we only store the values below the diagonal.
    if (IS_SQUARE_TILE and IS_BELOW_DIAG) or (not IS_SQUARE_TILE and not IS_ABOVE_DIAG):
        if not SKIP_UPPER_TRIANGLE:
            d_t_desc.store([offs_col, offs_row], d.T)


def tsyrk_ex(
    a: torch.Tensor, c: torch.Tensor = None, alpha: float = 1.0, beta: float = 0.0, skip_upper_triangle: bool = False
) -> torch.Tensor:
    """Triton implementation of bf16 syrk operation, following cuBLAS naming conventions with 't' denoting bf16.

    Note:
        If beta != 0, then a must be a symmetric matrix (i.e., a == a.T)

    Args:
        a: Input tensor of shape (N, K)
        c: None or symmetric input tensor of shape (N, N)
        alpha: Scaling factor for the matrix multiplication
        beta: Scaling factor for the matrix addition
        skip_upper_triangle: Whether to skip the upper triangle part of the output

    Returns:
        Output tensor of shape (N, N)
    """
    sm_version = torch.cuda.get_device_capability()
    assert sm_version in ((8, 0), (9, 0), (10, 0), (11, 0)), (
        f"Correctness of Triton kernel on SM {sm_version} can not be guaranteed."
    )
    assert a.dtype == torch.bfloat16, "Input tensor must be bfloat16"
    assert a.dim() == 2, "Input tensor must be 2D"
    assert a.is_contiguous() or a.T.is_contiguous(), "invalid input tensor layout. a or a.T must be contiguous."

    N, K = a.shape
    assert (c is None and beta == 0.0) or (c is not None and c.shape == (N, N)), (
        "if c is provided, c must be of shape (N, N)"
    )
    assert c is None or c.is_contiguous() or c.T.is_contiguous(), "if c is provided, c or c.T must be contiguous"

    d = torch.empty((N, N), device=a.device, dtype=a.dtype)

    dummy_block = [1, 1]

    is_trans = a.T.is_contiguous()

    if is_trans:
        # the descriptor relys on contiguous tensor to load the data
        a = a.T
    # descriptor to load [TILE_M, TILE_K] from a
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    # descriptor to load [TILE_K, TILE_N] from a.T
    a_t_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    # descriptor to store [TILE_M, TILE_N] to d
    d_desc = TensorDescriptor(d, d.shape, d.stride(), dummy_block)
    # descriptor to store [TILE_M, TILE_N] to d.T
    d_t_desc = TensorDescriptor(d, d.shape, d.stride(), dummy_block)

    if beta != 0.0:
        c = c.T if c.T.is_contiguous() else c
        # descriptor to load [TILE_M, TILE_N] from a
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    else:
        c_desc = None

    def grid(META):
        return (triton.cdiv(N, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),)

    syrk_kernel_bf16[grid](
        d_desc,
        d_t_desc,
        a_desc,
        a_t_desc,
        c_desc,
        alpha,
        beta,
        skip_upper_triangle,
        is_trans,
        N,
        K,
        WARP_SPECIALIZE=False,
    )
    return d
