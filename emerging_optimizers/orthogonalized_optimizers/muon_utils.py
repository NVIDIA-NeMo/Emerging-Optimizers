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
from typing import Any, Literal

import torch
from absl import logging

from emerging_optimizers import triton_kernels


__all__ = ["newton_schulz", "newton_schulz_tp"]

_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # optimized for a quintic iteration.
        # Source: https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        # Numbers from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L44
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # Polar Express iteration from: https://arxiv.org/abs/2505.16932
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ],
    "aol": [
        # from https://github.com/thib-s/flash-newton-schulz/blob/main/newton_schulz_triton.py#L511
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}


def distributed_normalize_p2(x: torch.Tensor, eps: float, group: torch.distributed.ProcessGroup) -> torch.Tensor:
    """Normalize a tensor in a distributed way."""
    x_sq_sum = (x * x).sum()
    torch.distributed.all_reduce(x_sq_sum, op=torch.distributed.ReduceOp.SUM, group=group)
    return x / torch.sqrt(x_sq_sum).clamp_min(eps)


def newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    eps: float = 1e-7,
    transpose: bool | None = None,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Use Newton-Schulz iteration to compute the zeroth power / orthogonalization of x.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of x. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero and minimize variance.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the
    slope at zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce :math:`UV^T` but rather something like :math:`US'V^T`
    where :math:`S'` is diagonal with noisy values around 1, which turns out not to hurt model performance
    at all relative to :math:`UV^T`, where :math:`USV^T = G` is the SVD.


    Parameter ``coefficient_type`` can be one of the following
      - "simple": Default coefficient set.
      - "quintic": Quintic iteration with optimized coefficients.
      - "polar_express": Polar Express iteration with optimized coefficients.
      - "custom": Custom coefficient sets.

    Arguments:
        x: The tensor to be orthogonalized.
        steps: Number of Newton-Schulz iterations.
        coefficient_type: Type of coefficient set to use for the Newton-Schulz iteration.
        custom_coefficient_sets: Custom coefficient sets to use for the Newton-Schulz iteration.
        eps: Small constant to avoid division by zero.
        transpose: Whether to transpose the tensor to perform whitening on the smaller dimension.
            If None, will be determined based on the size of the tensor.
        tp_group: The process group for communication if input is distributed.

    Returns:
        The orthogonalization of x.
    """
    # Muon is not for 1d parameters
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions since Muon is not for 1d parameters.")
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")

    # transpose tensor to perform whitening on the smaller dimension
    if transpose is None:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    # Ensure spectral norm is at most 1
    if tp_group is not None:
        X = distributed_normalize_p2(x, eps, tp_group)
    else:
        X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)

    if coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    elif coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise ValueError("custom_coefficient_sets must be provided when coefficient_type is 'custom'.")
        coefficient_sets = custom_coefficient_sets
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    if steps % len(coefficient_sets) != 0:
        raise ValueError(f"steps ({steps}) must be multiple of len(coefficient_sets) ({len(coefficient_sets)}).")

    # Perform the NS iterations
    if torch.get_float32_matmul_precision() == "medium":
        # PyTorch doesn't really have FP32 I/O BF16 compute kernels for precision "medium"
        # We explicitly convert to BF16 and back to FP32.
        # NOTE: There is a small difference to calling FP32 I/O BF16 compute kernels because the final result
        # is converted to BF16 before converting back to FP32. The rest should be the same as long as epilogue
        # is always in FP32.
        X = X.to(torch.bfloat16)
        logging.log_first_n(logging.INFO, "Using BF16 I/O kernels for Newton-Schulz iteration.", 1)

    for i in range(steps):
        a, b, c = coefficient_sets[i % len(coefficient_sets)]
        X = newton_schulz_step(X, a, b, c, tp_group=tp_group)

    # Convert back to FP32. This is a noop if X is already in FP32.
    X = X.to(torch.float32)

    # undo transpose if necessary
    if transpose:
        X = X.mT
    return X


def newton_schulz_tp(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str,
    tp_group: torch.distributed.ProcessGroup,
    partition_dim: int | None = None,
    mode: Literal["duplicated", "distributed"] = "duplicated",
) -> torch.Tensor:
    """Tensor Parallel Newton-Schulz iteration.

    This function uses partition_dim to determine along which dimension the input tensor is sharded. Transpose is
    set based on the partition_dim. If partition_dim is None, the input tensor is not sharded and the function will
    fall back to the non-TP path.

    Warning:
        If partition_dim is the smaller dim of the input tensor, `distributed` mode will run Newton-Schulz along the
        long dimension which wastes compute.
        Although we reuse the partition_dim name, the default value is None which means no partition instead of -1.

    Note:
        This function is designed to provide tensor parallel support for most common use of Newton-Schulz.
        Many arguments, e.g. custom coefficient sets and custom eps, are not supported.

    ``mode`` can be one of the following:
        - "duplicated": The input tensor is duplicated and orthogonalized on each rank.
        - "distributed": The input tensor is partitioned along the partition_dim and orthogonalized on each rank.

    Args:
        x: The tensor to be orthogonalized. Must has partition_dim and tensor_model_parallel set by TransformerEngine.
        steps: Number of Newton-Schulz iterations.
        coefficient_type: Type of coefficient set to use for the Newton-Schulz iteration.
        partition_dim: The dimension to partition the tensor.
        tp_group: The process group for communication if input is distributed.
        mode: The mode to use for the Newton-Schulz iteration.
    """
    if partition_dim is None:
        # Fallback path for non TP params.
        return newton_schulz(x, steps, coefficient_type)

    kwargs: Any = {
        "steps": steps,
        "coefficient_type": coefficient_type,
    }

    if mode == "duplicated":
        x_shards = [torch.empty_like(x) for _ in range(tp_group.size())]
        torch.distributed.all_gather(x_shards, x, tp_group)
        global_x = torch.cat(x_shards, dim=partition_dim)

        orthogonalized_x = newton_schulz(global_x, tp_group=None, **kwargs)
        output = orthogonalized_x.chunk(tp_group.size(), dim=partition_dim)[tp_group.rank()]
    elif mode == "distributed":
        if partition_dim == 0:
            transpose = True
        elif partition_dim == 1:
            transpose = False
        else:
            raise ValueError(f"Invalid partition_dim: {partition_dim}")
        output = newton_schulz(x, **kwargs, transpose=transpose, tp_group=tp_group)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return output


def newton_schulz_step(
    X: torch.Tensor, a: float, b: float, c: float, tp_group: torch.distributed.ProcessGroup | None = None
) -> torch.Tensor:
    """Perform a single Newton-Schulz iteration step.

    This function performs a single Newton-Schulz iteration step. It supports distributed input that's sharded
    along the smaller (orthogonalize) dimension.

    Warning:
        If distributed, this function doesn't have the information to verify that X is sharded along the smaller
        (orthogonalize) dimension. It is user's responsibility to ensure that X is sharded correctly.

    Arguments:
        X: The tensor to be orthogonalized.
        a: The a coefficient.
        b: The b coefficient.
        c: The c coefficient.
        tp_group: The process group to use for the all-reduce.

    Returns:
        The orthogonalization of X.
    """
    A = X @ X.mT
    if tp_group is not None:
        torch.distributed.all_reduce(A, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    B = torch.addmm(A, A, A, beta=b, alpha=c)
    X = torch.addmm(X, B, X, beta=a, alpha=1.0)
    return X


def newton_schulz_step_tsyrk(
    X: torch.Tensor, a: float, b: float, c: float, tp_group: torch.distributed.ProcessGroup | None = None
) -> torch.Tensor:
    """Perform a single Newton-Schulz iteration step.

    This function performs a single Newton-Schulz iteration step using the Triton kernel for extended syrk.

    Arguments:
        X: The tensor to be orthogonalized. Must be bfloat16.
        a: The a coefficient.
        b: The b coefficient.
        c: The c coefficient.
        tp_group: The process group to use for the all-reduce.

    Returns:
        The orthogonalization of X.
    """
    A = triton_kernels.tsyrk_ex(X)  # type: ignore[attr-defined]
    if tp_group is not None:
        torch.distributed.all_reduce(A, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    B = triton_kernels.tsyrk_ex(A, A, alpha=c, beta=b)  # type: ignore[attr-defined]
    X = torch.addmm(X, B, X, alpha=a, beta=1.0)
    return X
