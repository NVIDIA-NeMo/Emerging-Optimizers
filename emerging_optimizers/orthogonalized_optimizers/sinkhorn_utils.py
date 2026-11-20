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
import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


__all__ = ["sinkhorn_balance", "sinkhorn_balance_row_sharded"]


@torch.no_grad()  # type: ignore[misc]
def sinkhorn_balance(
    update: torch.Tensor,
    *,
    eps: float = 1e-20,
    num_normalization_steps: int = 11,
    zero_row_threshold: float = 1e-3,
) -> torch.Tensor:
    r"""Balance a signed 2D update along its row and column axes.

    Given an update ``G`` with row norms ``rho_i = ||G[i, :]||_2``, rows satisfying
    ``rho_i <= zero_row_threshold * mean(rho)`` are first set to zero. Starting from the masked update
    ``U``, an odd number ``K = num_normalization_steps`` of alternating normalizations applies

    ``U[i, :] <- U[i, :] / (||U[i, :]||_2 + eps)`` for odd steps, and
    ``U[:, j] <- U[:, j] / (||U[:, j]||_2 + eps)`` for even steps.

    The returned update is ``Delta = sqrt(n) * U``, where ``n`` is the number of columns. When no rows
    are masked, this approximately targets ``(1 / n) * sum_j Delta[i, j]^2 = 1`` for every row and
    ``(1 / m) * sum_i Delta[i, j]^2 = 1`` for every column, where ``m`` is the number of rows.

    Args:
        update: Dense signed update with shape ``(num_rows, num_columns)``.
        eps: Numerical stability term added to each row or column norm.
        num_normalization_steps: Total positive odd number of individual axis-normalization steps.
        zero_row_threshold: Masking threshold relative to the mean pre-balancing row norm.

    Returns:
        A new Sinkhorn-balanced tensor with the input shape, dtype, and device.

    Raises:
        ValueError: If ``update`` is not a nonempty, tall 2D tensor in BF16, FP16, or FP32, or if a
            numerical argument is invalid.
    """
    if update.ndim != 2:
        raise ValueError(f"sinkhorn_balance requires a 2D tensor, got {update.ndim}D")
    if update.size(0) == 0 or update.size(1) == 0:
        raise ValueError("sinkhorn_balance requires nonempty matrix dimensions")
    if update.size(0) < update.size(1):
        raise ValueError("sinkhorn_balance expects rows to be the larger matrix dimension")
    if update.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"sinkhorn_balance only supports bfloat16, float16, and float32 tensors, got {update.dtype}")
    _validate_sinkhorn_parameters(eps, num_normalization_steps, zero_row_threshold)
    return _sinkhorn_balance(
        update,
        eps=eps,
        num_normalization_steps=num_normalization_steps,
        zero_row_threshold=zero_row_threshold,
        row_group=None,
        global_num_rows=update.size(0),
    )


def _validate_sinkhorn_parameters(eps: float, num_normalization_steps: int, zero_row_threshold: float) -> None:
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError(f"eps must be positive and finite, got {eps}")
    if num_normalization_steps < 1 or num_normalization_steps % 2 == 0:
        raise ValueError(f"num_normalization_steps must be a positive odd integer, got {num_normalization_steps}")
    if zero_row_threshold < 0.0 or not math.isfinite(zero_row_threshold):
        raise ValueError(f"zero_row_threshold must be nonnegative and finite, got {zero_row_threshold}")


def _sinkhorn_balance(
    update: torch.Tensor,
    *,
    eps: float,
    num_normalization_steps: int,
    zero_row_threshold: float,
    row_group: dist.ProcessGroup | None,
    global_num_rows: int,
) -> torch.Tensor:
    balanced_update = update.to(dtype=torch.float32, copy=True)
    if row_group is None and update.size(0) == 0:
        return balanced_update.to(update.dtype)

    # DeepSeek Algorithm 1 measures each row norm before balancing and masks rows whose norm is at most
    # zero_row_threshold times the mean row norm. This keeps inactive or near-zero token rows at zero
    # instead of amplifying their numerical noise during normalization.
    row_norms = torch.linalg.vector_norm(balanced_update, dim=1, keepdim=True)
    if row_group is None:
        mean_row_norm = row_norms.mean()
    else:
        norm_sum = row_norms.sum()
        dist.all_reduce(norm_sum, op=dist.ReduceOp.SUM, group=row_group)
        mean_row_norm = norm_sum / global_num_rows
    balanced_update.masked_fill_(row_norms <= zero_row_threshold * mean_row_norm, 0.0)

    # Apply the first row step separately, then express each remaining iteration as a column/row pair.
    balanced_update.div_(row_norms.add_(eps))
    num_column_row_pairs = (num_normalization_steps - 1) // 2
    for _ in range(num_column_row_pairs):
        if row_group is None:
            column_norms = torch.linalg.vector_norm(balanced_update, dim=0, keepdim=True)
        else:
            column_squares = balanced_update.square().sum(dim=0, keepdim=True)
            dist.all_reduce(column_squares, op=dist.ReduceOp.SUM, group=row_group)
            column_norms = column_squares.sqrt_()
        balanced_update.div_(column_norms.add_(eps))

        row_norms = torch.linalg.vector_norm(balanced_update, dim=1, keepdim=True)
        balanced_update.div_(row_norms.add_(eps))

    # A unit-L2 row has RMS 1 / sqrt(n); this scale converts the balanced update to unit row-wise RMS.
    balanced_update.mul_(math.sqrt(update.size(1)))
    return balanced_update.to(update.dtype)


@torch.no_grad()
def sinkhorn_balance_row_sharded(
    update: torch.Tensor,
    *,
    eps: float = 1e-20,
    num_normalization_steps: int = 11,
    zero_row_threshold: float = 1e-3,
    row_group: dist.ProcessGroup | None,
    global_num_rows: int,
    use_shard_local_stats: bool = False,
) -> torch.Tensor:
    r"""Balance a signed matrix over distinct row shards without gathering its entries.

    Each local row must contain all feature columns. The logical matrix has ``global_num_rows`` rows and must be
    nonempty and tall (i.e., ``global_num_rows >= num_features > 0``); a local shard may be empty or wide. Balancing
    uses FP32 workspace and statistics, independently of whether the input is BF16, FP16, or FP32.

    In global mode, the initial sum of row L2 norms and each column sum of squares are reduced over ``row_group``.
    Square roots and epsilon are applied after reduction. 11 individual normalization steps require 1 scalar and 5
    feature-vector SUM reductions, with no matrix all-gather. ``row_group=None`` never selects ``WORLD``: it requires
    the complete matrix locally, as does a size-one group. ``use_shard_local_stats=True`` ignores the group and uses
    shard-local statistics, deliberately changing the optimization rule.

    The caller must supply the same global row count, feature width, numerical settings, and collective call order on
    every group member. Local row counts must sum to the supplied global count, and shards must not overlap or include
    replicas or padding. Coverage is a caller contract, not checked with another collective. Every row owner must call,
    including owners with zero local rows. Momentum, Nesterov blending, LR correction, and weight updates are not
    applied.

    Args:
        update: Dense strided local tensor with shape ``(local_rows, num_features)``. Pass local tensors rather than
            DTensors with unresolved placements.
        eps: Positive finite stability term added to each row or column norm.
        num_normalization_steps: Positive odd number of individual alternating row/column normalizations, beginning and
            ending with a row step.
        zero_row_threshold: Nonnegative finite masking threshold relative to the mean pre-balancing row norm, global
            unless ``use_shard_local_stats=True``.
        row_group: Initialized process group of distinct row owners, or None for no communication. The current rank must
            be a member in global mode.
        global_num_rows: Positive row count of the entire logical matrix, not the local shard. The global shape is
            validated even in local mode.
        use_shard_local_stats: Compute the masking threshold and column norms from the local shard only, without
            collectives. The resulting approximate update depends on the row partition.

    Returns:
        A new, detached balanced tensor with the input shape, dtype, and device. Retained rows have approximately unit
        RMS; masked rows stay zero. Finite iteration count and masking do not guarantee unit RMS for every column. The
        input is not modified.

    Raises:
        ValueError: If tensor layout, dtype, shape, numerical settings, or group membership/initialization are invalid,
            or a global shard lacks a multi-rank group. Collective backend errors propagate from PyTorch.
    """
    if update.ndim != 2 or update.layout != torch.strided:
        raise ValueError("sinkhorn_balance_row_sharded requires a dense strided 2D local tensor")
    if dist.is_available():
        if isinstance(update, DTensor):
            raise ValueError(
                "Pass a local tensor, not a DTensor, to sinkhorn_balance_row_sharded. "
                "For a row-sharded DTensor with all feature columns present locally and no partial values, "
                "pass update.to_local(), global_num_rows=update.size(0), and a row_group of distinct row owners. "
                "Resolve partial values or redistribute column shards first."
            )
    if update.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("sinkhorn_balance_row_sharded only supports bfloat16, float16, and float32 tensors")
    if global_num_rows <= 0:
        raise ValueError("global_num_rows must be a positive integer")
    if update.size(1) == 0 or global_num_rows < update.size(1):
        raise ValueError("sinkhorn_balance_row_sharded requires a nonempty globally tall matrix")
    if update.size(0) > global_num_rows:
        raise ValueError("Local row count exceeds global_num_rows")
    _validate_sinkhorn_parameters(eps, num_normalization_steps, zero_row_threshold)
    effective_group = None
    if not use_shard_local_stats:
        if row_group is not None:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("row_group requires initialized torch.distributed")
            world_size = dist.get_world_size(row_group)
            if world_size < 1:
                raise ValueError("The current rank must belong to row_group")
            if world_size > 1:
                effective_group = row_group
        if effective_group is None and update.size(0) != global_num_rows:
            raise ValueError("Global row shards require an explicit multi-rank row_group")
    rows = update.size(0) if use_shard_local_stats else global_num_rows
    return _sinkhorn_balance(
        update,
        eps=eps,
        num_normalization_steps=num_normalization_steps,
        zero_row_threshold=zero_row_threshold,
        row_group=effective_group,
        global_num_rows=rows,
    )
