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


__all__ = ["sinkhorn_balance"]


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
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError(f"eps must be positive and finite, got {eps}")
    if num_normalization_steps < 1 or num_normalization_steps % 2 == 0:
        raise ValueError(f"num_normalization_steps must be a positive odd integer, got {num_normalization_steps}")
    if zero_row_threshold < 0.0 or not math.isfinite(zero_row_threshold):
        raise ValueError(f"zero_row_threshold must be nonnegative and finite, got {zero_row_threshold}")

    balanced_update = update.to(dtype=torch.float32, copy=True)

    # DeepSeek Algorithm 1 measures each row norm before balancing and masks rows whose norm is at most
    # zero_row_threshold times the mean row norm. This keeps inactive or near-zero token rows at zero
    # instead of amplifying their numerical noise during normalization.
    row_norms = torch.linalg.vector_norm(balanced_update, dim=1, keepdim=True)
    balanced_update.masked_fill_(row_norms <= zero_row_threshold * row_norms.mean(), 0.0)

    # Apply the first row step separately, then express each remaining iteration as a column/row pair.
    balanced_update.div_(row_norms.add_(eps))
    num_column_row_pairs = (num_normalization_steps - 1) // 2
    for _ in range(num_column_row_pairs):
        column_norms = torch.linalg.vector_norm(balanced_update, dim=0, keepdim=True)
        balanced_update.div_(column_norms.add_(eps))

        row_norms = torch.linalg.vector_norm(balanced_update, dim=1, keepdim=True)
        balanced_update.div_(row_norms.add_(eps))

    # A unit-L2 row has RMS 1 / sqrt(n); this scale converts the balanced update to unit row-wise RMS.
    balanced_update.mul_(math.sqrt(update.size(1)))
    return balanced_update.to(update.dtype)
