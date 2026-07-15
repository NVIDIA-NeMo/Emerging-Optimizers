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

import torch


__all__ = ["row_norm_fn"]


def row_norm_fn(
    grad: torch.Tensor,
    *,
    center_rows: bool = False,
    eps: float = 1e-8,
    extra_scale_factor: float = 1.0,
) -> torch.Tensor:
    r"""Row-wise normalization of the update.

    Scales each row of ``G`` by the inverse of its norm:

    .. math::
        u_i = G_i \, / \max(\lVert G_i \rVert_2, \epsilon)

    Usable as ``scaled_orthogonalize_fn`` of
    :class:`~emerging_optimizers.orthogonalized_optimizers.OrthogonalizedOptimizer`.

    Args:
        grad: The (momentum) tensor to normalize.
        center_rows: If True, subtract the per-column mean (the average over the row axis, ``dim=0``)
            before and after the update, so each column is zero-mean.
        eps: Floor on the row norms.
        extra_scale_factor: Extra multiplier on the update.

    Returns:
        The row-normalized update, same shape and dtype as ``grad``.
    """
    m = grad.to(torch.float32)
    if center_rows:
        m = m - m.mean(dim=0, keepdim=True)

    update = m / m.norm(dim=-1, keepdim=True).clamp_min(eps) * extra_scale_factor

    if center_rows:
        update = update - update.mean(dim=0, keepdim=True)
    return update.to(grad.dtype)
