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

from typing import Literal

import torch

from emerging_optimizers.contrib.row_norm import row_norm_fn
from emerging_optimizers.orthogonalized_optimizers.polargrad import one_sided_polargrad_orth_fn


__all__ = ["hybrid_polargrad_fn"]


def hybrid_polargrad_fn(
    grad: torch.Tensor,
    *,
    side: Literal["left", "right"] = "right",
    order: Literal["polar_then_row", "row_then_polar"] = "polar_then_row",
    alpha: float = 1.0,
    center_rows: bool = False,
    eps: float = 1e-8,
    polar_eps: float = 1e-15,
    extra_scale_factor: float = 1.0,
) -> torch.Tensor:
    r"""Hybrid one-sided spectral polar and row-normalization update.

    Composes :func:`~emerging_optimizers.orthogonalized_optimizers.polargrad.one_sided_polargrad_orth_fn`
    and :func:`~emerging_optimizers.contrib.row_norm.row_norm_fn` in either order:

    - ``order="polar_then_row"``: applies one-sided polar orthogonalization first and then row-wise
      normalization. Non-degenerate rows have norm ``extra_scale_factor`` when ``center_rows=False``
      (for rows whose scaled polar norm exceeds ``eps``, the global ``||G||_*^alpha`` factor cancels
      in the row normalization; ``alpha`` still scales inputs near the ``eps`` floor).
    - ``order="row_then_polar"``: applies row-wise normalization first and then one-sided polar
      orthogonalization scaled by ``||row_norm(G)||_*^alpha * extra_scale_factor``.

    Usable as ``scaled_orthogonalize_fn`` of
    :class:`~emerging_optimizers.orthogonalized_optimizers.OrthogonalizedOptimizer`.

    Args:
        grad: The (momentum) tensor to transform, must be 2D.
        side: Which polar factor to orthogonalize, ``"left"`` or ``"right"``.
        order: Composition order, ``"polar_then_row"`` or ``"row_then_polar"``.
        alpha: Exponent applied to the nuclear-norm scale factor in the polar step.
        center_rows: If True, subtract the per-column mean (the average over the row axis, ``dim=0``)
            before and after each stage, keeping the update in the zero-column-mean subspace.
        eps: Floor on the row norms in the row-normalization step.
        polar_eps: Floor on the Gram eigenvalues in the polar step.
        extra_scale_factor: Extra multiplier applied to the final update.

    Returns:
        The hybrid update, same shape and dtype as ``grad``.

    Raises:
        ValueError: If ``grad`` is not 2D, ``side`` is not ``"left"`` or ``"right"``, or
            ``order`` is not ``"polar_then_row"`` or ``"row_then_polar"``.
    """
    if grad.ndim != 2:
        raise ValueError(f"Only 2D tensors are supported, got {grad.ndim}D")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if order not in ("polar_then_row", "row_then_polar"):
        raise ValueError(f"order must be 'polar_then_row' or 'row_then_polar', got {order!r}")

    m = grad.to(torch.float32)
    if order == "polar_then_row":
        u = one_sided_polargrad_orth_fn(
            m,
            side=side,
            alpha=alpha,
            center_rows=center_rows,
            eps=polar_eps,
        )
        update = row_norm_fn(
            u,
            center_rows=center_rows,
            eps=eps,
            extra_scale_factor=extra_scale_factor,
        )
    else:
        work_row = row_norm_fn(
            m,
            center_rows=center_rows,
            eps=eps,
        )
        update = one_sided_polargrad_orth_fn(
            work_row,
            side=side,
            alpha=alpha,
            center_rows=center_rows,
            eps=polar_eps,
            extra_scale_factor=extra_scale_factor,
        )

    return update.to(grad.dtype)
