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
from collections.abc import Callable
from typing import Any

import torch

from emerging_optimizers import registry
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon, get_muon_scale_factor
from emerging_optimizers.weight_update_hooks.base import HookStateT


__all__ = ["Aurora", "aurora_orthogonalize"]


def _validate_aurora_args(iterations: int, beta: float) -> None:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError(f"Aurora iterations must be a positive integer, got {iterations}")
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"Aurora beta must be in (0, 1], got {beta}")


def aurora_orthogonalize(
    momentum: torch.Tensor,
    orthogonalize_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    iterations: int = 2,
    beta: float = 0.5,
) -> torch.Tensor:
    r"""Balance tall update rows by diagonally preconditioning an unscaled polar solver.

    For a matrix :math:`M` with shape :math:`(m, n)`, where :math:`m > n`, start with
    :math:`D_i = 1 / \max(\|M_{i,:}\|_2, 10^{-7})` and compute :math:`U = \mathrm{NS}(DM)`.
    Between solves, refine :math:`D_i \leftarrow D_i ((n/m) / \max(\|U_{i,:}\|_2^2, 10^{-14}))^\beta`.
    Every solve uses the original momentum and the refined diagonal, not the preceding solver output.
    Exact zero rows remain zero. Square and wide matrices call the supplied solver once, unchanged.

    This follows Aurora's released diagonal refinement, using the caller's existing unscaled solver.
    Finite Newton-Schulz and outer iterations only approximately balance the row norms; this is not an
    exact polar decomposition. Shape-dependent update scaling belongs after this function returns.

    Args:
        momentum: Unscaled FP32 matrix in ``[out, in]`` layout, or a batch ``[batch, out, in]``.
            Matrices must be complete, unsharded, and have nonempty matrix dimensions.
        orthogonalize_fn: Unscaled solver, such as :func:`muon_utils.newton_schulz` with its options bound.
        iterations: Positive number of outer solver calls for tall matrices.
        beta: Damping exponent for diagonal refinement, in ``(0, 1]``.

    Returns:
        The unscaled solver output, with batch elements handled independently and input left unchanged.

    Raises:
        TypeError: If the input is not a 2D or 3D FP32 tensor.
        ValueError: If a numerical argument is invalid or a matrix dimension is empty.
    """
    _validate_aurora_args(iterations, beta)
    if momentum.ndim not in (2, 3):
        raise TypeError(f"Aurora requires a 2D or 3D tensor, got {momentum.ndim}D")
    if momentum.dtype != torch.float32:
        raise TypeError(f"Aurora requires float32 input, got {momentum.dtype}")
    rows, cols = momentum.shape[-2:]
    if rows == 0 or cols == 0:
        raise ValueError("Aurora requires nonempty matrix dimensions")
    if rows <= cols:
        return orthogonalize_fn(momentum)

    row_sq = momentum.square().sum(dim=-1, keepdim=True)
    diagonal = row_sq.clamp_min(1e-14).rsqrt().masked_fill(row_sq == 0, 0)
    orthogonalized = orthogonalize_fn(momentum * diagonal)
    for _ in range(1, iterations):
        row_sq = orthogonalized.float().square().sum(dim=-1, keepdim=True)
        diagonal = diagonal * ((cols / rows) / row_sq.clamp_min(1e-14)).pow(beta)
        orthogonalized = orthogonalize_fn(momentum * diagonal)
    return orthogonalized


@registry.register_optimizer("aurora")
class Aurora(Muon[HookStateT]):
    """Muon with Aurora row balancing for tall matrix updates.

    Replaces ``scaled_orthogonalize_fn`` while retaining Muon's momentum, Nesterov blending, weight
    decay, update hooks, and optimizer state. Tall matrices use :func:`aurora_orthogonalize` with the
    existing Newton-Schulz coefficients, precision, and backend. The Muon shape scale and extra scale
    factor are applied once, after all refinements. Square and wide matrices retain the Muon update.

    See :class:`Muon` for the inherited optimizer options and 2D FP32 parameter requirements. This
    implementation retains Muon's defaults, including Polar Express, rather than switching to the
    released Aurora solver's coefficient schedule. No additional persistent tensor state is introduced.

    Args:
        *args: Positional arguments passed to Muon.
        num_aurora_steps: Number of outer Newton-Schulz solves for tall matrices. Each solve runs
            ``num_ns_steps`` inner iterations; defaults to two solves.
        aurora_beta: Damping exponent for diagonal refinement, in ``(0, 1]``; defaults to 0.5.
        **kwargs: Keyword arguments passed to Muon, including ``weight_update_hook``.

    References:
        - Aurora's released optimizer implementation.
          [`GitHub <https://github.com/tilde-research/aurora-release/blob/12e30810e1a2226fba77dcb7d6e23f3c4c79726b/src/aurora.py>`_]
    """

    def __init__(
        self,
        *args: Any,
        num_aurora_steps: int = 2,
        aurora_beta: float = 0.5,
        **kwargs: Any,
    ) -> None:
        _validate_aurora_args(num_aurora_steps, aurora_beta)
        super().__init__(*args, **kwargs)
        self._num_aurora_steps = num_aurora_steps
        self._aurora_beta = aurora_beta
        muon_scaled_orthogonalize_fn = self.scaled_orthogonalize_fn

        def orthogonalize_fn(momentum: torch.Tensor) -> torch.Tensor:
            return muon_utils.newton_schulz(
                momentum,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
                use_syrk=self.use_syrk,
            )

        def scaled_orthogonalize_fn(momentum: torch.Tensor) -> torch.Tensor:
            if momentum.size(-2) <= momentum.size(-1):
                return muon_scaled_orthogonalize_fn(momentum)
            update = aurora_orthogonalize(momentum, orthogonalize_fn, iterations=num_aurora_steps, beta=aurora_beta)
            scale = get_muon_scale_factor(momentum.size(-2), momentum.size(-1), mode=self.scale_mode)
            return update * scale * self.extra_scale_factor

        self.scaled_orthogonalize_fn = scaled_orthogonalize_fn

    @property
    def num_aurora_steps(self) -> int:
        return self._num_aurora_steps

    @property
    def aurora_beta(self) -> float:
        return self._aurora_beta
