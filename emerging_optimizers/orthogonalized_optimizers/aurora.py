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

# Aurora refinement adapted from https://github.com/tilde-research/aurora-release.
# The original implementation is provided under the following license:
#
# MIT License
#
# Copyright (c) 2026 Tilde Research
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import inspect
import math
from collections.abc import Callable

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry, triton_kernels
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import MuonScaleT, get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc
from emerging_optimizers.utils import FP32MatmulPrecT
from emerging_optimizers.weight_update_hooks import WeightUpdateHook
from emerging_optimizers.weight_update_hooks.base import HookStateT


__all__ = ["Aurora", "aurora_orthogonalize"]


def _validate_aurora_args(iterations: int, beta: float, eps: float) -> None:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError(f"Aurora iterations must be a positive integer, got {iterations}")
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"Aurora beta must be in (0, 1], got {beta}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"Aurora eps must be finite and positive, got {eps}")


def aurora_orthogonalize(
    momentum: torch.Tensor,
    orthogonalize_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    iterations: int = 2,
    beta: float = 0.5,
    eps: float = 1e-14,
) -> torch.Tensor:
    r"""Balance tall update rows by diagonally preconditioning an unscaled polar solver.

    For a matrix :math:`M` with shape :math:`(m, n)`, where :math:`m > n`, start with
    :math:`D_i = 1 / \sqrt{\max(\|M_{i,:}\|_2^2, \varepsilon)}` and compute :math:`U = \mathrm{NS}(DM)`.
    Between solves, refine :math:`D_i \leftarrow D_i ((n/m) / \max(\|U_{i,:}\|_2^2, \varepsilon))^\beta`.
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
        eps: Finite positive floor for squared row norms; defaults to ``1e-14``.

    Returns:
        The unscaled solver output, with batch elements handled independently and input left unchanged.

    Raises:
        TypeError: If the input is not a 2D or 3D FP32 tensor.
        ValueError: If a numerical argument is invalid or a matrix dimension is empty.
    """
    _validate_aurora_args(iterations, beta, eps)
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
    diagonal = row_sq.clamp_min(eps).rsqrt().masked_fill(row_sq == 0, 0)
    orthogonalized = orthogonalize_fn(momentum * diagonal)
    for _ in range(1, iterations):
        row_sq = orthogonalized.float().square().sum(dim=-1, keepdim=True)
        diagonal = diagonal * ((cols / rows) / row_sq.clamp_min(eps)).pow(beta)
        orthogonalized = orthogonalize_fn(momentum * diagonal)
    return orthogonalized


@registry.register_optimizer("aurora")
class Aurora(OrthogonalizedOptimizer[HookStateT]):
    """Orthogonalized optimizer with Aurora row balancing for tall matrix updates.

    Supplies ``scaled_orthogonalize_fn`` to :class:`OrthogonalizedOptimizer`, retaining its momentum,
    Nesterov blending, weight decay, update hooks, and optimizer state. Tall matrices use
    :func:`aurora_orthogonalize` with the configured Newton-Schulz coefficients, precision, and backend.
    The Muon shape scale and extra scale
    factor are applied once, after all refinements. Square and wide matrices retain the Muon update.

    Parameters must be 2D and FP32. This implementation retains Muon's defaults, including Polar
    Express. No additional persistent tensor state is introduced.

    Args:
        {_args_doc}
        coefficient_type: Coefficient set for Newton-Schulz; defaults to ``"polar_express"``.
        num_ns_steps: Number of inner Newton-Schulz iterations per solve; defaults to 8.
        scale_mode: Shape-dependent update scaling; defaults to ``"spectral"``.
        extra_scale_factor: Additional multiplier applied after refinement; defaults to 0.2.
        use_syrk: Whether to use the Triton Newton-Schulz kernel on a supported GPU.
        num_aurora_steps: Number of outer Newton-Schulz solves for tall matrices. Each solve runs
            ``num_ns_steps`` inner iterations; defaults to two solves.
        aurora_beta: Damping exponent for diagonal refinement, in ``(0, 1]``; defaults to 0.5.
        eps: Finite positive floor for Aurora's squared row norms; defaults to ``1e-14``.

    References:
        - Aurora's released optimizer implementation.
          [`GitHub <https://github.com/tilde-research/aurora-release/blob/12e30810e1a2226fba77dcb7d6e23f3c4c79726b/src/aurora.py>`_]
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.1,
        *,
        nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "polar_express",
        num_ns_steps: int = 8,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 0.2,
        use_syrk: bool = False,
        num_aurora_steps: int = 2,
        aurora_beta: float = 0.5,
        eps: float = 1e-14,
        weight_update_hook: WeightUpdateHook[HookStateT] | None = None,
    ) -> None:
        _validate_aurora_args(num_aurora_steps, aurora_beta, eps)
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        self._num_ns_steps = num_ns_steps
        self._coefficient_type = coefficient_type
        self._scale_mode = scale_mode
        self._extra_scale_factor = extra_scale_factor
        self._num_aurora_steps = num_aurora_steps
        self._aurora_beta = aurora_beta
        self._eps = eps

        if use_syrk:
            sm_version = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            if not triton_kernels.HAS_TRITON_340:  # type: ignore[attr-defined]
                logging.error("Triton 3.4.0 or higher is required for use_syrk to be True.")
                use_syrk = False
            elif sm_version not in ((8, 0), (9, 0), (10, 0), (10, 3)):
                logging.error(
                    "Correctness of Triton kernel on SM %s cannot be guaranteed. Setting use_syrk to False.",
                    sm_version,
                )
                use_syrk = False
        self._use_syrk = use_syrk

        def orthogonalize_fn(momentum: torch.Tensor) -> torch.Tensor:
            return muon_utils.newton_schulz(
                momentum,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,
                use_syrk=self.use_syrk,
            )

        def scaled_orthogonalize_fn(momentum: torch.Tensor) -> torch.Tensor:
            update = aurora_orthogonalize(
                momentum, orthogonalize_fn, iterations=num_aurora_steps, beta=aurora_beta, eps=eps
            )
            scale = get_muon_scale_factor(momentum.size(-2), momentum.size(-1), mode=self.scale_mode)
            return update * scale * self.extra_scale_factor

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
            weight_update_hook=weight_update_hook,
        )

    @property
    def num_ns_steps(self) -> int:
        return self._num_ns_steps

    @property
    def coefficient_type(self) -> NSCoeffT:
        return self._coefficient_type

    @property
    def scale_mode(self) -> MuonScaleT:
        return self._scale_mode

    @property
    def extra_scale_factor(self) -> float:
        return self._extra_scale_factor

    @property
    def use_syrk(self) -> bool:
        return self._use_syrk

    @property
    def num_aurora_steps(self) -> int:
        return self._num_aurora_steps

    @property
    def aurora_beta(self) -> float:
        return self._aurora_beta

    @property
    def eps(self) -> float:
        return self._eps


Aurora.__doc__ = inspect.cleandoc(Aurora.__doc__).format(_args_doc=_args_doc)  # type: ignore[arg-type]
