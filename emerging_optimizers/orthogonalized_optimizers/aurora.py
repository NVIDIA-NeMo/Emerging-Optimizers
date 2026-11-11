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

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc
from emerging_optimizers.utils import FP32MatmulPrecT
from emerging_optimizers.weight_update_hooks import WeightUpdateHook
from emerging_optimizers.weight_update_hooks.base import HookStateT


__all__ = ["Aurora"]


@torch.compile(fullgraph=True)
def _precondition_rows(momentum: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A common positive scale leaves Aurora unchanged and keeps FP32 norm reductions
    # away from underflow/overflow. Each matrix in a batch has its own scale and floor.
    scale = momentum.abs().amax(dim=(-2, -1), keepdim=True)
    momentum = momentum / torch.where(scale > 0, scale, 1.0)
    row_norm = torch.linalg.vector_norm(momentum, dim=-1, keepdim=True)
    floor = eps * row_norm.mean(dim=-2, keepdim=True)
    denominator = torch.maximum(row_norm, floor)
    zero_rows = (momentum == 0).all(dim=-1, keepdim=True)
    diagonal = torch.where(denominator > 0, denominator, 1.0).reciprocal().masked_fill(zero_rows, 0)
    return momentum, diagonal, momentum * diagonal


@torch.compile(fullgraph=True)
def _refine_rows(
    momentum: torch.Tensor,
    orthogonalized: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_norm = torch.linalg.vector_norm(orthogonalized.float(), dim=-1, keepdim=True)
    floor = eps * row_norm.mean(dim=-2, keepdim=True)
    denominator = torch.maximum(row_norm, floor)
    denominator = torch.where((denominator > 0) & (diagonal != 0), denominator, 1.0)
    rows, cols = momentum.shape[-2:]
    diagonal = diagonal * (math.sqrt(cols / rows) / denominator).pow(2 * beta)
    return diagonal, momentum * diagonal


@registry.register_optimizer("aurora")
class Aurora(OrthogonalizedOptimizer[HookStateT]):
    """Orthogonalized optimizer with Aurora row balancing for tall matrix updates.

    Supplies ``scaled_orthogonalize_fn`` to :class:`OrthogonalizedOptimizer`, retaining its momentum,
    weight decay, hooks, and state. Tall updates use a diagonal row preconditioner, refined between
    Newton-Schulz solves of the original momentum. Square and wide updates retain the Muon path.
    Uses Polar Express with Muon's spectral scale and 0.2 multiplier, applied once after refinement.
    Parameters must be complete, unsharded 2D FP32 matrices. No persistent tensor state is added.

    Args:
        {_args_doc}
        num_ns_steps: Number of inner Newton-Schulz iterations per solve; defaults to 8.
        num_aurora_steps: Number of outer solves for tall matrices; defaults to 2.
        aurora_beta: Damping exponent for diagonal refinement, in ``(0, 1]``; defaults to 0.5.
        eps: Relative row L2 norm floor; defaults to ``1e-6``. Each balancing stage floors
            row norms at ``eps * mean(row_norms)`` for that matrix. Must be finite and in
            the normal FP32 range. Exact zero rows and all-zero matrices remain zero.

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
        num_ns_steps: int = 8,
        num_aurora_steps: int = 2,
        aurora_beta: float = 0.5,
        eps: float = 1e-6,
        weight_update_hook: WeightUpdateHook[HookStateT] | None = None,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if isinstance(num_aurora_steps, bool) or not isinstance(num_aurora_steps, int) or num_aurora_steps < 1:
            raise ValueError(f"num_aurora_steps must be a positive integer, got {num_aurora_steps}")
        if not 0.0 < aurora_beta <= 1.0:
            raise ValueError(f"aurora_beta must be in (0, 1], got {aurora_beta}")
        fp32 = torch.finfo(torch.float32)
        if not math.isfinite(eps) or not fp32.tiny <= eps <= fp32.max:
            raise ValueError(f"eps must be finite and in the normal FP32 range [{fp32.tiny}, {fp32.max}], got {eps}")

        self._num_ns_steps = num_ns_steps
        self._num_aurora_steps = num_aurora_steps
        self._aurora_beta = aurora_beta
        self._eps = eps

        def scaled_orthogonalize_fn(momentum: torch.Tensor) -> torch.Tensor:
            rows, cols = momentum.shape[-2:]
            preconditioned = momentum
            if rows > cols:
                momentum, diagonal, preconditioned = _precondition_rows(momentum, eps=eps)
            update = muon_utils.newton_schulz(preconditioned, steps=num_ns_steps, coefficient_type="polar_express")
            if rows > cols:
                for _ in range(1, num_aurora_steps):
                    diagonal, preconditioned = _refine_rows(momentum, update, diagonal, beta=aurora_beta, eps=eps)
                    update = muon_utils.newton_schulz(
                        preconditioned, steps=num_ns_steps, coefficient_type="polar_express"
                    )
            return update * get_muon_scale_factor(rows, cols) * 0.2

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
    def num_aurora_steps(self) -> int:
        return self._num_aurora_steps

    @property
    def aurora_beta(self) -> float:
        return self._aurora_beta

    @property
    def eps(self) -> float:
        return self._eps


Aurora.__doc__ = inspect.cleandoc(Aurora.__doc__).format(_args_doc=_args_doc)  # type: ignore[arg-type]
