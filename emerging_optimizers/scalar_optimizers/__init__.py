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
from typing import Any, override

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.registry import register_optimizer
from emerging_optimizers.scalar_optimizers.base import (
    SingleMomentumOptimizer,
    TwoMomentsOptimizer,
    _validate_common_hparams,
)
from emerging_optimizers.scalar_optimizers.update_functions import (
    calculate_laprop_update,
    calculate_lion_update,
    calculate_signum_update,
    calculate_sim_ademamix_update,
)


__all__ = [
    "LaProp",
    "Lion",
    "Signum",
    "SimplifiedAdEMAMix",
    "SingleMomentumOptimizer",
    "TwoMomentsOptimizer",
]


@register_optimizer("lion")
class Lion(SingleMomentumOptimizer):
    """Lion optimizer (Chen et al., 2023): sign-based update with a single first-moment EMA."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
        *,
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        _validate_common_hparams(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(
            params,
            defaults=dict(lr=lr, betas=betas, weight_decay=weight_decay),
            update_fn=calculate_lion_update,
            update_kwarg_names=("betas",),
            weight_decay_method=weight_decay_method,
        )


@register_optimizer("signum")
class Signum(SingleMomentumOptimizer):
    """Sign-SGD / Signum optimizer (Bernstein et al., 2018): sign of a bias-corrected single-moment EMA."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        *,
        correct_bias: bool = True,
        nesterov: bool = False,
        use_shape_scaling: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        _validate_common_hparams(lr=lr, weight_decay=weight_decay)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        super().__init__(
            params,
            defaults=dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                correct_bias=correct_bias,
                nesterov=nesterov,
                use_shape_scaling=use_shape_scaling,
            ),
            update_fn=calculate_signum_update,
            update_kwarg_names=("momentum", "correct_bias", "nesterov", "use_shape_scaling"),
            weight_decay_method=weight_decay_method,
        )


@register_optimizer("laprop")
class LaProp(TwoMomentsOptimizer):
    """LaProp optimizer (Ziyin et al., 2020): Adam with the gradient normalized before the first-moment update."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        correct_bias: bool = True,
        frob_normalize: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        _validate_common_hparams(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if frob_normalize and weight_decay != 0.0:
            logging.error("LaProp with frob_normalize=True is intended to be used with weight_decay=0.0.")
        self.frob_normalize = frob_normalize
        super().__init__(
            params,
            defaults=dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                correct_bias=correct_bias,
            ),
            update_fn=calculate_laprop_update,
            update_kwarg_names=("betas", "eps", "correct_bias"),
            weight_decay_method=weight_decay_method,
        )

    @override
    def pre_step_inplace(self, p: torch.Tensor, group: dict) -> Any:
        return p.data.norm() if self.frob_normalize else None

    @override
    def post_step_inplace(self, p: torch.Tensor, group: dict, ctx: Any) -> None:
        if self.frob_normalize:
            pre_norm = ctx
            p.data.mul_(pre_norm / p.data.norm().clamp_min(group["eps"]))


@register_optimizer("sim_ademamix")
class SimplifiedAdEMAMix(TwoMomentsOptimizer):
    """Simplified AdEMAMix: two-buffer variant mixing alpha-scaled current gradient into a theory-style first-moment EMA."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9999, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        correct_bias: bool = True,
        num_beta_fast_warmup_steps: int | None = None,
        min_beta_fast: float = 0.9,
        alpha: float = 2.0,
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        _validate_common_hparams(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if not 0.0 <= min_beta_fast < 1.0:
            raise ValueError(f"Invalid min_beta_fast: {min_beta_fast}")
        if num_beta_fast_warmup_steps is not None and num_beta_fast_warmup_steps <= 0:
            raise ValueError(f"Invalid num_beta_fast_warmup_steps: {num_beta_fast_warmup_steps}")
        super().__init__(
            params,
            defaults=dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                correct_bias=correct_bias,
                num_beta_fast_warmup_steps=num_beta_fast_warmup_steps,
                min_beta_fast=min_beta_fast,
                alpha=alpha,
            ),
            update_fn=calculate_sim_ademamix_update,
            update_kwarg_names=(
                "betas",
                "eps",
                "correct_bias",
                "num_beta_fast_warmup_steps",
                "min_beta_fast",
                "alpha",
            ),
            weight_decay_method=weight_decay_method,
        )
