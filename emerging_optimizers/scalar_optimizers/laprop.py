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
from typing import TYPE_CHECKING, override


if TYPE_CHECKING:
    from typing import overload

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.mixin import WeightDecayMixin, WeightDecayT
from emerging_optimizers.scalar_optimizers.update_functions import calculate_laprop_update


__all__ = [
    "LaProp",
]


@registry.register_optimizer("laprop")
class LaProp(WeightDecayMixin, torch.optim.Optimizer):
    """LaProp optimizer.

    LAProp can be seen as RMSProp with a momentum term, or normalized SGD with momentum.
    This optimizer tracks Adam-style first and second moments, but normalizes the gradient
    before the first-moment update.

    The update rule below assumes ``weight_decay_method="decoupled"`` (the default).
    See :class:`~emerging_optimizers.mixin.WeightDecayMixin` for the other modes.

    .. math::
        p = p \\cdot (1 - \\text{lr} \\cdot \\lambda) \\\\
        v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\
        g'_t = \\frac{g_t}{\\sqrt{\\hat{v}_t} + \\epsilon} \\\\
        m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g'_t \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\\\
        p = p - \\text{lr} \\cdot \\hat{m}_t

    References:
        - Ziyin, L., Wang, Z. T., & Ueda, M. *LaProp: Separating Momentum and
          Adaptivity in Adam.* arXiv:2002.04839 (2020).
          [`arXiv:2002.04839 <https://arxiv.org/abs/2002.04839>`_]

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        betas: Coefficients (beta1, beta2) for first and second moment EMAs.
        eps: Term added to the denominator for numerical stability.
        weight_decay: Weight decay coefficient.
        correct_bias: Whether to apply bias correction to the first and second moment EMAs.
        normalize: Whether to normalize each updated parameter back to its pre-update Frobenius norm.
        normalize_eps: Term used to avoid division by zero during parameter normalization.
        weight_decay_method: Method to apply weight decay, see
            :class:`~emerging_optimizers.mixin.WeightDecayMixin` for more details.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        normalize: bool = False,
        normalize_eps: float = 1e-8,
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 < normalize_eps:
            raise ValueError(f"Invalid normalize_eps value: {normalize_eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if normalize and weight_decay != 0.0:
            logging.warning("LaProp with normalize=True is intended to be used with weight_decay=0.0.")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        self.weight_decay_method = weight_decay_method
        self.normalize = normalize
        self.normalize_eps = normalize_eps
        super().__init__(params, defaults)

    @torch.no_grad()
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        """Performs lazy state initialization for parameters.

        Args:
            group: Parameter group dictionary.
            skip_non_grad_params: If True, skip parameters without gradients.
        """
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        Note:
            When ``weight_decay_method="l2"``, ``p.grad`` is modified in-place
            (the L2 penalty ``weight_decay * p`` is added to the gradient).
            If you need the original gradient after this call, clone it beforehand.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
            The loss from the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._init_group(group)

            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            correct_bias = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue  # pragma: no cover

                grad = p.grad
                state = self.state[p]
                state["step"] += 1
                pre_norm = p.data.norm() if self.normalize else None

                self._apply_weight_decay_inplace(p.data, grad, lr, weight_decay)

                update = calculate_laprop_update(
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    correct_bias,
                    betas,
                    state["step"],
                    eps,
                )
                p.data.add_(update, alpha=-lr)
                if pre_norm is not None:
                    p.data.mul_(pre_norm / p.data.norm().clamp_min(self.normalize_eps))

        return loss
