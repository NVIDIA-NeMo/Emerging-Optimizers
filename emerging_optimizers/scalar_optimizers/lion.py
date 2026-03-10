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
from typing import override

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.mixin import WeightDecayMixin, WeightDecayT


__all__ = [
    "calculate_lion_update",
    "Lion",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_lion_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    betas: tuple[float, float],
) -> torch.Tensor:
    """Performs the Lion update.

    This function performs the computation of 1 step of Lion update.

    The update rule is as follows:

    .. math::
        \\text{update} = \\text{sign}(\\beta_1 m_{t-1} + (1 - \\beta_1) g_t) \\\\
        m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        betas: The EMA beta coefficients (beta1, beta2) for the Lion update.

    Returns:
        The Lion update.
    """

    beta1, beta2 = betas

    # Compute update using interpolation (Lion's beta1)
    update_momentum = beta1 * exp_avg + (1 - beta1) * grad

    # Update the momentum state (Lion's beta2)
    exp_avg.lerp_(grad, 1 - beta2)

    # Return signed update (no shape scaling for Lion)
    return torch.sign(update_momentum)


@registry.register_optimizer("lion")
class Lion(WeightDecayMixin, torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023).

    A memory-efficient optimizer that uses only sign updates and tracks a single
    exponential moving average (no second moment), resulting in lower memory usage
    than Adam.

    The update rule is:

    .. math::
        p = p \\cdot (1 - \\text{lr} \\cdot \\lambda) \\\\
        \\text{update} = \\text{sign}(\\beta_1 m_{t-1} + (1 - \\beta_1) g_t) \\\\
        m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t \\\\
        p = p - \\text{lr} \\cdot \\text{update}

    where :math:`\\lambda` is the weight decay coefficient.

    References:
        - Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X.,
          Luber, T., Cho, T., Le, Q. V., & Henaff, O. J. *Symbolic Discovery of Optimization Algorithms.*
          arXiv:2302.06675 (2023).
          [`arXiv:2302.06675 <https://arxiv.org/abs/2302.06675>`_]

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        betas: Coefficients (beta1, beta2) for computing the update and running average.
            beta1 is used for the sign update interpolation, beta2 for the momentum EMA update.
        weight_decay: Weight decay coefficient.
        weight_decay_method: Method to apply weight decay, see
            :class:`~emerging_optimizers.mixin.WeightDecayMixin` for more details.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, weight_decay_method=weight_decay_method)
        self.weight_decay_method = weight_decay_method
        super().__init__(params, defaults)

    @override
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

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
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']

                # Weight decay
                self._apply_weight_decay_inplace(p.data, grad, lr, weight_decay)

                # Lion update: sign(beta1 * m + (1-beta1) * g)
                update = calculate_lion_update(grad, exp_avg, betas)
                p.data.add_(update, alpha=-lr)

        return loss
