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
import torch

from ._schedulers import _linear_half_life_warmup_scheduler


__all__ = [
    "calculate_sim_ademamix_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_sim_ademamix_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    num_beta_fast_warmup_steps: int | None,
    min_beta_fast: float,
    betas: tuple[float, float],
    step: int,
    eps: float,
    correct_bias: bool,
    alpha: float = 2,
) -> torch.Tensor:
    """Performs simplified AdEMAMix update.

    This function performs the computation of 1 step of simplified AdEMAMix.
    Based on https://github.com/DepenM/Simplified-AdEMAMix/blob/main/simplified_AdEMAMix.py
    and https://arxiv.org/abs/2409.03137.

    The update rule is as follows:

    .. math::
        m_t = \\beta_{\\text{fast}} m_{t-1} + g_t \\\\
        v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{m}_t = \\frac{m_t}{(1 - \\beta_{\\text{fast}}^t) / (1 - \\beta_{\\text{fast}})} \\\\
        \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\
        \\text{update} = \\frac{\\alpha g_t + \\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        exp_avg_sq: The accumulated second moment of the gradient.
        num_beta_fast_warmup_steps: Number of warmup steps used to increase beta_fast
        min_beta_fast: The minimum beta_fast value used at initialization
        betas: The EMA beta coefficients for the Adam update.
        step: The current step of the optimizer, used to compute the bias correction terms.
        eps: The epsilon for the Adam second moment update.
        correct_bias: Whether to correct the bias of the AdEMAMix update.
        alpha: Coeficient for mixing the current gradient and EMA.

    Returns:
        The simplified-AdEMAMix update.
    """
    beta_fast_final, beta2 = betas

    # Compute beta_fast based on scheduler
    if num_beta_fast_warmup_steps is not None:
        beta_fast = _linear_half_life_warmup_scheduler(
            step, beta_end=beta_fast_final, beta_start=min_beta_fast, num_warmup_steps=num_beta_fast_warmup_steps
        )
    else:
        beta_fast = beta_fast_final

    # Decay the first moment "theory style": https://arxiv.org/abs/2502.02431
    exp_avg.mul_(beta_fast).add_(grad, alpha=1.0)

    # Decay the second moment exponential moving average
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

    if correct_bias:
        # theory style bias correction
        bias_correction1 = (1 - beta_fast**step) / (1 - beta_fast)
        bias_correction2 = 1 - beta2**step
    else:
        bias_correction1 = 1
        bias_correction2 = 1

    # step size correction for optimizer states EMA
    momentum = exp_avg / bias_correction1
    adam_second_moment = exp_avg_sq / bias_correction2
    adam_second_moment = adam_second_moment.sqrt() + eps

    return (alpha * grad + momentum) / adam_second_moment
