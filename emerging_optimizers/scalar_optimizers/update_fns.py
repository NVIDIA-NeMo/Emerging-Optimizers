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
import math

import torch


__all__ = [
    "calculate_adam_update",
    "calculate_ademamix_update",
    "calculate_laprop_update",
    "calculate_lion_update",
    "calculate_signum_update",
    "calculate_sim_ademamix_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    betas: tuple[float, float],
    correct_bias: bool,
    nesterov: bool,
    step: int,
    eps: float,
) -> torch.Tensor:
    """Performs the Adam update.

    This function performs the computation of 1 step of Adam.

    The update rule is as follows:

    .. math::
        m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
        v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\\\
        \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\
        \\text{update} = \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} \\\\

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        exp_avg_sq: The accumulated second moment of the gradient.
        betas: The EMA beta coefficients for the Adam update.
        correct_bias: Whether to correct the bias of the Adam update.
        nesterov: Whether to use nesterov momentum.
        step: The current step of the optimizer, used to compute the bias correction terms.
        eps: The epsilon for the Adam second moment update.

    Returns:
        The Adam-update.
    """

    beta1, beta2 = betas

    # Decay the first and second moment running average coefficient
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

    # step size correction for optimizer states EMA
    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if correct_bias:
        # step size correction for ADAM moments EMA
        bias_correction1 = 1.0 - beta1 ** (step)
        bias_correction2 = 1.0 - beta2 ** (step)

    if nesterov:
        # Apply nesterov momentum correction, optionally with bias correction
        bias_correction_nesterov = (1 - beta1 ** (step + 1)) if correct_bias else 1.0
        momentum = beta1 * exp_avg / bias_correction_nesterov + (1 - beta1) * grad / bias_correction1
    else:
        # Use standard momentum, optionally with bias correction
        momentum = exp_avg / bias_correction1

    # construct the denominator of the inner ADAM optimizer
    adam_second_moment = exp_avg_sq / bias_correction2
    adam_second_moment = adam_second_moment.sqrt() + eps
    return momentum / adam_second_moment


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


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_signum_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    momentum: float,
    correct_bias: bool,
    nesterov: bool,
    step: int,
    use_shape_scaling: bool = False,
) -> torch.Tensor:
    """Performs the sign-SGD or Signum update.

    This function performs the computation of 1 step of sign-SGD or Signum.
    Based on https://arxiv.org/abs/1802.04434.
    When using signSGD with shape scaling, general recommendation is to
    scale :math:`lr = \\text{adam lr} \\cdot \\text{network width} \\cdot \\frac{2}{\\text{rows} + \\text{cols}}`.
    This is for learning rate transfer with width scaling (https://arxiv.org/abs/2506.07254v1).

    The update rule is as follows:

    .. math::
        m_t = \\beta m_{t-1} + (1 - \\beta) g_t \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta^t} \\\\
        \\text{update} = \\text{sign}(\\hat{m}_t)

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        momentum: The EMA beta coefficients for the momentum update.
        correct_bias: Whether to correct the bias of the momentum update.
        nesterov: Whether to use nesterov momentum.
        step: The current step of the optimizer, used to compute the bias correction terms.
        use_shape_scaling: Whether to scale the update by the shape of the tensor.

    Returns:
        The sign-SGD/Signum update.
    """

    # Standard SignSGD: update momentum first, then compute signed update
    # Decay the momentum with exponential moving average
    exp_avg.lerp_(grad, 1 - momentum)

    if correct_bias:
        bias_correction1 = 1 - momentum**step
    else:
        bias_correction1 = 1

    if nesterov:
        # Apply nesterov momentum correction, optionally with bias correction
        bias_correction_nesterov = (1 - momentum ** (step + 1)) if correct_bias else 1.0
        new_momentum = momentum * exp_avg / bias_correction_nesterov + (1 - momentum) * grad / bias_correction1
    else:
        # Use standard momentum, optionally with bias correction
        new_momentum = exp_avg / bias_correction1

    # scale update by shape of tensor to ensure consistent update size: https://arxiv.org/abs/2506.07254
    if use_shape_scaling:
        m, n = grad.shape
        return torch.sign(new_momentum) * (2 / (m + n))
    else:
        return torch.sign(new_momentum)


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_laprop_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    correct_bias: bool,
    betas: tuple[float, float],
    step: int,
    eps: float,
) -> torch.Tensor:
    """Performs the LAProp/Normalized SGD with momentum update.

    LAProp can be seen as RMSProp with a momentum term, or normalized SGD with momentum.
    Based on https://github.com/Z-T-WANG/LaProp-Optimizer/blob/master/laprop.py
    and https://arxiv.org/abs/2002.04839.

    The update rule is as follows:

    .. math::
        v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\
        g'_t = \\frac{g_t}{\\sqrt{\\hat{v}_t} + \\epsilon} \\\\
        m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g'_t \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\\\
        \\text{update} = \\hat{m}_t

    Args:
        grad: The gradient tensor.
        exp_avg: The exponential moving average of the gradient.
        exp_avg_sq: The exponential moving average of the gradient squared.
        correct_bias: Whether to correct the bias of the Adam update.
        betas: The betas for the exponential moving average.
        step: The current step.
        eps: The epsilon for the second moment update.

    Returns:
        The LAProp update.
    """
    beta1, beta2 = betas

    # Decay the second moment running average coefficient
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

    # step size correction for optimizer states EMA
    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if correct_bias:
        # step size correction for ADAM moments EMA
        bias_correction1 = 1.0 - beta1 ** (step)
        bias_correction2 = 1.0 - beta2 ** (step)

    # construct the denominator of the inner ADAM optimizer
    second_moment = exp_avg_sq / bias_correction2
    second_moment = second_moment.sqrt() + eps

    normalized_grad = grad / second_moment

    # update the exponential moving average of the gradient
    exp_avg.lerp_(normalized_grad, 1 - beta1)

    # return the LAProp update
    return exp_avg / bias_correction1


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


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_ademamix_update(
    grad: torch.Tensor,
    exp_avg_fast: torch.Tensor,
    exp_avg_slow: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    num_beta_slow_warmup_steps: int | None,
    num_alpha_warmup_steps: int | None,
    betas: tuple[float, float, float],
    step: int,
    eps: float,
    correct_bias: bool,
    alpha: float = 2,
) -> torch.Tensor:
    """Performs AdEMAMix update.

    This function performs the computation of 1 step of AdEMAMix.
    Based on https://github.com/apple/ml-ademamix/blob/main/pytorch/ademamix.py
    and https://arxiv.org/abs/2409.03137.

    The update rule is as follows:

    .. math::
        m_t^{\\text{fast}} = \\beta_{\\text{fast}} m_{t-1}^{\\text{fast}} + (1 - \\beta_{\\text{fast}}) g_t \\\\
        m_t^{\\text{slow}} = \\beta_{\\text{slow}} m_{t-1}^{\\text{slow}} + (1 - \\beta_{\\text{slow}}) g_t \\\\
        v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{m}_t^{\\text{fast}} = \\frac{m_t^{\\text{fast}}}{1 - \\beta_{\\text{fast}}^t} \\\\
        \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\
        \\text{update} = \\frac{\\hat{m}_t^{\\text{fast}} + \\alpha m_t^{\\text{slow}}}{\\sqrt{\\hat{v}_t} + \\epsilon}

    Args:
        grad: The gradient tensor.
        exp_avg_fast: The accumulated first moment of the gradient with fast time constant.
        exp_avg_slow: The accumulated first moment of the gradient with slow time constant.
        exp_avg_sq: The accumulated second moment of the gradient.
        num_beta_slow_warmup_steps: Number of warmup steps used to increase beta_slow
        num_alpha_warmup_steps: Number of warmup steps used to increase alpha
        betas: The EMA beta coefficients for the Adam update.
        step: The current step of the optimizer, used to compute the bias correction terms.
        eps: The epsilon for the Adam second moment update.
        correct_bias: Whether to correct the bias of the AdEMAMix update.
        alpha: Coeficient for mixing the current gradient and EMA, the final value to use in case of scheduling.

    Returns:
        The AdEMAMix update.
    """
    beta_fast, beta2, beta_slow_final = betas

    if num_alpha_warmup_steps is not None:
        alpha = _linear_warmup_scheduler(step, alpha_end=alpha, alpha_start=0, num_warmup_steps=num_alpha_warmup_steps)
    else:
        alpha = alpha

    # Compute beta_slow based on scheduler with half-life linear warmup
    # beta_start is usually set to beta_fast
    if num_beta_slow_warmup_steps is not None:
        beta_slow = _linear_half_life_warmup_scheduler(
            step, beta_end=beta_slow_final, beta_start=beta_fast, num_warmup_steps=num_beta_slow_warmup_steps
        )
    else:
        beta_slow = beta_slow_final

    if correct_bias:
        bias_correction1 = 1 - beta_fast**step
        bias_correction2 = 1 - beta2**step
    else:
        bias_correction1 = 1
        bias_correction2 = 1

    # Decay the fast first moment, slow first moment and second moment with an exponential moving average
    if beta_fast != 0.0:
        exp_avg_fast.lerp_(grad, 1 - beta_fast)
    else:
        exp_avg_fast = grad
    exp_avg_slow.lerp_(grad, 1 - beta_slow)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

    # Correct biases of fast moment and adam second moment, slow moment is not corrected
    fast_moment = exp_avg_fast / bias_correction1
    adam_second_moment = exp_avg_sq / bias_correction2
    adam_second_moment = adam_second_moment.sqrt() + eps

    return (fast_moment + alpha * exp_avg_slow) / adam_second_moment


def _half_life_steps(beta: float, eps: float = 1e-8) -> float:
    """Function that maps beta to the number of steps to reach 0.5.

    Equation:
        f(beta) = log(0.5) / log(beta + eps) - 1

    Args:
        beta: The beta parameter.
        eps: A small constant to avoid division by zero.

    Returns:
        The number of steps to reach 0.5.
    """
    return math.log(0.5) / math.log(beta + eps) - 1


def _inverse_half_life_beta(t: float) -> float:
    """Maps number of steps to reach 0.5 to beta.

    Equation:
        f_inv(t) = 0.5^(1 / (t + 1))

    Args:
        t: The number of steps to reach 0.5.

    Returns:
        The beta parameter.
    """
    return math.pow(0.5, 1 / (t + 1))


def _linear_half_life_warmup_scheduler(
    step: int, beta_end: float, beta_start: float = 0, num_warmup_steps: int = 1
) -> float:
    """Half-life linear warmup scheduler for the beta parameter.

    Equation:
        beta = f_inv((1 - step / num_warmup_steps) * f(beta_start) + (step / num_warmup_steps) * f(beta_end))


    Args:
        step: The current step of the optimizer.
        beta_end: The final value of the beta parameter.
        beta_start: The initial value of the beta parameter.
        num_warmup_steps: The number of warmup steps.

    Returns:
        The value of the beta parameter at the current step.
    """

    if step < num_warmup_steps:
        a = step / float(num_warmup_steps)
        return _inverse_half_life_beta((1.0 - a) * _half_life_steps(beta_start) + a * _half_life_steps(beta_end))
    return beta_end


def _linear_warmup_scheduler(step: int, alpha_end: float, alpha_start: float = 0, num_warmup_steps: int = 1) -> float:
    """Linear warmup scheduler for the alpha parameter.

    Equation:
        alpha = (1 - step / num_warmup_steps) * alpha_start + (step / num_warmup_steps) * alpha_end

    Args:
        step: The current step of the optimizer.
        alpha_end: The final value of the alpha parameter.
        alpha_start: The initial value of the alpha parameter.
        num_warmup_steps: The number of warmup steps.

    Returns:
        The value of the alpha parameter at the current step.
    """
    if step < num_warmup_steps:
        a = step / float(num_warmup_steps)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end
