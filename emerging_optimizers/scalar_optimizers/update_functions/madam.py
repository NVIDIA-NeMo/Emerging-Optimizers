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


__all__ = [
    "calculate_madam_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_madam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq_scaled: torch.Tensor,
    *,
    betas: tuple[float, float],
    correct_bias: bool,
    step: int,
    scale_log2: float,
) -> torch.Tensor:
    """Performs the magnitude-aware Adam (MAdam) update.

    Vanilla Adam adds an ``eps`` to ``sqrt(v_hat)`` so the denominator never goes to
    zero. The cost is that when the natural scale of ``sqrt(v_hat)`` approaches
    ``eps``, ``eps`` quietly reshapes the update (see ``docs/primer/epsilon.md``).
    MAdam removes ``eps`` entirely and prevents zero division two other ways:

    1. **Magnitude-aware storage.** The second moment is stored in scaled form
       ``v'_t = s * EMA(g_t^2) = EMA((sqrt(s) * g_t)^2)``, with ``s = 2 **
       scale_log2``. Multiplying ``g`` by ``sqrt(s)`` *before* squaring lifts
       tiny gradients above the fp32 underflow boundary, so ``v'_t`` stays
       non-zero whenever any prior gradient was non-zero. ``scale_log2`` is
       constrained to even integers so that ``sqrt(s) = 2 ** (scale_log2 // 2)``
       is itself an exact power of two — the pre-square multiplication is then
       a bare exponent shift with no rounding.
    2. **All-zero mask from ``exp_avg``.** Parameters whose gradient has been
       exactly zero from step 1 onward have ``exp_avg == 0`` (the EMA started at
       zero and never received non-zero input). For those entries both the
       numerator and the denominator are zero; we mask the update to zero rather
       than dividing.

    The update rule is:

    .. math::
        v'_t = \\beta_2 v'_{t-1} + (1 - \\beta_2) \\left( \\sqrt{s}\\, g_t \\right)^2 \\\\
        m_t  = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}'_t = \\frac{v'_t}{1 - \\beta_2^t} \\\\
        \\text{update} = \\begin{cases}
            \\dfrac{\\sqrt{s}\\, \\hat{m}_t}{\\sqrt{\\hat{v}'_t}} & m_t \\ne 0 \\\\
            0 & m_t = 0
        \\end{cases}


    Note:
        The mask uses ``exp_avg`` rather than ``grad`` so a parameter whose
        gradient is zero on the current step but was non-zero earlier still
        receives a momentum-driven update.

    Note:
        If a non-zero gradient ever produced a squared value that even
        ``sqrt(s) * g`` could not lift above underflow, ``exp_avg_sq_scaled`` can
        become zero while ``exp_avg`` is non-zero. The update will then be ``inf``
        / ``nan`` for those entries — i.e. it is the caller's job to pick a
        ``scale_log2`` large enough for the model's gradient regime.

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient (modified in place).
        exp_avg_sq_scaled: The accumulated **scaled** second moment, storing
            ``s * EMA(g_t^2)`` (modified in place). Allocate as ``zeros_like(p)``;
            the caller is responsible for using the same ``scale`` across steps.
        betas: The EMA beta coefficients ``(beta1, beta2)``.
        correct_bias: Whether to apply Adam-style bias correction.
        step: Current optimizer step (1-based), used for bias correction.
        scale_log2: ``log2`` of the magnitude scaling factor ``s = 2 **
            scale_log2`` used for the second-moment storage. When it is an even
            integer, ``sqrt(s) = 2 ** (scale_log2 // 2)`` is exactly representable
            in floating point.

    Returns:
        The MAdam update.
    """

    beta1, beta2 = betas
    assert scale_log2 // 2 == scale_log2 / 2, "scale_log2 should be an even integer"
    grad_scale = 2.0 ** (scale_log2 // 2)

    # First moment as usual; second moment stored scaled. Multiply before squaring
    # so small gradients are lifted above the fp32 underflow boundary first.
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq_scaled.lerp_((grad * grad_scale).square(), 1 - beta2)

    # Step-size correction for the EMAs
    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if correct_bias:
        bias_correction1 = 1.0 - beta1 ** (step)
        bias_correction2 = 1.0 - beta2 ** (step)

    momentum = exp_avg / bias_correction1

    second_moment_scaled = exp_avg_sq_scaled / bias_correction2
    out = momentum / second_moment_scaled.sqrt() * grad_scale

    zero_mask = exp_avg == 0
    out.masked_fill_(zero_mask, 0.0)
    return out
