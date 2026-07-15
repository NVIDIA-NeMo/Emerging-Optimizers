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
    "calculate_rmsprop_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_rmsprop_update(
    grad: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    alpha: float,
    eps: float,
) -> torch.Tensor:
    """Performs the RMSProp update.

    This function performs the computation of 1 step of RMSProp, matching
    ``torch.optim.RMSprop`` with ``momentum=0`` and ``centered=False``.

    The update rule is as follows:

    .. math::
        v_t = \\alpha v_{t-1} + (1 - \\alpha) g_t^2 \\\\
        \\text{update} = \\frac{g_t}{\\sqrt{v_t} + \\epsilon} \\\\

    Args:
        grad: The gradient tensor.
        exp_avg_sq: The accumulated second moment of the gradient (modified in place).
        alpha: The EMA coefficient for the second moment.
        eps: Epsilon for the second-moment denominator.

    Returns:
        The RMSProp update.
    """
    exp_avg_sq.lerp_(grad.square(), 1 - alpha)
    return grad / (exp_avg_sq.sqrt() + eps)
