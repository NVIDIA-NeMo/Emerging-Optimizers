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

from typing import Literal

import torch


WeightDecayT = Literal["decoupled", "independent", "l2"]
SecondMomentT = Literal["adamuon", "normuon"]
SecondMomentOptionalT = Literal["adamuon", "normuon", None]


class WeightDecayMixin:
    """Mixin for weight decay

    Supports different types of weight decay:

    - "decoupled": weight decay is applied directly to params without changing gradients
    - "independent": similar as decoupled weight decay, but without tying weight decay and learning rate
    - "l2": classic L2 regularization
    """

    def _apply_weight_decay_inplace(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Depends on the weight decay option, p or grad will be updated in place"""
        if weight_decay == 0.0:
            return

        weight_decay_method = getattr(self, "weight_decay_method", "l2")
        if weight_decay_method == "decoupled":
            p.add_(p, alpha=(-weight_decay * lr))
        elif weight_decay_method == "independent":
            p.add_(p, alpha=-weight_decay)
        elif weight_decay_method == "l2":
            grad.add_(p, alpha=weight_decay)
        else:
            raise ValueError(f"Invalid weight decay method: {weight_decay_method}")


class SecondMomentMixin:
    """Mixin for second moment accumulation and adaptive learning rates.

    This mixin provides functionality similar to Adam's second moment (exp_avg_sq),
    which can be applied after other transformations (e.g., orthogonalization).
    It maintains an exponential moving average of squared gradients and applies
    element-wise adaptive scaling.
    """

    def _initialize_second_moment(
        self,
        state: dict[str, torch.Tensor],
        grad: torch.Tensor,
    ) -> None:
        """Initialize the second moment buffer if it doesn't exist.

        The shape of the buffer depends on the second_moment_method:
        - "adamuon": Full elementwise buffer with same shape as grad
        - "normuon": Reduced shape buffer (averaged along -1 if shape[-2] >= shape[-1], else -2)

        Args:
            state: The optimizer state dict for a parameter.
            grad: The gradient tensor (used for shape/dtype).
        """
        second_moment_method = getattr(self, "second_moment_method", "adamuon")
        if "second_moment_buffer" not in state:
            if second_moment_method == "adamuon":
                # Full elementwise second moment
                second_moment = torch.zeros_like(grad)
            elif second_moment_method == "normuon":
                # Row/column-wise second moment - reduced along one dimension
                # Determine which dimension to reduce based on parameter shape
                avg_dim = -1 if grad.shape[-2] >= grad.shape[-1] else -2
                # Specify the shape with reduced dimension
                second_moment_shape = list(grad.shape)
                second_moment_shape[avg_dim] = 1
                second_moment = torch.zeros(second_moment_shape, dtype=grad.dtype, device=grad.device)
            else:
                raise ValueError(f"Invalid second moment method: {second_moment_method}")

            state["second_moment_buffer"] = second_moment

    def _apply_second_moment_normalization(
        self,
        orth_grad: torch.Tensor,
        second_moment: torch.Tensor,
        beta2: float,
        eps: float,
    ) -> torch.Tensor:
        """Apply AdamW-style second moment accumulation and normalization.

        This method supports two variants:
        - "adamuon": Full elementwise second moment (like AdamW, https://arxiv.org/abs/2507.11005)
        - "normuon": Row or column-wise second moment (https://arxiv.org/abs/2510.05491)

        For both methods:
        1. Updates the second moment as an EMA of squared gradients
        2. Returns the adaptively scaled gradient

        Args:
            orth_grad: The orthogonalized gradient tensor.
            second_moment: The second moment buffer from state.
            beta2: The exponential decay rate for second moment.
            eps: Small constant for numerical stability.

        Returns:
            The adaptively scaled weight update tensor.
        """

        second_moment_method = getattr(self, "second_moment_method", "adamuon")

        if second_moment_method == "adamuon":
            # AdamMuon: Full elementwise second moment like AdamW
            # Update second moment with EMA of squared gradient
            second_moment.lerp_(orth_grad.square(), 1 - beta2)

            # AdamW-style division: grad / (sqrt(second_moment) + eps)
            denom = second_moment.sqrt() + eps
            return orth_grad / denom

        elif second_moment_method == "normuon":
            # NorMuon: Row or column-wise second moment
            # Compute mean of squared gradients along one dimension based on shape
            # Average along the longer dimension to preserve structure along shorter dim
            avg_dim = -1 if orth_grad.shape[-2] >= orth_grad.shape[-1] else -2
            v_mean = orth_grad.square().mean(dim=avg_dim, keepdim=True)

            # Update second moment with EMA
            second_moment.lerp_(v_mean, 1 - beta2)

            # NorMuon uses reciprocal square root with clamping
            step_size = second_moment.clamp_min(eps).rsqrt_()
            return orth_grad * step_size

        else:
            raise ValueError(f"Invalid second moment method: {second_moment_method}")
