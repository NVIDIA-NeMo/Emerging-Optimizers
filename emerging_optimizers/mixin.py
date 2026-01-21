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
WeightUpdateT = Literal["sgd", "hyperball"]


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


class WeightUpdateMixin:
    """Mixin for weight update strategies.

    Supports different types of weight updates:

    - "simple": Standard SGD-style update: W_{t+1} = W_t - lr * update
    - "hyperball": Norm-preserving update on the sphere manifold:
        W_{t+1} = R * normalize(W_t - lr * R * normalize(update))
      where R is the Frobenius norm of W_t. This keeps the weight matrix
      at constant scale while updating.

    The normalized update is useful for training with normalized weights,
    as it performs gradient descent on the sphere manifold while preserving
    the weight norm.
    """

    def _apply_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
        lr: float,
        eps: float = 1e-8,
    ) -> None:
        """Apply weight update to parameter tensor in-place.

        Args:
            p: Parameter tensor to update
            update: The update/gradient tensor (already processed by optimizer)
            lr: Learning rate
            eps: Epsilon for numerical stability in normalization
        """
        weight_update_method = getattr(self, "weight_update_method", "sgd")

        if weight_update_method == "sgd":
            # Standard SGD-style update: W_{t+1} = W_t - lr * update
            p.add_(update, alpha=-lr)

        elif weight_update_method == "hyperball":
            # Norm-preserving update: W_{t+1} = R * normalize(W_t - lr * R * normalize(update))
            # Compute R = ||W_t||_F (Frobenius norm)
            R = p.norm()

            # Normalize the update (Frobenius norm over all elements)
            normalized_update = update / (update.norm() + eps)

            # Compute W_t - lr * R * normalize(update)
            p.add_(normalized_update, alpha=-lr * R)

            # Normalize the result and scale back by R: p = R * (p / ||p||)
            p.mul_(R / (p.norm() + eps))

        else:
            raise ValueError(f"Invalid weight update method: {weight_update_method}")
