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

from emerging_optimizers.orthogonalized_optimizers.muon import Muon


__all__ = ["MuonHyperball"]


class MuonHyperball(Muon):
    """Muon optimizer with hyperball-style norm-preserving weight updates.

    This optimizer extends Muon by performing gradient descent on the sphere manifold
    while preserving the weight norm. The update rule is:

        W_{t+1} = R * normalize(W_t - lr * R * normalize(update))

    where R is the Frobenius norm of W_t. This keeps the weight matrix at constant
    scale while updating.

    See :class:`~emerging_optimizers.orthogonalized_optimizers.muon.Muon` for full documentation
    of the base Muon optimizer.
    """

    def __init__(self, *args: Any, hyperball_eps: float = 1e-8, **kwargs: Any) -> None:
        """Initialize MuonHyperball optimizer.

        Args:
            *args: Arguments passed to Muon.
            hyperball_eps: Epsilon for numerical stability in normalization.
            **kwargs: Keyword arguments passed to Muon.
        """
        super().__init__(*args, **kwargs)
        self.hyperball_eps = hyperball_eps
        self._hyperball_R: float = 0.0

    @override
    def pre_weight_update(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Store the original weight norm and normalize the update using Frobenius norm.

        Args:
            p: The parameter tensor.
            update: The orthogonalized gradient tensor.
        """
        # Store R = ||W_t||_F (Frobenius norm)
        self._hyperball_R = p.norm().item()

        # Normalize the update in-place and scale by R
        # This modifies update to be: R * normalize(update) using Frobenius norm.
        update_norm = update.norm()
        if update_norm > self.hyperball_eps:
            update.mul_(self._hyperball_R / update_norm)

    @override
    def post_weight_update(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Normalize the updated weights and scale back to original norm using Frobenius norm.

        Args:
            p: The parameter tensor (already updated).
            update: The orthogonalized gradient tensor that was applied.
        """
        # Normalize the result and scale back by R: p = R * (p / ||p||_F) using Frobenius norm.
        p_norm = p.norm()
        if p_norm > self.hyperball_eps:
            p.mul_(self._hyperball_R / p_norm)
