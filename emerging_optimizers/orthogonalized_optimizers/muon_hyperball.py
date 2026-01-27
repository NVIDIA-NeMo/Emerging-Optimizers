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

    .. math::

        W_{t+1} = R \\cdot \\text{normalize}(W_t - \\text{lr} \\cdot R \\cdot \\text{normalize}(\\text{update}))

    where :math:`R` is the Frobenius norm of :math:`W_t`. This keeps the weight matrix at constant
    scale while updating.

    Warning:
        This optimizer is experimental and may change in future versions.

    See :class:`~emerging_optimizers.orthogonalized_optimizers.muon.Muon` for full documentation
    of the base Muon optimizer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize MuonHyperball optimizer.

        Args:
            *args: Arguments passed to Muon.
            hyperball_eps: Epsilon for numerical stability in normalization. Defaults to 1e-8.
            **kwargs: Keyword arguments passed to Muon.
        """
        self.hyperball_eps: float = kwargs.pop("hyperball_eps", 1e-8)
        # TODO(mkhona): Allow user to pick hyperball R
        super().__init__(*args, **kwargs)

    @override
    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Store the original weight norm and normalize the update using Frobenius norm.

        Args:
            p: The parameter tensor.
            update: The orthogonalized gradient tensor.
        """
        # Store R = ||W_t||_F (Frobenius norm) in per-parameter state
        R = p.norm().item()
        self.state[p]["hyperball_R"] = R

        # Normalize the update in-place and scale by R
        # This modifies update to be: R * normalize(update) using Frobenius norm.
        update_norm = update.norm()
        if update_norm > self.hyperball_eps:
            update.mul_(R / update_norm)

    @override
    def post_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Normalize the updated weights and scale back to original norm using Frobenius norm.

        Args:
            p: The parameter tensor (already updated).
            update: The orthogonalized gradient tensor that was applied.
        """
        # Retrieve R from per-parameter state
        R = self.state[p]["hyperball_R"]

        # Normalize the result and scale back by R: p = R * (p / ||p||_F) using Frobenius norm.
        p_norm = p.norm()
        if p_norm > self.hyperball_eps:
            p.mul_(R / p_norm)
