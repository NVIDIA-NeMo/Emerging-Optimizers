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

from typing import Any

import torch

from emerging_optimizers import registry
from emerging_optimizers.orthogonalized_optimizers import muon
from emerging_optimizers.weight_update_hooks import Hyperball


__all__ = ["MuonHyperball"]


@registry.register_optimizer("muon_hyperball")
class MuonHyperball(muon.Muon):
    """Muon optimizer with hyperball-style norm-preserving weight updates.

    This optimizer extends Muon by performing gradient descent on the sphere manifold
    while preserving the weight norm. The update rule is:

    .. math::

        W_{t+1} = R \\cdot \\text{normalize}(W_t - \\text{lr} \\cdot R \\cdot \\text{normalize}(\\text{update}))

    where :math:`R` is the Frobenius norm of :math:`W_t` (or a user-specified radius). This keeps
    the weight matrix at constant scale while updating.

    Warning:
        This optimizer is experimental and may change in future versions.

    See :class:`~emerging_optimizers.orthogonalized_optimizers.muon.Muon` for full documentation
    of the base Muon optimizer.


    Args:
        *args: Arguments passed to Muon.
        hyperball_eps: Epsilon for numerical stability in normalization.
            Default: ``1e-8``.
        hyperball_radius: Fixed radius for the hyperball. If ``None`` (default),
            uses each parameter's initial Frobenius norm as its radius. If specified, all
            parameters will be rescaled to have this radius at initialization.
        **kwargs: Keyword arguments passed to Muon.

    """

    def __init__(
        self,
        *args: Any,
        hyperball_eps: float = 1e-8,
        hyperball_radius: float | None = None,
        **kwargs: Any,
    ) -> None:
        if "weight_update_hook" in kwargs:
            raise TypeError(
                "MuonHyperball does not accept a 'weight_update_hook' argument; "
                "it manages its own Hyperball hook internally."
            )
        kwargs["weight_update_hook"] = Hyperball(radius=hyperball_radius, eps=hyperball_eps)
        super().__init__(*args, **kwargs)

        # Validate and optionally rescale parameters based on hyperball_radius.
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    p_norm = p.norm()
                    # Validate that parameter has non-zero norm.
                    if p_norm.item() == 0:
                        raise ValueError(
                            "MuonHyperball requires all parameters to have non-zero norm. Found parameter with zero norm."
                        )
                    # Rescale parameter to have the specified radius if provided.
                    if hyperball_radius is not None:
                        p.mul_(hyperball_radius / p_norm.clamp_min(hyperball_eps))
