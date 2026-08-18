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
from emerging_optimizers.weight_update_hooks import HyperballHook


__all__ = ["MuonHyperball"]


@registry.register_optimizer("muon_hyperball")
class MuonHyperball(muon.Muon[None]):
    """Muon optimizer with hyperball-style norm-preserving weight updates.

    This optimizer extends Muon by performing gradient descent on the sphere manifold
    while preserving the weight norm. The update rule is:

    .. math::

        W_{t+1} = R \\cdot \\text{normalize}(W_t - \\text{lr} \\cdot R \\cdot \\text{normalize}(\\text{update}))

    where :math:`R` is the user-specified Frobenius norm. This keeps the weight matrix at
    constant scale while updating.

    Warning:
        This optimizer is experimental and may change in future versions.

    See :class:`~emerging_optimizers.orthogonalized_optimizers.muon.Muon` for full documentation
    of the base Muon optimizer.


    Args:
        *args: Arguments passed to Muon.
        hyperball_radius: Fixed radius for the hyperball. All parameters must
            already have this Frobenius norm at construction time.
        hyperball_eps: Epsilon for numerical stability in normalization.
        **kwargs: Keyword arguments passed to Muon.

    Raises:
        ValueError: If any parameter has zero norm, or if a parameter's
            Frobenius norm does not match ``hyperball_radius``.

    """

    def __init__(
        self,
        *args: Any,
        hyperball_radius: float,
        hyperball_eps: float = 1e-15,
        **kwargs: Any,
    ) -> None:
        if "weight_update_hook" in kwargs:
            raise KeyError(
                "MuonHyperball does not accept a 'weight_update_hook' argument; "
                "it manages its own HyperballHook internally."
            )
        kwargs["weight_update_hook"] = HyperballHook(radius=hyperball_radius, eps=hyperball_eps)
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    p_norm = torch.linalg.vector_norm(p, dtype=torch.float32)
                    if p_norm < hyperball_eps:  # p_norm is non-negative, abs() is not needed
                        raise ValueError(
                            "MuonHyperball requires all parameters to have non-zero norm. "
                            "Found parameter with almost zero norm."
                        )
                    if not torch.isclose(
                        p_norm,
                        torch.tensor(hyperball_radius, dtype=p_norm.dtype, device=p_norm.device),
                        atol=0,
                        rtol=1e-5,
                    ):
                        raise ValueError(
                            f"hyperball_radius={hyperball_radius} was specified but a parameter "
                            f"has Frobenius norm {p_norm.item()}. Rescale your model parameters to the "
                            f"desired radius before constructing the optimizer."
                        )
