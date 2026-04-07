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

from emerging_optimizers import registry
from emerging_optimizers.orthogonalized_optimizers import muon


__all__ = ["MuonHyperball"]


@registry.register_optimizer("muon_hyperball")
class MuonHyperball(muon.Muon):
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
        hyperball_eps: Epsilon for numerical stability in normalization.
            Default: ``1e-8``.
        hyperball_radius: Fixed radius for the hyperball. All parameters must
            already have this Frobenius norm at construction time.
        **kwargs: Keyword arguments passed to Muon.

    Raises:
        ValueError: If any parameter has zero norm, or if a parameter's
            Frobenius norm does not match ``hyperball_radius``.

    """

    def __init__(
        self,
        *args: Any,
        hyperball_eps: float = 1e-8,
        hyperball_radius: float,
        **kwargs: Any,
    ) -> None:
        self.hyperball_eps = hyperball_eps
        self.hyperball_radius = hyperball_radius
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    p_norm = p.norm()
                    if p_norm == 0:
                        raise ValueError(
                            "MuonHyperball requires all parameters to have non-zero norm. "
                            "Found parameter with zero norm."
                        )
                    if not torch.isclose(
                        p_norm,
                        torch.tensor(self.hyperball_radius, dtype=p.dtype, device=p.device),
                        rtol=1e-5,
                        atol=1e-8,
                    ):
                        raise ValueError(
                            f"hyperball_radius={self.hyperball_radius} was specified but a parameter "
                            f"has Frobenius norm {p_norm.item()}. Rescale your model parameters to the "
                            f"desired radius before constructing the optimizer."
                        )

    @override
    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Normalize the update using Frobenius norm, scaled by R.

        Args:
            p: The parameter tensor.
            update: The orthogonalized gradient tensor.
        """
        if "hyperball_R" not in self.state[p]:
            self.state[p]["hyperball_R"] = torch.tensor(
                self.hyperball_radius, dtype=p.dtype, device=p.device
            )
        R = self.state[p]["hyperball_R"]

        update_norm = update.norm().clamp_min(self.hyperball_eps)
        update.mul_(R / update_norm)

    @override
    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Normalize the updated weights and scale back to original norm using Frobenius norm.

        Args:
            p: The parameter tensor (already updated).
        """
        # Retrieve R from per-parameter state
        R = self.state[p]["hyperball_R"]

        # Normalize the result and scale back by R: p = R * (p / ||p||_F) using Frobenius norm.
        p_norm = p.norm().clamp_min(self.hyperball_eps)
        p.mul_(R / p_norm)

