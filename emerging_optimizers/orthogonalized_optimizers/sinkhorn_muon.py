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
from emerging_optimizers.utils.sinkhorn_mapper import SinkhornMapper


__all__ = ["SinkhornMuon"]


@registry.register_optimizer("sinkhorn_muon")
class SinkhornMuon(muon.Muon):
    """Sinkhorn-Muon optimizer

    This optimizer extends Muon by performing a Sinkhorn-Knopp mapping after the weight update.
    The Sinkhorn-Knopp mapping is an iterative technique for normalizing the rows and columns of a matrix to sum to 1.

    Args:
        *args: Arguments passed to Muon.
        **kwargs: Keyword arguments passed to Muon.
        t_max: The number of iterations to run the Sinkhorn-Knopp mapping.
        epsilon: The epsilon value to use for the Sinkhorn-Knopp mapping.
    """

    def __init__(
        self,
        *args: Any,
        t_max: int = 20,
        epsilon: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Validate sinkhorn mapper parameters
        if t_max < 1:
            raise ValueError(f"t_max must be at least 1, got {t_max}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.sinkhorn_mapper = SinkhornMapper(t_max=t_max, epsilon=epsilon)

        for group in self.param_groups:
            for p in group["params"]:
                # Validate parameter is 2D
                if p.dim() != 2:
                    raise ValueError(
                        f"{self.__class__.__name__} only supports 2D parameters, "
                        f"but got parameter with shape {p.shape} (dim={p.dim()})"
                    )
                # Initialize weights as doubly-stochastic matrices.
                # Apply sigmoid to map values to (0, 1) range, ensuring safe exp() evaluation
                # and preserving relative ordering. Then apply Sinkhorn-Knopp normalization
                # to enforce row and column sum constraints.
                torch.sigmoid_(p)
                self.sinkhorn_mapper(p)

    @override
    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Normalize the updated weights in-place using Sinkhorn-Knopp mapping.

        Args:
            p: The parameter tensor (already updated).
        """
        self.sinkhorn_mapper(p)
