# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        sinkhorn_iters: The number of iterations to run the Sinkhorn-Knopp mapping.
        sinkhorn_eps: The epsilon value to use for the Sinkhorn-Knopp mapping.
        doubly_stochastic_tolerance: Tolerance for validating that parameters are close to
            doubly-stochastic. Defaults to 0.1.
    """

    def __init__(
        self,
        *args: Any,
        sinkhorn_iters: int = 20,
        sinkhorn_eps: float = 1e-8,
        doubly_stochastic_tolerance: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Validate sinkhorn mapper parameters
        if sinkhorn_iters < 1:
            raise ValueError(f"sinkhorn_iters must be at least 1, got {sinkhorn_iters}")
        if sinkhorn_eps <= 0:
            raise ValueError(f"sinkhorn_eps must be positive, got {sinkhorn_eps}")
        if doubly_stochastic_tolerance <= 0:
            raise ValueError(f"doubly_stochastic_tolerance must be positive, got {doubly_stochastic_tolerance}")

        self.sinkhorn_mapper = SinkhornMapper(sinkhorn_iters=sinkhorn_iters, epsilon=sinkhorn_eps)
        self.doubly_stochastic_tolerance = doubly_stochastic_tolerance

        for group in self.param_groups:
            for p in group["params"]:
                # Validate parameter is 2D
                if p.dim() != 2:
                    raise ValueError(
                        f"{self.__class__.__name__} only supports 2D parameters, "
                        f"but got parameter with shape {p.shape} (dim={p.dim()})"
                    )
                # Validate that parameter is close to doubly-stochastic.
                # A doubly-stochastic matrix has non-negative entries with row sums and column sums equal to 1.

                # Check non-negativity
                if (p < 0).any():
                    min_val = p.min().item()
                    raise ValueError(
                        f"Parameter with shape {p.shape} contains negative values (min={min_val:.6f}). "
                        f"Doubly-stochastic matrices must have non-negative entries."
                    )

                # Check row and column sum constraints
                row_sums = p.sum(dim=-1)
                col_sums = p.sum(dim=-2)
                max_row_deviation = (row_sums - 1.0).abs().max().item()
                max_col_deviation = (col_sums - 1.0).abs().max().item()

                if (
                    max_row_deviation > self.doubly_stochastic_tolerance
                    or max_col_deviation > self.doubly_stochastic_tolerance
                ):
                    raise ValueError(
                        f"Parameter with shape {p.shape} is not close to doubly-stochastic. "
                        f"Max row sum deviation: {max_row_deviation:.6f}, "
                        f"max column sum deviation: {max_col_deviation:.6f}. "
                        f"Expected deviations < {self.doubly_stochastic_tolerance}. "
                        f"Please initialize parameters as doubly-stochastic matrices."
                    )

    @override
    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Normalize the updated weights in-place using Sinkhorn-Knopp mapping.

        Args:
            p: The parameter tensor (already updated).
        """
        p.copy_(self.sinkhorn_mapper(p))
