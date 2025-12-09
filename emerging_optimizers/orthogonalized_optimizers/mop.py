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


from typing import Optional

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc


__all__ = ["MOP"]


class MOP(OrthogonalizedOptimizer):
    """MOP: Momentum Orthogonalized by Polar decomposition

    warning:
        This optimizer is experimental and not yet thoroughly tested.


    Args:
        {_args_doc}
        scale_mode: The type of scale factor to use for the update. Defaults to "spectral" style scaling.
        extra_scale_factor: The additional scale factor to use for the update.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        weight_decay: float = 0.01,
        *,
        use_nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: str = "highest",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
    ) -> None:
        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            orth_grad, _ = polar_via_svd(grad, False)

            scale_factor = muon.get_muon_scale_factor(grad.size(-2), grad.size(-1), mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )


MOP.__doc__ = MOP.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]


def polar_via_svd(A: torch.Tensor, return_p: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute polar decomposition via SVD"""

    U_svd, S, Vh = torch.linalg.svd(A, full_matrices=False)
    U_polar = U_svd @ Vh

    if not return_p:
        return U_polar, None
    else:
        H = Vh.mH @ torch.diag(S) @ Vh
        return U_polar, H
