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

import torch
import torch.nn.functional as F


__all__ = [
    "SinkhornMapper",
]


class SinkhornMapper:
    """
    Applies the Sinkhorn-Knopp mapping in place on the input tensor:
    Input -> [Exp] -> [Iterative Row/Col Normalization]

    Based on Deepseek's Manifold-Constrained Hyperconnections (https://arxiv.org/abs/2512.24880)

    Args:
        t_max: The number of iterations to run the Sinkhorn-Knopp mapping.
        epsilon: The epsilon value to use for the Sinkhorn-Knopp mapping for numerical stability.
    """

    def __init__(self, t_max: int = 20, epsilon: float = 1e-8):
        self.t_max = t_max
        self.epsilon = epsilon

    @torch.no_grad()
    def _sinkhorn_inplace(self, x: torch.Tensor) -> None:
        # Enforce positivity via exp in place
        x.exp_()

        # Iterative normalization of rows and columns
        for _ in range(self.t_max):
            # Normalize columns
            F.normalize(x, p=1, dim=-2, eps=self.epsilon, out=x)
            # Normalize rows
            F.normalize(x, p=1, dim=-1, eps=self.epsilon, out=x)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> None:
        self._sinkhorn_inplace(x)
