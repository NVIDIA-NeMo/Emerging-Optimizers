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
import math

import torch


__all__ = ["HyperballHook"]


class HyperballHook:
    """Normalize update and post-update weights to a fixed Frobenius norm.

    This hook mirrors the hyperball-style behavior used by MuonHyperball: before the weight update, normalize the
    update to the target radius; after the weight update, project the parameter back to that radius.
    """

    def __init__(
        self,
        radius: float,
        eps: float = 1e-15,
    ) -> None:
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}")
        if not math.isfinite(radius) or radius < eps:
            raise ValueError(f"radius must be finite and at least eps={eps}, got {radius}")
        self.radius = radius
        self.eps = eps

    def _scale_to_radius_inplace(self, tensor: torch.Tensor) -> None:
        norm = torch.linalg.vector_norm(tensor, dtype=torch.float32)
        is_numerical_zero = norm < self.eps
        radius = torch.as_tensor(self.radius, device=tensor.device, dtype=torch.float32)
        scale = torch.where(is_numerical_zero, torch.zeros_like(norm), radius / norm.clamp_min(self.eps))
        tensor.mul_(scale.to(dtype=tensor.dtype))

    def pre_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
    ) -> None:
        self._scale_to_radius_inplace(update)
        return None

    def post_weight_update_inplace(
        self,
        p: torch.Tensor,
        pre_update_state: None,
    ) -> None:
        self._scale_to_radius_inplace(p)
