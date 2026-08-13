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


__all__ = ["RadialBrakeHook"]


class RadialBrakeHook:
    """Dampen radial norm changes after an optimizer update.

    The optimizer first applies its usual update ``w = w_prev + dw``. This hook then rescales ``w`` so that

    .. math::

        \\|w_{brake}\\| = \\|w_{prev}\\| + s(\\|w\\| - \\|w_{prev}\\|)

    where ``s`` is ``outward_scale`` when the update increases the norm, otherwise
    ``inward_scale``.

    Args:
        outward_scale: Fraction of an outward norm change to retain.
        inward_scale: Fraction of an inward norm change to retain.
        eps: Norm threshold below which values are treated as numerical zero.
    """

    def __init__(
        self,
        outward_scale: float = 0.5,
        inward_scale: float = 1.0,
        eps: float = 1e-15,
    ) -> None:
        if not 0.0 <= outward_scale <= 1.0:
            raise ValueError(f"outward_scale must be in [0, 1], got {outward_scale}")
        if not 0.0 <= inward_scale <= 1.0:
            raise ValueError(f"inward_scale must be in [0, 1], got {inward_scale}")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}")
        self.outward_scale = outward_scale
        self.inward_scale = inward_scale
        self.eps = eps

    def _norm_or_zero(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(tensor, dtype=torch.float32)
        return torch.where(norm < self.eps, torch.zeros_like(norm), norm)

    def pre_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        return self._norm_or_zero(p)

    def post_weight_update_inplace(
        self,
        p: torch.Tensor,
        pre_update_state: torch.Tensor,
    ) -> None:
        pre_norm = pre_update_state
        post_norm = self._norm_or_zero(p)
        norm_delta = post_norm - pre_norm
        outward_scale = torch.as_tensor(self.outward_scale, device=p.device, dtype=torch.float32)
        inward_scale = torch.as_tensor(self.inward_scale, device=p.device, dtype=torch.float32)
        scale = torch.where(
            norm_delta >= self.eps,
            outward_scale,
            torch.where(norm_delta <= -self.eps, inward_scale, torch.zeros_like(norm_delta)),
        )
        target_norm = pre_norm + scale * norm_delta
        projection_scale = torch.where(
            post_norm < self.eps,
            torch.zeros_like(post_norm),
            target_norm / post_norm.clamp_min(self.eps),
        )
        p.mul_(projection_scale.to(dtype=p.dtype))
