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
import torch


__all__ = ["RadialBrake"]


class RadialBrake:
    """Dampen radial norm changes after an optimizer update.

    The optimizer first applies its usual update ``w = w_prev + dw``. This hook then rescales ``w`` so that

    .. math::

        \\|w_{brake}\\| = \\|w_{prev}\\| + s(\\|w\\| - \\|w_{prev}\\|)

    where ``s`` is ``outward_scale_factor`` when the update increases the norm, otherwise
    ``inward_scale_factor``.
    """

    def __init__(
        self,
        outward_scale_factor: float = 0.5,
        inward_scale_factor: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        if not 0.0 <= outward_scale_factor <= 1.0:
            raise ValueError(f"outward_scale_factor must be in [0, 1], got {outward_scale_factor}")
        if not 0.0 <= inward_scale_factor <= 1.0:
            raise ValueError(f"inward_scale_factor must be in [0, 1], got {inward_scale_factor}")
        self.outward_scale_factor = outward_scale_factor
        self.inward_scale_factor = inward_scale_factor
        self.eps = eps

    def pre_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        return torch.linalg.vector_norm(p.detach().to(torch.float32))

    def post_weight_update_inplace(
        self,
        p: torch.Tensor,
        pre_update_state: torch.Tensor | None,
    ) -> None:
        if pre_update_state is None:
            raise RuntimeError("RadialBrake requires pre-update norm state")
        pre_norm = pre_update_state
        post_norm = torch.linalg.vector_norm(p.detach().to(torch.float32))
        norm_delta = post_norm - pre_norm
        scale_factor = self.outward_scale_factor if norm_delta.item() > 0 else self.inward_scale_factor
        target_norm = pre_norm + scale_factor * norm_delta
        p.mul_((target_norm / post_norm.clamp_min(self.eps)).to(dtype=p.dtype))
