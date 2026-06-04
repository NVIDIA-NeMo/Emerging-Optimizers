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


__all__ = ["Hyperball"]


class Hyperball:
    """Normalize update and post-update weights to a fixed Frobenius norm.

    This hook mirrors the hyperball-style behavior used by MuonHyperball: before the weight update, normalize the
    update to the target radius; after the weight update, project the parameter back to that radius.
    """

    def __init__(
        self,
        radius: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.radius = radius
        self.eps = eps

    def pre_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        current_norm = torch.linalg.vector_norm(p.detach().to(torch.float32))

        if self.radius is not None:
            radius = torch.as_tensor(self.radius, device=p.device, dtype=torch.float32)
        else:
            if current_norm.item() == 0:
                raise ValueError("Hyperball requires all parameters to have non-zero norm when radius is not fixed.")
            radius = current_norm

        update_norm = torch.linalg.vector_norm(update.to(torch.float32)).clamp_min(self.eps)
        update.mul_((radius / update_norm).to(dtype=update.dtype))
        return radius

    def post_weight_update_inplace(
        self,
        p: torch.Tensor,
        pre_update_state: torch.Tensor | None,
    ) -> None:
        if pre_update_state is None:
            raise RuntimeError("Hyperball requires radius state")
        radius = pre_update_state
        post_norm = torch.linalg.vector_norm(p.detach().to(torch.float32)).clamp_min(self.eps)
        p.mul_((radius / post_norm).to(dtype=p.dtype))
