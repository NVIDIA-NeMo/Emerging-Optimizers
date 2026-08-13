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


__all__ = ["RelativeUpdateHook"]


class RelativeUpdateHook:
    r"""Scale the update norm to the current weight norm without projecting the weight.

    Before the optimizer applies its learning rate, this hook performs

    .. math::

        U \leftarrow \frac{\lVert W\rVert_F}{\lVert U\rVert_F} U.

    Therefore the learning rate directly controls the relative update norm. Unlike
    :class:`HyperballHook`, this hook does not project the weight after the update.
    """

    def __init__(self, eps: float = 1e-15) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps

    def pre_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
    ) -> None:
        weight_norm = torch.linalg.vector_norm(p, dtype=torch.float32)
        update_norm = torch.linalg.vector_norm(update, dtype=torch.float32)
        is_numerical_zero = (weight_norm < self.eps) | (update_norm < self.eps)
        scale = torch.where(
            is_numerical_zero,
            torch.zeros_like(update_norm),
            weight_norm / update_norm.clamp_min(self.eps),
        )
        update.mul_(scale.to(dtype=update.dtype))
        return None

    def post_weight_update_inplace(
        self,
        p: torch.Tensor,
        pre_update_state: None,
    ) -> None:
        pass
