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
from typing import Protocol

import torch


__all__ = ["NoOpWeightUpdateHook", "WeightUpdateHook"]


class WeightUpdateHook(Protocol):
    """Protocol for behavior around an optimizer's final in-place weight update."""

    def pre_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor | None:
        """Called immediately before ``p.add_(update, alpha=-lr)`` and returns pre-update state."""

    def post_weight_update_inplace(
        self,
        p: torch.Tensor,
        pre_update_state: torch.Tensor | None,
    ) -> None:
        """Called after the optimizer's final update and optimizer-specific post-update hook."""


class NoOpWeightUpdateHook:
    """Default hook that leaves the optimizer update unchanged."""

    def pre_weight_update_inplace(
        self,
        p: torch.Tensor,
        update: torch.Tensor,
    ) -> None:
        return None

    def post_weight_update_inplace(
        self,
        p: torch.Tensor,
        pre_update_state: torch.Tensor | None,
    ) -> None:
        pass
