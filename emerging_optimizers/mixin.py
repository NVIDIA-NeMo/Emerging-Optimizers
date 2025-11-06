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


class WeightDecayMixin:
    """Mixin for weight decay

    Supports different types of weight decay:
    - Decoupled weight decay: weight decay is applied directly to params without changing gradients
    - Independent weight decay: similar as decoupled weight decay, but without tying weight decay and learning rate
    - Classic weight decay: L2 regularization
    """

    def _apply_weight_decay_inplace(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        weight_decay: float,
        use_decoupled_wd: bool,
        use_independent_wd: bool,
    ) -> None:
        """Depends on the weight decay option, p or grad will be updated in place"""
        assert not (use_decoupled_wd and use_independent_wd), (
            "use_decoupled_wd and use_independent_wd cannot be True at the same time"
        )
        if weight_decay == 0.0:
            return

        if use_decoupled_wd:
            p.add_(p, alpha=(-weight_decay * lr))
        elif use_independent_wd:
            p.add_(p, alpha=-weight_decay)
        else:
            grad.add_(p, alpha=weight_decay)
