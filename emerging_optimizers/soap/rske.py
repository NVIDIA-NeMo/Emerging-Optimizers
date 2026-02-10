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

from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry
from emerging_optimizers.soap import soap


__all__ = ["RSKE"]


@registry.register_optimizer("rske")
class RSKE(soap.SOAP):
    """RSKE (Realtime Soap with Kullback–Leibler minimization and Eigen decomposition) optimizer.

    Note:
        RSKE (pronounces "risky") is a variant of SOAP.
        Refer to :class:`~emerging_optimizers.soap.soap.SOAP` for detailed documentation.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            shampoo_beta,
            eps,
            weight_decay,
            weight_decay_method=weight_decay_method,
            use_eigh=True,
            use_kl_shampoo=True,
        )
