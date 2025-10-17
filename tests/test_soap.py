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
from absl.testing import absltest, parameterized

from emerging_optimizers.soap import soap


class SoapTest(parameterized.TestCase):
    def setUp(self):
        self.default_config = {
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "precondition_frequency": 1,
            "shampoo_beta": 0.95,
            "precondition_1d": False,
            "adam_warmup_steps": 1,
            "fp32_matmul_prec": "highest",
            "use_adaptive_criteria": False,
            "trace_normalization": False,
            "power_iter_steps": 1,
        }

    def test_10steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device="cuda")
        optimizer = soap.SOAP(
            [param],
            **self.default_config,
        )

        for _ in range(10):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None


if __name__ == "__main__":
    absltest.main()
