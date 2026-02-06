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
from absl.testing import absltest, parameterized

from emerging_optimizers import psgd, registry, soap
from emerging_optimizers.orthogonalized_optimizers import adaptive_muon, mop, muon, scion


class TestRegistry(parameterized.TestCase):
    def test_register_optimizer(self):
        @registry.register_optimizer("test_optimizer")
        class TestOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=0.01):
                super().__init__(params, lr)

        self.assertIs(registry.get_optimizer("test_optimizer"), TestOptimizer)

        with self.assertRaisesRegex(ValueError, "already registered"):

            @registry.register_optimizer("test_optimizer")
            class TestOptimizer(torch.optim.Optimizer):
                def __init__(self, params, lr=0.01):
                    super().__init__(params, lr)

    @parameterized.parameters(
        ("muon", muon.Muon),
        ("mop", mop.MOP),
        ("adaptive_muon", adaptive_muon.AdaptiveMuon),
        ("psgd_pro", psgd.PSGDPro),
        ("scion", scion.Scion),
        ("soap", soap.SOAP),
    )
    def test_get_optimizer(self, opt_name, expected_opt_cls):
        opt_cls = registry.get_optimizer(opt_name)
        self.assertIs(opt_cls, expected_opt_cls)

    def test_raise_error_for_unknown_optimizer(self):
        with self.assertRaisesRegex(ValueError, "not found in the registry"):
            registry.get_optimizer("unknown_optimizer")

    def test_get_configured_optimizer_smoke(self):
        opt_cls = registry.get_configured_optimizer("muon", lr=0.01)
        assert opt_cls is not muon.Muon
        _ = opt_cls([torch.randn(10, 10)], extra_scale_factor=0.2)


if __name__ == "__main__":
    absltest.main()
