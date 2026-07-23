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
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.soap import SOAP
from emerging_optimizers.soap.stacked_soap import StackedSoap, _stack_2d, _unstack


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class StackedSoapTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(shape=[(8, 5), (4, 6, 3), (4, 3, 6)])
    def test_smoke(self, shape) -> None:
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        opt = StackedSoap([p], lr=1e-2, weight_decay=0.01)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt.step()
        self.assertTrue(torch.isfinite(p).all())

    @parameterized.product(shape=[(8, 5), (4, 6, 3), (4, 3, 6), (4, 5, 5)])
    def test_stack_unstack_shapes_and_roundtrip(self, shape) -> None:
        x = torch.randn(shape, device=self.device)

        if x.ndim == 2:
            expected_2d = shape
        else:
            b, m, n = shape
            expected_2d = (m, b * n) if n <= m else (b * m, n)

        stacked = _stack_2d(x)
        self.assertEqual(stacked.shape, torch.Size(expected_2d))

        restored = _unstack(stacked, x.shape)
        self.assertEqual(restored.shape, x.shape)
        assert_equal(restored, x)

    @parameterized.product(shape=[(8, 5), (16, 16), (5, 7)])
    def test_2d_input_7steps_matches_vanilla_soap(self, shape) -> None:
        x = torch.randn(shape, device=self.device)
        p_stacked = torch.nn.Parameter(x.clone())
        p_ref = torch.nn.Parameter(x.clone())

        opt_stacked = StackedSoap([p_stacked], lr=1e-2, weight_decay=0.01)
        opt_ref = SOAP(
            [p_ref],
            1e-2,
            weight_decay=0.01,
            weight_decay_method="decoupled",
            nesterov=False,
            correct_bias=True,
            use_eigh=False,
            power_iter_steps=1,
            use_kl_shampoo=True,
        )

        for _ in range(7):
            grad = torch.randn(shape, device=self.device)
            p_stacked.grad = grad.clone()
            p_ref.grad = grad.clone()
            opt_stacked.step()
            opt_ref.step()
            assert_equal(
                p_stacked.detach(),
                p_ref.detach(),
                msg=lambda m: f"StackedSoap must match stock SOAP exactly on 2D params.\n\n{m}",
            )

    @parameterized.product(shape=[(4, 6, 3), (4, 3, 6)])
    def test_3d_input_5steps_matches_vanilla_soap(self, shape) -> None:
        """StackedSoap on a 3D param must match vanilla SOAP run on the manually stacked 2D param."""
        x = torch.randn(shape, device=self.device)
        p_stacked = torch.nn.Parameter(x.clone())
        # Reference is vanilla SOAP on the 2D stacking of the same parameter.
        p_ref = torch.nn.Parameter(_stack_2d(x).clone())

        opt_stacked = StackedSoap([p_stacked], lr=1e-2, weight_decay=0.01)
        opt_ref = SOAP(
            [p_ref],
            1e-2,
            weight_decay=0.01,
            weight_decay_method="decoupled",
            nesterov=False,
            correct_bias=True,
            use_eigh=False,
            power_iter_steps=1,
            use_kl_shampoo=True,
        )

        for _ in range(5):
            grad = torch.randn(shape, device=self.device)
            p_stacked.grad = grad.clone()
            p_ref.grad = _stack_2d(grad)
            opt_stacked.step()
            opt_ref.step()
            assert_equal(
                _stack_2d(p_stacked.detach()),
                p_ref.detach(),
                msg=lambda m: f"StackedSoap on a 3D param must match vanilla SOAP on its 2D stacking.\n\n{m}",
            )


if __name__ == "__main__":
    absltest.main()
