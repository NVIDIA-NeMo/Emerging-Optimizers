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
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import registry
from emerging_optimizers.soap import MOSO


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class MOSOTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        {"shape": (5, 3)},
        {"shape": (3, 5)},
        {"shape": (4, 4)},
    )
    def test_3steps_smoke(self, shape: tuple[int, int]) -> None:
        param = torch.randn(shape, requires_grad=True, device=FLAGS.device)
        optimizer = MOSO(
            [param],
            lr=0.001,
            weight_decay=0.01,
            momentum=0.9,
            shampoo_beta=0.95,
        )

        for _ in range(3):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

    def test_registry(self) -> None:
        self.assertIs(registry.get_optimizer_cls("moso"), MOSO)

    @parameterized.parameters(
        {"shape": (3, 5)},
        {"shape": (5, 3)},
    )
    def test_accumulates_momentum_covariance_on_smaller_side(self, shape: tuple[int, int]) -> None:
        grad = torch.randn(shape, device=FLAGS.device)
        param = torch.zeros(shape, requires_grad=True, device=FLAGS.device)
        param.grad = grad.clone()
        optimizer = MOSO(
            [param],
            lr=0.0,
            momentum=0.0,
            shampoo_beta=0.0,
            weight_decay=0.0,
        )

        optimizer.step()

        state = optimizer.state[param]
        expected_momentum_factor = grad @ grad.T if shape[0] <= shape[1] else grad.T @ grad
        torch.testing.assert_close(
            state["M"],
            expected_momentum_factor,
            atol=1e-6,
            rtol=1e-6,
            msg=lambda msg: f"Momentum covariance mismatch for shape {shape}:\n{msg}",
        )
        self.assertEqual(state["M"].shape, (min(shape), min(shape)))
        self.assertEqual(state["Q_M"].shape, (min(shape), min(shape)))

    @parameterized.parameters(
        {"shape": (4, 8)},
        {"shape": (8, 4)},
    )
    def test_no_ema_is_close_to_one_sided_adam_in_eigenbasis(self, shape: tuple[int, int]) -> None:
        torch.manual_seed(7)
        grad = torch.randn(shape, device=FLAGS.device)
        param = torch.zeros(shape, requires_grad=True, device=FLAGS.device)
        param.grad = grad.clone()
        lr = 0.125
        optimizer = MOSO(
            [param],
            lr=lr,
            momentum=0.0,
            rms_beta=0.0,
            shampoo_beta=0.0,
            eps=1e-12,
            weight_decay=0.0,
        )

        optimizer.step()

        state = optimizer.state[param]
        eigenbasis = state["Q_M"]
        if shape[0] <= shape[1]:
            projected = eigenbasis.T @ grad
            adam_projected = projected / (projected.abs() + optimizer.param_groups[0]["eps"])
            expected_update = eigenbasis @ adam_projected
        else:
            projected = grad @ eigenbasis
            adam_projected = projected / (projected.abs() + optimizer.param_groups[0]["eps"])
            expected_update = adam_projected @ eigenbasis.T

        applied_update = -param.detach() / lr
        torch.testing.assert_close(
            applied_update,
            expected_update,
            atol=0.0,
            rtol=1e-4,
            msg=lambda msg: f"MOSO no-EMA update did not match projected Adam update for shape {shape}:\n{msg}",
        )

    def test_non_2d_param_raises_type_error(self) -> None:
        param = torch.randn(10, requires_grad=True, device=FLAGS.device)
        optimizer = MOSO([param], lr=0.001)
        param.grad = torch.randn_like(param)

        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            optimizer.step()

    def test_betas_raise_type_error(self) -> None:
        param = torch.randn((4, 4), requires_grad=True, device=FLAGS.device)
        with self.assertRaises(TypeError):
            MOSO([param], lr=0.001, betas=(0.9, 0.95))


if __name__ == "__main__":
    absltest.main()
