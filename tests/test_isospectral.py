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
from emerging_optimizers.rl_optimizers.isospectral import Iso


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class IsospectralTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("qr_tall", "qr", (8, 5)),
        ("qr_wide", "qr", (5, 8)),
        ("polar_tall", "polar", (8, 5)),
        ("polar_wide", "polar", (5, 8)),
        ("cayley_tall", "cayley", (8, 5)),
        ("cayley_wide", "cayley", (5, 8)),
    )
    def test_preserves_singular_values(self, retraction: str, shape: tuple[int, int]) -> None:
        param = torch.nn.Parameter(torch.randn(shape, device=FLAGS.device))
        initial_singular_values = torch.linalg.svdvals(param).clone()
        optimizer = Iso([param], lr=1e-2, momentum=0.9, retraction=retraction)

        for _ in range(5):
            param.grad = torch.randn_like(param)
            optimizer.step()

        torch.testing.assert_close(
            torch.linalg.svdvals(param),
            initial_singular_values,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_factor_state_is_orthonormal(self) -> None:
        param = torch.nn.Parameter(torch.randn((7, 4), device=FLAGS.device))
        optimizer = Iso([param], lr=1e-2)
        param.grad = torch.randn_like(param)
        optimizer.step()

        state = optimizer.state[param]
        identity = torch.eye(4, device=FLAGS.device)
        torch.testing.assert_close(state["u"].mT @ state["u"], identity, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(state["v"].mT @ state["v"], identity, atol=1e-5, rtol=1e-5)

    @parameterized.named_parameters(
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    )
    def test_cpu_low_precision_uses_fp32_factor_state(self, dtype: torch.dtype) -> None:
        param = torch.nn.Parameter(torch.randn((6, 4), dtype=dtype, device="cpu"))
        optimizer = Iso([param], lr=1e-2)
        param.grad = torch.randn_like(param)

        optimizer.step()

        state = optimizer.state[param]
        self.assertEqual(param.dtype, dtype)
        self.assertEqual(state["u"].dtype, torch.float32)
        self.assertEqual(state["sigma"].dtype, torch.float32)
        self.assertEqual(state["v"].dtype, torch.float32)
        self.assertEqual(state["momentum_u"].dtype, torch.float32)
        self.assertEqual(state["momentum_v"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(param).all())

    @parameterized.named_parameters(
        ("decoupled", "decoupled", 1.0 - 0.2 * 0.1),
        ("independent", "independent", 1.0 - 0.1),
        ("palm", "palm", 1.0 - 0.2**2 * 0.1),
    )
    def test_direct_weight_decay_scales_singular_values(
        self,
        weight_decay_method: str,
        expected_scale: float,
    ) -> None:
        param = torch.nn.Parameter(torch.randn((6, 4), device=FLAGS.device))
        initial_singular_values = torch.linalg.svdvals(param).clone()
        optimizer = Iso(
            [param],
            lr=0.2,
            momentum=0.0,
            weight_decay=0.1,
            weight_decay_method=weight_decay_method,
        )
        param.grad = torch.zeros_like(param)

        optimizer.step()
        optimizer.step()

        torch.testing.assert_close(
            torch.linalg.svdvals(param),
            initial_singular_values * expected_scale**2,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_rejects_non_matrix_parameter(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, device=FLAGS.device))
        optimizer = Iso([param])
        param.grad = torch.randn_like(param)

        with self.assertRaisesRegex(ValueError, "only supports 2D"):
            optimizer.step()

    @parameterized.named_parameters(
        ("negative_lr", {"lr": -1e-3}, "learning rate"),
        ("negative_momentum", {"momentum": -0.1}, "momentum"),
        ("unit_momentum", {"momentum": 1.0}, "momentum"),
        ("unknown_retraction", {"retraction": "invalid"}, "retraction"),
        ("negative_weight_decay", {"weight_decay": -0.1}, "weight_decay"),
    )
    def test_rejects_invalid_hyperparameters(self, kwargs: dict[str, object], message: str) -> None:
        param = torch.nn.Parameter(torch.randn((2, 2), device=FLAGS.device))
        with self.assertRaisesRegex(ValueError, message):
            Iso([param], **kwargs)

    def test_registered_as_iso(self) -> None:
        self.assertIs(registry.get_optimizer_cls("iso"), Iso)


if __name__ == "__main__":
    absltest.main()
