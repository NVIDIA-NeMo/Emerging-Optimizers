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

from emerging_optimizers.scalar_optimizers import LaProp
from emerging_optimizers.scalar_optimizers.update_functions import calculate_laprop_update


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class LaPropOptimizerTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_smoke(self, shape) -> None:
        """LaProp optimizer can be instantiated and stepped."""
        param = torch.nn.Parameter(torch.randn(*shape, device=self.device))
        optimizer = LaProp([param], lr=1e-4)
        param.grad = torch.randn_like(param)
        optimizer.step()

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_state_initialization(self, shape) -> None:
        """LaProp initializes first moment, second moment, and step state."""
        beta1, beta2 = 0.5, 0.75
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = LaProp([param], lr=0.25, betas=(beta1, beta2), weight_decay=0.0, correct_bias=True)
        grad = torch.randint_like(param, 1, 5)
        param.grad = grad.clone()
        optimizer.step()

        self.assertEqual(optimizer.state[param]["step"], 1)
        self.assertIn("exp_avg", optimizer.state[param])
        self.assertIn("exp_avg_sq", optimizer.state[param])

        expected_exp_avg_sq = (1 - beta2) * grad.square()
        normalized_grad = grad / (grad.abs() + optimizer.param_groups[0]["eps"])
        expected_exp_avg = (1 - beta1) * normalized_grad
        torch.testing.assert_close(optimizer.state[param]["exp_avg_sq"], expected_exp_avg_sq)
        torch.testing.assert_close(optimizer.state[param]["exp_avg"], expected_exp_avg)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_optimizer_step_matches_update_function(self, shape) -> None:
        """LaProp optimizer delegates update math to calculate_laprop_update."""
        lr = 0.25
        betas = (0.5, 0.75)
        eps = 1e-8
        param = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        grad = torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32)
        optimizer = LaProp([param], lr=lr, betas=betas, eps=eps, weight_decay=0.0)

        old_param = param.detach().clone()
        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        expected_update = calculate_laprop_update(grad, exp_avg, exp_avg_sq, True, betas, 1, eps)

        param.grad = grad.clone()
        optimizer.step()

        torch.testing.assert_close(param, old_param - lr * expected_update)
        torch.testing.assert_close(optimizer.state[param]["exp_avg"], exp_avg)
        torch.testing.assert_close(optimizer.state[param]["exp_avg_sq"], exp_avg_sq)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_no_grad_no_update_params_unchanged(self, shape) -> None:
        """Parameters without gradients are not updated."""
        param = torch.nn.Parameter(torch.randn(*shape, device=self.device))
        original = param.detach().clone()
        optimizer = LaProp([param], lr=1e-4)
        optimizer.step()
        torch.testing.assert_close(param, original, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_normalize_preserves_parameter_norm(self, shape) -> None:
        """LaProp can normalize updated parameters back to their pre-update norm."""
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = LaProp([param], lr=0.25, weight_decay=0.0, normalize=True)
        param.grad = torch.randint(1, 5, shape, device=self.device, dtype=torch.float32)
        original_norm = param.norm()

        optimizer.step()

        torch.testing.assert_close(param.norm(), original_norm)

    @parameterized.parameters(True, False)
    def test_init_group_skip_non_grad_params(self, skip_non_grad_params) -> None:
        """Test _init_group with skip_non_grad_params flag."""
        param_with_grad = torch.nn.Parameter(torch.randn(5, 7, device=self.device))
        param_without_grad = torch.nn.Parameter(torch.randn(5, 7, device=self.device))
        param_with_grad.grad = torch.randn_like(param_with_grad)

        opt = LaProp([param_with_grad, param_without_grad], lr=1e-4)
        opt._init_group(opt.param_groups[0], skip_non_grad_params=skip_non_grad_params)

        self.assertIn("exp_avg", opt.state[param_with_grad])
        self.assertIn("exp_avg_sq", opt.state[param_with_grad])
        self.assertEqual(opt.state[param_with_grad]["step"], 0)
        self.assertEqual("exp_avg" in opt.state[param_without_grad], not skip_non_grad_params)

    def test_negative_lr_raises_value_error(self) -> None:
        """Test that LaProp raises ValueError for negative learning rate."""
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid learning rate"):
            LaProp([param], lr=-1.0)

    def test_beta0_out_of_range_raises_value_error(self) -> None:
        """Test that LaProp raises ValueError for invalid beta at index 0."""
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid beta at index 0"):
            LaProp([param], betas=(1.0, 0.999))

    def test_beta1_out_of_range_raises_value_error(self) -> None:
        """Test that LaProp raises ValueError for invalid beta at index 1."""
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid beta at index 1"):
            LaProp([param], betas=(0.9, 1.0))

    def test_negative_eps_raises_value_error(self) -> None:
        """Test that LaProp raises ValueError for negative eps."""
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid epsilon"):
            LaProp([param], eps=-1e-8)

    def test_non_positive_normalize_eps_raises_value_error(self) -> None:
        """Test that LaProp raises ValueError for non-positive normalize_eps."""
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid normalize_eps"):
            LaProp([param], normalize_eps=0.0)

    def test_negative_weight_decay_raises_value_error(self) -> None:
        """Test that LaProp raises ValueError for negative weight_decay."""
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay"):
            LaProp([param], weight_decay=-0.1)


if __name__ == "__main__":
    absltest.main()
