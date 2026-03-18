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
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.scalar_optimizers import Lion


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class LionOptimizerTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_smoke(self, shape) -> None:
        """Lion optimizer can be instantiated and stepped."""
        param = torch.nn.Parameter(torch.randn(*shape, device=self.device))
        optimizer = Lion([param], lr=1e-4)
        param.grad = torch.randn_like(param)
        optimizer.step()

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_state_initialization(self, shape) -> None:
        """Lion initializes exp_avg state to zeros on first step."""
        beta2 = 0.75
        param = torch.nn.Parameter(torch.randint(-3, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion([param], lr=0.25, betas=(0.5, beta2), weight_decay=0.0)
        grad = torch.randint_like(param, -3, 5)
        param.grad = grad.clone()
        optimizer.step()
        self.assertIn("exp_avg", optimizer.state[param])
        # exp_avg is initialized to zero then updated: 0 * beta2 + (1 - beta2) * grad
        expected = (1 - beta2) * grad
        torch.testing.assert_close(optimizer.state[param]["exp_avg"], expected, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_no_grad_no_update_params_unchanged(self, shape) -> None:
        """Parameters without gradients are not updated."""
        param = torch.nn.Parameter(torch.randn(*shape, device=self.device))
        original = param.data.clone()
        optimizer = Lion([param], lr=1e-4)
        optimizer.step()
        torch.testing.assert_close(param.data, original, atol=0, rtol=0)

    @parameterized.product(
        betas=[(0.9, 0.99), (0.95, 0.98)],
        shape=[(3, 3), (15, 31), (127, 255)],
    )
    def test_update_is_sign_based(self, betas, shape) -> None:
        """Lion updates should be +/- lr (sign-based)."""
        param = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion([param], lr=0.25, betas=betas, weight_decay=0.0)
        # Use a fixed, non-zero gradient to guarantee sign(g) != 0 for every element.
        param.grad = torch.randint(1, 5, shape, device=self.device, dtype=torch.float32)
        old_param = param.data.clone()
        optimizer.step()

        # The change should be exactly +/- lr since Lion uses sign updates
        diff = old_param - param.data
        torch.testing.assert_close(diff.abs(), torch.full_like(diff, 0.25), atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_weight_decay_decoupled_matches_analytical(self, shape) -> None:
        """Decoupled weight decay shrinks parameters toward zero."""
        lr = 0.25
        wd = 0.5
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="decoupled")
        param.grad = torch.zeros(*shape, device=self.device)

        old_param = param.data.clone()
        optimizer.step()

        # With zero grad, sign update is 0. Decoupled weight decay: p = p * (1 - lr * wd)
        expected = old_param * (1 - lr * wd)
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_weight_decay_l2(self, shape) -> None:
        """L2 weight decay folds into gradient before sign(), so it can be masked."""
        lr = 0.25
        wd = 0.5
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="l2")
        # Use zero gradient so that the only gradient contribution is from L2: grad += wd * p
        param.grad = torch.zeros(*shape, device=self.device)

        old_param = param.data.clone()
        optimizer.step()

        # After L2, grad becomes 0 + wd * p (all positive since p > 0).
        # First step: exp_avg is zero, so update = sign(beta1 * 0 + (1-beta1) * wd * p) = sign(positive) = 1
        # p = p - lr * sign = p - lr
        expected = old_param - lr
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_weight_decay_l2_masked_by_gradient(self, shape) -> None:
        """L2 decay penalty can be masked when the gradient dominates the sign."""
        lr = 0.25
        wd = 0.125
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="l2")
        # Large negative gradient dominates: grad + wd*p is still negative, sign = -1
        param.grad = torch.randint(-10, -5, shape, device=self.device, dtype=torch.float32)

        old_param = param.data.clone()
        optimizer.step()

        # sign(negative) = -1, so p = p - lr * (-1) = p + lr, parameter grows
        # L2 cannot guarantee shrinkage when gradient dominates
        expected = old_param + lr
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_weight_decay_independent_matches_analytical(self, shape) -> None:
        """Independent weight decay shrinks params without lr scaling."""
        lr = 0.25
        wd = 0.5
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="independent")
        param.grad = torch.zeros(*shape, device=self.device)

        old_param = param.data.clone()
        optimizer.step()

        # Independent: p = p * (1 - wd). With zero grad, sign update is 0.
        expected = old_param * (1 - wd)
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_exp_avg_evolves_correctly(self, shape) -> None:
        """Verify exp_avg state matches analytical values after deterministic steps."""
        beta1, beta2 = 0.9, 0.99
        param = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion([param], lr=0.01, betas=(beta1, beta2), weight_decay=0.0)

        grads = [
            torch.randint(-3, 3, shape, device=self.device, dtype=torch.float32),
            torch.randint(-3, 3, shape, device=self.device, dtype=torch.float32),
            torch.randint(-3, 3, shape, device=self.device, dtype=torch.float32),
        ]

        # exp_avg starts at 0. Each step: exp_avg = lerp(exp_avg, grad, 1 - beta2)
        # i.e. exp_avg = beta2 * exp_avg + (1 - beta2) * grad
        expected_exp_avg = torch.zeros(*shape, device=self.device)
        for grad in grads:
            param.grad = grad.clone()
            optimizer.step()
            expected_exp_avg = beta2 * expected_exp_avg + (1 - beta2) * grad

        torch.testing.assert_close(optimizer.state[param]["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6)

    @parameterized.parameters(True, False)
    def test_init_group_skip_non_grad_params(self, skip_non_grad_params) -> None:
        """Test _init_group with skip_non_grad_params flag."""
        param_with_grad = torch.nn.Parameter(torch.randn(5, 7, device=self.device))
        param_without_grad = torch.nn.Parameter(torch.randn(5, 7, device=self.device))
        param_with_grad.grad = torch.randn_like(param_with_grad)

        opt = Lion([param_with_grad, param_without_grad], lr=1e-4)

        opt._init_group(opt.param_groups[0], skip_non_grad_params=skip_non_grad_params)

        self.assertIn("exp_avg", opt.state[param_with_grad])
        self.assertEqual(opt.state[param_with_grad]["exp_avg"].shape, param_with_grad.data.shape)

        self.assertEqual("exp_avg" in opt.state[param_without_grad], not skip_non_grad_params)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_param_groups_large_lr_moves_more(self, shape) -> None:
        """Lion supports multiple parameter groups with different hyperparameters."""
        p1 = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        p2 = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        optimizer = Lion(
            [
                {"params": [p1], "lr": 0.01},
                {"params": [p2], "lr": 0.001},
            ],
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )
        p1_original = p1.data.clone()
        p2_original = p2.data.clone()
        grad = torch.randint(1, 5, shape, device=self.device, dtype=torch.float32)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        optimizer.step()

        # Both should have state initialized
        self.assertIn("exp_avg", optimizer.state[p1])
        self.assertIn("exp_avg", optimizer.state[p2])

        # p1 (lr=0.01) should have moved more than p2 (lr=0.001)
        p1_change = (p1.data - p1_original).abs().mean()
        p2_change = (p2.data - p2_original).abs().mean()
        self.assertGreater(p1_change.item(), p2_change.item())


if __name__ == "__main__":
    absltest.main()
