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

    def test_smoke(self) -> None:
        """Lion optimizer can be instantiated and stepped."""
        param = torch.nn.Parameter(torch.randn(4, 4, device=self.device))
        optimizer = Lion([param], lr=1e-4)
        param.grad = torch.randn_like(param)
        optimizer.step()

    def test_state_initialization(self) -> None:
        """Lion initializes exp_avg state to zeros on first step."""
        beta2 = 0.99
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        optimizer = Lion([param], lr=1e-4, betas=(0.9, beta2), weight_decay=0.0)
        grad = torch.randn_like(param)
        param.grad = grad.clone()
        optimizer.step()
        self.assertIn("exp_avg", optimizer.state[param])
        # exp_avg is initialized to zero then updated: 0 * beta2 + (1 - beta2) * grad
        expected = (1 - beta2) * grad
        torch.testing.assert_close(optimizer.state[param]["exp_avg"], expected, atol=1e-6, rtol=1e-6)

    def test_no_grad_no_update(self) -> None:
        """Parameters without gradients are not updated."""
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        original = param.data.clone()
        optimizer = Lion([param], lr=1e-4)
        optimizer.step()
        torch.testing.assert_close(param.data, original, atol=0, rtol=0)

    @parameterized.parameters(
        {"betas": (0.9, 0.99)},
        {"betas": (0.95, 0.98)},
    )
    def test_update_is_sign_based(self, betas) -> None:
        """Lion updates should be +/- lr (sign-based)."""
        param = torch.nn.Parameter(torch.full((5, 5), 4.0, device=self.device))
        optimizer = Lion([param], lr=0.25, betas=betas, weight_decay=0.0)
        # Use a fixed, non-zero gradient to guarantee sign(g) != 0 for every element.
        param.grad = torch.full((5, 5), 2.0, device=self.device)
        old_param = param.data.clone()
        optimizer.step()

        # The change should be exactly +/- lr since Lion uses sign updates
        diff = old_param - param.data
        torch.testing.assert_close(diff.abs(), torch.full_like(diff, 0.25), atol=0, rtol=0)

    def test_weight_decay_decoupled(self) -> None:
        """Decoupled weight decay shrinks parameters toward zero."""
        lr = 0.25
        wd = 0.5
        param = torch.nn.Parameter(torch.ones(3, 3, device=self.device))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="decoupled")
        param.grad = torch.zeros(3, 3, device=self.device)

        old_param = param.data.clone()
        optimizer.step()

        # With zero grad, sign update is 0. Decoupled weight decay: p = p * (1 - lr * wd)
        expected = old_param * (1 - lr * wd)
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    def test_weight_decay_l2(self) -> None:
        """L2 weight decay folds into gradient before sign(), so it can be masked."""
        lr = 0.25
        wd = 0.5
        param = torch.nn.Parameter(torch.ones(3, 3, device=self.device))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="l2")
        # Use zero gradient so that the only gradient contribution is from L2: grad += wd * p
        param.grad = torch.zeros(3, 3, device=self.device)

        old_param = param.data.clone()
        optimizer.step()

        # After L2, grad becomes 0 + wd * 1 = wd (all positive).
        # First step: exp_avg is zero, so update = sign(beta1 * 0 + (1-beta1) * wd) = sign(wd) = 1
        # p = p - lr * sign = 1 - lr
        expected = old_param - lr * torch.ones(3, 3, device=self.device)
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    def test_weight_decay_l2_masked_by_gradient(self) -> None:
        """L2 decay penalty can be masked when the gradient dominates the sign."""
        lr = 0.25
        wd = 0.125
        param = torch.nn.Parameter(torch.ones(3, 3, device=self.device))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="l2")
        # Large negative gradient dominates: grad + wd*p = -8 + 0.125, sign = -1
        param.grad = torch.full((3, 3), -8.0, device=self.device)

        old_param = param.data.clone()
        optimizer.step()

        # sign(negative) = -1, so p = p - lr * (-1) = p + lr, parameter grows
        # L2 cannot guarantee shrinkage when gradient dominates
        expected = old_param + lr * torch.ones(3, 3, device=self.device)
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    def test_weight_decay_independent(self) -> None:
        """Independent weight decay shrinks params without lr scaling."""
        lr = 0.25
        wd = 0.5
        param = torch.nn.Parameter(torch.ones(3, 3, device=self.device))
        optimizer = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="independent")
        param.grad = torch.zeros(3, 3, device=self.device)

        old_param = param.data.clone()
        optimizer.step()

        # Independent: p = p * (1 - wd). With zero grad, sign update is 0.
        expected = old_param * (1 - wd)
        torch.testing.assert_close(param.data, expected, atol=0, rtol=0)

    def test_exp_avg_evolves_correctly(self) -> None:
        """Verify exp_avg state matches analytical values after deterministic steps."""
        beta1, beta2 = 0.9, 0.99
        param = torch.nn.Parameter(torch.ones(2, 2, device=self.device))
        optimizer = Lion([param], lr=0.01, betas=(beta1, beta2), weight_decay=0.0)

        grads = [
            torch.full((2, 2), 1.0, device=self.device),
            torch.full((2, 2), -2.0, device=self.device),
            torch.full((2, 2), 0.5, device=self.device),
        ]

        # exp_avg starts at 0. Each step: exp_avg = lerp(exp_avg, grad, 1 - beta2)
        # i.e. exp_avg = beta2 * exp_avg + (1 - beta2) * grad
        expected_exp_avg = torch.zeros(2, 2, device=self.device)
        for grad in grads:
            param.grad = grad.clone()
            optimizer.step()
            expected_exp_avg = beta2 * expected_exp_avg + (1 - beta2) * grad

        torch.testing.assert_close(optimizer.state[param]["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6)

    def test_convergence_on_quadratic(self) -> None:
        """Lion should minimize a simple quadratic f(x) = ||x||^2."""
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(10, device=self.device) * 5)
        optimizer = Lion([param], lr=1e-2, betas=(0.9, 0.99))

        for _ in range(2000):
            optimizer.zero_grad()
            loss = (param**2).sum()
            loss.backward()
            optimizer.step()

        self.assertLess(param.data.abs().max().item(), 0.1)

    def test_param_groups(self) -> None:
        """Lion supports multiple parameter groups with different hyperparameters."""
        p1 = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        p2 = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        optimizer = Lion(
            [
                {"params": [p1], "lr": 0.01},
                {"params": [p2], "lr": 0.001},
            ],
            betas=(0.9, 0.99),
        )
        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        optimizer.step()

        # Both should have state initialized
        self.assertIn("exp_avg", optimizer.state[p1])
        self.assertIn("exp_avg", optimizer.state[p2])


if __name__ == "__main__":
    absltest.main()
