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

from emerging_optimizers.mixin import WeightDecayMixin


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class _WeightDecayTestHelper(WeightDecayMixin):
    """Thin wrapper so we can set weight_decay_method and call the mixin."""

    def __init__(self, method: str):
        self.weight_decay_method = method


class WeightDecayMixinTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = FLAGS.device

    @parameterized.parameters("decoupled", "independent", "l2", "palm")
    def test_zero_weight_decay_is_noop(self, method):
        """Neither p nor grad should change when weight_decay is 0."""
        helper = _WeightDecayTestHelper(method)
        p = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        grad = torch.tensor([0.5, -0.5, 1.0], device=self.device)
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=0.1, weight_decay=0.0)

        torch.testing.assert_close(p, p_orig, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    @parameterized.parameters(
        {"lr": 0.25, "wd": 0.5},
        {"lr": 0.025, "wd": 1.0},
        {"lr": 1.0, "wd": 0.05},
    )
    def test_decoupled(self, lr, wd):
        """Decoupled: p <- p * (1 - wd * lr), grad untouched."""
        helper = _WeightDecayTestHelper("decoupled")
        p = torch.tensor([4.0, -2.0, 0.0, 8.0], device=self.device)
        grad = torch.tensor([1.0, 2.0, 1.0, 1.0], device=self.device)
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_p = p_orig * (1 - wd * lr)
        torch.testing.assert_close(p, expected_p, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    @parameterized.parameters(
        {"lr": 0.25, "wd": 0.5},
        {"lr": 0.025, "wd": 1.0},
        {"lr": 1.0, "wd": 0.05},
    )
    def test_independent(self, lr, wd):
        """Independent: p <- p * (1 - wd), grad untouched, lr irrelevant."""
        helper = _WeightDecayTestHelper("independent")
        p = torch.tensor([4.0, -2.0, 0.0, 7.0], device=self.device)
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_p = p_orig * (1 - wd)
        torch.testing.assert_close(p, expected_p, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    def test_independent_ignores_lr(self):
        """Two different lr values must produce identical results for independent decay."""
        wd = 0.3
        p1 = torch.tensor([5.0, -3.0, 1.0], device=self.device)
        p2 = p1.clone()
        grad1 = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        grad2 = grad1.clone()

        _WeightDecayTestHelper("independent")._apply_weight_decay_inplace(p1, grad1, lr=0.001, weight_decay=wd)
        _WeightDecayTestHelper("independent")._apply_weight_decay_inplace(p2, grad2, lr=100.0, weight_decay=wd)

        torch.testing.assert_close(p1, p2, atol=0, rtol=0)

    @parameterized.parameters(
        {"lr": 0.1, "wd": 0.5},
        {"lr": 0.01, "wd": 1.0},
        {"lr": 1.0, "wd": 0.01},
    )
    def test_l2(self, lr, wd):
        """L2: grad <- grad + p * wd, p untouched."""
        helper = _WeightDecayTestHelper("l2")
        p = torch.tensor([4.0, -2.0, 0.0, 7.0], device=self.device)
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_grad = grad_orig + p_orig * wd
        torch.testing.assert_close(p, p_orig, atol=0, rtol=0)
        torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)

    def test_l2_ignores_lr(self):
        """Two different lr values must produce identical results for L2 decay."""
        wd = 0.3
        p1 = torch.tensor([5.0, -3.0, 1.0], device=self.device)
        p2 = p1.clone()
        grad1 = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        grad2 = grad1.clone()

        _WeightDecayTestHelper("l2")._apply_weight_decay_inplace(p1, grad1, lr=0.001, weight_decay=wd)
        _WeightDecayTestHelper("l2")._apply_weight_decay_inplace(p2, grad2, lr=100.0, weight_decay=wd)

        torch.testing.assert_close(grad1, grad2, atol=0, rtol=0)

    @parameterized.parameters(
        {"lr": 0.25, "wd": 0.5},
        {"lr": 0.025, "wd": 1.0},
        {"lr": 1.0, "wd": 0.05},
    )
    def test_palm(self, lr, wd):
        """PaLM: p <- p * (1 - wd * lr^2), grad untouched."""
        helper = _WeightDecayTestHelper("palm")
        p = torch.tensor([4.0, -2.0, 0.0, 7.0], device=self.device)
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_p = p_orig * (1 - wd * lr * lr)
        torch.testing.assert_close(p, expected_p, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    def test_default_method_is_l2(self):
        """When weight_decay_method attribute is absent, default to L2."""
        helper = WeightDecayMixin()
        p = torch.tensor([4.0, -2.0, 0.0, 7.0], device=self.device)
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        p_orig, grad_orig = p.clone(), grad.clone()

        wd = 0.5
        helper._apply_weight_decay_inplace(p, grad, lr=0.1, weight_decay=wd)

        expected_grad = grad_orig + p_orig * wd
        torch.testing.assert_close(p, p_orig, atol=0, rtol=0)
        torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)

    def test_invalid_method_raises(self):
        """An unrecognized weight_decay_method must raise ValueError."""
        helper = _WeightDecayTestHelper("bogus")
        p = torch.tensor([1.0], device=self.device)
        grad = torch.tensor([1.0], device=self.device)
        with self.assertRaises(ValueError):
            helper._apply_weight_decay_inplace(p, grad, lr=0.1, weight_decay=0.1)


if __name__ == "__main__":
    absltest.main()
