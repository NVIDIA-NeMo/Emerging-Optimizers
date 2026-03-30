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

from emerging_optimizers.mixin import WeightDecayMixin


class _WeightDecayHelper(WeightDecayMixin):
    """Thin wrapper so we can set weight_decay_method and call the mixin."""

    def __init__(self, method: str):
        self.weight_decay_method = method


class WeightDecayMixinTest(parameterized.TestCase):
    # ------------------------------------------------------------------
    # Zero weight decay – nothing should change regardless of method
    # ------------------------------------------------------------------
    @parameterized.parameters("decoupled", "independent", "l2", "palm")
    def test_zero_weight_decay_is_noop(self, method):
        helper = _WeightDecayHelper(method)
        p = torch.tensor([1.0, 2.0, 3.0])
        grad = torch.tensor([0.5, -0.5, 1.0])
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=0.1, weight_decay=0.0)

        torch.testing.assert_close(p, p_orig, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Decoupled: p <- p + p * (-wd * lr)  =>  p * (1 - wd * lr)
    # grad is untouched
    # ------------------------------------------------------------------
    @parameterized.parameters(
        {"lr": 0.1, "wd": 0.5},
        {"lr": 0.01, "wd": 1.0},
        {"lr": 1.0, "wd": 0.01},
    )
    def test_decoupled(self, lr, wd):
        helper = _WeightDecayHelper("decoupled")
        p = torch.tensor([4.0, -2.0, 0.0, 7.0])
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_p = p_orig * (1 - wd * lr)
        torch.testing.assert_close(p, expected_p, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Independent: p <- p + p * (-wd)  =>  p * (1 - wd)
    # grad is untouched; lr is irrelevant
    # ------------------------------------------------------------------
    @parameterized.parameters(
        {"lr": 0.1, "wd": 0.5},
        {"lr": 0.01, "wd": 1.0},
        {"lr": 1.0, "wd": 0.01},
    )
    def test_independent(self, lr, wd):
        helper = _WeightDecayHelper("independent")
        p = torch.tensor([4.0, -2.0, 0.0, 7.0])
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_p = p_orig * (1 - wd)
        torch.testing.assert_close(p, expected_p, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Independent does NOT depend on lr – verify two different lr values
    # produce the same result
    # ------------------------------------------------------------------
    def test_independent_ignores_lr(self):
        wd = 0.3
        p1 = torch.tensor([5.0, -3.0, 1.0])
        p2 = p1.clone()
        grad1 = torch.tensor([1.0, 1.0, 1.0])
        grad2 = grad1.clone()

        _WeightDecayHelper("independent")._apply_weight_decay_inplace(p1, grad1, lr=0.001, weight_decay=wd)
        _WeightDecayHelper("independent")._apply_weight_decay_inplace(p2, grad2, lr=100.0, weight_decay=wd)

        torch.testing.assert_close(p1, p2, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # L2: grad <- grad + p * wd
    # p is untouched
    # ------------------------------------------------------------------
    @parameterized.parameters(
        {"lr": 0.1, "wd": 0.5},
        {"lr": 0.01, "wd": 1.0},
        {"lr": 1.0, "wd": 0.01},
    )
    def test_l2(self, lr, wd):
        helper = _WeightDecayHelper("l2")
        p = torch.tensor([4.0, -2.0, 0.0, 7.0])
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_grad = grad_orig + p_orig * wd
        torch.testing.assert_close(p, p_orig, atol=0, rtol=0)
        torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # L2 does NOT depend on lr
    # ------------------------------------------------------------------
    def test_l2_ignores_lr(self):
        wd = 0.3
        p1 = torch.tensor([5.0, -3.0, 1.0])
        p2 = p1.clone()
        grad1 = torch.tensor([1.0, 1.0, 1.0])
        grad2 = grad1.clone()

        _WeightDecayHelper("l2")._apply_weight_decay_inplace(p1, grad1, lr=0.001, weight_decay=wd)
        _WeightDecayHelper("l2")._apply_weight_decay_inplace(p2, grad2, lr=100.0, weight_decay=wd)

        torch.testing.assert_close(grad1, grad2, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # PaLM: p <- p + p * (-(wd * lr^2))  =>  p * (1 - wd * lr^2)
    # grad is untouched
    # ------------------------------------------------------------------
    @parameterized.parameters(
        {"lr": 0.1, "wd": 0.5},
        {"lr": 0.01, "wd": 1.0},
        {"lr": 1.0, "wd": 0.01},
    )
    def test_palm(self, lr, wd):
        helper = _WeightDecayHelper("palm")
        p = torch.tensor([4.0, -2.0, 0.0, 7.0])
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
        p_orig, grad_orig = p.clone(), grad.clone()

        helper._apply_weight_decay_inplace(p, grad, lr=lr, weight_decay=wd)

        expected_p = p_orig * (1 - wd * lr * lr)
        torch.testing.assert_close(p, expected_p, atol=0, rtol=0)
        torch.testing.assert_close(grad, grad_orig, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Default method (no attribute set) should be "l2"
    # ------------------------------------------------------------------
    def test_default_method_is_l2(self):
        helper = WeightDecayMixin()
        p = torch.tensor([4.0, -2.0, 0.0, 7.0])
        grad = torch.tensor([1.0, 1.0, 1.0, 1.0])
        p_orig, grad_orig = p.clone(), grad.clone()

        wd = 0.5
        helper._apply_weight_decay_inplace(p, grad, lr=0.1, weight_decay=wd)

        expected_grad = grad_orig + p_orig * wd
        torch.testing.assert_close(p, p_orig, atol=0, rtol=0)
        torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)

    # ------------------------------------------------------------------
    # Invalid method raises ValueError
    # ------------------------------------------------------------------
    def test_invalid_method_raises(self):
        helper = _WeightDecayHelper("bogus")
        p = torch.tensor([1.0])
        grad = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            helper._apply_weight_decay_inplace(p, grad, lr=0.1, weight_decay=0.1)


if __name__ == "__main__":
    absltest.main()
