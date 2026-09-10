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
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import registry
from emerging_optimizers.riemannian_optimizers.normalized_optimizer import (
    ObliqueAdam,
    ObliqueSGD,
    ObliqueSteepestAdam,
    ObliqueSteepestSGD,
)


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")

FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class NormalizedOptimizerFunctionalTest(parameterized.TestCase):
    """Tests for Oblique manifold optimizers and steepest descent variants."""

    def setUp(self):
        self.device = FLAGS.device

    @parameterized.parameters(
        (ObliqueSGD, 0),
        (ObliqueSGD, 1),
        (ObliqueAdam, 0),
        (ObliqueAdam, 1),
    )
    def test_unit_l2_norm_preservation(self, opt_cls, dim: int) -> None:
        """Test that L2 norm scaling preserves unit vector norms (norm=1.0)."""
        param = torch.randn(4, 6, dtype=torch.float32, device=self.device)
        torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=1e-8, out=param)
        param = torch.nn.Parameter(param)
        optimizer = opt_cls([param], lr=0.01, dim=dim, scale_mode="unit_l2_norm")

        param.grad = torch.randn_like(param.data)
        optimizer.step()

        final_norms = param.norm(dim=dim)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=0, rtol=1e-6)

    @parameterized.parameters(
        (ObliqueSGD, 0),
        (ObliqueSGD, 1),
        (ObliqueAdam, 0),
        (ObliqueAdam, 1),
        (ObliqueSteepestSGD, 0),
        (ObliqueSteepestSGD, 1),
        (ObliqueSteepestAdam, 0),
        (ObliqueSteepestAdam, 1),
    )
    def test_unit_rms_norm_preservation(self, opt_cls, dim: int) -> None:
        """Test that unit RMS scaling preserves vector RMS=1.0 (L2 norm = sqrt(dim_size))."""
        matrix_size = (4, 6)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)
        m = float(param.size(dim))

        # Initialize to unit RMS
        torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=1e-8, out=param)
        param.mul_(m**0.5)
        param = torch.nn.Parameter(param)

        optimizer = opt_cls([param], lr=0.01, dim=dim, scale_mode="unit_rms_norm")
        param.grad = torch.randn_like(param.data)
        optimizer.step()

        # Check L2 norm equals sqrt(m)
        final_norms = param.norm(dim=dim)
        expected_norms = torch.full_like(final_norms, m**0.5)
        torch.testing.assert_close(final_norms, expected_norms, atol=0, rtol=1e-6)

        # Explicitly verify RMS norm is 1.0
        rms_norm = torch.sqrt(torch.mean(param**2, dim=dim))
        expected_rms = torch.ones_like(rms_norm)
        torch.testing.assert_close(rms_norm, expected_rms, atol=0, rtol=1e-6)

    @parameterized.parameters(
        (0),
        (1),
    )
    def test_steepest_sgd_gradient_scale_invariance(self, dim: int) -> None:
        """Test that 1 -> RMS norm LMO makes updates invariant to gradient magnitude."""
        torch.manual_seed(42)
        base_param = torch.randn(5, 7, dtype=torch.float32, device=self.device)
        torch.nn.functional.normalize(base_param, p=2.0, dim=dim, eps=1e-8, out=base_param)
        base_param.mul_(float(base_param.size(dim)) ** 0.5)

        param1 = torch.nn.Parameter(base_param.clone())
        param2 = torch.nn.Parameter(base_param.clone())

        opt1 = ObliqueSteepestSGD([param1], lr=1e-3, momentum=0.0, dim=dim)
        opt2 = ObliqueSteepestSGD([param2], lr=1e-3, momentum=0.0, dim=dim)

        grad = torch.randn_like(base_param)
        param1.grad = grad.clone()
        param2.grad = grad.clone() * 250.0  # 250x larger gradient

        opt1.step()
        opt2.step()

        # Both parameter matrices must be identical despite 250x difference in gradient scale
        torch.testing.assert_close(param1.data, param2.data, atol=1e-6, rtol=1e-6)

    def test_riemannian_gradient_tangent_orthogonality(self) -> None:
        """Verify that the computed Riemannian update vector lies strictly in the tangent space."""
        matrix_size = (8, 12)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)
        torch.nn.functional.normalize(param, p=2.0, dim=0, eps=1e-8, out=param)
        param = torch.nn.Parameter(param)

        opt = ObliqueSteepestSGD([param], lr=1e-3, dim=0)
        grad = torch.randn_like(param.data)

        # Tangent vector calculation before retraction
        riem_grad = opt._get_riem_grad(param, grad, dim=0, scale=1.0, eps=1e-8)

        # Slice-wise dot product <W, riem_grad> must be zero
        dot_products = (param * riem_grad).sum(dim=0)
        zeros = torch.zeros_like(dot_products)
        torch.testing.assert_close(dot_products, zeros, atol=1e-6, rtol=1e-6)

    def test_oblique_sgd_zero_gradient(self) -> None:
        """Test that ObliqueSGD handles zero gradients correctly."""
        torch.manual_seed(1234)
        matrix_size = (2, 4)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)
        torch.nn.functional.normalize(param, p=2.0, dim=0, eps=1e-8, out=param)
        initial_param = param.clone()

        param = torch.nn.Parameter(param)
        optimizer = ObliqueSGD([param], lr=0.1, dim=0)
        param.grad = torch.zeros_like(param.data)
        optimizer.step()

        torch.testing.assert_close(param.data, initial_param, atol=0, rtol=1e-7)
        final_norms = param.norm(dim=0)
        torch.testing.assert_close(final_norms, torch.ones_like(final_norms), atol=0, rtol=1e-6)

    def test_oblique_sgd_momentum_buffer_accumulates_across_steps(self) -> None:
        """Test that ObliqueSGD persists momentum state across optimization steps."""
        param = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=self.device)
        param = torch.nn.Parameter(param)
        optimizer = ObliqueSGD([param], lr=0.1, momentum=0.8, dim=0)

        first_grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=self.device)
        second_grad = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32, device=self.device)

        param.grad = first_grad.clone()
        optimizer.step()
        assert_equal(optimizer.state[param]["momentum_buffer"], first_grad)

        param.grad = second_grad.clone()
        optimizer.step()
        assert_equal(optimizer.state[param]["momentum_buffer"], second_grad + 0.8 * first_grad)

    def test_oblique_adam_zero_gradient(self) -> None:
        """Test that ObliqueAdam handles zero gradients correctly."""
        matrix_size = (2, 3)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)
        torch.nn.functional.normalize(param, p=2.0, dim=1, eps=1e-8, out=param)
        initial_param = param.clone()

        param = torch.nn.Parameter(param)
        optimizer = ObliqueAdam([param], lr=0.01, dim=1)
        param.grad = torch.zeros_like(param.data)
        optimizer.step()

        torch.testing.assert_close(param.data, initial_param, atol=0, rtol=1e-6)
        final_norms = param.norm(dim=1)
        torch.testing.assert_close(final_norms, torch.ones_like(final_norms), atol=0, rtol=1e-6)

    @parameterized.named_parameters(
        ("oblique_sgd", "oblique_sgd", ObliqueSGD),
        ("oblique_adam", "oblique_adam", ObliqueAdam),
        ("oblique_steepest_sgd", "oblique_steepest_sgd", ObliqueSteepestSGD),
        ("oblique_steepest_adam", "oblique_steepest_adam", ObliqueSteepestAdam),
    )
    def test_registry_registration(self, name: str, expected_cls) -> None:
        """Test that all oblique optimizers are properly registered in the central registry."""
        resolved_cls = registry.get_optimizer_cls(name)
        self.assertIs(resolved_cls, expected_cls)

    @parameterized.named_parameters(
        ("sgd_negative_lr", ObliqueSGD, {"lr": -1.0}, "Invalid learning rate"),
        ("sgd_momentum_out_of_range", ObliqueSGD, {"momentum": 1.0}, "Invalid momentum"),
        ("sgd_negative_weight_decay", ObliqueSGD, {"weight_decay": -0.1}, "Invalid weight_decay"),
        ("steepest_sgd_negative_lr", ObliqueSteepestSGD, {"lr": -1.0}, "Invalid learning rate"),
        ("adam_negative_lr", ObliqueAdam, {"lr": -1.0}, "Invalid learning rate"),
        ("adam_beta1_out_of_range", ObliqueAdam, {"betas": (1.0, 0.99)}, "Invalid beta1"),
        ("adam_beta2_out_of_range", ObliqueAdam, {"betas": (0.9, 1.0)}, "Invalid beta2"),
        ("adam_negative_weight_decay", ObliqueAdam, {"weight_decay": -0.1}, "Invalid weight_decay"),
        ("steepest_adam_negative_lr", ObliqueSteepestAdam, {"lr": -1.0}, "Invalid learning rate"),
    )
    def test_invalid_param_raises_value_error(self, opt_cls, kwargs, error_msg) -> None:
        param = torch.nn.Parameter(torch.randn(3, 4, device=self.device))
        with self.assertRaisesRegex(ValueError, error_msg):
            opt_cls([param], **kwargs)

    @parameterized.named_parameters(
        ("sgd", ObliqueSGD),
        ("adam", ObliqueAdam),
        ("steepest_sgd", ObliqueSteepestSGD),
        ("steepest_adam", ObliqueSteepestAdam),
    )
    def test_non_2d_param_raises_value_error(self, opt_cls) -> None:
        param = torch.nn.Parameter(torch.randn(8, device=self.device))
        param.grad = torch.randn_like(param)
        opt = opt_cls([param])
        with self.assertRaisesRegex(ValueError, "only supports 2D"):
            opt.step()


if __name__ == "__main__":
    absltest.main()
