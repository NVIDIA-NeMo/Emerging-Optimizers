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
from absl import flags
from absl.testing import absltest, parameterized

from emerging_optimizers.riemannian_optimizers.normalized_optimizer import ObliqueAdam, ObliqueSGD


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS


class NormalizedOptimizerFunctionalTest(parameterized.TestCase):
    """Tests for ObliqueSGD and ObliqueAdam optimizers that preserve row/column norms."""

    def setUp(self):
        """Set random seed before each test."""
        # Set seed for PyTorch
        torch.manual_seed(1234)
        # Set seed for CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        self.device = FLAGS.device

    @parameterized.parameters(
        (0),
        (1),
    )
    def test_oblique_sgd_preserves_norms(self, dim: int) -> None:
        """Test that ObliqueSGD preserves row or column norms after one optimization step."""
        # Create a 4x6 matrix for testing
        matrix_size = (4, 6)

        # Initialize with random values then normalize
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize according to dim
        torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=1e-8, out=param)

        # Create optimizer
        param = torch.nn.Parameter(param)
        optimizer = ObliqueSGD([param], lr=0.1, momentum=0.9, dim=dim)

        # Generate random gradient
        torch.manual_seed(1234)  # For reproducible gradients
        param.grad = torch.randn_like(param.data, device=self.device)

        # Perform one optimization step
        optimizer.step()

        # Check that norms are preserved (should be 1.0 within tolerance)
        final_norms = param.norm(dim=dim)

        # All norms should be approximately 1.0 (unit norm constraint)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=0,
            rtol=1e-6,
        )

    @parameterized.parameters(
        (0),
        (1),
    )
    def test_oblique_adam_preserves_norms(self, dim: int) -> None:
        """Test that ObliqueAdam preserves row or column norms after one optimization step."""
        # Create a 3x5 matrix for testing
        matrix_size = (3, 5)

        # Initialize with random values then normalize
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize
        torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=1e-8, out=param)
        # Create optimizer
        param = torch.nn.Parameter(param)
        optimizer = ObliqueAdam([param], lr=0.01, betas=(0.9, 0.999), dim=dim)

        # Generate random gradient
        torch.manual_seed(1234)  # For reproducible gradients
        param.grad = torch.randn_like(param.data, device=self.device)

        # Perform one optimization step
        optimizer.step()

        # Check that norms are preserved (should be 1.0 within tolerance)
        final_norms = param.norm(dim=dim)

        # All norms should be approximately 1.0 (unit norm constraint)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=0,
            rtol=1e-6,
        )

    def test_oblique_sgd_zero_gradient(self) -> None:
        """Test that ObliqueSGD handles zero gradients correctly."""
        matrix_size = (2, 4)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize
        torch.nn.functional.normalize(param, p=2.0, dim=0, eps=1e-8, out=param)
        initial_param = param.clone()

        param = torch.nn.Parameter(param)
        optimizer = ObliqueSGD([param], lr=0.1, dim=0)

        # Set zero gradient
        param.grad = torch.zeros_like(param.data, device=self.device)

        # Perform optimization step
        optimizer.step()

        # Parameter should remain unchanged with zero gradient
        torch.testing.assert_close(param.data, initial_param, atol=0, rtol=1e-8)

        # Norms should still be 1.0
        final_norms = param.norm(dim=0)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=0, rtol=1e-6)

    def test_oblique_adam_zero_gradient(self) -> None:
        """Test that ObliqueAdam handles zero gradients correctly."""
        matrix_size = (2, 3)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize
        torch.nn.functional.normalize(param, p=2.0, dim=1, eps=1e-8, out=param)
        initial_param = param.clone()

        # Keep as tensor, not parameter, but enable gradients
        param.requires_grad_(True)
        optimizer = ObliqueAdam([param], lr=0.01, dim=1)

        # Set zero gradient
        param.grad = torch.zeros_like(param.data, device=self.device)

        # Perform optimization step
        optimizer.step()

        # Parameter should remain unchanged with zero gradient
        torch.testing.assert_close(param.data, initial_param, atol=0, rtol=1e-6)

        # Norms should still be 1.0
        final_norms = param.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=0, rtol=1e-6)

    def test_oblique_sgd_large_gradient(self) -> None:
        """Test that ObliqueSGD handles large gradients correctly."""
        matrix_size = (3, 4)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize
        param = param / param.norm(dim=0, keepdim=True).clamp(min=1e-8)

        param = torch.nn.Parameter(param)
        optimizer = ObliqueSGD([param], lr=0.1, dim=0)

        # Set large gradient
        param.grad = 100.0 * torch.randn_like(param.data, device=self.device)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved despite large gradient
        final_norms = param.norm(dim=0)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=0, rtol=1e-6)

    def test_oblique_adam_large_gradient(self) -> None:
        """Test that ObliqueAdam handles large gradients correctly."""
        matrix_size = (2, 5)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize rows
        param = param / param.norm(dim=1, keepdim=True).clamp(min=1e-8)

        param = torch.nn.Parameter(param)
        optimizer = ObliqueAdam([param], lr=0.01, dim=1)

        # Set large gradient
        param.grad = 1000.0 * torch.randn_like(param.data, device=self.device)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved despite large gradient
        final_norms = param.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=0,
            rtol=1e-6,
        )

    def test_multiple_optimization_steps_preserve_norms(self) -> None:
        """Test that norms are preserved across multiple optimization steps."""
        matrix_size = (4, 4)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize
        param = param / param.norm(dim=0, keepdim=True).clamp(min=1e-8)

        param = torch.nn.Parameter(param)
        optimizer = ObliqueSGD([param], lr=0.05, momentum=0.8, dim=0)

        # Perform multiple optimization steps
        for step in range(10):
            param.grad = torch.randn_like(param.data, device=self.device)
            optimizer.step()

            # Check norms after each step
            final_norms = param.norm(dim=0)
            expected_norms = torch.ones_like(final_norms)
            torch.testing.assert_close(
                final_norms,
                expected_norms,
                atol=0,
                rtol=1e-6,
            )

    def test_weight_decay_with_norm_preservation(self) -> None:
        """Test that weight decay doesn't break norm preservation."""
        matrix_size = (3, 3)
        param = torch.randn(matrix_size, dtype=torch.float32, device=self.device)

        # Normalize
        param = param / param.norm(dim=1, keepdim=True).clamp(min=1e-8)

        param = torch.nn.Parameter(param)
        optimizer = ObliqueAdam([param], lr=0.01, weight_decay=0.01, dim=1)

        # Generate random gradient
        param.grad = torch.randn_like(param.data, device=self.device)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved with weight decay
        final_norms = param.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=0,
            rtol=1e-6,
        )


if __name__ == "__main__":
    absltest.main()
