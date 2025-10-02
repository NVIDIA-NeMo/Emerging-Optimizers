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

from emerging_optimizers.riemannian_optimizers.normalized_optimizer import ObliqueAdam, ObliqueSGD


# Base class for tests requiring seeding for determinism
class BaseTestCase(parameterized.TestCase):
    def setUp(self):
        """Set random seed before each test."""
        # Set seed for PyTorch
        torch.manual_seed(42)
        # Set seed for CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)


class NormalizedOptimizerTest(BaseTestCase):
    """Tests for ObliqueSGD and ObliqueAdam optimizers that preserve row/column norms."""

    @parameterized.named_parameters(
        ("col_mode", "col"),
        ("row_mode", "row"),
    )
    def test_oblique_sgd_preserves_norms(self, mode):
        """Test that ObliqueSGD preserves row or column norms after one optimization step."""
        # Create a 4x6 matrix for testing
        matrix_size = (4, 6)

        # Initialize with random values then normalize
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize according to mode
        if mode == "col":
            # Normalize columns (each column has unit norm)
            param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        else:  # mode == "row"
            # Normalize rows (each row has unit norm)
            param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Create optimizer
        optimizer = ObliqueSGD([param], lr=0.1, momentum=0.9, mode=mode)

        # Generate random gradient
        torch.manual_seed(123)  # For reproducible gradients
        param.grad = torch.randn_like(param.data)

        # Perform one optimization step
        optimizer.step()

        # Check that norms are preserved (should be 1.0 within tolerance)
        if mode == "col":
            final_norms = param.data.norm(dim=0)
        else:  # mode == "row"
            final_norms = param.data.norm(dim=1)

        # All norms should be approximately 1.0 (unit norm constraint)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )

    @parameterized.named_parameters(
        ("col_mode", "col"),
        ("row_mode", "row"),
    )
    def test_oblique_adam_preserves_norms(self, mode):
        """Test that ObliqueAdam preserves row or column norms after one optimization step."""
        # Create a 3x5 matrix for testing
        matrix_size = (3, 5)

        # Initialize with random values then normalize
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize according to mode
        if mode == "col":
            # Normalize columns (each column has unit norm)
            param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        else:  # mode == "row"
            # Normalize rows (each row has unit norm)
            param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Create optimizer
        optimizer = ObliqueAdam([param], lr=0.01, betas=(0.9, 0.999), mode=mode)

        # Generate random gradient
        torch.manual_seed(456)  # For reproducible gradients
        param.grad = torch.randn_like(param.data)

        # Perform one optimization step
        optimizer.step()

        # Check that norms are preserved (should be 1.0 within tolerance)
        if mode == "col":
            final_norms = param.data.norm(dim=0)
        else:  # mode == "row"
            final_norms = param.data.norm(dim=1)

        # All norms should be approximately 1.0 (unit norm constraint)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_oblique_sgd_zero_gradient(self):
        """Test that ObliqueSGD handles zero gradients correctly."""
        matrix_size = (2, 4)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize columns
        param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        initial_param = param.data.clone()

        optimizer = ObliqueSGD([param], lr=0.1, mode="col")

        # Set zero gradient
        param.grad = torch.zeros_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Parameter should remain unchanged with zero gradient
        torch.testing.assert_close(param.data, initial_param, atol=1e-8, rtol=1e-8)

        # Norms should still be 1.0
        final_norms = param.data.norm(dim=0)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=1e-6, rtol=1e-6)

    def test_oblique_adam_zero_gradient(self):
        """Test that ObliqueAdam handles zero gradients correctly."""
        matrix_size = (2, 3)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize rows
        param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        initial_param = param.data.clone()

        optimizer = ObliqueAdam([param], lr=0.01, mode="row")

        # Set zero gradient
        param.grad = torch.zeros_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Parameter should remain unchanged with zero gradient
        torch.testing.assert_close(param.data, initial_param, atol=1e-8, rtol=1e-8)

        # Norms should still be 1.0
        final_norms = param.data.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=1e-6, rtol=1e-6)

    def test_oblique_sgd_large_gradient(self):
        """Test that ObliqueSGD handles large gradients correctly."""
        matrix_size = (3, 4)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize columns
        param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueSGD([param], lr=0.1, mode="col")

        # Set large gradient
        param.grad = 100.0 * torch.randn_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved despite large gradient
        final_norms = param.data.norm(dim=0)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms, expected_norms, atol=1e-6, rtol=1e-6, msg="Large gradients should not break norm preservation"
        )

    def test_oblique_adam_large_gradient(self):
        """Test that ObliqueAdam handles large gradients correctly."""
        matrix_size = (2, 5)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize rows
        param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueAdam([param], lr=0.01, mode="row")

        # Set large gradient
        param.grad = 1000.0 * torch.randn_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved despite large gradient
        final_norms = param.data.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_multiple_optimization_steps_preserve_norms(self):
        """Test that norms are preserved across multiple optimization steps."""
        matrix_size = (4, 4)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize columns
        param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueSGD([param], lr=0.05, momentum=0.8, mode="col")

        # Perform multiple optimization steps
        for step in range(10):
            torch.manual_seed(step)  # Different gradient each step
            param.grad = torch.randn_like(param.data)
            optimizer.step()

            # Check norms after each step
            final_norms = param.data.norm(dim=0)
            expected_norms = torch.ones_like(final_norms)
            torch.testing.assert_close(
                final_norms,
                expected_norms,
                atol=1e-6,
                rtol=1e-6,
            )

    def test_weight_decay_with_norm_preservation(self):
        """Test that weight decay doesn't break norm preservation."""
        matrix_size = (3, 3)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize rows
        param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueAdam([param], lr=0.01, weight_decay=0.01, mode="row")

        # Generate random gradient
        param.grad = torch.randn_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved with weight decay
        final_norms = param.data.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    absltest.main()
