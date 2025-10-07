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
from absl import flags, testing
from absl.testing import parameterized

from emerging_optimizers.psgd.psgd_utils import (
    norm_lower_bound_skew,
    norm_lower_bound_spd,
    uniformize_q_in_place,
)


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS


class BalanceQTest(parameterized.TestCase):
    """Test cases for uniformize_q_in_place function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = FLAGS.device

    def test_normalization_on_empty_list(self) -> None:
        """Test uniformize_q_in_place with empty list."""
        Q_list = []
        uniformize_q_in_place(Q_list)  # Should not raise any errors
        self.assertEqual(len(Q_list), 0)

    def test_normalization_on_single_tensor(self) -> None:
        """Test uniformize_q_in_place with single tensor."""
        Q = torch.randn(3, 3, device=self.device)
        original_Q = Q.clone()
        uniformize_q_in_place([Q])
        # for a single tensor, the result should be the same as the original
        torch.testing.assert_close(Q, original_Q)

    def test_normalization_on_two_tensors(self) -> None:
        """Test uniformize_q_in_place with two tensors."""
        Q1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        Q2 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=self.device)

        orig_max1 = torch.max(torch.abs(Q1))
        orig_max2 = torch.max(torch.abs(Q2))

        uniformize_q_in_place([Q1, Q2])

        new_max1 = torch.max(torch.abs(Q1))
        new_max2 = torch.max(torch.abs(Q2))

        # Should be equal to geometric mean of original maxima
        expected_max = (orig_max1 * orig_max2) ** 0.5
        self.assertAlmostEqual(new_max1.item(), expected_max.item(), places=5)
        self.assertAlmostEqual(new_max2.item(), expected_max.item(), places=5)

    @parameterized.parameters(
        (32, 32, 32),
        (256, 256, 256),
        (4096, 4096, 4096),
    )
    def test_normalization_on_three_tensors(self, size1: int, size2: int, size3: int) -> None:
        """Test uniformize_q_in_place with multiple tensors of different dynamic ranges."""
        Q1 = torch.randn(size1, size1, device=self.device) * 10.0
        Q2 = torch.randn(size2, size2, device=self.device) * 0.01
        Q3 = torch.randn(size3, size3, device=self.device) * 1.0

        orig_max1 = torch.max(torch.abs(Q1))
        orig_max2 = torch.max(torch.abs(Q2))
        orig_max3 = torch.max(torch.abs(Q3))

        uniformize_q_in_place([Q1, Q2, Q3])

        # All tensors should have the same max absolute value
        new_max1 = torch.max(torch.abs(Q1))
        new_max2 = torch.max(torch.abs(Q2))
        new_max3 = torch.max(torch.abs(Q3))

        # Should be equal to geometric mean
        expected_max = (orig_max1 * orig_max2 * orig_max3) ** (1.0 / 3.0)
        self.assertAlmostEqual(new_max1.item(), expected_max.item(), places=5)
        self.assertAlmostEqual(new_max2.item(), expected_max.item(), places=5)
        self.assertAlmostEqual(new_max3.item(), expected_max.item(), places=5)

    def test_modifies_in_place_on_three_tensors(self) -> None:
        """Test that uniformize_q_in_place modifies tensors in place."""
        Q = torch.randn(3, 3, device=self.device)
        original_id = id(Q)
        uniformize_q_in_place([Q, torch.randn(2, 2, device=self.device)])

        # Should be the same object (modified in place)
        self.assertEqual(id(Q), original_id)


class NormLowerBoundSpdTest(parameterized.TestCase):
    """Test cases for norm_lower_bound_spd function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = FLAGS.device

    def test_diagonal_matrix(self) -> None:
        """Test norm_lower_bound_spd with diagonal matrix."""
        # For diagonal matrix, spectral norm equals largest diagonal entry
        diag_values = torch.tensor([1.0, 3.0, 2.0], device=self.device)
        A = torch.diag(diag_values)

        bound = norm_lower_bound_spd(A)
        actual_norm = torch.max(diag_values)

        # Bound should be <= actual norm
        self.assertLessEqual(bound.item(), actual_norm.item() + 1e-5)
        # For diagonal matrix, bound should be reasonably tight
        self.assertGreater(bound.item(), 0.5 * actual_norm.item())

    def test_identity_matrix(self) -> None:
        """Test norm_lower_bound_spd with identity matrix."""
        A = torch.eye(3, device=self.device)
        bound = norm_lower_bound_spd(A)

        # For identity matrix, spectral norm is 1
        self.assertAlmostEqual(bound.item(), 1.0, places=5)

    def test_zero_matrix(self) -> None:
        """Test norm_lower_bound_spd with zero matrix."""
        A = torch.zeros(3, 3, device=self.device)
        bound = norm_lower_bound_spd(A)

        # For zero matrix, bound should be 0
        self.assertAlmostEqual(bound.item(), 0.0, places=5)

    @parameterized.product(
        dtype=[torch.float32, torch.bfloat16],
        size=[32, 256, 4096],
    )
    def test_norm_lower_bound_spd_is_lower_bound(self, dtype: torch.dtype, size: int) -> None:
        """Test that norm_lower_bound_spd provides a valid lower bound."""
        # Create a random SPD matrix
        B = torch.randn(size, size, dtype=dtype, device=self.device)
        A = B @ B.T + 1e-3 * torch.eye(
            size, dtype=dtype, device=self.device
        )  # Ensure positive definite and well-conditioned

        bound = norm_lower_bound_spd(A)
        # Spectral norm (largest singular value)
        # Pytorch's matrix norm does not support bfloat16, so we convert to float32
        actual_norm = torch.linalg.matrix_norm(A.to(torch.float32), ord=2)

        # Bound should be <= actual norm
        self.assertLessEqual(bound.item(), actual_norm.item() + 1e-5)
        # Bound should be positive for positive definite matrix
        self.assertGreater(bound.item(), 0.0)


class NormLowerBoundSkewTest(parameterized.TestCase):
    """Test cases for norm_lower_bound_skew function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = FLAGS.device

    def test_zero_matrix(self) -> None:
        """Test norm_lower_bound_skew with zero matrix."""
        A = torch.zeros(3, 3, device=self.device)
        bound = norm_lower_bound_skew(A)

        # For zero matrix, bound should be 0
        self.assertAlmostEqual(bound.item(), 0.0, places=5)

    def test_small_skew_symmetric_matrix(self) -> None:
        """Test norm_lower_bound_skew with a simple skew-symmetric matrix."""
        # Create a simple 3x3 skew-symmetric matrix
        A = torch.tensor([[0.0, 1.0, -2.0], [-1.0, 0.0, 3.0], [2.0, -3.0, 0.0]], device=self.device)

        bound = norm_lower_bound_skew(A)
        # Compute actual spectral norm
        actual_norm = torch.linalg.matrix_norm(A, ord=2)

        # Bound should be <= actual norm
        self.assertLessEqual(bound.item(), actual_norm.item() + 1e-5)
        # Bound should be positive for non-zero matrix
        self.assertGreater(bound.item(), 0.0)

    def test_identity_based_skew_matrix(self) -> None:
        """Test norm_lower_bound_skew with matrix based on identity structure."""
        # Create skew-symmetric matrix from anti-symmetric part of random matrix
        n = 4
        B = torch.randn(n, n, device=self.device)
        A = B - B.T  # This creates a skew-symmetric matrix

        bound = norm_lower_bound_skew(A)
        actual_norm = torch.linalg.matrix_norm(A, ord=2)

        # Bound should be <= actual norm
        self.assertLessEqual(bound.item(), actual_norm.item() + 1e-5)

    @parameterized.product(
        dtype=[torch.float32, torch.float64],
        size=[32, 128, 256],
    )
    def test_norm_lower_bound_skew_is_lower_bound(self, dtype: torch.dtype, size: int) -> None:
        """Test that norm_lower_bound_skew provides a valid lower bound."""
        # Create a random skew-symmetric matrix
        B = torch.randn(size, size, dtype=dtype, device=self.device)
        A = B - B.T  # Ensure skew-symmetric property: A^T = -A

        bound = norm_lower_bound_skew(A)
        # Compute actual spectral norm
        actual_norm = torch.linalg.matrix_norm(A.to(torch.float32), ord=2)

        # Bound should be <= actual norm (with small tolerance for numerical errors)
        self.assertLessEqual(bound.item(), actual_norm.item() + 1e-4)

        # Bound should be non-negative
        self.assertGreaterEqual(bound.item(), 0.0)

    @parameterized.parameters([4, 16, 32])
    def test_different_subspace_dimensions(self, rank: int) -> None:
        """Test norm_lower_bound_skew with different subspace dimensions."""
        # Create a skew-symmetric matrix
        B = torch.randn(64, 64, device=self.device)
        A = B - B.T

        bound = norm_lower_bound_skew(A, k=rank, half_iters=2)

        self.assertGreaterEqual(bound.item(), 0.0)

        actual_norm = torch.linalg.matrix_norm(A, ord=2)
        self.assertLessEqual(bound.item(), actual_norm.item() + 1e-5)


if __name__ == "__main__":
    torch.manual_seed(42)
    testing.absltest.main()
