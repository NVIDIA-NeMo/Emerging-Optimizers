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
import math

import torch
from absl import flags, testing
from absl.testing import parameterized

from emerging_optimizers.psgd.procrustes_step import procrustes_step
from emerging_optimizers.utils import fp32_matmul_precision


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS


class ProcrustesStepTest(parameterized.TestCase):
    """Test cases for procrustes_step function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = FLAGS.device

    def _procrustes_objective(self, Q: torch.Tensor) -> torch.Tensor:
        """Helper function to compute Procrustes objective ||Q^H Q - I||_F^2."""
        return torch.linalg.matrix_norm(Q.H @ Q - torch.eye(Q.size(0), dtype=Q.dtype, device=Q.device), ord="fro") ** 2

    def test_improves_orthogonality_simple_case(self) -> None:
        """Test that procrustes_step doesn't worsen orthogonality for a simple case."""

        # Make a SPD non-orthogonal matrix
        Q = torch.randn(2, 2, device=self.device, dtype=torch.float32)
        Q = Q @ Q.T

        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=1 / 16)

        final_obj = self._procrustes_objective(Q)

        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-6)

    @parameterized.product(
        size=[8, 128, 1024],
        order=[2, 3],
    )
    def test_minimal_change_when_already_orthogonal(self, size: int, order: int) -> None:
        """Test that procrustes_step makes minimal changes to an already orthogonal matrix."""
        # Create an orthogonal matrix using QR decomposition
        A = torch.randn(size, size, device=self.device, dtype=torch.float32)
        with fp32_matmul_precision("highest"):
            Q, _ = torch.linalg.qr(A)

        initial_obj = self._procrustes_objective(Q)

        Q = procrustes_step(Q, max_step_size=1 / 16, order=order)

        final_obj = self._procrustes_objective(Q)

        # For already orthogonal matrices, the objective should remain small
        self.assertLess(final_obj.item(), 1e-5)
        self.assertLess(final_obj.item(), initial_obj.item() + 1e-5)

    def test_handles_small_norm_gracefully(self) -> None:
        """Test that procrustes_step handles matrices with small R norm improvement."""
        # Create a matrix very close to orthogonal
        A = torch.randn(3, 3, device=self.device, dtype=torch.float32)
        with fp32_matmul_precision("highest"):
            Q, _ = torch.linalg.qr(A)
        # Add tiny perturbation
        Q += 1e-10 * torch.randn_like(Q, dtype=torch.float32)

        initial_obj = self._procrustes_objective(Q)

        Q = procrustes_step(Q, max_step_size=0.0625)

        final_obj = self._procrustes_objective(Q)

        self.assertLess(final_obj.item(), 1e-6)
        self.assertLess(final_obj.item(), initial_obj.item() + 1e-6)

    @parameterized.product(
        max_step_size=[0.015625, 0.03125, 0.0625, 0.125],
        order=[2, 3],
    )
    def test_different_step_sizes_reduces_objective(self, max_step_size: float, order: int) -> None:
        """Test procrustes_step improvement with different step sizes."""
        perturbation = 1e-1 * torch.randn(10, 10, device=self.device, dtype=torch.float32) / math.sqrt(10)
        Q = torch.linalg.qr(torch.randn(10, 10, device=self.device, dtype=torch.float32)).Q + perturbation
        initial_obj = self._procrustes_objective(Q)

        Q = procrustes_step(Q, max_step_size=max_step_size, order=order)

        final_obj = self._procrustes_objective(Q)

        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-4)

    @parameterized.parameters(
        (8,),
        (64,),
        (512,),
        (8192,),
    )
    def test_different_matrix_sizes_reduces_objective(self, size: int) -> None:
        """Test procrustes_step improvement with different matrix sizes."""
        # Create a non-orthogonal matrix by scaling an orthogonal one
        A = torch.randn(size, size, device=self.device, dtype=torch.float32)
        with fp32_matmul_precision("highest"):
            Q_orth, _ = torch.linalg.qr(A)
        # Add perturbation, we choose 1e-2 to be small enough to not affect the objective too much
        # but large enough to make the matrix non-orthogonal.
        Q = Q_orth + 1e-2 * torch.randn(size, size, device=self.device, dtype=torch.float32) / math.sqrt(size)
        max_step_size = 0.5 * size ** (-1 / 3)
        initial_obj = self._procrustes_objective(Q)

        Q = procrustes_step(Q, max_step_size=max_step_size)

        final_obj = self._procrustes_objective(Q)

        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-3)

    def test_preserves_determinant_sign_for_real_matrices(self) -> None:
        """Test that procrustes_step preserves the sign of determinant for real matrices."""
        # Create real matrices with positive and negative determinants
        Q_pos = torch.tensor([[2.0, 0.1], [0.1, 1.5]], device=self.device, dtype=torch.float32)  # det > 0
        Q_neg = torch.tensor([[-2.0, 0.1], [0.1, 1.5]], device=self.device, dtype=torch.float32)  # det < 0

        initial_det_pos = torch.det(Q_pos)
        initial_det_neg = torch.det(Q_neg)

        Q_pos = procrustes_step(Q_pos, max_step_size=0.0625)
        Q_neg = procrustes_step(Q_neg, max_step_size=0.0625)

        final_det_pos = torch.det(Q_pos)
        final_det_neg = torch.det(Q_neg)

        # Signs should be preserved
        self.assertGreater(initial_det_pos.item() * final_det_pos.item(), 0)
        self.assertGreater(initial_det_neg.item() * final_det_neg.item(), 0)

    @parameterized.parameters(
        (0.015625,),
        (0.03125,),
        (0.0625,),
        (0.125,),
    )
    def test_order3_converges_faster_amplitude_recovery(self, max_step_size: float = 0.0625) -> None:
        """Test that order 3 converges faster than order 2 in amplitude recovery setting."""
        # Use amplitude recovery setup to compare convergence speed
        n = 10
        Q_init = torch.randn(n, n, device=self.device, dtype=torch.float32)
        u, s, vh = torch.linalg.svd(Q_init)
        amplitude = vh.mH @ torch.diag(s) @ vh

        # Start from the same initial point for both orders
        Q_order2 = torch.clone(Q_init)
        Q_order3 = torch.clone(Q_init)

        max_steps = 200
        tolerance = 0.01
        err_order2_list = []
        err_order3_list = []

        # Track convergence steps directly
        steps_to_converge_order2 = max_steps  # Default to max_steps if doesn't converge
        steps_to_converge_order3 = max_steps
        step_count = 0

        while step_count < max_steps:
            Q_order2 = procrustes_step(Q_order2, order=2, max_step_size=max_step_size)
            Q_order3 = procrustes_step(Q_order3, order=3, max_step_size=max_step_size)

            err_order2 = torch.max(torch.abs(Q_order2 - amplitude)) / torch.max(torch.abs(amplitude))
            err_order3 = torch.max(torch.abs(Q_order3 - amplitude)) / torch.max(torch.abs(amplitude))

            err_order2_list.append(err_order2.item())
            err_order3_list.append(err_order3.item())
            step_count += 1

            # Record convergence step for each order (only record the first time)
            if err_order2 < tolerance and steps_to_converge_order2 == max_steps:
                steps_to_converge_order2 = step_count
            if err_order3 < tolerance and steps_to_converge_order3 == max_steps:
                steps_to_converge_order3 = step_count

            # Stop if both have converged
            if err_order2 < tolerance and err_order3 < tolerance:
                break

        # Order 3 should converge in fewer steps or at least as fast
        self.assertLessEqual(
            steps_to_converge_order3,
            steps_to_converge_order2,
            f"Order 3 converged in {steps_to_converge_order3} steps, "
            f"order 2 in {steps_to_converge_order2} steps. Order 3 should be faster.",
        )

    @parameterized.product(
        order=[2, 3],
    )
    def test_recovers_amplitude_with_sign_ambiguity(self, order: int) -> None:
        """Test procrustes_step recovers amplitude of real matrix up to sign ambiguity.

        This is the main functional test for procrustes_step. It must recover the amplitude
        of a real matrix up to a sign ambiguity with probability 1.
        """
        for trial in range(10):
            n = 10
            Q = torch.randn(n, n, device=self.device, dtype=torch.float32)
            u, s, vh = torch.linalg.svd(Q)
            amplitude = vh.mH @ torch.diag(s) @ vh
            Q1, Q2 = torch.clone(Q), torch.clone(Q)
            Q2[1] *= -1  # add a reflection to Q2 to get Q2'

            err1, err2 = float("inf"), float("inf")
            max_iterations = 1000
            tolerance = 0.01
            step_count = 0

            while step_count < max_iterations and err1 >= tolerance and err2 >= tolerance:
                Q1 = procrustes_step(Q1, order=order)
                Q2 = procrustes_step(Q2, order=order)
                err1 = torch.max(torch.abs(Q1 - amplitude)) / torch.max(torch.abs(amplitude))
                err2 = torch.max(torch.abs(Q2 - amplitude)) / torch.max(torch.abs(amplitude))
                step_count += 1

            # Record convergence information
            converged = err1 < tolerance or err2 < tolerance
            final_error = min(err1, err2)

            self.assertTrue(
                converged,
                f"Trial {trial} (order={order}): procrustes_step failed to recover amplitude after {step_count} steps. "
                f"Final errors: err1={err1:.4f}, err2={err2:.4f}, best_error={final_error:.4f}",
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    testing.absltest.main()
