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

from emerging_optimizers.utils import eig as eig_utils


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


# Base class for tests requiring determinism (seeding is handled by setUpModule when --seed is set)
class BaseTestCase(parameterized.TestCase):
    pass


class EigUtilsTest(BaseTestCase):
    def setUp(self) -> None:
        self.device = FLAGS.device

    def test_adaptive_criteria_met(self) -> None:
        """Tests the adaptive_criteria_met function for determining when to update eigenbasis."""
        # Create a diagonal matrix (should not trigger update with small tolerance)
        n = 4
        diagonal_matrix = torch.eye(n, device=self.device)

        # Test with small tolerance - should not update since matrix is diagonal
        self.assertTrue(
            eig_utils.met_approx_eigvals_criteria(
                diagonal_matrix,
                diagonal_matrix.diag(),
                tolerance=0.1,
            ),
            msg="Should not update for diagonal matrix with small tolerance",
        )

        # Create a matrix with significant off-diagonal elements
        off_diagonal_matrix = torch.tensor(
            [
                [1.0, 0.5, 0.3, 0.2],
                [0.5, 1.0, 0.4, 0.3],
                [0.3, 0.4, 1.0, 0.5],
                [0.2, 0.3, 0.5, 1.0],
            ],
            device=self.device,
        )

        # Test with small tolerance - should update since matrix has significant off-diagonal elements
        self.assertFalse(
            eig_utils.met_approx_eigvals_criteria(
                off_diagonal_matrix,
                off_diagonal_matrix.diag(),
                tolerance=0.1,
            ),
            msg="Should update for matrix with significant off-diagonal elements and small tolerance",
        )

        # Test with large tolerance - should not update even with off-diagonal elements
        self.assertTrue(
            eig_utils.met_approx_eigvals_criteria(
                off_diagonal_matrix,
                off_diagonal_matrix.diag(),
                tolerance=10.0,
            ),
            msg="Should not update for any matrix with large tolerance",
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"N": 4, "power_iter_steps": 1},
        {"N": 8, "power_iter_steps": 2},
        {"N": 16, "power_iter_steps": 3},
    )
    def test_update_eigenbasis_with_QR(self, N: int, power_iter_steps: int) -> None:
        """Tests the update_eigenbasis_with_QR function.

        Args:
            N: Size of the matrices to test
            power_iter_steps: Number of power iteration steps to perform
        """

        # Create test kronecker factor and eigenbasis
        kronecker_factor = torch.randn(N, N, device=self.device)
        # Make it symmetric positive definite
        kronecker_factor = kronecker_factor @ kronecker_factor.T
        # make a random orthonormal matrix
        eigenbasis = torch.randn(N, N, device=self.device)
        eigenbasis = torch.linalg.qr(eigenbasis).Q

        # Create inner adam second moment (should be positive)
        exp_avg_sq = torch.abs(torch.randn(N, N, device=self.device))

        # Create estimated eigenvalue matrix by projecting kronecker_factor onto eigenbasis's basis
        approx_eigenvalue_matrix = eigenbasis.T.mm(kronecker_factor).mm(eigenbasis)
        # Extract eigenvalues from the diagonal of the estimated eigenvalue matrix
        approx_eigvals = torch.diag(approx_eigenvalue_matrix)

        # Call the QR function to update the eigenbases and re-order the inner adam second moment
        Q_new, exp_avg_sq_new = eig_utils.orthogonal_iteration(
            approx_eigvals=approx_eigvals,
            kronecker_factor=kronecker_factor,
            eigenbasis=eigenbasis,
            ind=0,  # Test with first dimension
            exp_avg_sq=exp_avg_sq,
            power_iter_steps=power_iter_steps,
        )

        # Test 1: Check output shapes
        self.assertEqual(Q_new.shape, (N, N))
        self.assertEqual(exp_avg_sq_new.shape, exp_avg_sq.shape)

        # Test 2: Check orthogonality (Q^T Q ≈ I)
        expected_identity = torch.eye(N, dtype=Q_new.dtype, device=self.device)
        torch.testing.assert_close(
            Q_new.t() @ Q_new,
            expected_identity,
            atol=1e-5,
            rtol=1e-5,
            msg="Orthogonalization failed: Q^T Q is not close enough to the identity matrix.",
        )

        # Test 3: Check that exp_avg_sq is properly sorted based on eigenvalues
        # The sorting should be based on the diagonal elements of estimated_eigenvalue_matrix
        sort_idx = torch.argsort(approx_eigvals, descending=True)
        expected_exp_avg_sq = exp_avg_sq.index_select(0, sort_idx)
        torch.testing.assert_close(
            exp_avg_sq_new,
            expected_exp_avg_sq,
            atol=1e-5,
            rtol=1e-5,
            msg="exp_avg_sq was not properly sorted based on eigenvalues.",
        )

        # Test 4: Check that Q_new is different from input (transformation occurred)
        # This is a basic check - in practice they should be different due to power iteration
        self.assertFalse(torch.allclose(Q_new, eigenbasis))

    def test_eigh_with_fallback_descending_order(self) -> None:
        """Tests that eigenvalues are returned in descending order."""
        x = torch.tensor(
            [[4.0, 1.0], [1.0, 2.0]],
            device=self.device,
        )
        L, Q = eig_utils.eigh_with_fallback(x)
        # Eigenvalues should be in descending order
        self.assertTrue(torch.all(L[:-1] >= L[1:]))

    @parameterized.product(
        shape=[(8, 8), (16, 16), (31, 31)],
        force_double=[True, False],
    )
    def test_eigh_with_fallback_reconstruction_close_to_original(
        self,
        shape: tuple[int, int],
        force_double: bool,
    ) -> None:
        """Tests that Q @ diag(L) @ Q^T reconstructs the original matrix."""
        x = torch.randn(shape, device=self.device)
        x = x @ x.T  # symmetric positive semi-definite

        L, Q = eig_utils.eigh_with_fallback(
            x,
            force_double=force_double,
        )

        self.assertEqual(L.dtype, x.dtype)
        self.assertEqual(Q.dtype, x.dtype)

        # Reconstructing in double precision to avoid precision loss. The goal is to compare
        # output of eigh.
        reconstructed = Q.double() @ torch.diag(L.double()) @ Q.T.double()
        if not force_double:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 1e-6, 1e-6
        torch.testing.assert_close(reconstructed.to(x.dtype), x, atol=atol, rtol=rtol)

    def test_eigh_with_fallback_diagonal_input_smoke(self) -> None:
        """Tests that eigh_with_fallback works correctly with diagonal input."""
        x = torch.randn(4, 4, device=self.device)
        L, Q = eig_utils.eigh_with_fallback(x.diag().diag())
        self.assertEqual(L.shape, (4,))
        self.assertEqual(Q.shape, (4, 4))

    def test_conjugate_assert_2d_input(self) -> None:
        """Tests the conjugate function."""
        a = torch.randn(2, 3, 4, device=self.device)
        with self.assertRaises(TypeError):
            eig_utils.conjugate(a, a)

    def test_conjugate_match_reference(self) -> None:
        x = torch.randn(15, 17, device=self.device)
        a = x @ x.T
        _, p = torch.linalg.eigh(a)

        ref = p.T @ a @ p
        torch.testing.assert_close(eig_utils.conjugate(a, p), ref, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
