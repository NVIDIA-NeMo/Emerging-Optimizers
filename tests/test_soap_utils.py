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

from emerging_optimizers import utils
from emerging_optimizers.soap import soap_utils


# Base class for tests requiring seeding for determinism
class BaseTestCase(parameterized.TestCase):
    def setUp(self):
        """Set random seed before each test."""
        # Set seed for PyTorch
        torch.manual_seed(42)
        # Set seed for CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)


class SoapUtilsTest(BaseTestCase):
    def test_adaptive_criteria_met(self) -> None:
        """Tests the adaptive_criteria_met function for determining when to update eigenbasis."""
        # Create a diagonal matrix (should not trigger update with small tolerance)
        n = 4
        diagonal_matrix = torch.eye(n)

        # Test with small tolerance - should not update since matrix is diagonal
        self.assertFalse(
            soap_utils._adaptive_criteria_met(
                approx_eigenvalue_matrix=diagonal_matrix,
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
            ]
        )

        # Test with small tolerance - should update since matrix has significant off-diagonal elements
        self.assertTrue(
            soap_utils._adaptive_criteria_met(
                approx_eigenvalue_matrix=off_diagonal_matrix,
                tolerance=0.1,
            ),
            msg="Should update for matrix with significant off-diagonal elements and small tolerance",
        )

        # Test with large tolerance - should not update even with off-diagonal elements
        self.assertFalse(
            soap_utils._adaptive_criteria_met(
                approx_eigenvalue_matrix=off_diagonal_matrix,
                tolerance=10.0,
            ),
            msg="Should not update for any matrix with large tolerance",
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"N": 4, "M": 8},
        {"N": 16, "M": 8},
        {"N": 32, "M": 8},
    )
    def test_get_eigenbasis_qr(self, N: int, M: int) -> None:
        """Tests the get_eigenbasis_qr function with a simplified state dict."""
        # Construct a preconditioner matrices for testing
        torch.manual_seed(0)
        g = torch.randn(M, N)
        L = g.mm(g.t()).float()
        R = g.t().mm(g).float()
        # Fake preconditioner list and orth list in the state
        state = {
            "GG": [L, R],  # precondition matrix
            "Q": [
                torch.randn(M, M),
                torch.randn(N, N),
            ],  # an existing Q (we'll refine it using QR)
            "exp_avg_sq": torch.abs(torch.randn(M, N)),  # Some arbitrary tensor for example
        }

        # We'll call get_eigenbasis_qr
        Q_new_list, exp_avg_sq_new = soap_utils.get_eigenbasis_qr(
            kronecker_factor_list=state["GG"],
            eigenbasis_list=state["Q"],
            exp_avg_sq=state["exp_avg_sq"],
            convert_to_float=True,
            use_adaptive_criteria=False,
            adaptive_update_tolerance=float("inf"),
            power_iter_steps=1,
        )

        self.assertEqual(len(Q_new_list), 2)
        Q_new_L = Q_new_list[0]
        # Check that Q_new is MxM
        self.assertEqual(Q_new_L.shape, (M, M))
        Q_new_R = Q_new_list[1]
        # Check that Q_new is NxN
        self.assertEqual(Q_new_R.shape, (N, N))

        # check Q^T Q ~ I
        expected_identity = torch.eye(M, dtype=Q_new_L.dtype, device=Q_new_L.device)
        torch.testing.assert_close(
            Q_new_L.t() @ Q_new_L,
            expected_identity,
            atol=1e-5,
            rtol=1e-5,
            msg="Orthogonalization failed: Q^T Q is not close enough to the identity matrix.",
        )

        expected_identity = torch.eye(N, dtype=Q_new_R.dtype, device=Q_new_R.device)
        torch.testing.assert_close(
            Q_new_R.t() @ Q_new_R,
            expected_identity,
            atol=1e-5,
            rtol=1e-5,
            msg="Orthogonalization failed: Q^T Q is not close enough to the identity matrix.",
        )

        # Also check that "exp_avg_sq" remains in the state with same shape if not merging
        self.assertEqual(exp_avg_sq_new.shape, (M, N))

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
        # Set random seed for reproducibility
        torch.manual_seed(0)

        # Create test kronecker factor and eigenbasis
        kronecker_factor = torch.randn(N, N)
        # Make it symmetric positive definite
        kronecker_factor = kronecker_factor @ kronecker_factor.T
        # make a random orthonormal matrix
        eigenbasis = torch.randn(N, N)
        eigenbasis = torch.linalg.qr(eigenbasis).Q

        # Create inner adam second moment (should be positive)
        exp_avg_sq = torch.abs(torch.randn(N, N))

        # Create estimated eigenvalue matrix by projecting kronecker_factor onto eigenbasis's basis
        approx_eigenvalue_matrix = eigenbasis.T.mm(kronecker_factor).mm(eigenbasis)
        # Extract eigenvalues from the diagonal of the estimated eigenvalue matrix
        approx_eigvals = torch.diag(approx_eigenvalue_matrix)

        # Call the QR function to update the eigenbases and re-order the inner adam second moment
        Q_new, exp_avg_sq_new = soap_utils._orthogonal_iteration(
            approx_eigvals=approx_eigvals,
            kronecker_factor=kronecker_factor,
            eigenbasis=eigenbasis,
            ind=0,  # Test with first dimension
            exp_avg_sq=exp_avg_sq,
            convert_to_float=True,
            power_iter_steps=power_iter_steps,
        )

        # Test 1: Check output shapes
        self.assertEqual(Q_new.shape, (N, N))
        self.assertEqual(exp_avg_sq_new.shape, exp_avg_sq.shape)

        # Test 2: Check orthogonality (Q^T Q ≈ I)
        expected_identity = torch.eye(N, dtype=Q_new.dtype, device=Q_new.device)
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

    @parameterized.parameters(  # type: ignore[misc]
        {"dims": [128, 512]},
        {"dims": []},
    )
    def test_get_eigenbasis_eigh(self, dims: list[int]) -> None:
        """Tests the get_eigenbasis_eigh function."""
        kronecker_factor_list = []
        for dim in dims:
            if dim == 0:
                kronecker_factor_list.append(torch.empty(0, 0))
                continue

            k_factor = torch.randn(dim, dim, device="cuda")
            k_factor = k_factor @ k_factor.T + torch.eye(dim, device="cuda") * 1e-5
            kronecker_factor_list.append(k_factor)

        Q_list = soap_utils.get_eigenbasis_eigh(kronecker_factor_list, convert_to_float=True)

        self.assertEqual(len(Q_list), len(kronecker_factor_list))

        for i, Q in enumerate(Q_list):
            orig_dim = dims[i]
            if orig_dim == 0:
                self.assertEqual(Q.shape, (0, 0))
                continue

            self.assertEqual(Q.shape, (orig_dim, orig_dim))

            # Check orthogonality: Q.T @ Q should be close to identity due to orthonormal matrix property
            identity = torch.eye(orig_dim, dtype=Q.dtype, device=Q.device)
            with utils.fp32_matmul_precision("highest"):
                orthogonality_check = Q.T @ Q

            torch.testing.assert_close(
                orthogonality_check,
                identity,
                atol=1e-3,
                rtol=1e-3,
            )
            with utils.fp32_matmul_precision("highest"):
                # Check that Q diagonalizes the original matrix, by checking if off-diagonal elements are close to zero
                diagonalized_matrix = Q.T @ kronecker_factor_list[i].float() @ Q
            num_off_diagonal_elements = orig_dim * (orig_dim - 1)
            off_diagonal_mask = ~torch.eye(orig_dim, dtype=torch.bool, device=diagonalized_matrix.device)
            off_diagonal_norm = torch.linalg.norm(diagonalized_matrix * off_diagonal_mask)
            scaled_off_diagonal_norm = off_diagonal_norm / (num_off_diagonal_elements**0.5)
            self.assertTrue(
                scaled_off_diagonal_norm < 1e-4,
                msg=f"Matrix {i} was not properly diagonalized. Off-diagonal norm: {off_diagonal_norm}",
            )

    def test_conjugate_assert_2d_input(self) -> None:
        """Tests the conjugate function."""
        a = torch.randn(2, 3, 4, device="cuda")
        with self.assertRaises(TypeError):
            soap_utils._conjugate(a, a)

    def test_conjugate_match_reference(self) -> None:
        x = torch.randn(15, 17, device="cuda")
        a = x @ x.T
        _, p = torch.linalg.eigh(a)

        ref = p.T @ a @ p
        torch.testing.assert_close(soap_utils._conjugate(a, p), ref, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
