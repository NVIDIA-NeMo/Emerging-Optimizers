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
from _comparison import assert_close_to_identity, assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.soap import soap_utils


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


class SoapUtilsTest(BaseTestCase):
    def setUp(self) -> None:
        self.device = FLAGS.device

    @parameterized.parameters(  # type: ignore[misc]
        {"N": 4, "M": 8},
        {"N": 16, "M": 8},
        {"N": 32, "M": 8},
    )
    def test_get_eigenbasis_qr(self, N: int, M: int) -> None:
        """Tests the get_eigenbasis_qr function with a simplified state dict."""
        # Construct a preconditioner matrices for testing
        torch.manual_seed(0)
        g = torch.randn(M, N, device=self.device)
        L = g.mm(g.t()).float()
        R = g.t().mm(g).float()
        # Fake preconditioner list and orth list in the state
        kronecker_factor_list = [L, R]
        eigenbasis_list = [
            torch.randn(M, M, device=self.device),
            torch.randn(N, N, device=self.device),
        ]
        eigvals_list, Q_new_list = soap_utils.get_eigenbasis_qr(
            kronecker_factor_list=kronecker_factor_list,
            eigenbasis_list=eigenbasis_list,
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
        assert_close_to_identity(Q_new_L.t() @ Q_new_L, diag_atol=1e-5, off_diag_atol=1e-5)
        assert_close_to_identity(Q_new_R.t() @ Q_new_R, diag_atol=1e-5, off_diag_atol=1e-5)

        self.assertEqual(len(eigvals_list), 2)
        for eigvals, kronecker_factor, Q_new in zip(eigvals_list, kronecker_factor_list, Q_new_list, strict=True):
            self.assertEqual(eigvals.shape, (kronecker_factor.shape[0],))
            torch.testing.assert_close(
                eigvals,
                torch.diag(Q_new.t() @ kronecker_factor @ Q_new),
                atol=1e-5,
                rtol=1e-5,
                msg=lambda msg: f"eigvals do not match the Rayleigh quotients diag(Q^T K Q)\n\n{msg}",
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"N": 4, "M": 8},
        {"N": 16, "M": 8},
    )
    def test_permute_eigenbasis_and_exp_avg_sq(self, N: int, M: int) -> None:
        """Permutes eigenbasis columns and exp_avg_sq slots consistently, in descending eigval order."""
        g = torch.randint(-5, 6, (M, N), device=self.device) / 16.0
        # Add a small ridge so K is full-rank (g @ g.t() is rank-deficient when M > N, etc.),
        # which keeps the approximate eigenvalues away from numerical zero where ordering becomes
        # ambiguous under float rounding.
        eps_eye = lambda n: 1e-3 * torch.eye(n, device=self.device)
        kronecker_factor_list = [g @ g.t() + eps_eye(M), g.t() @ g + eps_eye(N)]
        eigenbasis_list = [torch.linalg.qr(Q).Q for Q in kronecker_factor_list]
        exp_avg_sq = torch.abs(torch.randint(-5, 6, (M, N), device=self.device) / 16.0)

        permuted_eigenbasis_list, permuted_exp_avg_sq = soap_utils.permute_eigenbasis_and_exp_avg_sq(
            kronecker_factor_list,
            eigenbasis_list,
            exp_avg_sq,
        )

        # Compute the expected per-axis permutations from the originals.
        sort_idx_list = []
        for K, Q in zip(kronecker_factor_list, eigenbasis_list, strict=True):
            sort_idx_list.append(torch.argsort(torch.diag(Q.t() @ K @ Q), descending=True))

        # Each eigenbasis is column-permuted by its own sort_idx.
        for i, (Q_old, Q_permuted) in enumerate(zip(eigenbasis_list, permuted_eigenbasis_list, strict=True)):
            assert_equal(
                Q_permuted,
                Q_old[:, sort_idx_list[i]],
                msg=lambda m, i=i: f"eigenbasis i={i} not permuted by sort_idx\n\n{m}",
            )

        # exp_avg_sq is permuted along every axis cumulatively.
        expected_sq = exp_avg_sq
        for i, sort_idx in enumerate(sort_idx_list):
            expected_sq = expected_sq.index_select(i, sort_idx)
        assert_equal(
            permuted_exp_avg_sq,
            expected_sq,
            msg=lambda m: f"exp_avg_sq not permuted to match permuted eigenbases\n\n{m}",
        )

        # Permuted eigenbases yield descending approximate eigenvalues.
        for K, Q in zip(kronecker_factor_list, permuted_eigenbasis_list, strict=True):
            permuted_eigvals = torch.diag(Q.t() @ K @ Q)
            self.assertTrue(torch.all(permuted_eigvals[:-1] >= permuted_eigvals[1:]))

    def test_permute_eigenbasis_and_exp_avg_sq_batched_matches_loop(self) -> None:
        batch, m, n = 3, 8, 4
        basis_list = [torch.linalg.qr(torch.randn(batch, k, k, device=self.device)).Q for k in (m, n)]
        spectrum_list = [torch.arange(1.0, k + 1, device=self.device) for k in (m, n)]
        kronecker_factor_list = [
            (basis * spectrum.unsqueeze(-2)) @ basis.mT
            for basis, spectrum in zip(basis_list, spectrum_list, strict=True)
        ]
        eigenbasis_list = [torch.roll(basis, 1, dims=-1) for basis in basis_list]
        exp_avg_sq = torch.abs(torch.randn(batch, m, n, device=self.device))

        permuted_eigenbasis_list, permuted_exp_avg_sq = soap_utils.permute_eigenbasis_and_exp_avg_sq(
            kronecker_factor_list,
            eigenbasis_list,
            exp_avg_sq,
        )

        for b in range(batch):
            expected_eigenbasis_list, expected_exp_avg_sq = soap_utils.permute_eigenbasis_and_exp_avg_sq(
                [kronecker_factor[b] for kronecker_factor in kronecker_factor_list],
                [eigenbasis[b] for eigenbasis in eigenbasis_list],
                exp_avg_sq[b],
            )
            for i, expected_eigenbasis in enumerate(expected_eigenbasis_list):
                assert_equal(
                    permuted_eigenbasis_list[i][b],
                    expected_eigenbasis,
                    msg=lambda msg, b=b, i=i: f"eigenbasis {i} differs from per-slice reference at slice {b}\n\n{msg}",
                )
            assert_equal(
                permuted_exp_avg_sq[b],
                expected_exp_avg_sq,
                msg=lambda msg, b=b: f"exp_avg_sq differs from per-slice reference at slice {b}\n\n{msg}",
            )

    @parameterized.parameters(  # type: ignore[misc]
        {"dims": [128, 512]},
        {"dims": []},
    )
    def test_get_eigenbasis_eigh(self, dims: list[int]) -> None:
        """Tests the get_eigenbasis_eigh function."""
        kronecker_factor_list = []
        for dim in dims:
            k_factor = torch.randn(dim, dim, device=self.device)
            k_factor = k_factor @ k_factor.T + torch.eye(dim, device=self.device) * 1e-5
            kronecker_factor_list.append(k_factor)

        eigvals_list, Q_list = soap_utils.get_eigenbasis_eigh(kronecker_factor_list)

        self.assertEqual(len(Q_list), len(kronecker_factor_list))
        self.assertEqual(len(eigvals_list), len(kronecker_factor_list))

        for i, (eigvals, Q) in enumerate(zip(eigvals_list, Q_list, strict=True)):
            orig_dim = dims[i]
            self.assertEqual(Q.shape, (orig_dim, orig_dim))
            self.assertEqual(eigvals.shape, (orig_dim,))

            # Eigenvalues are returned in descending order
            self.assertTrue(torch.all(eigvals[:-1] >= eigvals[1:]))

            # Check orthogonality: Q.T @ Q should be close to identity due to orthonormal matrix property
            with utils.fp32_matmul_precision("highest"):
                orthogonality_check = Q.T @ Q

            assert_close_to_identity(orthogonality_check, diag_atol=1e-3, off_diag_atol=1e-3)
            with utils.fp32_matmul_precision("highest"):
                # Check that Q diagonalizes the original matrix, by checking if off-diagonal elements are close to zero
                diagonalized_matrix = Q.T @ kronecker_factor_list[i].float() @ Q
            num_off_diagonal_elements = orig_dim * (orig_dim - 1)
            off_diagonal_mask = ~torch.eye(orig_dim, dtype=torch.bool, device=self.device)
            off_diagonal_norm = torch.linalg.norm(diagonalized_matrix * off_diagonal_mask)
            scaled_off_diagonal_norm = off_diagonal_norm / (num_off_diagonal_elements**0.5)
            self.assertTrue(
                scaled_off_diagonal_norm < 1e-4,
                msg=f"Matrix {i} was not properly diagonalized. Off-diagonal norm: {off_diagonal_norm}",
            )


if __name__ == "__main__":
    absltest.main()
