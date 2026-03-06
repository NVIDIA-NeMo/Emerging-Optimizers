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
        state = {
            "GG": [L, R],  # precondition matrix
            "Q": [
                torch.randn(M, M, device=self.device),
                torch.randn(N, N, device=self.device),
            ],  # an existing Q (we'll refine it using QR)
            "exp_avg_sq": torch.abs(torch.randn(M, N, device=self.device)),  # Some arbitrary tensor for example
        }

        # We'll call get_eigenbasis_qr
        Q_new_list, exp_avg_sq_new = soap_utils.get_eigenbasis_qr(
            kronecker_factor_list=state["GG"],
            eigenbasis_list=state["Q"],
            exp_avg_sq=state["exp_avg_sq"],
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

    def test_get_eigenbasis_qr_empty_factor(self) -> None:
        """Tests get_eigenbasis_qr with an empty (numel()==0) kronecker factor."""
        torch.manual_seed(0)
        N = 4
        g = torch.randn(N, N, device=self.device)
        L = g.mm(g.t()).float()

        empty_factor = torch.empty(0, 0, device=self.device)
        kronecker_factor_list = [L, empty_factor]
        eigenbasis_list = [torch.randn(N, N, device=self.device), torch.empty(0, device=self.device)]
        exp_avg_sq = torch.abs(torch.randn(N, N, device=self.device))

        Q_new_list, exp_avg_sq_new = soap_utils.get_eigenbasis_qr(
            kronecker_factor_list=kronecker_factor_list,
            eigenbasis_list=eigenbasis_list,
            exp_avg_sq=exp_avg_sq,
        )

        self.assertEqual(len(Q_new_list), 2)
        self.assertEqual(Q_new_list[0].shape, (N, N))
        self.assertEqual(Q_new_list[1].numel(), 0)

    @parameterized.parameters(  # type: ignore[misc]
        {"dims": [128, 512]},
        {"dims": []},
        {"dims": [64, 0, 32]},
    )
    def test_get_eigenbasis_eigh(self, dims: list[int]) -> None:
        """Tests the get_eigenbasis_eigh function."""
        kronecker_factor_list = []
        for dim in dims:
            if dim == 0:
                kronecker_factor_list.append(torch.empty(0, 0))
                continue

            k_factor = torch.randn(dim, dim, device=self.device)
            k_factor = k_factor @ k_factor.T + torch.eye(dim, device=self.device) * 1e-5
            kronecker_factor_list.append(k_factor)

        Q_list = soap_utils.get_eigenbasis_eigh(kronecker_factor_list)

        self.assertEqual(len(Q_list), len(kronecker_factor_list))

        for i, Q in enumerate(Q_list):
            orig_dim = dims[i]
            if orig_dim == 0:
                self.assertEqual(Q.shape, (0, 0))
                continue

            self.assertEqual(Q.shape, (orig_dim, orig_dim))

            # Check orthogonality: Q.T @ Q should be close to identity due to orthonormal matrix property
            identity = torch.eye(orig_dim, dtype=Q.dtype, device=self.device)
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
            off_diagonal_mask = ~torch.eye(orig_dim, dtype=torch.bool, device=self.device)
            off_diagonal_norm = torch.linalg.norm(diagonalized_matrix * off_diagonal_mask)
            scaled_off_diagonal_norm = off_diagonal_norm / (num_off_diagonal_elements**0.5)
            self.assertTrue(
                scaled_off_diagonal_norm < 1e-4,
                msg=f"Matrix {i} was not properly diagonalized. Off-diagonal norm: {off_diagonal_norm}",
            )

    def test_all_eigenbases_met_criteria_empty_list_returns_true(self) -> None:
        kronecker_factor_list = []
        eigenbasis_list = []
        self.assertTrue(soap_utils.all_eigenbases_met_criteria(kronecker_factor_list, eigenbasis_list))

    @parameterized.parameters(
        {"N": 16},
        {"N": 33},
        {"N": 255},
    )
    def test_all_eigenbases_met_criteria_random_eigenbasis_returns_false(self, N: int) -> None:
        kronecker_factor_list = [torch.randn(N, N, device=self.device)]
        eigenbasis_list = [torch.diag(torch.randn(N, device=self.device))]
        self.assertFalse(soap_utils.all_eigenbases_met_criteria(kronecker_factor_list, eigenbasis_list))

    @parameterized.parameters(
        {"N": 16},
        {"N": 33},
        {"N": 255},
    )
    def test_all_eigenbases_met_criteria_true_eigenbasis_returns_true(self, N: int) -> None:
        kronecker_factor_list = [torch.randn(N, N, device=self.device)]

        eigenbasis_list = [torch.diag(torch.linalg.eigh(K).eigenvalues) for K in kronecker_factor_list]
        self.assertTrue(soap_utils.all_eigenbases_met_criteria(kronecker_factor_list, eigenbasis_list))


if __name__ == "__main__":
    absltest.main()
