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

        Q_new = eig_utils.orthogonal_iteration(
            kronecker_factor=kronecker_factor,
            eigenbasis=eigenbasis,
            power_iter_steps=power_iter_steps,
        )

        # Test 1: Check output shape
        self.assertEqual(Q_new.shape, (N, N))

        # Test 2: Check orthogonality (Q^T Q ≈ I)
        expected_identity = torch.eye(N, dtype=Q_new.dtype, device=self.device)
        torch.testing.assert_close(
            Q_new.t() @ Q_new,
            expected_identity,
            atol=1e-5,
            rtol=1e-5,
            msg="Orthogonalization failed: Q^T Q is not close enough to the identity matrix.",
        )

        # Test 3: Check that Q_new is different from input (power iteration ran)
        self.assertFalse(torch.allclose(Q_new, eigenbasis))

    def test_eigh_with_fallback_descending_order(self) -> None:
        """Tests that eigenvalues are returned in descending order."""
        x = torch.tensor(
            [[4.0, 1.0], [1.0, 2.0]],
            device=self.device,
        )
        eigenvalues, eigenvectors = eig_utils.eigh_with_fallback(x)
        # Eigenvalues should be in descending order
        self.assertTrue(torch.all(eigenvalues[:-1] >= eigenvalues[1:]))

    @parameterized.product(
        shape=[(8, 8), (16, 16), (31, 31)],
        force_double=[True, False],
    )
    def test_eigh_with_fallback_reconstruction_close_to_original(
        self,
        shape: tuple[int, int],
        force_double: bool,
    ) -> None:
        """Tests that eigenvectors @ diag(eigenvalues) @ eigenvectors^T reconstructs the original matrix."""
        a = torch.randint(-8, 10, shape, device=self.device) / 16.0

        # Create symmetric positive semi-definite matrix
        x = a @ a.T

        eigenvalues, eigenvectors = eig_utils.eigh_with_fallback(
            x,
            force_double=force_double,
        )

        self.assertEqual(eigenvalues.dtype, x.dtype)
        self.assertEqual(eigenvectors.dtype, x.dtype)

        # Reconstructing in double precision to avoid precision loss. The goal is to compare
        # output of eigh.
        reconstructed = eigenvectors.double() @ torch.diag(eigenvalues.double()) @ eigenvectors.T.double()
        if not force_double:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 1e-6, 1e-6
        torch.testing.assert_close(reconstructed.to(x.dtype), x, atol=atol, rtol=rtol)

    def test_eigh_with_fallback_diagonal_input_smoke(self) -> None:
        """Tests that eigh_with_fallback works correctly with diagonal input."""
        x = torch.randn(4, 4, device=self.device)
        eigenvalues, eigenvectors = eig_utils.eigh_with_fallback(x.diag().diag())
        self.assertEqual(eigenvalues.shape, (4,))
        self.assertEqual(eigenvectors.shape, (4, 4))

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
        assert_equal(eig_utils.conjugate(a, p), ref)

    def test_eigh_with_fallback_reraises_runtime_error_when_force_double(self) -> None:
        """Test that eigh_with_fallback re-raises when force_double=True and eigh fails."""
        from unittest.mock import patch

        x = torch.randn(4, 4, device=self.device)
        x = x @ x.T

        with patch("torch.linalg.eigh", side_effect=RuntimeError("mock eigh failure")):
            with self.assertRaisesRegex(RuntimeError, "mock eigh failure"):
                eig_utils.eigh_with_fallback(x, force_double=True)


if __name__ == "__main__":
    absltest.main()
