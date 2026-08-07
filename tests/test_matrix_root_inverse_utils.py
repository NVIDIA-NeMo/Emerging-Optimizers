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
from _comparison import assert_close_to_identity
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.soap.matrix_root_inverse_utils import inv_root_via_eigh, mat_root_inv_via_scaled_cans


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class MatrixRootInverseUtilsTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(4, 4), (2, 4, 4)],
        fp32_matmul_prec=["high", "highest"],
    )
    def test_mat_root_inv_via_scaled_cans_smoke(
        self,
        shape: tuple[int, ...],
        fp32_matmul_prec: utils.FP32MatmulPrecT,
    ) -> None:
        x = torch.randn(*shape, device=FLAGS.device)
        matrix = x @ x.mT + 0.1 * torch.eye(shape[-1], device=FLAGS.device)
        previous_precision = torch.get_float32_matmul_precision()

        with utils.fp32_matmul_precision(fp32_matmul_prec):
            inverse_root = mat_root_inv_via_scaled_cans(matrix)

        self.assertEqual(inverse_root.shape, matrix.shape)
        self.assertEqual(inverse_root.dtype, torch.float32)
        self.assertEqual(torch.get_float32_matmul_precision(), previous_precision)

    @parameterized.product(shape=[(8, 8), (16, 16), (2, 8, 8), (3, 16, 16)])
    def test_mat_root_inv_via_scaled_cans_accuracy_tridiag_toeplitz(self, shape: tuple[int, ...]) -> None:
        # Generate a symmetric positive-definite tridiagonal Toeplitz matrix, then scale each
        # batched copy by a positive integer to exercise matrices with different norms.
        matrix_size = shape[-1]
        base_matrix = 2.0 * torch.eye(matrix_size, device=FLAGS.device)
        base_matrix.diagonal(offset=1).fill_(0.25)
        base_matrix.diagonal(offset=-1).fill_(0.25)
        if len(shape) == 2:
            matrix = base_matrix
        else:
            batch_scale = torch.arange(
                1,
                shape[0] + 1,
                device=FLAGS.device,
                dtype=base_matrix.dtype,
            ).view(-1, 1, 1)
            matrix = base_matrix.unsqueeze(0) * batch_scale

        inverse_root = mat_root_inv_via_scaled_cans(matrix)
        whitened_matrix = inverse_root @ matrix @ inverse_root
        matrix_root = torch.linalg.inv(inverse_root)
        reconstructed_matrix = matrix_root @ matrix_root

        for whitened_matrix_slice in whitened_matrix.reshape(-1, matrix_size, matrix_size):
            assert_close_to_identity(whitened_matrix_slice, off_diag_atol=2e-4, diag_atol=2e-4)
        torch.testing.assert_close(reconstructed_matrix, matrix, atol=2e-4, rtol=2e-4)

    def test_mat_root_inv_via_scaled_cans_rejects_non_fp32_tensor(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be in float32"):
            mat_root_inv_via_scaled_cans(torch.eye(4, device=FLAGS.device, dtype=torch.bfloat16))

    def test_mat_root_inv_via_scaled_cans_rejects_medium_fp32_matmul_precision(self) -> None:
        with (
            utils.fp32_matmul_precision("medium"),
            self.assertRaisesRegex(RuntimeError, "`medium`.*insufficient for CANS"),
        ):
            mat_root_inv_via_scaled_cans(torch.eye(4, device=FLAGS.device))

    @parameterized.parameters(8, 16, 32)
    def test_inv_root_via_eigh_reconstruct_idendity(self, m) -> None:
        x = torch.randn(m, m)
        test_matrix = 0.5 * x @ x.mT + torch.eye(m) * 1e-3

        inverse_root = inv_root_via_eigh(test_matrix)
        assert_close_to_identity(inverse_root @ test_matrix @ inverse_root, off_diag_atol=1e-3, diag_atol=1e-3)


if __name__ == "__main__":
    absltest.main()
