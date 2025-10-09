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
from absl import logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon, get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import _COEFFICIENT_SETS, newton_schulz


def newton_schulz_ref(x: torch.Tensor, coefficient_sets: list[tuple[float, float, float]]) -> torch.Tensor:
    """Reference Newton-Schulz iteration to compute the zeroth power / orthogonalization of x."""
    # Muon is not for 1d parameters
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions since Muon is not for 1d parameters.")

    steps = len(coefficient_sets)

    # transpose tensor to perform whitening on the smaller dimension
    needs_transpose = x.size(-2) > x.size(-1)
    if needs_transpose:
        x = x.mT

    # Ensure spectral norm is at most 1
    X = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for i in range(steps):
        with utils.fp32_matmul_precision("highest"):
            a, b, c = coefficient_sets[i]
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

    # undo transpose if necessary
    if needs_transpose:
        X = X.mT
    return X


class TestMuonUtils(parameterized.TestCase):
    def setUp(self):
        self.prev_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision(self.prev_precision)

    @parameterized.parameters(
        (512, 512),
        (512, 256),
        (256, 512),
    )
    def test_newtonschulz5_svd_close(self, dim1, dim2):
        shape = (dim1, dim2)
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        out_zeropowerns = newton_schulz(x, steps=5, coefficient_type="quintic")
        U, _, V = torch.linalg.svd(x, full_matrices=False)
        out_zeropower_svd = (U @ V).float()
        # Check that the outputs are close.
        # Note: Due to the nature of the approximation and different computation paths and the bfloat16 conversion
        # will lead to differences and need a higher tolerance.
        # This is expected behavior, see https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        torch.testing.assert_close(
            out_zeropowerns.float(),
            out_zeropower_svd.float(),
            atol=1e-1,
            rtol=1e-7,
        )

    @parameterized.parameters(
        (512, 512),
        (512, 256),
        (256, 512),
    )
    def test_newtonschulz5_close_to_reference(self, dim1, dim2):
        x = torch.randn(dim1, dim2, device="cuda", dtype=torch.float32)
        out_zeropower_test = newton_schulz(x, steps=5, coefficient_type="quintic")
        out_zeropowerns_ref = newton_schulz_ref(
            x,
            coefficient_sets=_COEFFICIENT_SETS["quintic"],
        )

        torch.testing.assert_close(
            out_zeropower_test,
            out_zeropowerns_ref,
            atol=1e-6,
            rtol=1e-7,
        )

    @parameterized.parameters(
        (511, 513),
        (511, 257),
        (257, 513),
    )
    def test_newtonschulz_custom_coeff_close_to_reference(self, dim1, dim2):
        x = 2 ** torch.randint(-10, -5, (dim1, dim2), device="cuda", dtype=torch.float32)

        test_coefficient_sets = [
            (3, 5, 7),
            (11, 13, 17),
        ]
        out_zeropower_test = newton_schulz(
            x,
            steps=2,
            coefficient_type="custom",
            custom_coefficient_sets=test_coefficient_sets,
        )
        out_zeropowerns_ref = newton_schulz_ref(
            x,
            coefficient_sets=test_coefficient_sets,
        )

        torch.testing.assert_close(
            out_zeropower_test,
            out_zeropowerns_ref,
            atol=0,
            rtol=1e-6,
        )

    @parameterized.parameters(
        (512, 512),
        (512, 256),
        (256, 512),
    )
    def test_polar_express_better_than_quintic(self, dim1, dim2):
        # Create a matrix with terrible condition number
        min_dim = min(dim1, dim2)

        # Generate proper random orthogonal matrices for SVD structure
        random_left = torch.randn(dim1, min_dim, device="cuda", dtype=torch.float32)
        random_right = torch.randn(dim2, min_dim, device="cuda", dtype=torch.float32)
        # orthogonalize the random matrices using QR decomposition
        u, _ = torch.linalg.qr(random_left)
        v, _ = torch.linalg.qr(random_right)

        # Create singular values with terrible condition number (range from 1e6 to 1e-6)
        singular_values = torch.logspace(6, -6, min_dim, device="cuda", dtype=torch.float32)

        # Construct the matrix with terrible condition number using proper SVD: U @ diag(S) @ V^T
        # condition number = 1e12
        x = u @ torch.diag(singular_values) @ v.T

        # Compare polar express vs quintic Newton-Schulz methods
        out_svd = (u @ v.T).float()
        out_polar_express = newton_schulz(x, steps=8, coefficient_type="polar_express")
        out_quintic = newton_schulz(x, steps=5, coefficient_type="quintic")

        l2_norm_diff_polar = torch.norm(out_polar_express.float() - out_svd.float(), p=2)
        l2_norm_diff_quintic = torch.norm(out_quintic.float() - out_svd.float(), p=2)

        logging.info(f"Polar express norm difference: {l2_norm_diff_polar:.6f}")
        logging.info(f"Quintic norm difference: {l2_norm_diff_quintic:.6f}")

        self.assertLess(
            l2_norm_diff_polar,
            l2_norm_diff_quintic,
            f"Polar Express norm is larger than Quintic norm: {l2_norm_diff_polar:.6f} > {l2_norm_diff_quintic:.6f}",
        )

    @parameterized.product(
        size_pairs=[(512, 512), (512, 256), (256, 512), (97, 37), (37, 97)],
        mode=["shape_scaling", "spectral", "unit_rms_norm"],
    )
    def test_get_scale_factor(self, size_pairs, mode):
        size_out, size_in = size_pairs
        scale = get_muon_scale_factor(size_out, size_in, mode)
        if mode == "shape_scaling":
            self.assertEqual(scale, math.sqrt(max(1, size_out / size_in)))
        elif mode == "spectral":
            self.assertEqual(scale, math.sqrt(max(size_out, size_in)))
        elif mode == "unit_rms_norm":
            self.assertEqual(scale, math.sqrt(size_out / size_in))
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def test_qkv_split_shapes_validation(self):
        """Test validation of qkv_split_shapes parameter"""
        dummy_param = torch.nn.Parameter(torch.randn(4, 4))
        dummy_args = dict(split_qkv=True, is_qkv_fn=lambda x: True)
        # Test non-integer values
        with self.assertRaises(ValueError) as cm:
            Muon([dummy_param], **dummy_args, qkv_split_shapes=(512.5, 256, 256))
        self.assertIn("must be integers", str(cm.exception))

        # Test negative values
        with self.assertRaises(ValueError) as cm:
            Muon([dummy_param], **dummy_args, qkv_split_shapes=(512, -256, 256))
        self.assertIn("must be positive", str(cm.exception))

        # Test wrong number of elements
        with self.assertRaises(ValueError) as cm:
            Muon([dummy_param], **dummy_args, qkv_split_shapes=(512, 256))
        self.assertIn("tuple of 3 integers", str(cm.exception))


class TestNewtonSchulzStepWithTsyrk(parameterized.TestCase):
    def setUp(self):
        self.prev_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision(self.prev_precision)

    @parameterized.parameters(
        (32, 32),
        (32, 64),
    )
    def test_match_newton_schulz_step_by_gemm(self, dim1, dim2):
        x = torch.randint(-2, 3, (dim1, dim2), device="cuda", dtype=torch.bfloat16)
        test_out = muon_utils.newton_schulz_step_tsyrk(x, 2**-1, 2**-2, 2**-3)
        test_ref = muon_utils.newton_schulz_step(x, 2**-1, 2**-2, 2**-3)

        torch.testing.assert_close(test_out, test_ref, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
