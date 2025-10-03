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
from absl import logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers.spectral_clipping_utils import (
    spectral_clip,
    spectral_hardcap,
)


class TestSpectralClipping(parameterized.TestCase):
    def setUp(self):
        self.prev_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        torch.manual_seed(1234)

    def tearDown(self):
        torch.set_float32_matmul_precision(self.prev_precision)

    @parameterized.product(
        dims=[(256, 128), (128, 256), (512, 512), (2048, 2048)],
        sigma_range=[(0.2, 0.8), (0.1, 20)],
    )
    def test_spectral_clipping(self, dims, sigma_range):
        """Test that spectral clipping properly clips singular values to the specified range."""

        sigma_min, sigma_max = sigma_range
        x = torch.randn(dims, device=self.device, dtype=torch.float32)

        _, original_singular_values, _ = torch.linalg.svd(x, full_matrices=False)
        original_min_sv = original_singular_values.min().item()
        original_max_sv = original_singular_values.max().item()

        clipped_x = spectral_clip(x, sigma_min=sigma_min, sigma_max=sigma_max)

        _, singular_values, _ = torch.linalg.svd(clipped_x, full_matrices=False)

        min_sv = singular_values.min().item()
        max_sv = singular_values.max().item()

        logging.info(f"Original matrix shape: {x.shape}")
        logging.info(f"Original singular values range: [{original_min_sv:.6f}, {original_max_sv:.6f}]")
        logging.info(f"Clipped singular values range: [{min_sv:.6f}, {max_sv:.6f}]")
        logging.info(f"Target range: [{sigma_min:.6f}, {sigma_max:.6f}]")
        logging.info(f"Shape preservation: input {x.shape} -> output {clipped_x.shape}")

        tolerance_upper = 1e-1
        tolerance_lower = 5e-1
        self.assertGreaterEqual(
            min_sv + tolerance_lower,
            sigma_min,
        )
        self.assertLessEqual(
            max_sv - tolerance_upper,
            sigma_max,
        )

        self.assertEqual(clipped_x.shape, x.shape)

    @parameterized.product(
        dims=[(256, 128), (128, 256), (512, 512), (100, 200)],
        beta=[0.5, 1.0, 0.8, 2.0],
    )
    def test_spectral_hardcap(self, dims, beta):
        """Test that spectral hardcap properly clips singular values from above to be less than beta."""
        x = torch.randn(dims, device=self.device, dtype=torch.float32)

        U_orig, original_singular_values, Vt_orig = torch.linalg.svd(x, full_matrices=False)
        original_min_sv = original_singular_values.min().item()
        original_max_sv = original_singular_values.max().item()
        logging.info(f"Original matrix shape: {x.shape}")
        logging.info(f"Original singular values range: [{original_min_sv:.6f}, {original_max_sv:.6f}]")

        hardcapped_x = spectral_hardcap(x, beta=beta)

        U_hard, singular_values, Vt_hard = torch.linalg.svd(hardcapped_x, full_matrices=False)

        tolerance_upper = 1e-1

        max_sv = singular_values.max().item()

        logging.info(f"Hardcapped max singular value: {max_sv:.6f}")
        logging.info(f"Beta (upper bound): {beta:.6f}")
        logging.info(f"Shape preservation: input {x.shape} -> output {hardcapped_x.shape}")

        self.assertLessEqual(
            max_sv - tolerance_upper,
            beta,
        )

        self.assertEqual(hardcapped_x.shape, x.shape)

        # Test that singular vectors are preserved (polar factor UV^T should be similar)
        polar_orig = U_orig @ Vt_orig
        polar_hard = U_hard @ Vt_hard

        # The polar factors should be very similar since hardcap only changes singular values, compute the relative difference
        relative_polar_frobenius_diff = torch.norm(polar_orig - polar_hard, "fro") / torch.norm(polar_orig, "fro")
        polar_tolerance = 1e-4

        logging.info(f"Polar factor Frobenius norm difference: {relative_polar_frobenius_diff:.6f}")

        self.assertLessEqual(
            relative_polar_frobenius_diff,
            polar_tolerance,
        )


if __name__ == "__main__":
    absltest.main()
