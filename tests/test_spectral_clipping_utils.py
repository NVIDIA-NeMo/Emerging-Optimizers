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

import emerging_optimizers.orthogonalized_optimizers as orthogonalized_optimizers


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class TestSpectralClipping(parameterized.TestCase):
    def setUp(self):
        self.prev_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.device = FLAGS.device
        logging.info("Using device: %s", self.device)

    def tearDown(self):
        torch.set_float32_matmul_precision(self.prev_precision)

    @parameterized.product(
        dims=[(256, 128), (128, 256), (512, 512), (2048, 2048)],
        sigma_range=[(0.2, 0.8), (0.1, 20)],
    )
    def test_spectral_clipping_clips_singular_values_to_range(self, dims, sigma_range):
        """Test that spectral clipping properly clips singular values to the specified range."""

        sigma_min, sigma_max = sigma_range
        x = torch.randn(dims, device=self.device, dtype=torch.float32)

        _, original_singular_values, _ = torch.linalg.svd(x, full_matrices=False)
        original_min_sv = original_singular_values.min().item()
        original_max_sv = original_singular_values.max().item()

        clipped_x = orthogonalized_optimizers.spectral_clip(x, sigma_min=sigma_min, sigma_max=sigma_max)

        singular_values = torch.linalg.svdvals(clipped_x.double())

        min_sv = singular_values.min().item()
        max_sv = singular_values.max().item()

        logging.debug("Original matrix shape: %s", x.shape)
        logging.debug("Original singular values range: [%.6f, %.6f]", original_min_sv, original_max_sv)
        logging.debug("Clipped singular values range: [%.6f, %.6f]", min_sv, max_sv)
        logging.debug("Target range: [%.6f, %.6f]", sigma_min, sigma_max)
        logging.debug("Shape preservation: input %s -> output %s", x.shape, clipped_x.shape)

        # use higher tolerance for lower singular values
        # typically, this algorithm introduces more error for lower singular values
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

        U_orig, original_singular_values, Vt_orig = torch.linalg.svd(x.double(), full_matrices=False)
        original_min_sv = original_singular_values.min().item()
        original_max_sv = original_singular_values.max().item()
        logging.debug("Original matrix shape: %s", x.shape)
        logging.debug("Original singular values range: [%.6f, %.6f]", original_min_sv, original_max_sv)

        hardcapped_x = orthogonalized_optimizers.spectral_hardcap(x, beta=beta)

        U_hard, singular_values, Vt_hard = torch.linalg.svd(hardcapped_x.double(), full_matrices=False)

        tolerance_upper = 1e-1

        max_sv = singular_values.max().item()

        logging.debug("Hardcapped max singular value: %.6f", max_sv)
        logging.debug("Beta (upper bound): %.6f", beta)
        logging.debug("Shape preservation: input %s -> output %s", x.shape, hardcapped_x.shape)

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

        logging.debug("Polar factor Frobenius norm difference: %.6f", relative_polar_frobenius_diff)

        self.assertLessEqual(
            relative_polar_frobenius_diff,
            polar_tolerance,
        )


if __name__ == "__main__":
    absltest.main()
