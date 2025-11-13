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
import torch.nn as nn
from absl import flags
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers import AdaptiveOrthogonalizedOptimizer


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")
flags.DEFINE_integer("seed", 42, "Random seed for reproducible tests")

FLAGS = flags.FLAGS


# Create a dummy class for testing second moment methods
class TestAdaptiveOptimizer(AdaptiveOrthogonalizedOptimizer):
    """Test optimizer that uses AdaptiveOrthogonalizedOptimizer."""

    def __init__(self, params, second_moment_method: str = "adamuon"):
        super().__init__(
            params=params,
            lr=0.001,
            momentum_beta=0.9,
            weight_decay=0.0,
            use_nesterov=False,
            second_moment_method=second_moment_method,
            beta2=0.999,
            eps=1e-8,
            weight_decay_method="decoupled",
            fp32_matmul_prec="highest",
        )


class SecondMomentTest(parameterized.TestCase):
    def setUp(self):
        """Set random seed and device before each test."""
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)
        self.device = FLAGS.device

    @parameterized.parameters(
        {"shape": (8, 16), "beta2": 0.999, "eps": 1e-8},
        {"shape": (32, 64), "beta2": 0.99, "eps": 1e-6},
        {"shape": (4, 4), "beta2": 0.9, "eps": 1e-10},
    )
    def test_adamuon_method(self, shape, beta2, eps):
        """Test AdamMuon (elementwise) second moment method."""
        # Create a dummy parameter for the optimizer
        dummy_param = nn.Parameter(torch.randn(shape, device=self.device))
        optimizer = TestAdaptiveOptimizer([dummy_param], second_moment_method="adamuon")

        orth_grad = torch.randn(shape, device=self.device)
        second_moment = torch.zeros_like(orth_grad)

        # Apply second moment normalization
        result = optimizer._apply_second_moment_normalization(
            orth_grad=orth_grad,
            second_moment=second_moment,
            beta2=beta2,
            eps=eps,
        )

        # Check that second moment was updated
        expected_second_moment = (1 - beta2) * orth_grad.square()
        torch.testing.assert_close(second_moment, expected_second_moment, rtol=1e-5, atol=1e-7)

        # Check result shape
        self.assertEqual(result.shape, orth_grad.shape)

        # Check that result is computed correctly (elementwise division)
        expected_result = orth_grad / (expected_second_moment.sqrt() + eps)
        torch.testing.assert_close(result, expected_result, rtol=1e-5, atol=1e-7)

    @parameterized.parameters(
        {"shape": (16, 8)},  # rows > cols, should average along -1
        {"shape": (8, 16)},  # cols > rows, should average along -2
        {"shape": (32, 32)},  # square, should average along -1
    )
    def test_normuon_method(self, shape):
        """Test NorMuon (row/column-wise) second moment method."""
        # Create a dummy parameter for the optimizer
        dummy_param = nn.Parameter(torch.randn(shape, device=self.device))
        optimizer = TestAdaptiveOptimizer([dummy_param], second_moment_method="normuon")

        orth_grad = torch.randn(shape, device=self.device)

        # Determine which dimension should be averaged
        avg_dim = -1 if shape[-2] >= shape[-1] else -2
        expected_v_mean = orth_grad.square().mean(dim=avg_dim, keepdim=True)

        # Initialize second moment to zeros with correct shape
        second_moment = torch.zeros_like(expected_v_mean)

        beta2 = 0.999
        eps = 1e-8

        # Apply second moment normalization
        result = optimizer._apply_second_moment_normalization(
            orth_grad=orth_grad,
            second_moment=second_moment,
            beta2=beta2,
            eps=eps,
        )

        # Check that second moment was updated with correct shape
        expected_second_moment = (1 - beta2) * expected_v_mean
        torch.testing.assert_close(second_moment, expected_second_moment, rtol=1e-5, atol=1e-7)

        # Check result shape matches input
        self.assertEqual(result.shape, orth_grad.shape)

        # Check that result uses reciprocal square root
        step_size = expected_second_moment.clamp_min(eps).rsqrt_()
        expected_result = orth_grad * step_size
        torch.testing.assert_close(result, expected_result, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    absltest.main()
