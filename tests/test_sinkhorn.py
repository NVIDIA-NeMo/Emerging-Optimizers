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

from emerging_optimizers.orthogonalized_optimizers.sinkhorn_muon import SinkhornMuon
from emerging_optimizers.utils.sinkhorn_mapper import SinkhornMapper


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS


class TestSinkhornMapper(parameterized.TestCase):
    """Tests for the SinkhornMapper class."""

    @parameterized.parameters(
        (1023, 1023),
        (2047, 4095),
        (4095, 2047),
    )
    def test_output_is_doubly_stochastic(self, num_rows, num_cols):
        """After sufficient iterations the output should be approximately doubly stochastic (rows and columns each sum to ~1)."""
        x = torch.randn(num_rows, num_cols, device=FLAGS.device, dtype=torch.float32)
        x = SinkhornMapper(num_iters=50)(x)

        # All entries should be non-negative (they come from exp then normalization)
        self.assertTrue((x >= 0).all().item())

        # Row sums should be approximately equal
        row_sums = x.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.full_like(row_sums, row_sums.mean()),
            atol=1e-5,
            rtol=1e-5,
        )

        # Column sums should be approximately equal
        col_sums = x.sum(dim=-2)
        torch.testing.assert_close(
            col_sums,
            torch.full_like(col_sums, col_sums.mean()),
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.parameters(
        (5, 5),
        (7, 11),
        (13, 9),
    )
    def test_output_is_exactly_doubly_stochastic_with_integer_input(self, num_rows, num_cols):
        """With integer inputs and sufficient iterations, output should be exactly doubly stochastic."""
        # Use integer inputs to get exact results
        x = torch.randint(1, 10, (num_rows, num_cols), device=FLAGS.device, dtype=torch.float32)
        x = SinkhornMapper(num_iters=100)(x)

        # All entries should be non-negative
        self.assertTrue((x >= 0).all().item())

        # Row and column sums should be exactly equal (no approximation needed)
        row_sums = x.sum(dim=-1)
        col_sums = x.sum(dim=-2)

        # All row sums should be equal to each other
        torch.testing.assert_close(
            row_sums,
            torch.full_like(row_sums, row_sums.mean()),
            atol=1e-6,
            rtol=1e-6,
        )

        # All column sums should be equal to each other
        torch.testing.assert_close(
            col_sums,
            torch.full_like(col_sums, col_sums.mean()),
            atol=1e-6,
            rtol=1e-6,
        )

    @parameterized.parameters(
        (1023, 1023),
        (1023, 2047),
        (2047, 1023),
    )
    def test_output_is_non_negative(self, num_rows, num_cols):
        """Output entries should always be non-negative since they originate from exp."""
        x = torch.randn(num_rows, num_cols, device=FLAGS.device, dtype=torch.float32)
        x = SinkhornMapper(num_iters=50)(x)
        self.assertTrue((x >= 0).all().item())

    def test_more_iterations_improves_convergence(self):
        """More iterations should yield lower variance in row/column sums."""
        x = torch.randn(7, 7, device=FLAGS.device, dtype=torch.float32)

        x_1 = SinkhornMapper(num_iters=1)(x.clone())
        x_50 = SinkhornMapper(num_iters=50)(x.clone())

        row_var_1 = x_1.sum(dim=-1).var().item()
        col_var_1 = x_1.sum(dim=-2).var().item()
        row_var_50 = x_50.sum(dim=-1).var().item()
        col_var_50 = x_50.sum(dim=-2).var().item()

        self.assertLess(row_var_50, row_var_1)
        self.assertLess(col_var_50, col_var_1)

    @parameterized.parameters(
        (3, 5, 5),
        (3, 7, 6),
    )
    def test_batched_input(self, batch, num_rows, num_cols):
        """SinkhornMapper should work on batched (3D) inputs, normalizing the last two dims."""
        x = torch.randn(batch, num_rows, num_cols, device=FLAGS.device, dtype=torch.float32)
        x = SinkhornMapper(num_iters=30)(x)

        self.assertEqual(x.shape, (batch, num_rows, num_cols))
        # Check each batch element is approximately doubly stochastic
        for b, x_batch in enumerate(x):
            row_sums = x_batch.sum(dim=-1)
            col_sums = x_batch.sum(dim=-2)
            torch.testing.assert_close(
                row_sums,
                torch.full_like(row_sums, row_sums.mean()),
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                col_sums,
                torch.full_like(col_sums, col_sums.mean()),
                atol=1e-5,
                rtol=1e-5,
            )

    @parameterized.parameters(
        (10, 10),
        (20, 15),
    )
    def test_inplace_modes(self, num_rows, num_cols):
        """Test both inplace=False and inplace=True modes produce same results."""
        mapper = SinkhornMapper(num_iters=20)

        # Test inplace=False: input should not be modified
        x_original = torch.randn(num_rows, num_cols, device=FLAGS.device, dtype=torch.float32)
        x_copy = x_original.clone()
        result_non_inplace = mapper._sinkhorn_map(x_copy, inplace=False)

        # Verify input was not modified
        torch.testing.assert_close(x_copy, x_original, atol=1e-7, rtol=1e-7)

        # Test inplace=True: input should be modified
        x_inplace = x_original.clone()
        result_inplace = mapper._sinkhorn_map(x_inplace, inplace=True)

        # Verify input was modified (should equal the result)
        torch.testing.assert_close(x_inplace, result_inplace, atol=1e-7, rtol=1e-7)

        # Verify both modes produce the same result
        torch.testing.assert_close(result_non_inplace, result_inplace, atol=1e-5, rtol=1e-5)


class TestSinkhornMuon(parameterized.TestCase):
    """Tests for the SinkhornMuon optimizer."""

    @parameterized.product(
        shape=[(5, 7), (15, 15), (31, 63)],
        weight_decay_method=["decoupled", "independent", "l2"],
        use_nesterov=[True, False],
    )
    def test_smoke(self, shape, weight_decay_method, use_nesterov) -> None:
        """Smoke test: SinkhornMuon should run without errors."""
        # Initialize parameter as doubly-stochastic using SinkhornMapper
        init_data = torch.randint(1, 10, shape, dtype=torch.float32, device=FLAGS.device)
        test_param = nn.Parameter(SinkhornMapper(num_iters=50)(init_data))
        
        # Create gradient that is also doubly-stochastic
        grad_data = torch.randint(1, 10, shape, dtype=torch.float32, device=FLAGS.device)
        test_param.grad = SinkhornMapper(num_iters=50)(grad_data)

        opt = SinkhornMuon(
            [test_param],
            weight_decay_method=weight_decay_method,
            use_nesterov=use_nesterov,
        )
        opt.step()

    def test_invalid_num_iters_raises(self) -> None:
        """num_iters < 1 should raise ValueError."""
        # Initialize parameter as doubly-stochastic
        init_data = torch.randint(1, 10, (5, 5), dtype=torch.float32, device=FLAGS.device)
        test_param = nn.Parameter(SinkhornMapper(num_iters=50)(init_data))
        with self.assertRaises(ValueError):
            SinkhornMuon([test_param], num_iters=0)

    def test_invalid_eps_raises(self) -> None:
        """eps <= 0 should raise ValueError."""
        # Initialize parameter as doubly-stochastic
        init_data = torch.randint(1, 10, (5, 5), dtype=torch.float32, device=FLAGS.device)
        test_param = nn.Parameter(SinkhornMapper(num_iters=50)(init_data))
        with self.assertRaises(ValueError):
            SinkhornMuon([test_param], eps=0.0)
        with self.assertRaises(ValueError):
            SinkhornMuon([test_param], eps=-1e-8)

    def test_zero_lr_only_weight_decay(self) -> None:
        """With lr=0 and independent weight decay, weight decay + sinkhorn should be applied."""
        shape = (8, 8)
        weight_decay = 0.25
        num_iters = 20
        eps = 1e-8
        
        # Initialize parameter as doubly-stochastic
        init_data = torch.randint(1, 10, shape, dtype=torch.float32, device=FLAGS.device)
        test_param = nn.Parameter(SinkhornMapper(num_iters=50)(init_data))
        
        # Create gradient that is also doubly-stochastic
        grad_data = torch.randint(1, 10, shape, dtype=torch.float32, device=FLAGS.device)
        test_param.grad = SinkhornMapper(num_iters=50)(grad_data)

        # With lr=0, only weight decay is applied, then sinkhorn mapping runs on top
        expected_param = (1 - weight_decay) * test_param.data.clone()
        expected_param = SinkhornMapper(num_iters=num_iters, eps=eps)(expected_param)

        opt = SinkhornMuon(
            [test_param],
            lr=0.0,
            weight_decay=weight_decay,
            weight_decay_method="independent",
            momentum_beta=0.0,
            num_iters=num_iters,
            eps=eps,
        )
        opt.step()

        torch.testing.assert_close(test_param.data, expected_param, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
