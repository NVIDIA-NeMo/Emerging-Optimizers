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
        (1024, 1024),
        (2048, 4096),
        (4096, 2048),
    )
    def test_output_is_doubly_stochastic(self, rows, cols):
        """After sufficient iterations the output should be approximately doubly stochastic
        (rows and columns each sum to ~1)."""
        x = torch.randn(rows, cols, device=FLAGS.device, dtype=torch.float32)
        SinkhornMapper(t_max=50)(x)

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
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
    )
    def test_output_is_non_negative(self, rows, cols):
        """Output entries should always be non-negative since they originate from exp."""
        x = torch.randn(rows, cols, device=FLAGS.device, dtype=torch.float32)
        SinkhornMapper(t_max=50)(x)
        self.assertTrue((x >= 0).all().item())

    def test_more_iterations_improves_convergence(self):
        """More iterations should yield lower variance in row/column sums."""
        x = torch.randn(8, 8, device=FLAGS.device, dtype=torch.float32)

        x_1 = x.clone()
        SinkhornMapper(t_max=1)(x_1)
        x_50 = x.clone()
        SinkhornMapper(t_max=50)(x_50)

        row_var_1 = x_1.sum(dim=-1).var().item()
        col_var_1 = x_1.sum(dim=-2).var().item()
        row_var_50 = x_50.sum(dim=-1).var().item()
        col_var_50 = x_50.sum(dim=-2).var().item()

        self.assertLess(row_var_50, row_var_1)
        self.assertLess(col_var_50, col_var_1)

    @parameterized.parameters(
        (3, 4, 4),
        (2, 8, 6),
    )
    def test_batched_input(self, batch, rows, cols):
        """SinkhornMapper should work on batched (3D) inputs, normalizing the last two dims."""
        x = torch.randn(batch, rows, cols, device=FLAGS.device, dtype=torch.float32)
        SinkhornMapper(t_max=30)(x)

        self.assertEqual(x.shape, (batch, rows, cols))
        # Check each batch element is approximately doubly stochastic
        for b in range(batch):
            row_sums = x[b].sum(dim=-1)
            col_sums = x[b].sum(dim=-2)
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


class TestSinkhornMuon(parameterized.TestCase):
    """Tests for the SinkhornMuon optimizer."""

    @parameterized.product(
        shape=[(5, 7), (16, 16), (32, 64)],
        weight_decay_method=["decoupled", "independent", "l2"],
        use_nesterov=[True, False],
    )
    def test_smoke(self, shape, weight_decay_method, use_nesterov) -> None:
        """Smoke test: SinkhornMuon should run without errors."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        opt = SinkhornMuon(
            [test_param],
            weight_decay_method=weight_decay_method,
            use_nesterov=use_nesterov,
        )
        opt.step()

    def test_invalid_t_max_raises(self) -> None:
        """t_max < 1 should raise ValueError."""
        test_param = nn.Parameter(torch.randn(4, 4, device=FLAGS.device, dtype=torch.float32))
        with self.assertRaises(ValueError):
            SinkhornMuon([test_param], t_max=0)

    def test_invalid_epsilon_raises(self) -> None:
        """epsilon <= 0 should raise ValueError."""
        test_param = nn.Parameter(torch.randn(4, 4, device=FLAGS.device, dtype=torch.float32))
        with self.assertRaises(ValueError):
            SinkhornMuon([test_param], epsilon=0.0)
        with self.assertRaises(ValueError):
            SinkhornMuon([test_param], epsilon=-1e-8)

    def test_zero_lr_only_weight_decay(self) -> None:
        """With lr=0 and independent weight decay, weight decay + sinkhorn should be applied."""
        shape = (8, 8)
        weight_decay = 0.25
        t_max = 20
        epsilon = 1e-8
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        # With lr=0, only weight decay is applied, then sinkhorn mapping runs on top
        expected_param = (1 - weight_decay) * test_param.data.clone()
        SinkhornMapper(t_max=t_max, epsilon=epsilon)(expected_param)

        opt = SinkhornMuon(
            [test_param],
            lr=0.0,
            weight_decay=weight_decay,
            weight_decay_method="independent",
            momentum_beta=0.0,
            t_max=t_max,
            epsilon=epsilon,
        )
        opt.step()

        torch.testing.assert_close(test_param.data, expected_param, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
