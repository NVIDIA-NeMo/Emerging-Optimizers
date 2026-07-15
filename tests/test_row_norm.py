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
import functools

import torch
import torch.nn as nn
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers import row_norm
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class RowNormFnTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.parameters((8, 4), (32, 8), (8, 32))
    def test_rows_close_to_unit_norm(self, num_rows, num_cols) -> None:
        grad = torch.randn((num_rows, num_cols), device=self.device)
        update = row_norm.row_norm_fn(grad)
        torch.testing.assert_close(
            update.norm(dim=-1),
            torch.ones(num_rows, device=self.device),
        )

    @parameterized.parameters((8, 4), (8, 32))
    def test_row_scale_invariance(self, num_rows, num_cols) -> None:
        """f(D G) == f(G) for a positive per-row scaling D."""
        grad = torch.randn((num_rows, num_cols), device=self.device)
        scales = torch.rand((num_rows, 1), device=self.device) + 0.5

        torch.testing.assert_close(
            row_norm.row_norm_fn(scales * grad),
            row_norm.row_norm_fn(grad),
        )

    @parameterized.product(shape=[(8, 4), (8, 32)], center_rows=[False, True])
    def test_row_permutation_equivariance(self, shape, center_rows) -> None:
        """f(P G) == P f(G) for a permutation P of the rows."""
        grad = torch.randn(shape, device=self.device)
        perm = torch.randperm(shape[0], device=self.device)

        torch.testing.assert_close(
            row_norm.row_norm_fn(grad[perm], center_rows=center_rows),
            row_norm.row_norm_fn(grad, center_rows=center_rows)[perm],
        )

    @parameterized.product(shape=[(8, 4), (32, 8)], center_rows=[False, True])
    def test_right_orthogonal_equivariance(self, shape, center_rows) -> None:
        """f(G Q) == f(G) Q for an orthogonal Q acting on the column (right) dimension."""
        grad = torch.randn(shape, device=self.device)
        q, _ = torch.linalg.qr(torch.randn(shape[1], shape[1], device=self.device))

        torch.testing.assert_close(
            row_norm.row_norm_fn(grad @ q, center_rows=center_rows),
            row_norm.row_norm_fn(grad, center_rows=center_rows) @ q,
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.parameters((8, 4), (8, 32))
    def test_center_rows_output_is_centered(self, num_rows, num_cols) -> None:
        """With center_rows=True the update stays in the zero-column-mean quotient subspace."""
        grad = torch.randn((num_rows, num_cols), device=self.device)
        update = row_norm.row_norm_fn(grad, center_rows=True)
        torch.testing.assert_close(
            update.mean(dim=0),
            torch.zeros(num_cols, device=self.device),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_zero_grad_gives_zero_update(self) -> None:
        update = row_norm.row_norm_fn(torch.zeros((8, 4), device=self.device))
        torch.testing.assert_close(update, torch.zeros((8, 4), device=self.device), atol=0.0, rtol=0.0)

    def test_usable_as_scaled_orthogonalize_fn(self) -> None:
        param = nn.Parameter(torch.randn((32, 8), device=self.device))
        scaled_orthogonalize_fn = functools.partial(row_norm.row_norm_fn, center_rows=True, extra_scale_factor=0.2)
        opt = OrthogonalizedOptimizer(
            [param],
            lr=1e-2,
            momentum=0.95,
            weight_decay=0.01,
            nesterov=False,
            weight_decay_method="decoupled",
            fp32_matmul_prec="highest",
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )
        for _ in range(3):
            param.grad = torch.randn_like(param)
            opt.step()
        self.assertTrue(torch.isfinite(param).all())


if __name__ == "__main__":
    absltest.main()
