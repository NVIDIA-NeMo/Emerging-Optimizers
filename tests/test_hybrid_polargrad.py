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

from emerging_optimizers.contrib import hybrid_polargrad
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


class HybridPolarGradFnTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    def test_non_2d_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only 2D tensors are supported"):
            hybrid_polargrad.hybrid_polargrad_fn(torch.randn((2, 4, 8), device=self.device))

    def test_invalid_side_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "side must be 'left' or 'right'"):
            hybrid_polargrad.hybrid_polargrad_fn(torch.randn((8, 4), device=self.device), side="both")

    def test_invalid_order_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "order must be 'polar_then_row' or 'row_then_polar'"):
            hybrid_polargrad.hybrid_polargrad_fn(torch.randn((8, 4), device=self.device), order="invalid")

    @parameterized.product(
        side=["right", "left"],
        order=["polar_then_row", "row_then_polar"],
        center_rows=[False, True],
    )
    def test_right_orthogonal_equivariance(self, side, order, center_rows) -> None:
        """f(G Q) == f(G) Q for an orthogonal Q acting on the column (right) dimension."""
        shape = (16, 4) if side == "right" else (4, 16)
        grad = torch.randn(shape, device=self.device)
        q, _ = torch.linalg.qr(torch.randn(shape[1], shape[1], device=self.device))

        rotated = hybrid_polargrad.hybrid_polargrad_fn(grad @ q, side=side, order=order, center_rows=center_rows)
        expected = hybrid_polargrad.hybrid_polargrad_fn(grad, side=side, order=order, center_rows=center_rows) @ q
        atol, rtol = (1e-2, 1e-3) if (side == "left" and center_rows) else (1e-4, 1e-4)
        torch.testing.assert_close(
            rotated,
            expected,
            atol=atol,
            rtol=rtol,
        )

    @parameterized.product(
        side=["right", "left"],
        order=["polar_then_row", "row_then_polar"],
        center_rows=[False, True],
    )
    def test_row_permutation_equivariance(self, side, order, center_rows) -> None:
        """f(P G) == P f(G) for a permutation P of the rows."""
        shape = (16, 4) if side == "right" else (4, 16)
        grad = torch.randn(shape, device=self.device)
        perm = torch.randperm(shape[0], device=self.device)

        permuted = hybrid_polargrad.hybrid_polargrad_fn(grad[perm], side=side, order=order, center_rows=center_rows)
        expected = hybrid_polargrad.hybrid_polargrad_fn(grad, side=side, order=order, center_rows=center_rows)[perm]
        atol, rtol = (1e-2, 1e-3) if (side == "left" and center_rows) else (1e-4, 1e-4)
        torch.testing.assert_close(
            permuted,
            expected,
            atol=atol,
            rtol=rtol,
        )

    @parameterized.parameters(("right", (16, 4)), ("left", (4, 16)))
    def test_row_then_polar_row_scale_invariance(self, side, shape) -> None:
        """For row_then_polar without centering, f(D G) == f(G) for power-of-two per-row scaling D."""
        grad = torch.randn(shape, device=self.device)
        scales = torch.pow(2.0, torch.randint(-2, 3, (shape[0], 1), device=self.device))

        torch.testing.assert_close(
            hybrid_polargrad.hybrid_polargrad_fn(scales * grad, side=side, order="row_then_polar"),
            hybrid_polargrad.hybrid_polargrad_fn(grad, side=side, order="row_then_polar"),
            atol=0.0,
            rtol=0.0,
        )

    @parameterized.product(
        side=["right", "left"],
        extra_scale_factor=[1.0, 0.25],
    )
    def test_polar_then_row_rows_have_expected_norm(self, side, extra_scale_factor) -> None:
        """With polar_then_row and center_rows=False, output rows have norm extra_scale_factor."""
        shape = (16, 4) if side == "right" else (4, 16)
        grad = torch.randn(shape, device=self.device)
        update = hybrid_polargrad.hybrid_polargrad_fn(
            grad, side=side, order="polar_then_row", extra_scale_factor=extra_scale_factor
        )
        torch.testing.assert_close(
            update.norm(dim=-1),
            torch.full((shape[0],), extra_scale_factor, device=self.device),
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.parameters(("right", (16, 4)), ("left", (4, 16)))
    def test_row_then_polar_is_orthonormal_and_scales_with_alpha(self, side, shape) -> None:
        """For row_then_polar, alpha=0 gives an orthonormal factor and alpha scales by nuclear norm."""
        grad = torch.randn(shape, device=self.device)
        u0 = hybrid_polargrad.hybrid_polargrad_fn(grad, side=side, order="row_then_polar", alpha=0.0)
        gram = u0.T @ u0 if side == "right" else u0 @ u0.T
        eye = torch.eye(gram.shape[0], device=self.device)
        torch.testing.assert_close(gram, eye, atol=1e-4, rtol=1e-4)

        u1 = hybrid_polargrad.hybrid_polargrad_fn(grad, side=side, order="row_then_polar", alpha=1.0)
        u_half = hybrid_polargrad.hybrid_polargrad_fn(
            grad, side=side, order="row_then_polar", alpha=0.5, extra_scale_factor=0.25
        )
        work_row = grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        nuclear_norm = torch.linalg.svdvals(work_row).sum()
        torch.testing.assert_close(u1, u0 * nuclear_norm, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(u_half, u0 * nuclear_norm.sqrt() * 0.25, atol=1e-4, rtol=1e-4)

    @parameterized.product(
        side=["right", "left"],
        order=["polar_then_row", "row_then_polar"],
    )
    def test_center_rows_output_is_centered(self, side, order) -> None:
        """With center_rows=True the update stays in the zero-column-mean quotient subspace."""
        shape = (16, 4) if side == "right" else (4, 16)
        grad = torch.randn(shape, device=self.device)
        update = hybrid_polargrad.hybrid_polargrad_fn(grad, side=side, order=order, center_rows=True)
        torch.testing.assert_close(
            update.mean(dim=0),
            torch.zeros(shape[1], device=self.device),
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.product(
        side=["right", "left"],
        order=["polar_then_row", "row_then_polar"],
        center_rows=[False, True],
    )
    def test_zero_grad_gives_zero_update(self, side, order, center_rows) -> None:
        shape = (8, 4) if side == "right" else (4, 8)
        update = hybrid_polargrad.hybrid_polargrad_fn(
            torch.zeros(shape, device=self.device), side=side, order=order, center_rows=center_rows
        )
        torch.testing.assert_close(update, torch.zeros(shape, device=self.device), atol=0.0, rtol=0.0)

    @parameterized.parameters(torch.float32, torch.bfloat16)
    def test_preserves_dtype(self, dtype) -> None:
        grad = torch.randn((16, 4), dtype=dtype, device=self.device)
        for order in ("polar_then_row", "row_then_polar"):
            update = hybrid_polargrad.hybrid_polargrad_fn(grad, order=order)
            self.assertEqual(update.dtype, dtype)
            self.assertTrue(torch.isfinite(update).all())

    @parameterized.product(
        side=["right", "left"],
        order=["polar_then_row", "row_then_polar"],
    )
    def test_usable_as_scaled_orthogonalize_fn(self, side, order) -> None:
        shape = (32, 8) if side == "right" else (8, 32)
        param = nn.Parameter(torch.randn(shape, device=self.device))
        scaled_orthogonalize_fn = functools.partial(
            hybrid_polargrad.hybrid_polargrad_fn,
            side=side,
            order=order,
            alpha=1.0,
            center_rows=True,
            extra_scale_factor=0.2,
        )
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
