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

from emerging_optimizers.orthogonalized_optimizers import polargrad
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


class PolarGradTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        extra_scale_factor=[1.0, 0.2],
    )
    def test_smoke(self, shape, extra_scale_factor) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        polargrad_opt = polargrad.PolarGrad(
            [test_param],
            extra_scale_factor=extra_scale_factor,
        )
        polargrad_opt.step()

    @parameterized.product(
        shape=[(4, 8), (16, 16), (32, 64), (13, 17)],
        extra_scale_factor=[0.25, 0.125],
    )
    def test_orthogonalize_fn_matches_ref(self, shape, extra_scale_factor) -> None:
        dummy_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        dummy_grad = torch.full(shape, 0.5, dtype=torch.float32, device=self.device)

        # Set num_ns_steps to 0 to skip Newton-Schulz iterations and only normalize the input gradient.
        polargrad_opt = polargrad.PolarGrad([dummy_param], num_ns_steps=0, extra_scale_factor=extra_scale_factor)
        norm_grad = torch.nn.functional.normalize(dummy_grad, p=2, dim=(-2, -1), eps=1e-7)

        # Assert normalization took effect
        self.assertFalse((norm_grad == 1).all())

        ref_scale = (norm_grad * dummy_grad).sum()
        ref_out = norm_grad * ref_scale * extra_scale_factor

        test_out = polargrad_opt.scaled_orthogonalize_fn(dummy_grad)

        torch.testing.assert_close(
            ref_out,
            test_out,
            atol=0,
            rtol=0,
        )

    def test_negative_num_ns_steps_raises_value_error(self) -> None:
        """Test that PolarGrad raises ValueError for negative num_ns_steps."""
        test_param = nn.Parameter(torch.randn(5, 7, dtype=torch.float32, device=self.device))
        with self.assertRaisesRegex(ValueError, "num_ns_steps must be positive"):
            polargrad.PolarGrad([test_param], num_ns_steps=-1)


class RightPolarGradOrthFnTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(shape=[(8, 4), (32, 8), (65, 33)])
    def test_orthonormal_columns(self, shape) -> None:
        """With alpha=0 the update is the right polar factor: its columns are orthonormal."""
        grad = torch.randn(shape, device=self.device)
        u = polargrad.right_polargrad_orth_fn(grad, alpha=0.0)
        n = shape[1]
        torch.testing.assert_close(
            u.transpose(-1, -2) @ u,
            torch.eye(n, device=self.device),
            atol=1e-5,
            rtol=1e-5,
            msg=lambda m: f"Right polar factor must have orthonormal columns.\n\n{m}",
        )

    @parameterized.product(shape=[(8, 4), (32, 8), (65, 33)])
    def test_nuclear_norm_scaling(self, shape) -> None:
        """alpha=1 scales the orthonormal-column update by the nuclear norm of the input."""
        grad = torch.randn(shape, device=self.device)
        u = polargrad.right_polargrad_orth_fn(grad, alpha=0.0)
        scaled = polargrad.right_polargrad_orth_fn(grad, alpha=1.0)
        nuclear_norm = torch.linalg.svdvals(grad).sum()
        torch.testing.assert_close(
            scaled,
            u * nuclear_norm,
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m: f"alpha=1 update must equal the orthonormal-column update times the nuclear norm.\n\n{m}",
        )

    @parameterized.product(extra_scale_factor=[0.2, 2.0])
    def test_extra_scale_factor_is_linear(self, extra_scale_factor) -> None:
        grad = torch.randn((16, 8), device=self.device)
        base = polargrad.right_polargrad_orth_fn(grad)
        scaled = polargrad.right_polargrad_orth_fn(grad, extra_scale_factor=extra_scale_factor)
        torch.testing.assert_close(scaled, base * extra_scale_factor, atol=1e-5, rtol=1e-5)

    def test_center_rows_zero_row_mean(self) -> None:
        """center_rows projects the update onto the zero-row-mean subspace (logit-shift removal)."""
        grad = torch.randn((32, 8), device=self.device)
        update = polargrad.right_polargrad_orth_fn(grad, center_rows=True)
        self.assertLess(update.mean(dim=0).abs().max().item(), 1e-5)

    @parameterized.product(shape=[(8, 4), (32, 8)], center_rows=[False, True])
    def test_right_orthogonal_equivariance(self, shape, center_rows) -> None:
        """f(G Q) == f(G) Q for an orthogonal Q acting on the hidden (right) dimension."""
        grad = torch.randn(shape, device=self.device)
        n = shape[1]
        q, _ = torch.linalg.qr(torch.randn(n, n, device=self.device))

        rotated = polargrad.right_polargrad_orth_fn(grad @ q, center_rows=center_rows)
        expected = polargrad.right_polargrad_orth_fn(grad, center_rows=center_rows) @ q
        torch.testing.assert_close(
            rotated,
            expected,
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m: f"Update must be equivariant to right-orthogonal (hidden) transforms.\n\n{m}",
        )

    @parameterized.product(shape=[(8, 4), (32, 8)], center_rows=[False, True])
    def test_row_permutation_equivariance(self, shape, center_rows) -> None:
        """f(P G) == P f(G) for a row (vocabulary) permutation P."""
        grad = torch.randn(shape, device=self.device)
        perm = torch.randperm(shape[0], device=self.device)

        permuted = polargrad.right_polargrad_orth_fn(grad[perm], center_rows=center_rows)
        expected = polargrad.right_polargrad_orth_fn(grad, center_rows=center_rows)[perm]
        torch.testing.assert_close(
            permuted,
            expected,
            atol=1e-4,
            rtol=1e-4,
            msg=lambda m: f"Update must be equivariant to row (vocabulary) permutations.\n\n{m}",
        )

    def test_usable_as_scaled_orthogonalize_fn(self) -> None:
        """The function can be partially applied as an OrthogonalizedOptimizer scaled_orthogonalize_fn."""
        param = nn.Parameter(torch.randn((32, 8), device=self.device))
        scaled_orthogonalize_fn = functools.partial(
            polargrad.right_polargrad_orth_fn, alpha=1.0, center_rows=True, extra_scale_factor=0.2
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
