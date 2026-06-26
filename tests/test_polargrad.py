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
from _comparison import assert_equal
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

        assert_equal(
            ref_out,
            test_out,
        )

    def test_negative_num_ns_steps_raises_value_error(self) -> None:
        """Test that PolarGrad raises ValueError for negative num_ns_steps."""
        test_param = nn.Parameter(torch.randn(5, 7, dtype=torch.float32, device=self.device))
        with self.assertRaisesRegex(ValueError, "num_ns_steps must be positive"):
            polargrad.PolarGrad([test_param], num_ns_steps=-1)


class RightPolarGradOrthFnTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

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
        )

    def test_usable_as_scaled_orthogonalize_fn(self) -> None:
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
