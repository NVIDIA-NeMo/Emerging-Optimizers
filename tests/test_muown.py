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
import torch
from absl import flags, logging
from absl.testing import absltest, parameterized
from muown_reference import MuownReference

from emerging_optimizers.orthogonalized_optimizers.muown import Muown


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class MuownTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = FLAGS.device

    @parameterized.product(shape=[(8, 16), (16, 8), (33, 65)])
    def test_smoke(self, shape):
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        opt = Muown([p], lr=1e-2, weight_decay=0.01)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt.step()
        self.assertTrue(torch.isfinite(p).all())

    def test_raises_on_non_2d(self):
        for shape in [(8,), (2, 3, 4)]:
            p = torch.nn.Parameter(torch.randn(shape, device=self.device))
            p.grad = torch.randn_like(p)
            opt = Muown([p], lr=1e-2)
            with self.assertRaises(TypeError):
                opt.step()

    def test_raises_on_closure(self):
        p = torch.nn.Parameter(torch.randn((8, 16), device=self.device))
        p.grad = torch.randn_like(p)
        opt = Muown([p], lr=1e-2)
        with self.assertRaises(ValueError):
            opt.step(lambda: 0.0)

    @parameterized.product(shape=[(8, 16), (16, 8)], weight_decay=[0.0, 0.1])
    def test_row_norm_equals_magnitude_state(self, shape, weight_decay):
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        opt = Muown([p], lr=1e-2, weight_decay=weight_decay)
        for _ in range(4):
            p.grad = torch.randn_like(p)
            opt.step()
            row_norm = p.detach().norm(dim=1, keepdim=True)
            torch.testing.assert_close(
                row_norm,
                opt.state[p]["g"],
                atol=0,
                rtol=1e-5,
            )

    @parameterized.product(shape=[(8, 16), (16, 8), (33, 65)], momentum=[0.0, 0.95])
    def test_close_reference(self, shape, momentum):
        """Muown matches the reference even though their momentum conventions differ.

        Muown uses EMA momentum (``buf = m*buf + (1-m)*grad_v``) while the reference uses heavy-ball
        (``buf = m*buf + grad_v``). The two buffers differ only by the constant factor ``(1 - m)``, which
        the scale-invariant Newton-Schulz orthogonalization removes, so the direction updates agree (to
        Newton-Schulz's eps-normalization tolerance). Both feed the same orthogonalization function.
        """
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        p_ref = torch.nn.Parameter(p.detach().clone())

        opt = Muown(
            [p],
            lr=0.125,
            momentum=momentum,
            weight_decay=0.0,
            extra_scale_factor=0.25,
            fp32_matmul_prec="highest",
        )
        # Hold orthogonalization identical: feed the reference Muown's own scaled_orthogonalize_fn.
        opt_ref = MuownReference(
            [p_ref],
            orthogonalize_fn=opt.scaled_orthogonalize_fn,
            lr=0.125,
            momentum=momentum,
            nesterov=False,
            weight_decay=0.0,
        )

        for _ in range(5):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            p_ref.grad = grad.clone()
            opt.step()
            opt_ref.step()

        torch.testing.assert_close(
            p.detach(),
            p_ref.detach(),
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            opt.state[p]["g"],
            opt_ref.state[p_ref]["g"],
            atol=1e-7,
            rtol=1e-5,
        )


if __name__ == "__main__":
    absltest.main()
