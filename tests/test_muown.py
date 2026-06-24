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
    # Use the most precise fp32 matmul path so the reference (which calls the optimizer's
    # orthogonalization outside the fp32_matmul_precision context) computes bit-comparable results.
    torch.set_float32_matmul_precision("highest")


class MuownTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = FLAGS.device

    @parameterized.product(shape=[(8, 16), (16, 8), (33, 65)])
    def test_smoke(self, shape):
        """A few steps run and keep the weight finite."""
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        opt = Muown([p], lr=1e-2, weight_decay=0.01)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt.step()
        self.assertTrue(torch.isfinite(p).all())

    def test_raises_on_non_2d(self):
        """Muown supports 2D parameters only."""
        for shape in [(8,), (2, 3, 4)]:
            p = torch.nn.Parameter(torch.randn(shape, device=self.device))
            p.grad = torch.randn_like(p)
            opt = Muown([p], lr=1e-2)
            with self.assertRaises(TypeError):
                opt.step()

    def test_raises_on_closure(self):
        """Closures are not supported."""
        p = torch.nn.Parameter(torch.randn((8, 16), device=self.device))
        p.grad = torch.randn_like(p)
        opt = Muown([p], lr=1e-2)
        with self.assertRaises(ValueError):
            opt.step(lambda: 0.0)

    @parameterized.product(shape=[(8, 16), (16, 8)], weight_decay=[0.0, 0.1])
    def test_row_norm_equals_magnitude_state(self, shape, weight_decay):
        """The reparameterization invariant ||W_row|| == g holds after each step."""
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        opt = Muown([p], lr=1e-2, weight_decay=weight_decay)
        for _ in range(4):
            p.grad = torch.randn_like(p)
            opt.step()
            row_norm = p.detach().norm(dim=1, keepdim=True)
            torch.testing.assert_close(
                row_norm,
                opt.state[p]["g"],
                atol=1e-5,
                rtol=1e-5,
                msg=lambda m: f"Row norms of W must equal the magnitude state g.\n\n{m}",
            )

    def test_weight_decay_shrinks_magnitude(self):
        """Decoupled weight decay shrinks the magnitude g relative to the no-decay run."""
        p_wd = torch.nn.Parameter(torch.randn((16, 32), device=self.device))
        p_no = torch.nn.Parameter(p_wd.detach().clone())
        opt_wd = Muown([p_wd], lr=1e-2, weight_decay=0.1)
        opt_no = Muown([p_no], lr=1e-2, weight_decay=0.0)
        for _ in range(5):
            grad = torch.randn_like(p_wd)
            p_wd.grad = grad.clone()
            p_no.grad = grad.clone()
            opt_wd.step()
            opt_no.step()
        self.assertLess(opt_wd.state[p_wd]["g"].sum().item(), opt_no.state[p_no]["g"].sum().item())

    @parameterized.product(shape=[(8, 16), (16, 8), (33, 65)], momentum=[0.0, 0.95])
    def test_agrees_with_reference(self, shape, momentum):
        """Muown matches the authors' reference implementation (no weight decay).

        The reference uses classic (heavy-ball) momentum while Muown uses EMA momentum; the two differ by
        a constant factor that the scale-invariant Newton-Schulz orthogonalization removes, so the updates
        agree. Both use the same injected orthogonalization, so only float rounding (lerp vs mul/add,
        Newton-Schulz normalization) separates them — hence a tight tolerance rather than bit-identity.
        """
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        p_ref = torch.nn.Parameter(p.detach().clone())

        opt = Muown(
            [p],
            lr=1e-2,
            momentum=momentum,
            weight_decay=0.0,
            extra_scale_factor=0.2,
            fp32_matmul_prec="highest",
        )
        # Hold orthogonalization identical: feed the reference Muown's own scaled_orthogonalize_fn.
        opt_ref = MuownReference(
            [p_ref],
            orthogonalize_fn=opt.scaled_orthogonalize_fn,
            lr=1e-2,
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
            atol=1e-5,
            rtol=1e-4,
            msg=lambda m: f"Muown weight diverged from the reference implementation.\n\n{m}",
        )
        torch.testing.assert_close(
            opt.state[p]["g"],
            opt_ref.state[p_ref]["g"],
            atol=1e-5,
            rtol=1e-4,
            msg=lambda m: f"Muown magnitude g diverged from the reference implementation.\n\n{m}",
        )


if __name__ == "__main__":
    absltest.main()
