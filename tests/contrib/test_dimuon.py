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

from emerging_optimizers.contrib.dimuon import Dimuon
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS

# Polar Express coefficients as written in https://github.com/anpaure/dimuon, before its 1.01 safety factor.
_RAW_POLAR_EXPRESS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


def _reference_direction(m_old: torch.Tensor, m_new: torch.Tensor, damping: float) -> torch.Tensor:
    """Dense FP64 Dimuon direction, with the Polar Express input scaled by 1 / (1.01 ||X||)."""

    def condition(energy: torch.Tensor) -> torch.Tensor:
        if not energy.any():
            return torch.ones_like(energy)
        return (energy / energy.max() + damping) ** -0.25

    left = condition(m_old.square().sum(dim=1))[:, None]
    right = condition(m_old.square().sum(dim=0))[None, :]
    x = left * m_new * right
    transpose = x.size(0) > x.size(1)
    x = (x.T if transpose else x) / (1.01 * x.norm())
    for a, b, c in _RAW_POLAR_EXPRESS:
        gram = x @ x.T
        x = a / 1.01 * x + (b / 1.01**3 * gram + c / 1.01**5 * gram @ gram) @ x
    d = left * (x.T if transpose else x) * right
    return d * min(d.shape) ** 0.5 / d.norm()


def _scaled_gradient(shape: tuple[int, int], device: str) -> torch.Tensor:
    """Gradient whose rows and columns differ in scale by up to 100x, so the conditioners matter."""
    row_scale = torch.logspace(-1, 1, shape[0], device=device)[torch.randperm(shape[0], device=device)]
    col_scale = torch.logspace(-1, 1, shape[1], device=device)[torch.randperm(shape[1], device=device)]
    return torch.randn(shape, device=device) * row_scale[:, None] * col_scale


class DimuonTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.device = FLAGS.device

    def _relative_update_errors(self, shape, *, fp32_matmul_prec, scale_mode="spectral") -> list[float]:
        """Runs five Dimuon steps and returns each update's error relative to the dense FP64 reference."""
        lr, beta, weight_decay, damping, extra_scale_factor = 0.02, 0.9, 0.1, 1e-4, 0.25
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        opt = Dimuon(
            [p],
            lr=lr,
            momentum=beta,
            weight_decay=weight_decay,
            fp32_matmul_prec=fp32_matmul_prec,
            damping=damping,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
        )
        update_scale = lr * extra_scale_factor * get_muon_scale_factor(*shape, mode=scale_mode)

        momentum = torch.zeros(shape, dtype=torch.float64, device=self.device)
        errors = []
        for _ in range(5):
            decayed = p.detach().double() * (1 - lr * weight_decay)
            grad = _scaled_gradient(shape, self.device)
            p.grad = grad
            opt.step()

            new_momentum = momentum.lerp(grad.double(), 1 - beta)
            expected = -update_scale * _reference_direction(momentum, new_momentum, damping)
            momentum = new_momentum
            errors.append(((p.detach().double() - decayed - expected).norm() / expected.norm()).item())
        return errors

    @parameterized.product(
        shape=[(8, 16), (16, 8), (33, 65)],
        scale_mode=["spectral", "shape_scaling"],
    )
    def test_fp32_matches_fp64_reference(self, shape, scale_mode):
        errors = self._relative_update_errors(shape, fp32_matmul_prec="highest", scale_mode=scale_mode)
        self.assertLess(max(errors), 1e-4, msg=f"per-step relative update errors {errors}")

    def test_bf16_close_to_fp32(self):
        """The default fp32_matmul_prec="medium" runs the Polar Express in BF16, within ~10% of FP32."""
        shape = (64, 96)
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        p_fp32 = torch.nn.Parameter(p.detach().clone())
        opt = Dimuon([p], lr=0.02, weight_decay=0.0)
        opt_fp32 = Dimuon([p_fp32], lr=0.02, weight_decay=0.0, fp32_matmul_prec="highest")
        for _ in range(3):
            before = p_fp32.detach().clone()
            grad = _scaled_gradient(shape, self.device)
            p.grad, p_fp32.grad = grad.clone(), grad.clone()
            opt.step()
            opt_fp32.step()
            expected = p_fp32.detach() - before
            error = ((p.detach() - before) - expected).norm() / expected.norm()
            self.assertLess(error.item(), 0.1)
            p.data.copy_(p_fp32.detach())

    @parameterized.product(shape=[(8, 16), (64, 96)])
    def test_zero_gradient_applies_only_weight_decay(self, shape):
        p = torch.nn.Parameter(torch.randn(shape, device=self.device))
        initial = p.detach().clone()
        opt = Dimuon([p], lr=0.1, weight_decay=0.5)
        p.grad = torch.zeros_like(p)
        opt.step()
        torch.testing.assert_close(p.detach(), initial * (1 - 0.1 * 0.5))

    def test_raises_on_non_2d(self):
        for shape in [(8,), (2, 3, 4)]:
            p = torch.nn.Parameter(torch.randn(shape, device=self.device))
            p.grad = torch.randn_like(p)
            opt = Dimuon([p], lr=1e-2)
            with self.assertRaises(ValueError):
                opt.step()

    def test_raises_on_closure(self):
        p = torch.nn.Parameter(torch.randn((8, 16), device=self.device))
        p.grad = torch.randn_like(p)
        opt = Dimuon([p], lr=1e-2)
        with self.assertRaises(ValueError):
            opt.step(lambda: 0.0)

    def test_raises_on_nonpositive_damping(self):
        p = torch.nn.Parameter(torch.randn((8, 16), device=self.device))
        with self.assertRaises(ValueError):
            Dimuon([p], lr=1e-2, damping=0.0)


if __name__ == "__main__":
    absltest.main()
