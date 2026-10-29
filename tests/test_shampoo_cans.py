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
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.shampoo import matrix_root_inverse_utils
from emerging_optimizers.shampoo.shampoo_cans import KlShampooCans, KlShampooCansPreconditioner


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class KlShampooCansPreconditionerTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    @parameterized.parameters((8, 16), (16, 8), (13, 15))
    def test_init_state_layout(self, m: int, n: int) -> None:
        state = KlShampooCansPreconditioner.init_state((m, n), self.device)

        expected_shapes = {"L": (m, m), "R": (n, n)}
        self.assertCountEqual(state, expected_shapes)
        for key, shape in expected_shapes.items():
            self.assertEqual(state[key].shape, shape, msg=key)
            self.assertEqual(state[key].dtype, torch.float32, msg=key)
            self.assertEqual(state[key].device.type, self.device.type, msg=key)
            assert_equal(state[key], torch.zeros(shape, device=self.device))

    @parameterized.parameters(1, 3, 8)
    def test_unsupported_p_root_inv_raises(self, p_root_inv: int) -> None:
        state = KlShampooCansPreconditioner.init_state((4, 4), self.device)

        with self.assertRaisesRegex(ValueError, "not supported by CANS"):
            KlShampooCansPreconditioner(state, p_root_inv=p_root_inv, eps=1e-8)

    @parameterized.product(shape=[(8, 16), (16, 8)], p_root_inv=[2, 4])
    def test_precondition_close_to_eigh_reference(self, shape: tuple[int, int], p_root_inv: int) -> None:
        m, n = shape
        eps = 1e-8
        state = KlShampooCansPreconditioner.init_state((m, n), self.device)
        for key, dim in (("L", m), ("R", n)):
            seed = torch.randn(dim, dim, device=self.device)
            state[key] = seed @ seed.mT + 0.1 * torch.eye(dim, device=self.device)
        preconditioner = KlShampooCansPreconditioner(state, p_root_inv=p_root_inv, eps=eps)
        x = torch.randn(m, n, device=self.device)

        with utils.fp32_matmul_precision("highest"):
            preconditioned = preconditioner.precondition(x)

        references = {}
        for key in ("L", "R"):
            w, v = torch.linalg.eigh(state[key].double())
            references[key] = (v * w.pow(-1.0 / p_root_inv)) @ v.mT
        expected = (references["L"] @ x.double() @ references["R"]).float()

        torch.testing.assert_close(
            preconditioned,
            expected,
            atol=2e-3,
            rtol=2e-3,
            msg=lambda default: f"p_root_inv={p_root_inv}\n\n{default}",
        )

    @parameterized.parameters((8, 16), (16, 8))
    def test_init_step_identity_correction(self, m: int, n: int) -> None:
        preconditioner = KlShampooCansPreconditioner(
            KlShampooCansPreconditioner.init_state((m, n), self.device), p_root_inv=2, eps=1e-8
        )
        grad = torch.randn(m, n, device=self.device)

        preconditioner.init_step(grad, shampoo_beta=0)

        with utils.fp32_matmul_precision("highest"):
            expected_L = grad @ grad.mT / n
            expected_R = grad.mT @ grad / m
        torch.testing.assert_close(preconditioner.kronecker_factor_pair.L, expected_L, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(preconditioner.kronecker_factor_pair.R, expected_R, atol=1e-6, rtol=1e-6)

    @parameterized.product(shape=[(8, 16), (16, 8)], shampoo_beta=[0.5, 0.95])
    def test_update_kronecker_factors_matches_manual(self, shape: tuple[int, int], shampoo_beta: float) -> None:
        m, n = shape
        state = KlShampooCansPreconditioner.init_state((m, n), self.device)
        for key, dim in (("L", m), ("R", n)):
            seed = torch.randn(dim, dim, device=self.device)
            state[key] = seed @ seed.mT + 0.1 * torch.eye(dim, device=self.device)
        preconditioner = KlShampooCansPreconditioner(state, p_root_inv=2, eps=1e-8)
        reference_L = state["L"].clone()
        reference_R = state["R"].clone()
        grad = torch.randn(m, n, device=self.device)

        preconditioner.update_kronecker_factors(grad, shampoo_beta)

        with utils.fp32_matmul_precision("highest"):
            root_inv_L = matrix_root_inverse_utils.mat_root_inv_via_scaled_cans(reference_L)
            root_inv_R = matrix_root_inverse_utils.mat_root_inv_via_scaled_cans(reference_R)
            expected_L = reference_L.lerp(grad @ (root_inv_R @ root_inv_R) @ grad.mT / n, 1 - shampoo_beta)
            expected_R = reference_R.lerp(grad.mT @ (root_inv_L @ root_inv_L) @ grad / m, 1 - shampoo_beta)

        torch.testing.assert_close(preconditioner.kronecker_factor_pair.L, expected_L, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(preconditioner.kronecker_factor_pair.R, expected_R, atol=1e-5, rtol=1e-5)

    def test_rebind_state_missing_key_raises(self) -> None:
        state = KlShampooCansPreconditioner.init_state((4, 4), self.device)
        preconditioner = KlShampooCansPreconditioner(state, p_root_inv=2, eps=1e-8)
        del state["R"]

        with self.assertRaisesRegex(KeyError, "missing keys"):
            preconditioner.rebind_state(state)


class KlShampooCansTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    @parameterized.product(shape=[(8, 5), (5, 8), (16, 16)], p_root_inv=[2, 4])
    def test_3steps_smoke(self, shape: tuple[int, int], p_root_inv: int) -> None:
        m, n = shape
        p = torch.nn.Parameter(torch.randn(m, n, device=self.device))
        initial = p.detach().clone()
        optimizer = KlShampooCans([p], lr=1e-2, p_root_inv=p_root_inv)

        for _ in range(3):
            p.grad = torch.randn_like(p)
            optimizer.step()

        self.assertTrue(torch.isfinite(p).all())
        self.assertFalse(torch.equal(p.detach(), initial))
        self.assertCountEqual(
            optimizer.state[p],
            {"step", "exp_avg", "L", "R"},
        )

    @parameterized.parameters(1, 3, 8)
    def test_unsupported_p_root_inv_raises(self, p_root_inv: int) -> None:
        with self.assertRaisesRegex(ValueError, "not supported by CANS"):
            KlShampooCans([torch.nn.Parameter(torch.randn(4, 4, device=self.device))], lr=1e-3, p_root_inv=p_root_inv)


if __name__ == "__main__":
    absltest.main()
