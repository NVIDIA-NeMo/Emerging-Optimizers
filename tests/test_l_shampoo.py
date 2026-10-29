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
from emerging_optimizers.shampoo.l_shampoo import LKlShampoo, LKlShampooPreconditioner, _pack_triu, _unpack_triu


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class PackTriuTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    @parameterized.parameters(1, 3, 7, 16, 33)
    def test_pack_unpack_round_trip(self, m: int) -> None:
        x = torch.randn(m, m, device=self.device)
        sym = (x + x.mT) * 0.5

        packed = _pack_triu(sym)

        self.assertEqual(packed.shape, (m * (m + 1) // 2,))
        assert_equal(_unpack_triu(packed), sym)

    @parameterized.parameters(1, 3, 7, 16, 33)
    def test_pack_matches_triu_indices_reference(self, m: int) -> None:
        x = torch.randn(m, m, device=self.device)
        sym = (x + x.mT) * 0.5

        tri = torch.triu_indices(m, m)
        assert_equal(_pack_triu(sym), sym[tri[0], tri[1]])


class LKlShampooPreconditionerTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    @staticmethod
    def _init_state_with_positive_definite_factors(m: int, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        state = LKlShampooPreconditioner.init_state((m, n), device)
        for key, dim in (("L", m), ("R", n)):
            x = torch.randint(-2, 3, (dim, dim), dtype=torch.float32, device=device)
            state[key] = _pack_triu(x @ x.mT + 1.0 / 128 * torch.eye(dim, device=device))
        return state

    @parameterized.parameters((8, 16), (16, 8), (13, 15))
    def test_init_state_layout(self, m: int, n: int) -> None:
        state = LKlShampooPreconditioner.init_state((m, n), self.device)

        expected_shapes = {"L": (m * (m + 1) // 2,), "R": (n * (n + 1) // 2,)}
        self.assertCountEqual(state, expected_shapes)
        for key, shape in expected_shapes.items():
            assert_equal(state[key], torch.zeros(shape, device=self.device))

    def test_init_state_rejects_non_2d(self) -> None:
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            LKlShampooPreconditioner.init_state((2, 3, 4), self.device)

    @parameterized.parameters(1, 3, 8)
    def test_unsupported_p_root_inv_raises(self, p_root_inv: int) -> None:
        state = LKlShampooPreconditioner.init_state((4, 4), self.device)

        with self.assertRaisesRegex(ValueError, "not supported by CANS"):
            LKlShampooPreconditioner(state, p_root_inv=p_root_inv, eps=1e-8)

    def test_init_unpacks_state_dense(self) -> None:
        m, n = 6, 4
        state = LKlShampooPreconditioner.init_state((m, n), self.device)

        preconditioner = LKlShampooPreconditioner(state, p_root_inv=2, eps=1e-8)

        self.assertEqual(state["L"].shape, (m, m))
        self.assertEqual(state["R"].shape, (n, n))
        self.assertIs(state["L"], preconditioner.kronecker_factor_pair.L)
        self.assertIs(state["R"], preconditioner.kronecker_factor_pair.R)

    @parameterized.product(shape=[(8, 8), (16, 8)], p_root_inv=[2, 4])
    def test_precondition_close_to_eigh_reference(self, shape: tuple[int, int], p_root_inv: int) -> None:
        m, n = shape
        state = self._init_state_with_positive_definite_factors(m, n, self.device)
        preconditioner = LKlShampooPreconditioner(state, p_root_inv=p_root_inv, eps=0)
        x = torch.randint(-3, 4, (m, n), dtype=torch.float32, device=self.device)

        with utils.fp32_matmul_precision("highest"):
            preconditioned = preconditioner.precondition(x)

        references = {}
        for key in ("L", "R"):
            w, v = torch.linalg.eigh(state[key])
            references[key] = (v * w.pow(-1.0 / p_root_inv)) @ v.mT
        expected = references["L"] @ x @ references["R"]

        torch.testing.assert_close(
            preconditioned,
            expected,
            atol=0,
            rtol=0.02,
            msg=lambda default: f"p_root_inv={p_root_inv}\n\n{default}",
        )

    @parameterized.parameters((8, 16), (16, 8))
    def test_init_step_identity_shape_scale(self, m: int, n: int) -> None:
        preconditioner = LKlShampooPreconditioner(
            LKlShampooPreconditioner.init_state((m, n), self.device), p_root_inv=2, eps=0
        )
        grad = torch.randn(m, n, device=self.device)

        preconditioner.init_step(grad, shampoo_beta=0)

        with utils.fp32_matmul_precision("highest"):
            expected_L = grad @ grad.mT / n
            expected_R = grad.mT @ grad / m
        torch.testing.assert_close(preconditioner.kronecker_factor_pair.L, expected_L, atol=0, rtol=1e-6)
        torch.testing.assert_close(preconditioner.kronecker_factor_pair.R, expected_R, atol=0, rtol=1e-6)

    @parameterized.product(shape=[(8, 16), (16, 8)], shampoo_beta=[0.5, 0.95], p_root_inv=[2, 4])
    def test_update_kronecker_factors_close_to_eigh_reference(
        self, shape: tuple[int, int], shampoo_beta: float, p_root_inv: int
    ) -> None:
        m, n = shape
        state = self._init_state_with_positive_definite_factors(m, n, self.device)
        preconditioner = LKlShampooPreconditioner(state, p_root_inv=p_root_inv, eps=1e-8)
        reference_L = state["L"].clone()
        reference_R = state["R"].clone()
        grad = torch.randn(m, n, device=self.device)

        preconditioner.update_kronecker_factors(grad, shampoo_beta)

        with utils.fp32_matmul_precision("highest"):
            inverses = {}
            for key, factor in (("L", reference_L), ("R", reference_R)):
                w, v = torch.linalg.eigh(factor)
                inverses[key] = (v * w.reciprocal()) @ v.mT
            expected_L = reference_L.lerp(grad @ inverses["R"] @ grad.mT / n, 1 - shampoo_beta)
            expected_R = reference_R.lerp(grad.mT @ inverses["L"] @ grad / m, 1 - shampoo_beta)

        torch.testing.assert_close(preconditioner.kronecker_factor_pair.L, expected_L, atol=0, rtol=0.01)
        torch.testing.assert_close(preconditioner.kronecker_factor_pair.R, expected_R, atol=0, rtol=0.01)

    def test_rebind_state_packs_upper_triangle(self) -> None:
        m, n = 6, 4
        state = LKlShampooPreconditioner.init_state((m, n), self.device)
        preconditioner = LKlShampooPreconditioner(state, p_root_inv=2, eps=1e-8)
        preconditioner.init_step(torch.randn(m, n, device=self.device), shampoo_beta=0)

        preconditioner.rebind_state(state)

        self.assertEqual(state["L"].shape, (m * (m + 1) // 2,))
        self.assertEqual(state["R"].shape, (n * (n + 1) // 2,))
        assert_equal(state["L"], _pack_triu(preconditioner.kronecker_factor_pair.L))
        assert_equal(state["R"], _pack_triu(preconditioner.kronecker_factor_pair.R))

    def test_rebind_state_missing_key_raises(self) -> None:
        state = LKlShampooPreconditioner.init_state((4, 4), self.device)
        preconditioner = LKlShampooPreconditioner(state, p_root_inv=2, eps=1e-8)
        del state["R"]

        with self.assertRaisesRegex(KeyError, "missing keys"):
            preconditioner.rebind_state(state)


class LKlShampooTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    @parameterized.product(shape=[(8, 5), (5, 8), (16, 16)], p_root_inv=[2, 4])
    def test_3steps_smoke(self, shape: tuple[int, int], p_root_inv: int) -> None:
        m, n = shape
        p = torch.nn.Parameter(torch.randn(m, n, device=self.device))
        initial = p.detach().clone()
        optimizer = LKlShampoo([p], lr=1e-2, p_root_inv=p_root_inv)

        for _ in range(3):
            p.grad = torch.randn_like(p)
            optimizer.step()

        self.assertTrue(torch.isfinite(p).all())
        self.assertFalse(torch.equal(p.detach(), initial))
        self.assertCountEqual(
            optimizer.state[p],
            {"step", "exp_avg", "L", "R"},
        )
        self.assertEqual(optimizer.state[p]["L"].shape, (m * (m + 1) // 2,))
        self.assertEqual(optimizer.state[p]["R"].shape, (n * (n + 1) // 2,))

    @parameterized.parameters(1, 3, 8)
    def test_unsupported_p_root_inv_raises(self, p_root_inv: int) -> None:
        with self.assertRaisesRegex(ValueError, "not supported by CANS"):
            LKlShampoo([torch.nn.Parameter(torch.randn(4, 4, device=self.device))], lr=1e-3, p_root_inv=p_root_inv)


if __name__ == "__main__":
    absltest.main()
