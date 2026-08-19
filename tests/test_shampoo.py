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
from typing import override

import torch
from _comparison import assert_close_to_identity, assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.legacy_soap import soap
from emerging_optimizers.shampoo.shampoo import Shampoo, ShampooBase, ShampooPreconditioner


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


def _root_inverse_reference(a: torch.Tensor, p_root_inv: float, eps: float) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(a)
    return (u * s.clamp_min(eps) ** (-1.0 / p_root_inv)) @ vh


def gen_signed_permutation(m: int):
    signs = torch.randint(0, 2, (m,), dtype=torch.float32) * 2 - 1
    Q = torch.zeros(m, m)
    Q[torch.randperm(m), torch.arange(m)] = signs

    return Q


class ShampooPreconditionerTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    @parameterized.parameters((8, 16), (16, 8), (13, 15))
    def test_init_state_layout(self, m: int, n: int) -> None:
        state = ShampooPreconditioner.init_state((m, n), self.device)

        expected_shapes = {"L": (m, m), "R": (n, n)}
        self.assertCountEqual(state, expected_shapes)
        for key, shape in expected_shapes.items():
            self.assertEqual(state[key].shape, shape, msg=key)
            self.assertEqual(state[key].dtype, torch.float32, msg=key)
            self.assertEqual(state[key].device.type, self.device.type, msg=key)
            assert_equal(state[key], torch.zeros(shape, device=self.device))

    def test_init_state_rejects_non_2d(self) -> None:
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            ShampooPreconditioner.init_state((2, 3, 4), self.device)

    @parameterized.parameters((8, 16), (16, 8), (13, 15))
    def test_rebind_state_binds_current_tensors(self, m: int, n: int) -> None:
        state = ShampooPreconditioner.init_state((m, n), self.device)
        preconditioner = ShampooPreconditioner(state, p_root_inv=4, eps=1e-8)
        preconditioner.step(torch.randn(m, n, device=self.device), 0.95)
        preconditioner.rebind_state(state)

        self.assertIs(state["L"], preconditioner.kronecker_factor_pair.L)
        self.assertIs(state["R"], preconditioner.kronecker_factor_pair.R)

    def test_rebind_state_missing_key_raises(self) -> None:
        state = ShampooPreconditioner.init_state((4, 4), self.device)
        preconditioner = ShampooPreconditioner(state, p_root_inv=4, eps=1e-8)
        del state["R"]

        with self.assertRaisesRegex(KeyError, "missing keys"):
            preconditioner.rebind_state(state)

    @parameterized.parameters((8, 16), (16, 8), (13, 15))
    def test_init_step_seeds_eps_identity(self, m: int, n: int) -> None:
        eps = 0.5
        preconditioner = ShampooPreconditioner(
            ShampooPreconditioner.init_state((m, n), self.device), p_root_inv=4, eps=eps
        )
        grad = torch.zeros(m, n, device=self.device)

        preconditioner.init_step(grad, shampoo_beta=1)

        assert_equal(preconditioner.kronecker_factor_pair.L, torch.eye(m, device=self.device) * eps)
        assert_equal(preconditioner.kronecker_factor_pair.R, torch.eye(n, device=self.device) * eps)

    @parameterized.product(shape=[(8, 16), (16, 8), (13, 15)], shampoo_beta=[0.5, 0.95])
    def test_update_kronecker_factors_matches_legacy(self, shape: tuple[int, int], shampoo_beta: float) -> None:
        m, n = shape
        preconditioner = ShampooPreconditioner(
            ShampooPreconditioner.init_state((m, n), self.device), p_root_inv=4, eps=1e-8
        )
        preconditioner.init_step(torch.randn(m, n, device=self.device), shampoo_beta)

        reference_factors = [
            preconditioner.kronecker_factor_pair.L.clone(),
            preconditioner.kronecker_factor_pair.R.clone(),
        ]
        grad = torch.randn(m, n, device=self.device)
        soap.update_kronecker_factors(reference_factors, grad, shampoo_beta)
        preconditioner.update_kronecker_factors(grad, shampoo_beta)

        assert_equal(preconditioner.kronecker_factor_pair.L, reference_factors[0])
        assert_equal(preconditioner.kronecker_factor_pair.R, reference_factors[1])

    def test_step_equals_update_kronecker_factors(self) -> None:
        eps = 1e-8
        m, n, shampoo_beta = 6, 4, 0.9
        grad = torch.randn(m, n, device=self.device)

        stepped = ShampooPreconditioner(ShampooPreconditioner.init_state((m, n), self.device), p_root_inv=4, eps=eps)
        updated = ShampooPreconditioner(ShampooPreconditioner.init_state((m, n), self.device), p_root_inv=4, eps=eps)
        stepped.step(grad, shampoo_beta)
        updated.update_kronecker_factors(grad, shampoo_beta)

        assert_equal(stepped.kronecker_factor_pair.L, updated.kronecker_factor_pair.L)
        assert_equal(stepped.kronecker_factor_pair.R, updated.kronecker_factor_pair.R)

    @parameterized.product(m=[4, 9, 16], p_root_inv=[2, 4])
    def test_get_root_inverse_close_to_svd_reference(self, m: int, p_root_inv: int) -> None:
        x = 2 ** torch.randint(-3, 2, (m, m), device=self.device, dtype=torch.float)
        factor = x @ x.T + 0.125 * torch.eye(m, device=self.device)
        preconditioner = ShampooPreconditioner(
            ShampooPreconditioner.init_state((m, m), self.device), p_root_inv=p_root_inv, eps=0
        )

        with utils.fp32_matmul_precision("highest"):
            root_inverse = preconditioner._get_root_inverse(factor)

        torch.testing.assert_close(
            root_inverse,
            _root_inverse_reference(factor, p_root_inv, 0),
            atol=1e-3,
            rtol=1e-3,
        )

    @parameterized.parameters(2, 4)
    def test_get_root_inverse_tikhonov_eps_effect(self, p_root_inv: int) -> None:
        eps = 2.0**-4
        preconditioner = ShampooPreconditioner(
            {"L": torch.eye(7, device=self.device), "R": torch.eye(7, device=self.device)},
            p_root_inv=p_root_inv,
            eps=eps,
        )

        root_inverse = preconditioner._get_root_inverse(preconditioner.kronecker_factor_pair.L)
        scale = 1 / (1 + eps ** (2 / p_root_inv))

        assert_close_to_identity(root_inverse / scale)

    @parameterized.parameters((6, 4), (4, 6), (5, 5))
    def test_precondition_identity_factors_is_noop(self, m: int, n: int) -> None:
        preconditioner = ShampooPreconditioner(
            {"L": torch.eye(m, device=self.device), "R": torch.eye(n, device=self.device)}, p_root_inv=4, eps=0
        )
        x = torch.randn(m, n, device=self.device)

        with utils.fp32_matmul_precision("highest"):
            preconditioned = preconditioner.precondition(x)

        assert_equal(preconditioned, x)

    @parameterized.parameters(6, 16, 33)
    def test_precondition_matches_inverse_of_known_spectrum(self, m: int) -> None:
        """Test designed to have exact match.

        Kronecker factors are created by integer eigen values and signed permutation eigven vectors.
        """
        p = gen_signed_permutation(m).to(self.device)
        eigvals = 2 ** torch.randint(-5, 0, (m,), dtype=torch.float32, device=self.device)
        A = p * eigvals @ p.mT

        init_kronecker_factors = {
            "L": A,
            "R": A.clone(),
        }
        inv_root_kwargs = {
            "p_root_inv": 2,
            "eps": 0,
        }
        preconditioner = ShampooPreconditioner(init_kronecker_factors, **inv_root_kwargs)

        scale = 7
        x = torch.eye(m, device=self.device, dtype=torch.float32) * scale
        with utils.fp32_matmul_precision("highest"):
            preconditioned = preconditioner.precondition(x).round()

        expected = (p * (eigvals**-1) @ p.mT * scale).round()

        assert_equal(preconditioned, expected)

    @parameterized.parameters((8, 3), (3, 8))
    def test_precondition_4steps_smoke(self, m: int, n: int) -> None:
        shampoo_beta = 0.95
        preconditioner = ShampooPreconditioner(
            ShampooPreconditioner.init_state((m, n), self.device), p_root_inv=4, eps=1e-8
        )
        preconditioner.init_step(torch.randn(m, n, device=self.device), shampoo_beta)
        for _ in range(4):
            preconditioner.step(torch.randn(m, n, device=self.device), shampoo_beta)

        preconditioned = preconditioner.precondition(torch.randn(m, n, device=self.device))

        self.assertEqual(preconditioned.shape, (m, n))


class _BypassPreconditioner:
    def __init__(self, state: dict, p_root_inv: float, eps: float) -> None:
        self.p_root_inv = p_root_inv
        self.eps = eps

        # Store shampoo beta for verifing its value recieved in step.
        self.shampoo_beta = None

    @staticmethod
    def init_state(shape: tuple[int, ...], device: torch.device) -> dict[str, torch.Tensor]:
        return {}

    def rebind_state(self, state: dict) -> None:
        state["shampoo_beta"] = self.shampoo_beta

    def init_step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        self.shampoo_beta = shampoo_beta

    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        pass

    def step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        self.shampoo_beta = shampoo_beta

    def precondition(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _SgdShampoo(ShampooBase):
    """A fake shampoo that bypass preconditioning for testing base class."""

    PreconditionerCls = _BypassPreconditioner

    @override
    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        *,
        momentum: float,
    ) -> torch.Tensor:
        exp_avg.mul_(momentum).add_(grad)
        return exp_avg


class ShampooBaseTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    def test_step_smoke(self) -> None:
        p = torch.nn.Parameter(torch.randn(4, 4, device=self.device))
        optimizer = _SgdShampoo([p], lr=1e-3)

        p.grad = torch.randn_like(p)

        optimizer.step()

    @parameterized.product(lr=[0.25, 0.125], momentum=[0.0, 75], weight_decay=[0.125, 0.05])
    def test_step_3steps_close_to_sgd(self, lr: float, momentum: float, weight_decay: float) -> None:
        p = torch.nn.Parameter(torch.randn(4, 3, device=self.device))
        expected = p.detach().clone()
        sgd_buffer = torch.zeros_like(expected)
        optimizer = _SgdShampoo([p], lr=lr, momentum=momentum, weight_decay=weight_decay)

        for _ in range(3):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            optimizer.step()
            sgd_buffer = momentum * sgd_buffer + grad
            expected.mul_(1 - lr * weight_decay).add_(sgd_buffer, alpha=-lr)

            torch.testing.assert_close(
                p.detach(),
                expected,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_rejects_non_2d(self) -> None:
        p = torch.nn.Parameter(torch.randn(2, 3, 4, device=self.device))
        p.grad = torch.randn_like(p)
        optimizer = _SgdShampoo([p], lr=1e-3)

        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            optimizer.step()

    @parameterized.parameters(
        {"kwargs": {"lr": -1.0}, "message": "Invalid learning rate"},
        {"kwargs": {"lr": 1e-3, "p_root_inv": -2}, "message": "p_root_inv must be positive integer"},
    )
    def test_invalid_arguments_raise(self, kwargs: dict, message: str) -> None:
        with self.assertRaisesRegex(ValueError, message):
            _SgdShampoo([torch.nn.Parameter(torch.randn(4, 4, device=self.device))], **kwargs)

    @parameterized.parameters(2, 4.0)
    def test_integral_p_root_inv_accepted(self, p_root_inv: float) -> None:
        _SgdShampoo([torch.nn.Parameter(torch.randn(4, 4, device=self.device))], lr=1e-3, p_root_inv=p_root_inv)


class ShampooTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(FLAGS.device)

    @parameterized.parameters((8, 5), (5, 8), (16, 16))
    def test_3steps_smoke(self, m: int, n: int) -> None:
        p = torch.nn.Parameter(torch.randn(m, n, device=self.device))
        initial = p.detach().clone()

        optimizer = Shampoo([p], lr=1e-2)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            optimizer.step()

        self.assertTrue(torch.isfinite(p).all())
        self.assertFalse(torch.equal(p.detach(), initial))
        state = optimizer.state[p]
        self.assertEqual(state["step"], 3)
        self.assertCountEqual(state, {"step", "exp_avg", "L", "R"})

    def test_shampoo_beta_bias_corrected_over_5steps(self) -> None:
        shampoo_beta = 0.75
        p = torch.nn.Parameter(torch.randn(4, 3, device=self.device))
        optimizer = _SgdShampoo([p], lr=1e-3, shampoo_beta=shampoo_beta)

        for curr_iter_1_based in range(1, 6):
            p.grad = torch.randn_like(p)
            optimizer.step()

            geometric_weight_sum = sum(shampoo_beta**i for i in range(curr_iter_1_based))
            self.assertEqual(
                optimizer.state[p]["shampoo_beta"],
                1 - 1 / geometric_weight_sum,
                msg=f"bias corrected shampoo_beta mismatch at step {curr_iter_1_based}",
            )

    @parameterized.parameters(True, False)
    def test_init_group_skip_non_grad_params(self, skip_non_grad_params: bool) -> None:
        with_grad = torch.nn.Parameter(torch.randn(4, 3, device=self.device))
        without_grad = torch.nn.Parameter(torch.randn(5, 2, device=self.device))
        with_grad.grad = torch.randn_like(with_grad)
        optimizer = Shampoo([with_grad, without_grad], lr=1e-3)

        optimizer._init_group(optimizer.param_groups[0], skip_non_grad_params=skip_non_grad_params)

        self.assertCountEqual(optimizer.state[with_grad], {"step", "exp_avg", "L", "R"})
        if skip_non_grad_params:
            self.assertEmpty(optimizer.state[without_grad])
        else:
            self.assertCountEqual(optimizer.state[without_grad], {"step", "exp_avg", "L", "R"})
            self.assertEqual(optimizer.state[without_grad]["L"].shape, (5, 5))
            self.assertEqual(optimizer.state[without_grad]["R"].shape, (2, 2))

    def test_zero_grad_applies_only_weight_decay(self) -> None:
        lr, weight_decay = 0.1, 0.05
        p = torch.nn.Parameter(torch.randn(5, 5, device=self.device))
        initial = p.detach().clone()
        optimizer = Shampoo([p], lr=lr, weight_decay=weight_decay)

        p.grad = torch.zeros_like(p)
        optimizer.step()

        torch.testing.assert_close(
            p.detach(),
            initial * (1 - lr * weight_decay),
            atol=1e-6,
            rtol=1e-6,
            msg=lambda default: f"A zero gradient should leave only decoupled weight decay\n\n{default}",
        )

    @parameterized.parameters(0.0, 0.5, 0.75)
    def test_scalar_update_close_to_sgd(self, momentum: float) -> None:
        p = torch.nn.Parameter(torch.randn(6, 6, device=self.device))
        optimizer = Shampoo([p], lr=0.125, momentum=momentum)
        exp_avg = torch.zeros_like(p)
        sgd_buffer = torch.zeros_like(p)

        for _ in range(3):
            grad = torch.randn_like(p)
            scalar_update = optimizer._scalar_update(grad, exp_avg, momentum=momentum)
            sgd_buffer.mul_(momentum).add_(grad, alpha=1 - momentum)

            torch.testing.assert_close(
                scalar_update,
                sgd_buffer,
                atol=1e-5,
                rtol=1e-5,
            )


if __name__ == "__main__":
    absltest.main()
