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
import copy
from functools import partial
from unittest.mock import patch

import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import registry, utils
from emerging_optimizers.orthogonalized_optimizers import Aurora, Muon, aurora_orthogonalize, muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


def _ns(matrix: torch.Tensor) -> torch.Tensor:
    return newton_schulz(matrix, steps=5, coefficient_type="quintic")


def _reference(
    matrix: torch.Tensor, iterations: int = 2, beta: float = 0.5, solver=_ns, eps: float = 1e-14
) -> torch.Tensor:
    """Independent norm/reciprocal expression of the released diagonal refinement."""
    if matrix.size(-2) <= matrix.size(-1):
        return solver(matrix)
    diagonal = torch.linalg.vector_norm(matrix, dim=-1, keepdim=True).clamp_min(eps**0.5).reciprocal()
    for index in range(iterations):
        update = solver(diagonal * matrix)
        if index + 1 < iterations:
            squared_row_norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True).square().clamp_min(eps)
            diagonal = diagonal * (matrix.size(-1) / matrix.size(-2) / squared_row_norm).pow(beta)
    return update


class AuroraTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = FLAGS.device
        self.precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, self.precision)

    def _matrix(self, shape: tuple[int, ...] = (96, 16)) -> torch.Tensor:
        return torch.randn(shape, device=self.device) * torch.logspace(-2, 2, shape[-2], device=self.device).unsqueeze(
            -1
        )

    def _optimizer(self, params, **kwargs) -> Aurora:
        options = dict(fp32_matmul_prec="highest", coefficient_type="quintic", num_ns_steps=5)
        options.update(kwargs)
        return Aurora(params, **options)

    @parameterized.product(iterations=[1, 2, 4], beta=[0.25, 0.5, 1.0])
    def test_matches_diagonal_refinement_reference(self, iterations, beta) -> None:
        matrix = self._matrix()
        before = matrix.clone()
        actual = aurora_orthogonalize(matrix, _ns, iterations=iterations, beta=beta)
        expected = _reference(matrix, iterations, beta)
        # The implementation's sum-of-squares/rsqrt and the independent norm/reciprocal
        # reference differ by reduction rounding, which five quintic NS steps amplify.
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(matrix, before, atol=0, rtol=0)

    def test_reduces_row_imbalance(self) -> None:
        matrix = self._matrix()
        muon_norms = torch.linalg.vector_norm(_ns(matrix), dim=-1)
        aurora_norms = torch.linalg.vector_norm(aurora_orthogonalize(matrix, _ns), dim=-1)
        self.assertLess(aurora_norms.std() / aurora_norms.mean(), 0.3 * muon_norms.std() / muon_norms.mean())

    def test_default_medium_precision_matches_independent_reference(self) -> None:
        matrix = self._matrix()
        optimizer = Aurora([torch.nn.Parameter(matrix.clone())])
        solver = partial(newton_schulz, steps=8, coefficient_type="polar_express")
        self.assertEqual(optimizer.fp32_matmul_prec, "medium")
        with utils.fp32_matmul_precision(optimizer.fp32_matmul_prec):
            actual = optimizer.scaled_orthogonalize_fn(matrix)
            expected = _reference(matrix, solver=solver)
        scale = get_muon_scale_factor(*matrix.shape, mode="spectral") * 0.2
        # NS deliberately rounds GEMMs to BF16 in medium precision. Different FP32
        # row-norm reductions can cross BF16 rounding boundaries before repeated NS
        # solves. Exact solver forwarding and final scaling are checked separately.
        torch.testing.assert_close(actual / scale, expected, atol=2e-2, rtol=2e-2)

    def test_medium_step_preserves_solver_configuration_and_scales_final_output_once(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros((32, 8), device=self.device))
        parameter.grad = self._matrix((32, 8))
        calls = []

        def record_solve(matrix, **kwargs):
            result = newton_schulz(matrix, **kwargs)
            calls.append((kwargs, torch.get_float32_matmul_precision(), matrix.dtype, result.clone()))
            return result

        optimizer = Aurora(
            [parameter],
            lr=1,
            momentum=0,
            weight_decay=0,
            fp32_matmul_prec="medium",
            coefficient_type="simple",
            num_ns_steps=3,
            num_aurora_steps=3,
            scale_mode="unit_rms_norm",
            extra_scale_factor=1.7,
        )
        with patch.object(muon_utils, "newton_schulz", side_effect=record_solve):
            optimizer.step()
        self.assertLen(calls, 3)
        for kwargs, precision, dtype, result in calls:
            self.assertEqual(kwargs, dict(steps=3, coefficient_type="simple", use_syrk=False))
            self.assertEqual(precision, "medium")
            self.assertEqual(dtype, torch.float32)
            self.assertTrue(torch.isfinite(result).all())
        scale = get_muon_scale_factor(32, 8, mode="unit_rms_norm")
        torch.testing.assert_close(parameter, -(calls[-1][-1] * scale * 1.7), atol=0, rtol=0)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    @parameterized.parameters(1, 2, 3)
    def test_each_solve_uses_original_momentum_with_refined_diagonal(self, iterations) -> None:
        matrix = torch.tensor([[3.0, 4.0], [0.0, 2.0], [4.0, 0.0]], device=self.device)
        transform = torch.tensor([[0.5, 0.2], [-0.1, 0.7]], device=self.device)
        calls = []

        def solve(value):
            calls.append(value.clone())
            return value @ transform

        actual = aurora_orthogonalize(matrix, solve, iterations=iterations, beta=0.5)
        diagonal = matrix.norm(dim=-1, keepdim=True).reciprocal()
        self.assertLen(calls, iterations)
        for index in range(iterations):
            expected_input = diagonal * matrix
            torch.testing.assert_close(calls[index], expected_input)
            expected_output = expected_input @ transform
            if index + 1 < iterations:
                diagonal = diagonal * (2 / 3 / expected_output.square().sum(-1, keepdim=True)).sqrt()
        torch.testing.assert_close(actual, expected_output)

    @parameterized.parameters((16, 96), (32, 32))
    def test_square_and_wide_call_original_solver_once(self, rows, columns) -> None:
        matrix = self._matrix((rows, columns))
        calls = []

        def solve(value):
            calls.append(value)
            return _ns(value)

        actual = aurora_orthogonalize(matrix, solve, iterations=4)
        self.assertLen(calls, 1)
        self.assertIs(calls[0], matrix)
        torch.testing.assert_close(actual, _ns(matrix), atol=0, rtol=0)

    @parameterized.product(shape=[(16, 32), (16, 16)], nesterov=[False, True])
    def test_square_and_wide_optimizer_match_muon_exactly(self, shape, nesterov) -> None:
        initial = torch.randn(shape, device=self.device)
        actual, expected = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
        kwargs = dict(fp32_matmul_prec="highest", nesterov=nesterov, weight_decay=0.1)
        aurora = Aurora([actual], num_aurora_steps=4, **kwargs)
        muon = Muon([expected], **kwargs)
        for _ in range(3):
            grad = torch.randn_like(initial)
            actual.grad, expected.grad = grad.clone(), grad.clone()
            aurora.step()
            muon.step()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                aurora.state[actual]["momentum_buffer"], muon.state[expected]["momentum_buffer"], atol=0, rtol=0
            )

    @parameterized.parameters(False, True)
    def test_zero_rows_and_rank_deficient_updates_stay_finite(self, all_zero) -> None:
        matrix = self._matrix()
        matrix[0] = 0
        matrix[:, 1] = matrix[:, 0]
        if all_zero:
            matrix.zero_()
        actual = aurora_orthogonalize(matrix, _ns, iterations=8)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)
        if all_zero:
            self.assertEqual(torch.count_nonzero(actual).item(), 0)

    def test_batched_helper_keeps_experts_independent(self) -> None:
        matrix = self._matrix((3, 48, 8))
        matrix[1].zero_()
        actual = aurora_orthogonalize(matrix, _ns)
        expected = torch.stack([aurora_orthogonalize(expert, _ns) for expert in matrix])
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(torch.count_nonzero(actual[1]).item(), 0)

    def test_custom_eps_controls_initial_and_refinement_floors(self) -> None:
        matrix = torch.tensor([[0.0], [0.01], [1.0]], device=self.device)
        # Both the tiny input row and the solver's small output require the custom floor.
        solver = lambda value: value * 0.01
        actual = aurora_orthogonalize(matrix, solver, eps=0.25)
        torch.testing.assert_close(actual, _reference(matrix, solver=solver, eps=0.25))
        self.assertEqual(actual[0].item(), 0)
        self.assertFalse(torch.allclose(actual, aurora_orthogonalize(matrix, solver)))

        optimizer = self._optimizer([torch.nn.Parameter(matrix.clone())], eps=0.25)
        with patch.object(muon_utils, "newton_schulz", side_effect=lambda value, **kwargs: solver(value)):
            scaled = optimizer.scaled_orthogonalize_fn(matrix)
        scale = get_muon_scale_factor(*matrix.shape, mode=optimizer.scale_mode) * optimizer.extra_scale_factor
        torch.testing.assert_close(scaled, actual * scale)

    @parameterized.parameters(0.0, -1e-14, float("inf"), float("nan"))
    def test_rejects_invalid_eps(self, eps) -> None:
        matrix = self._matrix()
        with self.assertRaisesRegex(ValueError, "eps"):
            aurora_orthogonalize(matrix, _ns, eps=eps)
        with self.assertRaisesRegex(ValueError, "eps"):
            self._optimizer([torch.nn.Parameter(matrix)], eps=eps)

    @parameterized.product(
        scale_mode=["spectral", "shape_scaling", "unit_rms_norm"],
        nesterov=[False, True],
        weight_decay_method=["decoupled", "independent", "l2"],
    )
    def test_full_steps_preserve_momentum_weight_decay_and_scale_once(
        self, scale_mode, nesterov, weight_decay_method
    ) -> None:
        initial = torch.randn((32, 8), device=self.device)
        parameter = torch.nn.Parameter(initial.clone())
        expected, momentum = initial.clone(), torch.zeros_like(initial)
        lr, mu, wd, extra = 0.003, 0.9, 0.1, 1.7
        optimizer = self._optimizer(
            [parameter],
            lr=lr,
            momentum=mu,
            weight_decay=wd,
            nesterov=nesterov,
            weight_decay_method=weight_decay_method,
            scale_mode=scale_mode,
            extra_scale_factor=extra,
        )
        for _ in range(3):
            gradient = torch.randn_like(initial)
            parameter.grad = gradient.clone()
            previous_buffer = optimizer.state.get(parameter, {}).get("momentum_buffer", torch.zeros_like(initial))
            state_gradient = gradient.clone()
            if weight_decay_method == "l2":
                state_gradient.add_(parameter.detach(), alpha=wd)
            expected_buffer = previous_buffer.clone().lerp_(state_gradient, 1 - mu)
            if weight_decay_method == "decoupled":
                expected.mul_(1 - lr * wd)
            elif weight_decay_method == "independent":
                expected.mul_(1 - wd)
            else:
                gradient = gradient.add(expected, alpha=wd)
            momentum.lerp_(gradient, 1 - mu)
            effective = gradient.lerp(momentum, mu) if nesterov else momentum
            update = _reference(effective) * get_muon_scale_factor(*initial.shape, mode=scale_mode) * extra
            expected.add_(update, alpha=-lr)
            self.assertIsNone(optimizer.step())
            torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], expected_buffer, atol=0, rtol=0)
        self.assertEqual(set(optimizer.state[parameter]), {"momentum_buffer"})

    @parameterized.parameters(False, True)
    def test_resume_existing_muon_state_and_round_trip_aurora_state(self, source_is_aurora) -> None:
        parameter = torch.nn.Parameter(torch.randn((32, 8), device=self.device))
        kwargs = dict(lr=0.01, momentum=0.95, nesterov=True, fp32_matmul_prec="highest")
        source = (Aurora if source_is_aurora else Muon)([parameter], **kwargs)
        parameter.grad = torch.randn_like(parameter)
        source.step()
        checkpoint = copy.deepcopy(source.state_dict())
        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = Aurora([resumed_parameter], **kwargs)
        resumed.load_state_dict(checkpoint)
        expected = Aurora([parameter], **kwargs)
        expected.load_state_dict(copy.deepcopy(checkpoint))
        for _ in range(2):
            gradient = torch.randn_like(parameter)
            parameter.grad, resumed_parameter.grad = gradient.clone(), gradient.clone()
            expected.step()
            resumed.step()
            torch.testing.assert_close(resumed_parameter, parameter, atol=0, rtol=0)
            torch.testing.assert_close(
                resumed.state[resumed_parameter]["momentum_buffer"],
                expected.state[parameter]["momentum_buffer"],
                atol=0,
                rtol=0,
            )
        self.assertEqual(set(resumed.state[resumed_parameter]), {"momentum_buffer"})

    def test_registered_public_optimizer_and_configured_factory(self) -> None:
        self.assertIs(registry.get_optimizer_cls("aurora"), Aurora)
        factory = registry.get_configured_optimizer_cls(
            "aurora", num_aurora_steps=3, aurora_beta=0.25, num_ns_steps=4, coefficient_type="simple"
        )
        optimizer = factory([torch.nn.Parameter(torch.randn((16, 8), device=self.device))])
        self.assertIsInstance(optimizer, Aurora)

    def test_aurora_configuration_properties_are_read_only(self) -> None:
        optimizer = self._optimizer(
            [torch.nn.Parameter(self._matrix())], num_aurora_steps=3, aurora_beta=0.25, eps=1e-10
        )
        self.assertEqual(optimizer.num_aurora_steps, 3)
        self.assertEqual(optimizer.aurora_beta, 0.25)
        self.assertEqual(optimizer.eps, 1e-10)
        for name, value in [("num_aurora_steps", 4), ("aurora_beta", 0.5), ("eps", 1e-8)]:
            with self.assertRaises(AttributeError):
                setattr(optimizer, name, value)

    def test_preserves_weight_update_hook_and_precision_context(self) -> None:
        parameter = torch.nn.Parameter(torch.randn((32, 8), device=self.device))
        parameter.grad = torch.randn_like(parameter)
        before = parameter.detach().clone()
        events = []

        class RecordingHook:
            def pre_weight_update_inplace(self, value, update):
                events.append(("pre", value.clone(), update.clone()))
                update.mul_(0.5)
                return value.clone()

            def post_weight_update_inplace(self, value, previous):
                events.append(("post", value.clone(), previous.clone()))

        optimizer = self._optimizer(
            [parameter], lr=0.01, momentum=0, weight_decay=0, weight_update_hook=RecordingHook()
        )
        with utils.fp32_matmul_precision("high"):
            optimizer.step()
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.assertEqual([entry[0] for entry in events], ["pre", "post"])
        torch.testing.assert_close(events[0][1], before, atol=0, rtol=0)
        torch.testing.assert_close(events[1][2], before, atol=0, rtol=0)
        torch.testing.assert_close(parameter, before - 0.005 * events[0][2])

    def test_rejects_closure_and_nonmatrix_parameter(self) -> None:
        parameter = torch.nn.Parameter(torch.randn(8, device=self.device))
        optimizer = self._optimizer([parameter])
        with self.assertRaisesRegex(ValueError, "closure is not supported"):
            optimizer.step(lambda: 1.0)
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(ValueError, "Only 2D"):
            optimizer.step()

    @parameterized.parameters(0, -1, 1.5, True)
    def test_invalid_outer_iteration_count(self, value) -> None:
        matrix = self._matrix()
        with self.assertRaises(ValueError):
            aurora_orthogonalize(matrix, _ns, iterations=value)
        with self.assertRaises(ValueError):
            self._optimizer([torch.nn.Parameter(matrix)], num_aurora_steps=value)

    @parameterized.parameters(0.0, -0.1, 1.1, float("nan"), float("inf"))
    def test_invalid_beta(self, value) -> None:
        matrix = self._matrix()
        with self.assertRaises(ValueError):
            aurora_orthogonalize(matrix, _ns, beta=value)
        with self.assertRaises(ValueError):
            self._optimizer([torch.nn.Parameter(matrix)], aurora_beta=value)

    @parameterized.parameters(torch.bfloat16, torch.float16, torch.float64, torch.int64)
    def test_requires_fp32_momentum_like_muon(self, dtype) -> None:
        with self.assertRaises(TypeError):
            aurora_orthogonalize(self._matrix().to(dtype), _ns)

    @parameterized.parameters(dict(shape=(8,)), dict(shape=(2, 2, 4, 8)))
    def test_helper_rejects_unsupported_dimensions(self, shape) -> None:
        with self.assertRaises(TypeError):
            aurora_orthogonalize(torch.zeros(shape, device=self.device), _ns)

    @parameterized.parameters(dict(shape=(0, 8)), dict(shape=(8, 0)), dict(shape=(2, 0, 8)), dict(shape=(2, 8, 0)))
    def test_helper_rejects_empty_matrix_axes(self, shape) -> None:
        with self.assertRaises(ValueError):
            aurora_orthogonalize(torch.zeros(shape, device=self.device), _ns)


if __name__ == "__main__":
    absltest.main()
