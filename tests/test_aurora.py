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
import math

import torch
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers import Aurora, Muon
from emerging_optimizers.orthogonalized_optimizers.aurora import _precondition_rows, _refine_rows
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


def _gen_row_scaled_matrix(shape: tuple[int, int], device: str) -> torch.Tensor:
    """Integer entries with exactly representable, power-of-two row scales."""
    entries = torch.randint(-8, 9, shape, device=device).float()
    exponents = torch.arange(shape[0], device=device) % 13 - 6
    return entries * torch.exp2(exponents.float()).unsqueeze(-1)


def _reference_aurora_update(
    matrix: torch.Tensor, *, iterations: int = 2, beta: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    """Independent FP32 reference without the implementation's initial rescaling."""
    matrix = matrix.float()
    row_norm = torch.linalg.vector_norm(matrix, dim=-1, keepdim=True)
    floor = eps * row_norm.mean(dim=-2, keepdim=True)
    denominator = torch.maximum(row_norm, floor)
    diagonal = torch.where(denominator > 0, denominator, 1.0).reciprocal()
    diagonal = diagonal.masked_fill(row_norm == 0, 0)
    for index in range(iterations):
        update = newton_schulz((diagonal * matrix).float(), steps=8, coefficient_type="polar_express")
        if index + 1 < iterations:
            row_norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
            floor = eps * row_norm.mean(dim=-2, keepdim=True)
            denominator = torch.maximum(row_norm, floor)
            denominator = torch.where((denominator > 0) & (diagonal != 0), denominator, 1.0)
            diagonal = diagonal * (matrix.size(-1) / matrix.size(-2) / denominator.square()).pow(beta)
    return update * math.sqrt(matrix.size(-2)) * 0.2


class AuroraTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = FLAGS.device
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, precision)

    @parameterized.named_parameters(
        ("default", 2, 0.5, 1e-6),
        ("more_refinements", 4, 0.25, 1e-6),
        ("custom_floor", 2, 0.5, 0.25),
    )
    def test_update_is_close_to_diagonal_refinement_reference(self, iterations, beta, eps) -> None:
        matrix = _gen_row_scaled_matrix((96, 16), self.device)
        before = matrix.clone()
        optimizer = Aurora(
            [torch.nn.Parameter(matrix.clone())],
            fp32_matmul_prec="highest",
            num_aurora_steps=iterations,
            aurora_beta=beta,
            eps=eps,
        )
        actual = optimizer.scaled_orthogonalize_fn(matrix)
        expected = _reference_aurora_update(matrix, iterations=iterations, beta=beta, eps=eps)
        # Independent FP32 norm reductions round differently before the NS solves.
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
        assert_equal(matrix, before)

    def test_default_medium_precision_update_is_close_to_reference(self) -> None:
        matrix = _gen_row_scaled_matrix((96, 16), self.device)
        optimizer = Aurora([torch.nn.Parameter(matrix.clone())])
        with utils.fp32_matmul_precision(optimizer.fp32_matmul_prec):
            actual = optimizer.scaled_orthogonalize_fn(matrix)
            expected = _reference_aurora_update(matrix)
        # BF16 GEMMs amplify differences between the independent FP32 reductions.
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-2)

    @parameterized.parameters((16, 32), (16, 16))
    def test_square_and_wide_updates_match_muon_exactly(self, rows, cols) -> None:
        matrix = _gen_row_scaled_matrix((rows, cols), self.device)
        aurora = Aurora([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest", num_aurora_steps=4)
        muon = Muon([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest")
        assert_equal(aurora.scaled_orthogonalize_fn(matrix), muon.scaled_orthogonalize_fn(matrix))

    @parameterized.parameters(1e-6, torch.finfo(torch.float32).tiny)
    def test_zero_rows_stay_zero_with_repeated_refinement(self, eps) -> None:
        matrix = _gen_row_scaled_matrix((96, 16), self.device)
        matrix[::2] = 0
        matrix[:, 1] = matrix[:, 0]  # Also exercise a rank-deficient matrix.
        optimizer = Aurora(
            [torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest", num_aurora_steps=4, eps=eps
        )
        actual = optimizer.scaled_orthogonalize_fn(matrix)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(torch.count_nonzero(actual[::2]).item(), 0)
        actual = optimizer.scaled_orthogonalize_fn(torch.zeros_like(matrix))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(torch.count_nonzero(actual).item(), 0)

    def test_tiny_nonzero_rows_are_not_masked_as_zero(self) -> None:
        matrix = torch.full((8, 4), 2.0**-80, device=self.device)
        self.assertEqual(torch.count_nonzero(matrix.square()).item(), 0)
        optimizer = Aurora([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest")
        actual = optimizer.scaled_orthogonalize_fn(matrix)
        # Use unit scale for the FP32 reference because the original norm underflows.
        expected = _reference_aurora_update(torch.ones_like(matrix))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(torch.count_nonzero(actual).item(), 0)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=2e-5)

    def test_relative_floor_uses_mean_row_l2_norm_at_both_stages(self) -> None:
        matrix = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.0, 1.0], [4.0, 0.0]], device=self.device)
        normalized, diagonal, actual = _precondition_rows(matrix, eps=0.25)
        # mean L2 norm=(0+.01+1+4)/4, rather than sqrt(mean squared L2 norm).
        floor = 0.25 * 5.01 / 4
        expected = torch.tensor([[0.0, 0.0], [0.01 / floor, 0.0], [0.0, 1.0], [1.0, 0.0]], device=self.device)
        torch.testing.assert_close(actual, expected, atol=0, rtol=1e-6)
        orthogonalized = matrix / 4
        refined_diagonal, actual = _refine_rows(normalized, orthogonalized, diagonal, beta=0.5, eps=0.25)
        denominator = torch.tensor([1.0, floor / 4, 0.25, 1.0], device=self.device).unsqueeze(-1)
        expected_diagonal = diagonal * (math.sqrt(0.5) / denominator)
        torch.testing.assert_close(refined_diagonal, expected_diagonal, atol=0, rtol=1e-6)
        torch.testing.assert_close(actual, normalized * expected_diagonal, atol=0, rtol=1e-6)

    def test_tall_update_is_invariant_to_momentum_scale(self) -> None:
        matrix = _gen_row_scaled_matrix((96, 16), self.device)
        matrix[0] = 0
        matrix[1] *= 1e-10  # Below the relative floor at every tested global scale.
        optimizer = Aurora([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest")
        self.assertEqual(optimizer.eps, 1e-6)
        expected = optimizer.scaled_orthogonalize_fn(matrix)
        for scale in (1e-8, 1e8, 2.0**-80, 2.0**80):
            with self.subTest(scale=scale):
                actual = optimizer.scaled_orthogonalize_fn(matrix * scale)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)

    def test_batched_matrices_have_independent_relative_floors(self) -> None:
        first_row = _gen_row_scaled_matrix((96, 16), self.device)
        second_row = _gen_row_scaled_matrix((96, 16), self.device) * 1e-8
        batch = torch.stack([first_row, second_row, torch.zeros_like(first_row)])
        optimizer = Aurora([torch.nn.Parameter(first_row.clone())], fp32_matmul_prec="highest", eps=0.25)
        actual = optimizer.scaled_orthogonalize_fn(batch)
        expected = torch.stack([optimizer.scaled_orthogonalize_fn(matrix) for matrix in batch])
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
        self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)

    def test_switch_from_muon_checkpoint_preserves_momentum(self) -> None:
        matrix = _gen_row_scaled_matrix((32, 8), self.device)
        parameter = torch.nn.Parameter(matrix.clone())
        optimizer_kwargs = dict(lr=0.01, momentum=0.9, weight_decay=0.1, nesterov=True, fp32_matmul_prec="highest")
        source = Muon([parameter], **optimizer_kwargs)
        parameter.grad = matrix.clone()
        source.step()
        state_dict_muon = copy.deepcopy(source.state_dict())
        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = Aurora([resumed_parameter], **optimizer_kwargs)
        resumed.load_state_dict(state_dict_muon)
        gradient = matrix.roll(1, dims=0)
        buffer = source.state[parameter]["momentum_buffer"].clone().lerp_(gradient, 0.1)
        expected_update = _reference_aurora_update(gradient.lerp(buffer, 0.9))
        expected_parameter = parameter.detach() * (1 - optimizer_kwargs["lr"] * optimizer_kwargs["weight_decay"])
        expected_parameter -= optimizer_kwargs["lr"] * expected_update
        resumed_parameter.grad = gradient
        resumed.step()
        torch.testing.assert_close(resumed_parameter, expected_parameter, atol=1e-8, rtol=1e-6)
        assert_equal(resumed.state[resumed_parameter]["momentum_buffer"], buffer)
        self.assertEqual(set(resumed.state[resumed_parameter]), {"momentum_buffer"})

    @parameterized.named_parameters(
        ("no_outer_steps", {"num_aurora_steps": 0}),
        ("fractional_outer_steps", {"num_aurora_steps": 1.5}),
        ("no_inner_steps", {"num_ns_steps": 0}),
        ("zero_beta", {"aurora_beta": 0.0}),
        ("excess_beta", {"aurora_beta": 1.1}),
        ("subnormal_floor", {"eps": 1e-40}),
        ("overflowing_floor", {"eps": 1e40}),
        ("nonfinite_floor", {"eps": float("nan")}),
    )
    def test_rejects_invalid_numerical_configuration(self, kwargs) -> None:
        parameter = torch.nn.Parameter(torch.zeros((4, 2), device=self.device))
        with self.assertRaises(ValueError):
            Aurora([parameter], **kwargs)


if __name__ == "__main__":
    absltest.main()
