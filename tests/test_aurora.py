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
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers import Aurora, Muon
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
    matrix: torch.Tensor, *, iterations: int = 2, beta: float = 0.5, eps: float = 1e-14
) -> torch.Tensor:
    """Independent norm/reciprocal expression of the released diagonal refinement."""
    diagonal = torch.linalg.vector_norm(matrix, dim=-1, keepdim=True).clamp_min(eps**0.5).reciprocal()
    for index in range(iterations):
        update = newton_schulz(diagonal * matrix, steps=8, coefficient_type="polar_express")
        if index + 1 < iterations:
            row_norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
            diagonal = diagonal * (matrix.size(-1) / matrix.size(-2) / row_norm.square().clamp_min(eps)).pow(beta)
    return update * math.sqrt(matrix.size(-2)) * 0.2


class AuroraTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = FLAGS.device
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, precision)

    @parameterized.named_parameters(
        ("default", 2, 0.5, 1e-14),
        ("more_refinements", 4, 0.25, 1e-14),
        ("custom_floor", 2, 0.5, 0.25),
    )
    def test_matches_diagonal_refinement_reference(self, iterations, beta, eps) -> None:
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
        # Norm/reciprocal and squared-sum/rsqrt round differently before the NS solves.
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(matrix, before, atol=0, rtol=0)

    def test_default_medium_precision_matches_reference(self) -> None:
        matrix = _gen_row_scaled_matrix((96, 16), self.device)
        optimizer = Aurora([torch.nn.Parameter(matrix.clone())])
        with utils.fp32_matmul_precision(optimizer.fp32_matmul_prec):
            actual = optimizer.scaled_orthogonalize_fn(matrix)
            expected = _reference_aurora_update(matrix)
        # BF16 GEMMs amplify differences between the independent FP32 reductions.
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_reduces_row_imbalance(self) -> None:
        matrix = _gen_row_scaled_matrix((96, 16), self.device)
        aurora = Aurora([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest")
        muon = Muon([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest")
        aurora_norms = torch.linalg.vector_norm(aurora.scaled_orthogonalize_fn(matrix), dim=-1)
        muon_norms = torch.linalg.vector_norm(muon.scaled_orthogonalize_fn(matrix), dim=-1)
        self.assertLess(aurora_norms.std() / aurora_norms.mean(), 0.3 * muon_norms.std() / muon_norms.mean())

    @parameterized.parameters((16, 32), (16, 16))
    def test_square_and_wide_updates_match_muon(self, rows, cols) -> None:
        matrix = _gen_row_scaled_matrix((rows, cols), self.device)
        aurora = Aurora([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest", num_aurora_steps=4)
        muon = Muon([torch.nn.Parameter(matrix.clone())], fp32_matmul_prec="highest")
        torch.testing.assert_close(
            aurora.scaled_orthogonalize_fn(matrix), muon.scaled_orthogonalize_fn(matrix), atol=0, rtol=0
        )

    @parameterized.parameters(1e-14, torch.finfo(torch.float32).tiny)
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
        expected = _reference_aurora_update(matrix)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(torch.count_nonzero(actual).item(), 0)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_switch_from_muon_checkpoint_preserves_momentum(self) -> None:
        matrix = _gen_row_scaled_matrix((32, 8), self.device)
        parameter = torch.nn.Parameter(matrix.clone())
        source = Muon([parameter], lr=0.01, momentum=0.9, weight_decay=0.1, nesterov=True, fp32_matmul_prec="highest")
        parameter.grad = matrix.clone()
        source.step()
        checkpoint = copy.deepcopy(source.state_dict())
        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed = Aurora(
            [resumed_parameter], lr=0.01, momentum=0.9, weight_decay=0.1, nesterov=True, fp32_matmul_prec="highest"
        )
        resumed.load_state_dict(checkpoint)
        gradient = matrix.roll(1, dims=0)
        buffer = source.state[parameter]["momentum_buffer"].clone().lerp_(gradient, 0.1)
        expected_update = _reference_aurora_update(gradient.lerp(buffer, 0.9))
        expected_parameter = parameter.detach() * (1 - 0.01 * 0.1) - 0.01 * expected_update
        resumed_parameter.grad = gradient
        resumed.step()
        torch.testing.assert_close(resumed_parameter, expected_parameter, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(resumed.state[resumed_parameter]["momentum_buffer"], buffer, atol=0, rtol=0)
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
