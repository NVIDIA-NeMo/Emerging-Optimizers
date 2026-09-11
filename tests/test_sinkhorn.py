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
import math

import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.embedding_optimizers import sinkhorn_balance


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


def _sinkhorn_balance_reference(
    update: torch.Tensor,
    *,
    eps: float,
    num_steps: int,
    zero_row_threshold: float,
) -> torch.Tensor:
    result = update.to(dtype=torch.float32, copy=True)
    row_norms = torch.linalg.vector_norm(result, dim=1, keepdim=True)
    result.masked_fill_(row_norms <= zero_row_threshold * row_norms.mean(), 0.0)

    for step in range(num_steps):
        if step % 2 == 0:
            row_norms = torch.linalg.vector_norm(result, dim=1, keepdim=True)
            result = result / (row_norms + eps)
        else:
            column_norms = torch.linalg.vector_norm(result, dim=0, keepdim=True)
            result = result / (column_norms + eps)

    result = result * math.sqrt(update.size(1))
    return result.to(update.dtype)


class SinkhornBalanceTest(parameterized.TestCase):
    def test_defaults_match_deepseek_v41(self) -> None:
        update = torch.randn((32, 8), device=FLAGS.device)

        actual = sinkhorn_balance(update)
        expected = sinkhorn_balance(update, eps=1e-20, num_steps=11, zero_row_threshold=1e-3)

        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    @parameterized.parameters(
        ((8, 4), torch.float32, 1, 0.0),
        ((9, 3), torch.float16, 3, 0.5),
        ((17, 5), torch.bfloat16, 11, 1.0),
        ((6, 6), torch.float32, 7, 0.0),
    )
    def test_matches_step_by_step_reference(self, shape, dtype, num_steps, zero_row_threshold) -> None:
        update = torch.randn(shape, device=FLAGS.device).to(dtype)

        actual = sinkhorn_balance(
            update,
            eps=1e-12,
            num_steps=num_steps,
            zero_row_threshold=zero_row_threshold,
        )
        expected = _sinkhorn_balance_reference(
            update,
            eps=1e-12,
            num_steps=num_steps,
            zero_row_threshold=zero_row_threshold,
        )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    @parameterized.parameters((64, 8), (128, 16), (16, 16))
    def test_balances_row_and_column_rms(self, num_rows, num_columns) -> None:
        update = torch.randn((num_rows, num_columns), device=FLAGS.device)

        result = sinkhorn_balance(update, eps=1e-12, num_steps=11, zero_row_threshold=0.0)

        torch.testing.assert_close(
            result.square().mean(dim=1),
            torch.ones(num_rows, device=FLAGS.device),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            result.square().mean(dim=0),
            torch.ones(num_columns, device=FLAGS.device),
            atol=0.05,
            rtol=0.05,
        )

    def test_single_step_converts_unit_row_l2_norm_to_unit_rms(self) -> None:
        update = torch.arange(1, 33, device=FLAGS.device, dtype=torch.float32).reshape(8, 4)

        result = sinkhorn_balance(update, eps=1e-20, num_steps=1, zero_row_threshold=0.0)

        torch.testing.assert_close(
            result.square().mean(dim=1),
            torch.ones(8, device=FLAGS.device),
            atol=1e-6,
            rtol=1e-6,
        )

    @parameterized.parameters(
        (0.5, [[0.0], [1.0]]),
        (0.49, [[1.0], [1.0]]),
    )
    def test_masks_rows_at_or_below_threshold_only(self, zero_row_threshold, expected) -> None:
        update = torch.tensor([[1.0], [3.0]], device=FLAGS.device)

        result = sinkhorn_balance(update, num_steps=1, zero_row_threshold=zero_row_threshold)

        torch.testing.assert_close(result, torch.tensor(expected, device=FLAGS.device), atol=0.0, rtol=0.0)

    def test_all_zero_update_stays_zero_and_finite(self) -> None:
        update = torch.zeros((8, 4), device=FLAGS.device)

        result = sinkhorn_balance(update)

        torch.testing.assert_close(result, torch.zeros_like(update), atol=0.0, rtol=0.0)
        self.assertTrue(torch.isfinite(result).all())

    def test_preserves_signs_of_unmasked_entries(self) -> None:
        update = torch.tensor(
            [[1.0, -2.0], [-3.0, 4.0], [5.0, -6.0], [-7.0, 8.0]],
            device=FLAGS.device,
        )

        result = sinkhorn_balance(update, zero_row_threshold=0.0)

        torch.testing.assert_close(torch.sign(result), torch.sign(update), atol=0.0, rtol=0.0)

    def test_is_invariant_to_positive_input_scale(self) -> None:
        update = torch.randn((32, 8), device=FLAGS.device)

        result = sinkhorn_balance(update, eps=1e-20, zero_row_threshold=0.0)
        scaled_result = sinkhorn_balance(update * 1e4, eps=1e-20, zero_row_threshold=0.0)

        torch.testing.assert_close(result, scaled_result, atol=2e-6, rtol=2e-6)

    @parameterized.parameters(torch.bfloat16, torch.float16, torch.float32)
    def test_preserves_shape_dtype_device_and_input(self, dtype) -> None:
        update = torch.linspace(-2.0, 2.0, 32, device=FLAGS.device).reshape(8, 4).to(dtype)
        update_before = update.clone()

        result = sinkhorn_balance(update, eps=1e-12, zero_row_threshold=0.0)

        self.assertEqual(result.shape, update.shape)
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.device, update.device)
        self.assertNotEqual(result.data_ptr(), update.data_ptr())
        torch.testing.assert_close(update, update_before, atol=0.0, rtol=0.0)

    @parameterized.parameters(torch.bfloat16, torch.float16)
    def test_low_precision_input_matches_fp32_workspace(self, dtype) -> None:
        update = torch.randn((32, 8), device=FLAGS.device).to(dtype)

        actual = sinkhorn_balance(update, eps=1e-12, zero_row_threshold=0.0)
        expected = sinkhorn_balance(update.float(), eps=1e-12, zero_row_threshold=0.0).to(dtype)

        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_supports_noncontiguous_input(self) -> None:
        update = torch.randn((4, 8), device=FLAGS.device).T
        self.assertFalse(update.is_contiguous())

        actual = sinkhorn_balance(update, eps=1e-12, zero_row_threshold=0.0)
        expected = sinkhorn_balance(update.contiguous(), eps=1e-12, zero_row_threshold=0.0)

        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-7)

    def test_does_not_participate_in_autograd(self) -> None:
        update = torch.randn((8, 4), device=FLAGS.device, requires_grad=True)

        result = sinkhorn_balance(update)

        self.assertFalse(result.requires_grad)

    @parameterized.parameters(torch.float64, torch.int64, torch.complex64)
    def test_rejects_unsupported_dtype(self, dtype) -> None:
        update = torch.zeros((8, 4), device=FLAGS.device, dtype=dtype)

        with self.assertRaisesRegex(ValueError, "only supports bfloat16, float16, and float32"):
            sinkhorn_balance(update)

    @parameterized.named_parameters(
        ("scalar", ()),
        ("vector", (8,)),
        ("batched_matrix", (2, 8, 4)),
    )
    def test_rejects_non_matrix_input(self, shape) -> None:
        update = torch.zeros(shape, device=FLAGS.device)

        with self.assertRaisesRegex(ValueError, "requires a 2D tensor"):
            sinkhorn_balance(update)

    @parameterized.parameters((0, 0), (8, 0), (0, 8))
    def test_rejects_empty_matrix_dimensions(self, num_rows, num_columns) -> None:
        update = torch.empty((num_rows, num_columns), device=FLAGS.device)

        with self.assertRaisesRegex(ValueError, "requires nonempty matrix dimensions"):
            sinkhorn_balance(update)

    def test_rejects_wide_matrix(self) -> None:
        update = torch.zeros((4, 8), device=FLAGS.device)

        with self.assertRaisesRegex(ValueError, "rows to be the larger"):
            sinkhorn_balance(update)

    @parameterized.parameters(0.0, -1.0, float("inf"), float("nan"))
    def test_rejects_invalid_eps(self, eps) -> None:
        update = torch.zeros((8, 4), device=FLAGS.device)

        with self.assertRaisesRegex(ValueError, "eps must be positive and finite"):
            sinkhorn_balance(update, eps=eps)

    @parameterized.parameters(-1, 0, 2, 12)
    def test_rejects_non_positive_or_even_num_steps(self, num_steps) -> None:
        update = torch.zeros((8, 4), device=FLAGS.device)

        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            sinkhorn_balance(update, num_steps=num_steps)

    @parameterized.parameters(-1.0, float("inf"), float("nan"))
    def test_rejects_invalid_zero_row_threshold(self, zero_row_threshold) -> None:
        update = torch.zeros((8, 4), device=FLAGS.device)

        with self.assertRaisesRegex(ValueError, "nonnegative and finite"):
            sinkhorn_balance(update, zero_row_threshold=zero_row_threshold)


if __name__ == "__main__":
    absltest.main()
