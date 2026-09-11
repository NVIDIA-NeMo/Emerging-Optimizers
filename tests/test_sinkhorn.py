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
import inspect
import math

import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils import sinkhorn_balance


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


def _sinkhorn_balance_serial_cpu_reference(
    update: torch.Tensor,
    *,
    eps: float,
    num_steps: int,
    zero_row_threshold: float,
) -> torch.Tensor:
    values = update.detach().to(device="cpu", dtype=torch.float32).tolist()
    num_rows = len(values)
    num_columns = len(values[0])

    initial_row_norms = [math.sqrt(sum(value * value for value in row)) for row in values]
    mean_row_norm = sum(initial_row_norms) / num_rows
    for row_index, row_norm in enumerate(initial_row_norms):
        if row_norm <= zero_row_threshold * mean_row_norm:
            values[row_index] = [0.0] * num_columns

    for step in range(num_steps):
        if step % 2 == 0:
            for row_index in range(num_rows):
                row_norm = math.sqrt(sum(value * value for value in values[row_index]))
                for column_index in range(num_columns):
                    values[row_index][column_index] /= row_norm + eps
        else:
            for column_index in range(num_columns):
                column_norm = math.sqrt(sum(values[row_index][column_index] ** 2 for row_index in range(num_rows)))
                for row_index in range(num_rows):
                    values[row_index][column_index] /= column_norm + eps

    result = torch.tensor(values, dtype=torch.float32)
    result.mul_(math.sqrt(num_columns))
    return result.to(device=update.device, dtype=update.dtype)


class SinkhornBalanceTest(parameterized.TestCase):
    def test_defaults_match_deepseek_v41(self) -> None:
        parameters = inspect.signature(sinkhorn_balance).parameters

        self.assertEqual(parameters["eps"].default, 1e-20)
        self.assertEqual(parameters["num_steps"].default, 11)
        self.assertEqual(parameters["zero_row_threshold"].default, 1e-3)

    @parameterized.parameters(
        ((8, 4), torch.float32, 1, 0.0),
        ((9, 3), torch.float16, 3, 0.5),
        ((17, 5), torch.bfloat16, 11, 1.0),
        ((6, 6), torch.float32, 7, 0.0),
    )
    def test_is_close_to_serial_cpu_reference(self, shape, dtype, num_steps, zero_row_threshold) -> None:
        update = torch.randn(shape, device=FLAGS.device).to(dtype)

        actual = sinkhorn_balance(
            update,
            eps=1e-12,
            num_steps=num_steps,
            zero_row_threshold=zero_row_threshold,
        )
        expected = _sinkhorn_balance_serial_cpu_reference(
            update,
            eps=1e-12,
            num_steps=num_steps,
            zero_row_threshold=zero_row_threshold,
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_balances_row_and_column_rms(self) -> None:
        update = torch.randn((64, 8), device=FLAGS.device)

        result = sinkhorn_balance(update, eps=1e-12, num_steps=11, zero_row_threshold=0.0)

        torch.testing.assert_close(
            result.square().mean(dim=1),
            torch.ones(64, device=FLAGS.device),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            result.square().mean(dim=0),
            torch.ones(8, device=FLAGS.device),
            atol=0.05,
            rtol=0.05,
        )

    def test_masks_rows_at_or_below_threshold(self) -> None:
        update = torch.tensor([[1.0], [3.0]], device=FLAGS.device)

        result = sinkhorn_balance(update, zero_row_threshold=0.5)

        torch.testing.assert_close(result, torch.tensor([[0.0], [1.0]], device=FLAGS.device), atol=0.0, rtol=0.0)

    def test_all_zero_update_stays_zero_and_finite(self) -> None:
        update = torch.zeros((8, 4), device=FLAGS.device)

        result = sinkhorn_balance(update)

        torch.testing.assert_close(result, torch.zeros_like(update), atol=0.0, rtol=0.0)
        self.assertTrue(torch.isfinite(result).all())

    @parameterized.parameters(torch.bfloat16, torch.float16, torch.float32)
    def test_preserves_output_contract_and_input(self, dtype) -> None:
        update = torch.linspace(-2.0, 2.0, 32, device=FLAGS.device).reshape(4, 8).T.to(dtype)
        update.requires_grad_()
        update_before = update.detach().clone()

        result = sinkhorn_balance(update, eps=1e-12, zero_row_threshold=0.0)

        self.assertEqual(result.shape, update.shape)
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.device, update.device)
        self.assertNotEqual(result.data_ptr(), update.data_ptr())
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(update, update_before, atol=0.0, rtol=0.0)

    def test_rejects_invalid_inputs(self) -> None:
        valid_update = torch.zeros((8, 4), device=FLAGS.device)
        invalid_inputs = (
            ("non_matrix", torch.zeros(8, device=FLAGS.device), {}, "requires a 2D tensor"),
            ("empty", torch.empty((0, 4), device=FLAGS.device), {}, "requires nonempty matrix dimensions"),
            ("wide", torch.zeros((4, 8), device=FLAGS.device), {}, "rows to be the larger"),
            (
                "unsupported_dtype",
                torch.zeros((8, 4), device=FLAGS.device, dtype=torch.float64),
                {},
                "only supports bfloat16, float16, and float32",
            ),
            ("non_positive_eps", valid_update, {"eps": 0.0}, "eps must be positive and finite"),
            ("non_finite_eps", valid_update, {"eps": float("nan")}, "eps must be positive and finite"),
            ("non_positive_num_steps", valid_update, {"num_steps": 0}, "positive odd integer"),
            ("even_num_steps", valid_update, {"num_steps": 2}, "positive odd integer"),
            (
                "negative_zero_row_threshold",
                valid_update,
                {"zero_row_threshold": -1.0},
                "nonnegative and finite",
            ),
            (
                "non_finite_zero_row_threshold",
                valid_update,
                {"zero_row_threshold": float("nan")},
                "nonnegative and finite",
            ),
        )

        for case_name, update, kwargs, error_message in invalid_inputs:
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(ValueError, error_message):
                    sinkhorn_balance(update, **kwargs)


if __name__ == "__main__":
    absltest.main()
