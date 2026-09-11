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

from emerging_optimizers import registry
from emerging_optimizers.embedding_optimizers import Sinkhorn


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class SinkhornTest(parameterized.TestCase):
    def test_deepseek_v41_defaults(self) -> None:
        param = torch.nn.Parameter(torch.zeros((8, 4), device=FLAGS.device))
        group = Sinkhorn([param]).param_groups[0]

        self.assertEqual(group["momentum"], 0.95)
        self.assertEqual(group["eps"], 1e-20)
        self.assertEqual(group["num_steps"], 11)
        self.assertEqual(group["zero_row_threshold"], 1e-3)
        self.assertEqual(group["lr_correction"], 0.18)

    def test_rejects_float64_parameters(self) -> None:
        param = torch.nn.Parameter(torch.zeros((8, 4), device=FLAGS.device, dtype=torch.float64))
        param.grad = torch.ones_like(param)
        optimizer = Sinkhorn([param])

        with self.assertRaisesRegex(ValueError, "only supports bfloat16, float16, and float32"):
            optimizer.step()

    @parameterized.parameters((64, 8), (128, 16))
    def test_balances_row_and_column_rms(self, num_rows, num_cols) -> None:
        lr = 0.125
        lr_correction = 0.25
        param = torch.nn.Parameter(torch.zeros((num_rows, num_cols), device=FLAGS.device))
        param.grad = torch.randn_like(param)
        optimizer = Sinkhorn(
            [param],
            lr=lr,
            momentum=0.0,
            eps=1e-12,
            num_steps=7,
            zero_row_threshold=0.0,
            lr_correction=lr_correction,
        )

        optimizer.step()
        applied_update = -param.detach() / (lr * lr_correction)

        torch.testing.assert_close(
            applied_update.square().mean(dim=1),
            torch.ones(num_rows, device=FLAGS.device),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            applied_update.square().mean(dim=0),
            torch.ones(num_cols, device=FLAGS.device),
            atol=0.1,
            rtol=0.1,
        )

    def test_masks_near_zero_rows(self) -> None:
        param = torch.nn.Parameter(torch.zeros((16, 4), device=FLAGS.device))
        grad = torch.ones_like(param)
        grad[0].fill_(1e-6)
        param.grad = grad
        optimizer = Sinkhorn([param], lr=1.0, momentum=0.0, zero_row_threshold=1e-3, lr_correction=1.0)

        optimizer.step()

        torch.testing.assert_close(param[0], torch.zeros(4, device=FLAGS.device), atol=0.0, rtol=0.0)
        self.assertTrue(torch.isfinite(param).all())

    def test_nesterov_momentum(self) -> None:
        param = torch.nn.Parameter(torch.zeros((8, 4), device=FLAGS.device))
        optimizer = Sinkhorn(
            [param],
            lr=1.0,
            momentum=0.5,
            eps=1e-12,
            num_steps=1,
            zero_row_threshold=0.0,
            lr_correction=1.0,
        )

        first_grad = torch.arange(1, 33, device=FLAGS.device, dtype=torch.float32).reshape(8, 4)
        param.grad = first_grad
        optimizer.step()
        expected_momentum = 0.5 * first_grad
        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"], expected_momentum)

        second_grad = first_grad.flip(1)
        param.grad = second_grad
        param_before = param.detach().clone()
        optimizer.step()
        expected_momentum = 0.5 * expected_momentum + 0.5 * second_grad
        expected_nesterov = 0.5 * expected_momentum + 0.5 * second_grad
        expected_update = expected_nesterov / expected_nesterov.norm(dim=1, keepdim=True) * 2.0

        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"], expected_momentum)
        torch.testing.assert_close(param_before - param.detach(), expected_update)

    @parameterized.parameters(0, 2, 4)
    def test_rejects_non_positive_or_even_num_steps(self, num_steps) -> None:
        param = torch.nn.Parameter(torch.zeros((8, 4), device=FLAGS.device))
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            Sinkhorn([param], num_steps=num_steps)

    @parameterized.parameters(
        {"lr": -1.0},
        {"momentum": -0.1},
        {"momentum": 1.0},
        {"eps": 0.0},
        {"zero_row_threshold": -1.0},
        {"lr_correction": -1.0},
    )
    def test_rejects_invalid_hyperparameters(self, **kwargs) -> None:
        param = torch.nn.Parameter(torch.zeros((8, 4), device=FLAGS.device))
        with self.assertRaises(ValueError):
            Sinkhorn([param], **kwargs)

    @parameterized.parameters((8,), (2, 4, 8))
    def test_rejects_non_matrix_parameters(self, shape) -> None:
        param = torch.nn.Parameter(torch.zeros(shape, device=FLAGS.device))
        param.grad = torch.ones_like(param)
        optimizer = Sinkhorn([param])
        with self.assertRaisesRegex(ValueError, "only supports 2D"):
            optimizer.step()

    def test_rejects_wide_matrices(self) -> None:
        param = torch.nn.Parameter(torch.zeros((4, 8), device=FLAGS.device))
        param.grad = torch.ones_like(param)
        optimizer = Sinkhorn([param])
        with self.assertRaisesRegex(ValueError, "rows to be the larger"):
            optimizer.step()

    def test_rejects_sparse_gradients(self) -> None:
        param = torch.nn.Parameter(torch.zeros((8, 4), device=FLAGS.device))
        indices = torch.tensor([[0, 1]], device=FLAGS.device)
        values = torch.ones((2, 4), device=FLAGS.device)
        param.grad = torch.sparse_coo_tensor(indices, values, param.shape, device=FLAGS.device)
        optimizer = Sinkhorn([param])
        with self.assertRaisesRegex(ValueError, "sparse gradients"):
            optimizer.step()

    def test_rejects_closure(self) -> None:
        param = torch.nn.Parameter(torch.zeros((8, 4), device=FLAGS.device))
        optimizer = Sinkhorn([param])
        with self.assertRaisesRegex(ValueError, "closure is not supported"):
            optimizer.step(lambda: 0.0)

    def test_registered(self) -> None:
        self.assertIs(registry.get_optimizer_cls("sinkhorn"), Sinkhorn)


if __name__ == "__main__":
    absltest.main()
