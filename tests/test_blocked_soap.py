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
from emerging_optimizers.soap import SOAP, BlockedSoap


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class BlockedSoapTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(13)
        cls.device = FLAGS.device

    def test_registered(self) -> None:
        self.assertIs(registry.get_optimizer_cls("blocked_soap"), BlockedSoap)

    @parameterized.parameters(
        {"shape": (8, 4), "num_blocks": 2, "block_size": 4},
        {"shape": (4, 8), "num_blocks": 2, "block_size": 4},
        {"shape": (6, 6), "num_blocks": 1, "block_size": 6},
    )
    def test_10steps_smoke_and_state_shapes(self, shape: tuple[int, int], num_blocks: int, block_size: int) -> None:
        param = torch.nn.Parameter(torch.randn(shape, device=self.device))
        optimizer = BlockedSoap([param], lr=1e-3)
        for _ in range(10):
            param.grad = torch.randn(shape, device=self.device)
            optimizer.step()

        self.assertEqual(param.shape, torch.Size(shape))
        self.assertEqual(param.grad.shape, torch.Size(shape))
        self.assertTrue(torch.isfinite(param).all())
        state = optimizer.state[param]
        for key in ("exp_avg", "exp_avg_sq", "L", "R", "Q_L", "Q_R"):
            self.assertEqual(state[key].shape, (num_blocks, block_size, block_size))
        for key in ("eigvals_L", "eigvals_R"):
            self.assertEqual(state[key].shape, (num_blocks, block_size))

    def test_tall_param_close_to_per_block_soap(self) -> None:
        torch.manual_seed(0)
        m, n, num_steps = 8, 4, 5
        weight = torch.randn(m, n, device=self.device)
        grad_list = [torch.randn(m, n, device=self.device) for _ in range(num_steps)]

        blocked_param = torch.nn.Parameter(weight.clone())
        blocked_optimizer = BlockedSoap([blocked_param], lr=1e-2)

        block_param_list = [torch.nn.Parameter(weight[i * n : (i + 1) * n].clone()) for i in range(m // n)]
        soap_optimizer = SOAP(block_param_list, lr=1e-2)

        for grad in grad_list:
            blocked_param.grad = grad.clone()
            blocked_optimizer.step()
            for i, block_param in enumerate(block_param_list):
                block_param.grad = grad[i * n : (i + 1) * n].clone()
            soap_optimizer.step()

        torch.testing.assert_close(
            blocked_param.detach(),
            torch.cat([block_param.detach() for block_param in block_param_list]),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda msg: f"BlockedSoap diverged from per-block SOAP on a tall parameter\n\n{msg}",
        )

    def test_wide_param_close_to_per_block_soap(self) -> None:
        torch.manual_seed(0)
        m, n, num_steps = 4, 8, 5
        weight = torch.randn(m, n, device=self.device)
        grad_list = [torch.randn(m, n, device=self.device) for _ in range(num_steps)]

        blocked_param = torch.nn.Parameter(weight.clone())
        blocked_optimizer = BlockedSoap([blocked_param], lr=1e-2)

        block_param_list = [torch.nn.Parameter(weight[:, i * m : (i + 1) * m].clone()) for i in range(n // m)]
        soap_optimizer = SOAP(block_param_list, lr=1e-2)

        for grad in grad_list:
            blocked_param.grad = grad.clone()
            blocked_optimizer.step()
            for i, block_param in enumerate(block_param_list):
                block_param.grad = grad[:, i * m : (i + 1) * m].clone()
            soap_optimizer.step()

        torch.testing.assert_close(
            blocked_param.detach(),
            torch.cat([block_param.detach() for block_param in block_param_list], dim=1),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda msg: f"BlockedSoap diverged from per-block SOAP on a wide parameter\n\n{msg}",
        )

    def test_square_param_close_to_soap(self) -> None:
        torch.manual_seed(0)
        n, num_steps = 6, 5
        weight = torch.randn(n, n, device=self.device)
        grad_list = [torch.randn(n, n, device=self.device) for _ in range(num_steps)]

        blocked_param = torch.nn.Parameter(weight.clone())
        blocked_optimizer = BlockedSoap([blocked_param], lr=1e-2)
        soap_param = torch.nn.Parameter(weight.clone())
        soap_optimizer = SOAP([soap_param], lr=1e-2)

        for grad in grad_list:
            blocked_param.grad = grad.clone()
            blocked_optimizer.step()
            soap_param.grad = grad.clone()
            soap_optimizer.step()

        torch.testing.assert_close(
            blocked_param.detach(),
            soap_param.detach(),
            atol=1e-4,
            rtol=1e-4,
            msg=lambda msg: f"BlockedSoap diverged from SOAP on a square parameter\n\n{msg}",
        )

    def test_non_divisible_shape_raises(self) -> None:
        param = torch.nn.Parameter(torch.randn(6, 4, device=self.device))
        param.grad = torch.randn(6, 4, device=self.device)
        optimizer = BlockedSoap([param], lr=1e-3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            optimizer.step()

    def test_non_2d_param_raises(self) -> None:
        param = torch.nn.Parameter(torch.randn(8, device=self.device))
        param.grad = torch.randn(8, device=self.device)
        optimizer = BlockedSoap([param], lr=1e-3)
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            optimizer.step()


if __name__ == "__main__":
    absltest.main()
