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
from absl.testing import absltest

from emerging_optimizers.psgd.psgd import PSGDPro


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class TestPSGDPro(absltest.TestCase):
    def setUp(self) -> None:
        self.device = FLAGS.device

    def test_step_counter_increments(self) -> None:
        """PSGDPro must increment state['step'] each optimization step."""
        param = torch.randn(4, 4, device=self.device, requires_grad=True)
        optimizer = PSGDPro([param], lr=0.01)

        for expected_step in range(1, 4):
            param.grad = torch.randn_like(param)
            optimizer.step()
            self.assertEqual(optimizer.state[param]["step"], expected_step)

    def test_step_with_float64_param(self) -> None:
        param = torch.randn(8, 4, device=self.device, dtype=torch.float64, requires_grad=True)
        optimizer = PSGDPro([param], lr=0.01)

        for _ in range(3):
            param.grad = torch.randn_like(param)
            optimizer.step()

        self.assertEqual(param.dtype, torch.float64)
        self.assertTrue(torch.isfinite(param).all())
        for q in optimizer.state[param]["Q"]:
            self.assertEqual(q.dtype, torch.float64)

    def test_bfloat16_param_matches_float32_reference(self) -> None:
        """bf16 params keep fp32 Kronecker factors and track an fp32 run of the same problem."""
        init = torch.randn(8, 4, device=self.device).bfloat16()
        grads = [torch.randn(8, 4, device=self.device).bfloat16() for _ in range(5)]
        param = init.clone().requires_grad_()
        ref_param = init.float().requires_grad_()
        optimizer = PSGDPro([param], lr=0.01, precond_lr=0.5)
        ref_optimizer = PSGDPro([ref_param], lr=0.01, precond_lr=0.5)

        for i, grad in enumerate(grads):
            param.grad = grad.clone()
            ref_param.grad = grad.float()
            # Both runs draw their damping noise in fp32, so the same seed gives the same noise.
            torch.manual_seed(i)
            optimizer.step()
            torch.manual_seed(i)
            ref_optimizer.step()

        self.assertEqual(param.dtype, torch.bfloat16)
        for q, ref_q in zip(optimizer.state[param]["Q"], ref_optimizer.state[ref_param]["Q"]):
            self.assertEqual(q.dtype, torch.float32)
            torch.testing.assert_close(q, ref_q, atol=5e-3, rtol=0)
        torch.testing.assert_close(param.float(), ref_param, atol=2e-2, rtol=0)


if __name__ == "__main__":
    absltest.main()
