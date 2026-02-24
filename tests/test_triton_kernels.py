# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from emerging_optimizers import triton_kernels


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class TsyrkTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        ({"n": 128, "k": 128, "atol": 0, "rtol": 0.05}, {"n": 256, "k": 64, "atol": 0.1, "rtol": 0.05}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_close_to_matmul(self, n: int, k: int, atol: float, rtol: float, trans: bool):
        a = torch.randn(n, k, device=self.device, dtype=torch.bfloat16)
        a_warmup = torch.randn_like(a, device=a.device, dtype=torch.bfloat16)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        ref = a @ a.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex(a_warmup)
        c = triton_kernels.tsyrk_ex(a)
        torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)

    @parameterized.product(
        ({"n": 128, "alpha": 0.4, "beta": 0.3, "rtol": 0.05}, {"n": 256, "alpha": 0.5, "beta": 0.5, "rtol": 0.05}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_close_to_addmm(self, n: int, alpha: float, beta: float, trans: bool, rtol: float):
        a = torch.randn(n, n, device=self.device, dtype=torch.bfloat16)
        # make a symmetric input matrix
        a = a + a.T
        a_warmup = torch.randn_like(a, device=a.device, dtype=torch.bfloat16)
        ref = torch.addmm(a, a, a, alpha=alpha, beta=beta)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex(a_warmup, a_warmup, alpha=alpha, beta=beta)
        c = triton_kernels.tsyrk_ex(a, a, alpha=alpha, beta=beta)
        torch.testing.assert_close(c, ref, atol=0, rtol=rtol)


class TsyrkIntegerInputTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        ({"n": 128, "k": 128}, {"n": 256, "k": 64}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_match_matmul(self, n: int, k: int, trans: bool):
        a = torch.randint(-3, 3, (n, k), device=self.device, dtype=torch.bfloat16)
        a_warmup = torch.randint_like(a, -3, 3, device=a.device, dtype=torch.bfloat16)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        ref = a @ a.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex(a_warmup)
        c = triton_kernels.tsyrk_ex(a)
        torch.testing.assert_close(c, ref, atol=0, rtol=0)

    @parameterized.product(
        ({"n": 128, "alpha": 0.5, "beta": 0.5}, {"n": 256, "alpha": 0.25, "beta": 0.25}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_match_addmm(self, n: int, alpha: float, beta: float, trans: bool):
        a = torch.randint(-3, 3, (n, n), device=self.device, dtype=torch.bfloat16)
        # make a symmetric input matrix
        a = a + a.T
        a_warmup = torch.randint_like(a, -3, 3, device=a.device, dtype=torch.bfloat16)
        ref = torch.addmm(a, a, a, alpha=alpha, beta=beta)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex(a_warmup, a_warmup, alpha=alpha, beta=beta)
        c = triton_kernels.tsyrk_ex(a, a, alpha=alpha, beta=beta)
        torch.testing.assert_close(c, ref, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
