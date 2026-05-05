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
import triton
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import triton_kernels
from emerging_optimizers.triton_kernels.syrk import prune_invalid_configs, prune_invalid_configs_for_small_matrix


flags.DEFINE_enum("device", "cuda", ["cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


def _make_config(tile_m: int, tile_n: int, tile_k: int) -> triton.Config:
    return triton.Config({"TILE_M": tile_m, "TILE_N": tile_n, "TILE_K": tile_k})


class PruneInvalidConfigsTest(parameterized.TestCase):
    @parameterized.parameters(
        {"n": 5123, "configs": [(128, 256, 64), (64, 128, 64), (256, 256, 128)], "expected": [(128, 256, 64)]},
        {
            "n": 3999,
            "configs": [(64, 128, 64), (128, 128, 128), (256, 256, 64)],
            "expected": [(64, 128, 64), (128, 128, 128)],
        },
        {"n": 1337, "configs": [(128, 64, 64)], "expected": []},
    )
    def test_prune_invalid_configs(self, n: int, configs: list, expected: list):
        triton_configs = [_make_config(*c) for c in configs]
        result = prune_invalid_configs(triton_configs, {"N": n})
        result_tuples = [(c.kwargs["TILE_M"], c.kwargs["TILE_N"], c.kwargs["TILE_K"]) for c in result]
        self.assertEqual(result_tuples, expected)


class PruneInvalidConfigsForSmallMatrixTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            "n": 7777,
            "configs": [(128, 128, 64), (256, 256, 64), (64, 64, 64), (128, 256, 64)],
            "expected": [(128, 128, 64), (256, 256, 64)],
        },
        {
            "n": 2345,
            "configs": [(64, 64, 64), (128, 128, 128), (128, 256, 64)],
            "expected": [(64, 64, 64), (128, 128, 128)],
        },
        {"n": 999, "configs": [(256, 256, 64)], "expected": [(256, 256, 64)]},
    )
    def test_prune_invalid_configs_for_small_matrix(self, n: int, configs: list, expected: list):
        triton_configs = [_make_config(*c) for c in configs]
        result = prune_invalid_configs_for_small_matrix(triton_configs, {"N": n})
        result_tuples = [(c.kwargs["TILE_M"], c.kwargs["TILE_N"], c.kwargs["TILE_K"]) for c in result]
        self.assertEqual(result_tuples, expected)


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
        ({"n": 128, "k": 128, "atol": 0, "rtol": 0.05}, {"n": 256, "k": 64, "atol": 0.1, "rtol": 0.05}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_small_matrix_close_to_matmul(self, n: int, k: int, atol: float, rtol: float, trans: bool):
        a = torch.randn(n, k, device=self.device, dtype=torch.bfloat16)
        a_warmup = torch.randn_like(a, device=a.device, dtype=torch.bfloat16)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        ref = a @ a.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex_small_matrix(a_warmup)
        c = triton_kernels.tsyrk_ex_small_matrix(a)
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

    @parameterized.product(
        ({"n": 128, "alpha": 0.4, "beta": 0.3, "rtol": 0.05}, {"n": 256, "alpha": 0.5, "beta": 0.5, "rtol": 0.05}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_small_matrix_close_to_addmm(self, n: int, alpha: float, beta: float, trans: bool, rtol: float):
        a = torch.randn(n, n, device=self.device, dtype=torch.bfloat16)
        # make a symmetric input matrix.
        a = a + a.T
        a_warmup = torch.randn_like(a, device=a.device, dtype=torch.bfloat16)
        ref = torch.addmm(a, a, a, alpha=alpha, beta=beta)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex_small_matrix(a_warmup, a_warmup, alpha=alpha, beta=beta)
        c = triton_kernels.tsyrk_ex_small_matrix(a, a, alpha=alpha, beta=beta)
        torch.testing.assert_close(c, ref, atol=0, rtol=rtol)

    @parameterized.product(
        ({"n": 128, "k": 128, "atol": 0, "rtol": 0.05}, {"n": 256, "k": 64, "atol": 0.1, "rtol": 0.05}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_small_matrix_with_out_tensor(self, n: int, k: int, atol: float, rtol: float, trans: bool):
        a = torch.randn(n, k, device=self.device, dtype=torch.bfloat16)
        if trans:
            a = a.T
        out = torch.empty((a.shape[0], a.shape[0]), device=self.device, dtype=torch.bfloat16)
        result_with_out = triton_kernels.tsyrk_ex_small_matrix(a, out=out)
        result_no_out = triton_kernels.tsyrk_ex_small_matrix(a)
        torch.testing.assert_close(result_with_out, result_no_out, atol=atol, rtol=rtol)


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
        ({"n": 128, "k": 128}, {"n": 256, "k": 64}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_small_matrix_match_matmul(self, n: int, k: int, trans: bool):
        a = torch.randint(-3, 3, (n, k), device=self.device, dtype=torch.bfloat16)
        a_warmup = torch.randint_like(a, -3, 3, device=a.device, dtype=torch.bfloat16)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        ref = a @ a.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex_small_matrix(a_warmup)
        c = triton_kernels.tsyrk_ex_small_matrix(a)
        torch.testing.assert_close(c, ref, atol=0, rtol=0)

    @parameterized.product(
        ({"n": 128, "alpha": 0.5, "beta": 0.5}, {"n": 256, "alpha": 0.25, "beta": 0.25}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_match_addmm(self, n: int, alpha: float, beta: float, trans: bool):
        a = torch.randint(-3, 3, (n, n), device=self.device, dtype=torch.bfloat16)
        # make a symmetric input matrix.
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

    @parameterized.product(
        ({"n": 128, "alpha": 0.5, "beta": 0.5}, {"n": 256, "alpha": 0.25, "beta": 0.25}),
        ({"trans": False}, {"trans": True}),
    )
    def test_tsyrk_ex_small_matrix_match_addmm(self, n: int, alpha: float, beta: float, trans: bool):
        a = torch.randint(-3, 3, (n, n), device=self.device, dtype=torch.bfloat16)
        # make a symmetric input matrix.
        a = a + a.T
        a_warmup = torch.randint_like(a, -3, 3, device=a.device, dtype=torch.bfloat16)
        ref = torch.addmm(a, a, a, alpha=alpha, beta=beta)
        if trans:
            a = a.T
            a_warmup = a_warmup.T
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.tsyrk_ex_small_matrix(a_warmup, a_warmup, alpha=alpha, beta=beta)
        c = triton_kernels.tsyrk_ex_small_matrix(a, a, alpha=alpha, beta=beta)
        torch.testing.assert_close(c, ref, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
