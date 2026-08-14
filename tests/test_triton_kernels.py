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
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import triton_kernels
from emerging_optimizers.triton_kernels.batched_syrk import prune_invalid_batched_configs
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


def _make_batched_config(
    tile_m: int, tile_n: int, tile_k: int, num_warps: int, num_stages: int, warp_specialize: bool = False
) -> triton.Config:
    return triton.Config(
        {"TILE_M": tile_m, "TILE_N": tile_n, "TILE_K": tile_k, "WARP_SPECIALIZE": warp_specialize},
        num_warps=num_warps,
        num_stages=num_stages,
    )


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


class PruneInvalidBatchedConfigsTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            "n": 512,
            "k": 256,
            "configs": [(64, 64, 64, 4, 3, False), (64, 64, 64, 4, 3, True), (128, 64, 64, 4, 2, False)],
            "expected": [(64, 64, 64, 4, 3, False)],
        },
        {
            "n": 1024,
            "k": 128,
            "configs": [(128, 128, 32, 4, 2, False), (64, 64, 64, 8, 4, True), (64, 64, 128, 4, 2, False)],
            "expected": [(128, 128, 32, 4, 2, False), (64, 64, 64, 8, 4, True)],
        },
        {
            "n": 2048,
            "k": 512,
            "configs": [(128, 128, 128, 4, 4, False), (128, 128, 64, 4, 2, False), (64, 64, 64, 4, 2, False)],
            "expected": [(128, 128, 64, 4, 2, False)],
        },
    )
    def test_prune_invalid_batched_configs(self, n: int, k: int, configs: list, expected: list):
        triton_configs = [_make_batched_config(*c) for c in configs]
        result = prune_invalid_batched_configs(triton_configs, {"N": n, "K": k})
        result_tuples = [
            (
                c.kwargs["TILE_M"],
                c.kwargs["TILE_N"],
                c.kwargs["TILE_K"],
                c.num_warps,
                c.num_stages,
                c.kwargs["WARP_SPECIALIZE"],
            )
            for c in result
        ]
        self.assertEqual(result_tuples, expected)

    def test_prune_invalid_batched_configs_falls_back_when_all_pruned(self):
        triton_configs = [_make_batched_config(128, 64, 64, 4, 2), _make_batched_config(64, 128, 64, 4, 2)]
        result = prune_invalid_batched_configs(triton_configs, {"N": 512, "K": 128})
        self.assertEqual(result, triton_configs)


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


class TsyrkExValidationTest(parameterized.TestCase):
    """Tests for input validation raises in tsyrk_ex and tsyrk_ex_small_matrix."""

    def setUp(self):
        self.device = FLAGS.device

    def test_tsyrk_ex_non_bf16_raises_type_error(self) -> None:
        a = torch.randn(4, 4, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "must be bfloat16"):
            triton_kernels.tsyrk_ex(a)

    def test_tsyrk_ex_non_2d_raises_type_error(self) -> None:
        a = torch.randn(4, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "must be 2D"):
            triton_kernels.tsyrk_ex(a)

    def test_tsyrk_ex_non_contiguous_raises_type_error(self) -> None:
        a = torch.randn(8, 8, device=self.device, dtype=torch.bfloat16)[:, ::2]
        with self.assertRaisesRegex(TypeError, "must be contiguous"):
            triton_kernels.tsyrk_ex(a)

    def test_tsyrk_ex_c_wrong_shape_raises_runtime_error(self) -> None:
        a = torch.randn(4, 4, device=self.device, dtype=torch.bfloat16)
        c = torch.randn(3, 3, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, r"c must be of shape \(N, N\)"):
            triton_kernels.tsyrk_ex(a, c, beta=1.0)

    def test_tsyrk_ex_c_non_contiguous_raises_runtime_error(self) -> None:
        a = torch.randn(8, 8, device=self.device, dtype=torch.bfloat16)
        c = torch.randn(16, 16, device=self.device, dtype=torch.bfloat16)[::2, ::2]
        with self.assertRaisesRegex(RuntimeError, "c or c.T must be contiguous"):
            triton_kernels.tsyrk_ex(a, c, beta=1.0)

    def test_tsyrk_ex_small_non_bf16_raises_type_error(self) -> None:
        a = torch.randn(4, 4, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "must be bfloat16"):
            triton_kernels.tsyrk_ex_small_matrix(a)

    def test_tsyrk_ex_small_non_2d_raises_type_error(self) -> None:
        a = torch.randn(4, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "must be 2D"):
            triton_kernels.tsyrk_ex_small_matrix(a)

    def test_tsyrk_ex_small_non_contiguous_raises_type_error(self) -> None:
        a = torch.randn(8, 8, device=self.device, dtype=torch.bfloat16)[:, ::2]
        with self.assertRaisesRegex(TypeError, "must be contiguous"):
            triton_kernels.tsyrk_ex_small_matrix(a)

    def test_tsyrk_ex_small_c_wrong_shape_raises_runtime_error(self) -> None:
        a = torch.randn(4, 4, device=self.device, dtype=torch.bfloat16)
        c = torch.randn(3, 3, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, r"c must be of shape \(N, N\)"):
            triton_kernels.tsyrk_ex_small_matrix(a, c, beta=1.0)

    def test_tsyrk_ex_small_c_non_contiguous_raises_runtime_error(self) -> None:
        a = torch.randn(8, 8, device=self.device, dtype=torch.bfloat16)
        c = torch.randn(16, 16, device=self.device, dtype=torch.bfloat16)[::2, ::2]
        with self.assertRaisesRegex(RuntimeError, "c or c.T must be contiguous"):
            triton_kernels.tsyrk_ex_small_matrix(a, c, beta=1.0)

    def test_tsyrk_ex_small_out_wrong_shape_raises_runtime_error(self) -> None:
        a = torch.randn(4, 4, device=self.device, dtype=torch.bfloat16)
        out = torch.empty(3, 3, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "out must be same shape/device/dtype"):
            triton_kernels.tsyrk_ex_small_matrix(a, out=out)

    def test_tsyrk_ex_small_out_non_contiguous_raises_runtime_error(self) -> None:
        a = torch.randn(4, 4, device=self.device, dtype=torch.bfloat16)
        out = torch.empty(8, 8, device=self.device, dtype=torch.bfloat16)[::2, ::2]
        with self.assertRaisesRegex(RuntimeError, "out must be contiguous"):
            triton_kernels.tsyrk_ex_small_matrix(a, out=out)


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
        assert_equal(c, ref)

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
        assert_equal(c, ref)

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
        assert_equal(c, ref)

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
        assert_equal(c, ref)


class BatchedTsyrkExValidationTest(parameterized.TestCase):
    """Tests for input validation in batched_tsyrk_ex and can_use_batched_tsyrk."""

    def setUp(self):
        self.device = FLAGS.device

    def test_batched_tsyrk_ex_non_bf16_raises_type_error(self) -> None:
        a = torch.randn(2, 16, 16, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "must be bfloat16"):
            triton_kernels.batched_tsyrk_ex(a)

    def test_batched_tsyrk_ex_non_3d_raises_type_error(self) -> None:
        a = torch.randn(16, 16, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "must be 3D"):
            triton_kernels.batched_tsyrk_ex(a)

    def test_batched_tsyrk_ex_non_contiguous_raises_type_error(self) -> None:
        a = torch.randn(2, 16, 32, device=self.device, dtype=torch.bfloat16)[:, :, ::2]
        with self.assertRaisesRegex(TypeError, "must be contiguous"):
            triton_kernels.batched_tsyrk_ex(a)

    @parameterized.parameters(
        (2, 12, 16),
        (2, 16, 12),
    )
    def test_batched_tsyrk_ex_unaligned_shape_raises_runtime_error(self, b: int, n: int, k: int) -> None:
        a = torch.randn(b, n, k, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "must be multiples of 8"):
            triton_kernels.batched_tsyrk_ex(a)

    def test_batched_tsyrk_ex_c_wrong_shape_raises_runtime_error(self) -> None:
        a = torch.randn(2, 16, 16, device=self.device, dtype=torch.bfloat16)
        c = torch.randn(2, 8, 8, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, r"c must be of shape \(B, N, N\)"):
            triton_kernels.batched_tsyrk_ex(a, c, beta=1.0)

    def test_batched_tsyrk_ex_c_non_contiguous_raises_runtime_error(self) -> None:
        a = torch.randn(2, 16, 16, device=self.device, dtype=torch.bfloat16)
        c = torch.randn(2, 32, 32, device=self.device, dtype=torch.bfloat16)[:, ::2, ::2]
        with self.assertRaisesRegex(RuntimeError, "c or c.mT must be contiguous"):
            triton_kernels.batched_tsyrk_ex(a, c, beta=1.0)

    def test_batched_tsyrk_ex_out_wrong_shape_raises_runtime_error(self) -> None:
        a = torch.randn(2, 16, 16, device=self.device, dtype=torch.bfloat16)
        out = torch.empty(2, 8, 8, device=self.device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "out must be same shape/device/dtype"):
            triton_kernels.batched_tsyrk_ex(a, out=out)

    def test_batched_tsyrk_ex_out_non_contiguous_raises_runtime_error(self) -> None:
        a = torch.randn(2, 16, 16, device=self.device, dtype=torch.bfloat16)
        out = torch.empty(2, 32, 32, device=self.device, dtype=torch.bfloat16)[:, ::2, ::2]
        with self.assertRaisesRegex(RuntimeError, "out must be contiguous"):
            triton_kernels.batched_tsyrk_ex(a, out=out)

    def test_can_use_batched_tsyrk(self) -> None:
        a = torch.randn(2, 128, 64, device=self.device, dtype=torch.bfloat16)
        c = torch.randn(2, 128, 128, device=self.device, dtype=torch.bfloat16)
        cases = {
            "contiguous": ((a, None), True),
            "transposed": ((torch.randn(2, 64, 128, device=self.device, dtype=torch.bfloat16).mT, None), True),
            "symmetric_c": ((a, c), True),
            "non_bf16": ((a.float(), None), False),
            "non_3d": ((a[0], None), False),
            "unaligned_n": ((torch.randn(2, 12, 64, device=self.device, dtype=torch.bfloat16), None), False),
            "unaligned_k": ((torch.randn(2, 128, 12, device=self.device, dtype=torch.bfloat16), None), False),
            "non_contiguous": ((c[:, :, ::2], None), False),
            "c_wrong_shape": ((a, torch.randn(2, 64, 64, device=self.device, dtype=torch.bfloat16)), False),
            "c_wrong_dtype": ((a, c.float()), False),
            "c_non_contiguous": (
                (a, torch.randn(2, 256, 256, device=self.device, dtype=torch.bfloat16)[:, ::2, ::2]),
                False,
            ),
        }
        for name, ((a_case, c_case), expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(triton_kernels.can_use_batched_tsyrk(a_case, c_case), expected)


class BatchedTsyrkIntegerInputTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        ({"b": 2, "n": 128, "k": 128}, {"b": 4, "n": 256, "k": 64}),
        ({"trans": False}, {"trans": True}),
    )
    def test_batched_tsyrk_ex_match_bmm(self, b: int, n: int, k: int, trans: bool):
        a = torch.randint(-3, 3, (b, n, k), device=self.device, dtype=torch.bfloat16)
        a_warmup = torch.randint_like(a, -3, 3)
        if trans:
            a = a.mT
            a_warmup = a_warmup.mT
        ref = torch.bmm(a, a.mT)
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.batched_tsyrk_ex(a_warmup)
        d = triton_kernels.batched_tsyrk_ex(a)
        assert_equal(d, ref)

    @parameterized.product(
        ({"b": 2, "n": 128, "alpha": 0.5, "beta": 0.5}, {"b": 4, "n": 256, "alpha": 0.25, "beta": 0.25}),
        ({"trans": False}, {"trans": True}),
    )
    def test_batched_tsyrk_ex_match_baddbmm(self, b: int, n: int, alpha: float, beta: float, trans: bool):
        a = torch.randint(-3, 3, (b, n, n), device=self.device, dtype=torch.bfloat16)
        # make symmetric input matrices.
        a = a + a.mT
        a_warmup = torch.randint_like(a, -3, 3)
        ref = torch.baddbmm(a, a, a, alpha=alpha, beta=beta)
        if trans:
            a = a.mT
            a_warmup = a_warmup.mT
        # warmup the triton kernel to avoid the wrong result from the first run.
        _ = triton_kernels.batched_tsyrk_ex(a_warmup, a_warmup, alpha=alpha, beta=beta)
        d = triton_kernels.batched_tsyrk_ex(a, a, alpha=alpha, beta=beta)
        assert_equal(d, ref)


if __name__ == "__main__":
    absltest.main()
