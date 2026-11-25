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
import os
import sys
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils import (
    sinkhorn_balance,
    sinkhorn_balance_row_sharded,
)


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if not dist.is_initialized():
        if FLAGS.device == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend="nccl" if FLAGS.device == "cuda" else "gloo", timeout=timedelta(seconds=120))
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


def _bounds(rows: int, rank: int, size: int) -> tuple[int, int]:
    base, remainder = divmod(rows, size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


def _matrix(rows: int, columns: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed((FLAGS.seed or 123) + rows * columns)
    return torch.randn((rows, columns), generator=generator, device="cpu").to(device=FLAGS.device, dtype=dtype)


class DistributedSinkhornBalanceCpuTest(parameterized.TestCase):
    @parameterized.product(dtype=(torch.float32, torch.bfloat16, torch.float16), steps=(1, 3, 11))
    def test_uneven_locally_wide_shards_match_full_matrix(self, dtype, steps):
        rows = dist.get_world_size() + 5
        full = _matrix(rows, 4, dtype)
        start, end = _bounds(rows, dist.get_rank(), dist.get_world_size())
        local = full[start:end].clone().requires_grad_()
        before = local.detach().clone()
        actual = sinkhorn_balance_row_sharded(
            local, row_group=dist.group.WORLD, global_num_rows=rows, num_normalization_steps=steps
        )
        expected = sinkhorn_balance(full, num_normalization_steps=steps)[start:end]
        tolerance = 1e-5 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(local, before, atol=0, rtol=0)
        self.assertEqual(actual.shape, local.shape)
        self.assertEqual(actual.dtype, local.dtype)
        self.assertFalse(actual.requires_grad)

    @parameterized.parameters(False, True)
    def test_empty_shards_participate(self, all_rows_on_one_rank):
        full = _matrix(5, 4, torch.float32)
        rank, size = dist.get_rank(), dist.get_world_size()
        if all_rows_on_one_rank:
            start, end = (0, 5) if rank == 0 else (5, 5)
        else:
            start, end = _bounds(5, rank, size)
        actual = sinkhorn_balance_row_sharded(full[start:end], row_group=dist.group.WORLD, global_num_rows=5)
        torch.testing.assert_close(actual, sinkhorn_balance(full)[start:end], atol=1e-5, rtol=1e-5)

    def test_noncontiguous_shards(self):
        rows = dist.get_world_size() + 5
        full = _matrix(4, rows, torch.float32).T
        start, end = _bounds(rows, dist.get_rank(), dist.get_world_size())
        local = full[start:end]
        actual = sinkhorn_balance_row_sharded(local, row_group=dist.group.WORLD, global_num_rows=rows)
        torch.testing.assert_close(actual, sinkhorn_balance(full)[start:end], atol=1e-5, rtol=1e-5)

    def test_global_mask_and_zero_updates(self):
        rows = dist.get_world_size() + 5
        full = torch.ones((rows, 4), device=FLAGS.device)
        full[0] *= 1e6
        full[-1] = 0
        start, end = _bounds(rows, dist.get_rank(), dist.get_world_size())
        actual = sinkhorn_balance_row_sharded(full[start:end], row_group=dist.group.WORLD, global_num_rows=rows)
        torch.testing.assert_close(actual, sinkhorn_balance(full)[start:end], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            actual[max(1 - start, 0) :], torch.zeros_like(actual[max(1 - start, 0) :]), atol=0, rtol=0
        )
        zeros = torch.zeros_like(full[start:end])
        zero_result = sinkhorn_balance_row_sharded(zeros, row_group=dist.group.WORLD, global_num_rows=rows)
        torch.testing.assert_close(zero_result, zeros, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(zero_result).all())

    def test_mask_threshold_equality(self):
        full = torch.tensor([[1.0], [3.0], [0.0], [0.0]], device=FLAGS.device)
        start, end = _bounds(4, dist.get_rank(), dist.get_world_size())
        actual = sinkhorn_balance_row_sharded(
            full[start:end], row_group=dist.group.WORLD, global_num_rows=4, zero_row_threshold=1.0
        )
        expected = torch.tensor([[0.0], [1.0], [0.0], [0.0]], device=FLAGS.device)[start:end]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_replicated_domains_do_not_enter_statistics(self):
        size = dist.get_world_size()
        if size < 4 or size % 2:
            self.skipTest("requires an even world size of at least four")
        halves = [list(range(size // 2)), list(range(size // 2, size))]
        groups = [dist.new_group(ranks) for ranks in halves]
        group = groups[int(dist.get_rank() >= size // 2)]
        try:
            full = torch.tensor([[0.75], [1.25], [1.25], [0.75]], device=FLAGS.device)
            start, end = _bounds(4, dist.get_rank(group), dist.get_world_size(group))
            actual = sinkhorn_balance_row_sharded(
                full[start:end],
                row_group=group,
                global_num_rows=4,
                num_normalization_steps=1,
                zero_row_threshold=1.0,
            )
            expected = sinkhorn_balance(full, num_normalization_steps=1, zero_row_threshold=1.0)[start:end]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        finally:
            dist.destroy_process_group(group)

    def test_local_mode_has_no_collectives(self):
        full = torch.tensor([[10.0, 1.0], [10.0, 1.0], [1.0, 10.0], [1.0, 10.0]], device=FLAGS.device)
        rank, size = dist.get_rank(), dist.get_world_size()
        start, end = (0, 2) if rank == 0 else (2, 4) if rank == size - 1 and size > 1 else (0, 0)
        local = full[start:end]
        expected = torch.ones_like(local)
        with mock.patch.object(dist, "all_reduce", side_effect=AssertionError("local mode communicated")):
            actual = sinkhorn_balance_row_sharded(
                local, row_group=dist.group.WORLD, global_num_rows=4, use_shard_local_stats=True
            )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        if size > 1 and local.numel():
            global_reference = sinkhorn_balance(full)[start:end]
            self.assertFalse(torch.allclose(actual, global_reference))

    def test_none_group_does_not_use_world(self):
        full = _matrix(7, 4, torch.float32)
        with mock.patch.object(dist, "all_reduce", side_effect=AssertionError("None used WORLD")):
            actual = sinkhorn_balance_row_sharded(full, row_group=None, global_num_rows=7)
        torch.testing.assert_close(actual, sinkhorn_balance(full), atol=0, rtol=0)

    def test_dtensor_is_rejected(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        mesh = init_device_mesh(FLAGS.device, (dist.get_world_size(),))
        update = distribute_tensor(torch.ones((dist.get_world_size() * 4, 4), device=FLAGS.device), mesh, [Shard(0)])
        with self.assertRaisesRegex(ValueError, r"local tensor, not a DTensor.*update\.to_local\(\)"):
            sinkhorn_balance_row_sharded(update, row_group=dist.group.WORLD, global_num_rows=dist.get_world_size() * 4)

    def test_nonmember_group_is_rejected(self):
        if dist.get_world_size() < 2:
            self.skipTest("requires at least two ranks")
        group = dist.new_group([0])
        try:
            full = torch.ones((4, 1), device=FLAGS.device)
            if dist.get_rank() == 0:
                actual = sinkhorn_balance_row_sharded(full, row_group=group, global_num_rows=4)
                torch.testing.assert_close(actual, sinkhorn_balance(full), atol=0, rtol=0)
            else:
                with self.assertRaisesRegex(ValueError, "belong to row_group"):
                    sinkhorn_balance_row_sharded(full, row_group=group, global_num_rows=4)
        finally:
            if dist.get_rank() == 0:
                dist.destroy_process_group(group)


if __name__ == "__main__":
    torch.set_num_threads(1)
    rank = int(os.environ.get("RANK", "0"))
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--xml_output_file="):
            base, extension = os.path.splitext(arg)
            sys.argv[i] = f"{base}_rank{rank}{extension}"
            break
    try:
        absltest.main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
