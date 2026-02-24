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
import os

import numpy as np
import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers import muon_utils


flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class DistributedNewtonSchulzStepCpuTest(parameterized.TestCase):
    def setUp(self):
        self.coefs = 3.4445, -4.7750, 2.0315

    @parameterized.parameters(
        {"shape": (21, 16)},
        {"shape": (16, 32)},
    )
    def test_close_to_non_distributed(self, shape):
        x = torch.nn.functional.normalize(torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32), dim=(-2, -1))
        # All-reduce ensures that every rank gets the same x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_x = x.chunk(world_size, dim=1)[rank]

        dist_out = muon_utils.newton_schulz_step(local_x, *self.coefs, tp_group=torch.distributed.group.WORLD)

        ref_out = muon_utils.newton_schulz_step(x, *self.coefs)

        torch.testing.assert_close(ref_out.chunk(world_size, dim=1)[rank], dist_out)

    @absltest.skipIf(int(os.environ.get("WORLD_SIZE", 1)) < 4, "test requires at least 2 ranks")
    @parameterized.product(
        shape=((21, 16), (16, 32)),
        tp_size=(2, 4),
    )
    def test_with_partial_tp(self, shape, tp_size):
        x = torch.nn.functional.normalize(torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32), dim=(-2, -1))
        # All-reduce ensures that every rank gets the same x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

        num_tp_groups = torch.distributed.get_world_size() // tp_size
        tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
            np.split(np.arange(torch.distributed.get_world_size()), num_tp_groups)
        )
        assert tp_group.size() == tp_size
        local_x = x.chunk(tp_group.size(), dim=1)[tp_group.rank()]

        dist_out = muon_utils.newton_schulz_step(local_x, *self.coefs, tp_group=tp_group)
        ref_out = muon_utils.newton_schulz_step(x, *self.coefs)
        torch.testing.assert_close(ref_out.chunk(tp_group.size(), dim=1)[tp_group.rank()], dist_out)


class DistributedNewtonSchulzCpuTest(parameterized.TestCase):
    @parameterized.parameters(
        {"shape": (21, 16)},
        {"shape": (16, 32)},
    )
    def test_distributed_normalize_close_to_non_distributed(self, shape):
        x = torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32)
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_x = x.chunk(world_size, dim=1)[rank]

        dist_out = muon_utils.distributed_normalize_p2(local_x, eps=1e-7, group=torch.distributed.group.WORLD)
        ref_out = torch.nn.functional.normalize(x, dim=(-2, -1), eps=1e-7)

        torch.testing.assert_close(ref_out.chunk(world_size, dim=1)[rank], dist_out)

    @parameterized.parameters(
        {"shape": (3, 32)},
        {"shape": (5, 100)},
    )
    def test_1step_close_to_non_distributed(self, shape):
        x = torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32)
        # All-reduce ensures that every rank gets the same x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_x = x.chunk(world_size, dim=1)[rank]

        dist_out = muon_utils.newton_schulz(
            local_x, steps=1, coefficient_type="simple", tp_group=torch.distributed.group.WORLD
        )
        ref_out = muon_utils.newton_schulz(x, steps=1, coefficient_type="simple")
        torch.testing.assert_close(ref_out.chunk(world_size, dim=1)[rank], dist_out)

    @parameterized.parameters(
        {"shape": (32, 3), "transpose": True},
        {"shape": (5, 100), "transpose": False},
    )
    def test_5steps_with_transpose_close_to_non_distributed(self, shape, transpose):
        x = torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32)
        # All-reduce ensures that every rank gets the same x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        chunk_dim = 0 if transpose else 1
        local_x = x.chunk(world_size, dim=chunk_dim)[rank]

        dist_out = muon_utils.newton_schulz(
            local_x, steps=5, tp_group=torch.distributed.group.WORLD, transpose=transpose
        )
        ref_out = muon_utils.newton_schulz(x, steps=5, transpose=transpose)
        torch.testing.assert_close(ref_out.chunk(world_size, dim=chunk_dim)[rank], dist_out)

    @parameterized.parameters(
        {"shape": (32, 3), "transpose": True, "tp_size": 2},
        {"shape": (5, 100), "transpose": False, "tp_size": 4},
    )
    def test_1step_with_partial_tp_close_to_non_distributed(self, shape, transpose, tp_size):
        x = torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32)
        # All-reduce ensures that every rank gets the same x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

        num_tp_groups = torch.distributed.get_world_size() // tp_size
        tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
            np.split(np.arange(torch.distributed.get_world_size()), num_tp_groups)
        )
        assert tp_group.size() == tp_size

        chunk_dim = 0 if transpose else 1
        local_x = x.chunk(tp_group.size(), dim=chunk_dim)[tp_group.rank()]

        dist_out = muon_utils.newton_schulz(
            local_x, steps=1, coefficient_type="simple", tp_group=tp_group, transpose=transpose
        )
        ref_out = muon_utils.newton_schulz(x, steps=1, coefficient_type="simple", transpose=transpose)
        torch.testing.assert_close(ref_out.chunk(tp_group.size(), dim=chunk_dim)[tp_group.rank()], dist_out)


class TestTensorParallelNewtonSchulz(parameterized.TestCase):
    @parameterized.parameters(
        {"shape": (21, 16)},
        {"shape": (16, 32)},
    )
    def test_fall_back_to_non_tp(self, shape):
        x = torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32)

        test_out = muon_utils.newton_schulz_tp(
            x, steps=5, coefficient_type="quintic", partition_dim=None, tp_group=None
        )
        ref_out = muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic")

        torch.testing.assert_close(test_out, ref_out, atol=0, rtol=0)

    @parameterized.product(
        shape=((20, 16), (16, 32)),
        partition_dim=(0, 1),
        mode=("distributed", "duplicated"),
    )
    def test_1step_close_to_non_distributed(self, shape, partition_dim, mode):
        if shape[partition_dim] % torch.distributed.get_world_size() != 0:
            self.skipTest("Skipping because incompatible shape and world size")
        x = torch.randint(-5, 5, shape, device="cpu", dtype=torch.float32)
        # All-reduce ensures that every rank gets the same x
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_x = x.chunk(world_size, dim=partition_dim)[rank]

        dist_out = muon_utils.newton_schulz_tp(
            local_x,
            steps=1,
            coefficient_type="simple",
            tp_group=torch.distributed.group.WORLD,
            partition_dim=partition_dim,
            mode=mode,
        )

        ref_out = muon_utils.newton_schulz(x, steps=1, coefficient_type="simple")

        torch.testing.assert_close(ref_out.chunk(world_size, dim=partition_dim)[rank], dist_out, atol=1e-6, rtol=0)


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="gloo")
    torch.set_float32_matmul_precision("highest")
    absltest.main()

    torch.distributed.destroy_process_group()
