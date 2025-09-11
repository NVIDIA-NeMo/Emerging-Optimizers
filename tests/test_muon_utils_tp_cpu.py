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
from absl.testing import parameterized, absltest

import torch

from llm_shower.orthogonalized_optimizers import muon_utils


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
