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
import sys

import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.soap import soap, tp_utils
from emerging_optimizers.utils import get_pg_rank, get_pg_size


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class AllGatherGradAndKroneckerFactorsTpCpuTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.tp_group = torch.distributed.group.WORLD
        self.world_size = get_pg_size(self.tp_group)
        self.rank = get_pg_rank(self.tp_group)

    @parameterized.product(
        shape=((16, 32), (32, 16), (96, 200)),
        partition_dim=(0, 1),
    )
    def test_matches_non_distributed(self, shape, partition_dim):
        m, n = shape
        # Grad is sharded along partition_dim; both kronecker factors are sharded along dim 0,
        # so m (rows of L) and n (rows of R) must each be divisible by the group size.
        assert m % self.world_size == 0 and n % self.world_size == 0, "shape must be divisible by world size"
        full_grad = torch.randint(-5, 5, shape)
        full_l = torch.randint(-5, 5, (m, m))
        full_r = torch.randint(-5, 5, (n, n))
        # All-reduce ensures that every rank starts from the same tensors.
        torch.distributed.all_reduce(full_grad)
        torch.distributed.all_reduce(full_l)
        torch.distributed.all_reduce(full_r)

        local_grad = full_grad.chunk(self.world_size, dim=partition_dim)[self.rank].contiguous()
        local_l = full_l.chunk(self.world_size, dim=0)[self.rank].contiguous()
        local_r = full_r.chunk(self.world_size, dim=0)[self.rank].contiguous()

        gathered_grad, gathered_factors = tp_utils.all_gather_grad_and_kronecker_factors_tp(
            kronecker_factor_list=[local_l, local_r],
            grad=local_grad,
            partition_dim=partition_dim,
            tp_group=self.tp_group,
        )

        torch.testing.assert_close(gathered_grad, full_grad, atol=0, rtol=0)
        torch.testing.assert_close(gathered_factors[0], full_l, atol=0, rtol=0)
        torch.testing.assert_close(gathered_factors[1], full_r, atol=0, rtol=0)

    @parameterized.product(
        shape=((16, 32), (32, 16), (96, 200)),
        partition_dim=(0, 1),
    )
    def test_updated_factors_match_non_distributed(self, shape, partition_dim):
        m, n = shape
        assert m % self.world_size == 0 and n % self.world_size == 0, "shape must be divisible by world size"
        # lerp_ requires a floating-point dtype, but values stay integer-valued so both paths
        # produce bit-identical results.
        full_grad = torch.randint(-5, 5, shape, dtype=torch.float32)
        full_l = torch.randint(-5, 5, (m, m), dtype=torch.float32)
        full_r = torch.randint(-5, 5, (n, n), dtype=torch.float32)
        torch.distributed.all_reduce(full_grad)
        torch.distributed.all_reduce(full_l)
        torch.distributed.all_reduce(full_r)

        shampoo_beta = 0.95

        ref_l = full_l.clone()
        ref_r = full_r.clone()
        soap.update_kronecker_factors([ref_l, ref_r], full_grad, shampoo_beta=shampoo_beta)

        local_grad = full_grad.chunk(self.world_size, dim=partition_dim)[self.rank].contiguous()
        local_l = full_l.chunk(self.world_size, dim=0)[self.rank].contiguous()
        local_r = full_r.chunk(self.world_size, dim=0)[self.rank].contiguous()

        gathered_grad, gathered_factors = tp_utils.all_gather_grad_and_kronecker_factors_tp(
            kronecker_factor_list=[local_l, local_r],
            grad=local_grad,
            partition_dim=partition_dim,
            tp_group=self.tp_group,
        )
        soap.update_kronecker_factors(gathered_factors, gathered_grad, shampoo_beta=shampoo_beta)

        torch.testing.assert_close(gathered_factors[0], ref_l, atol=0, rtol=0)
        torch.testing.assert_close(gathered_factors[1], ref_r, atol=0, rtol=0)


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="gloo")
    torch.set_float32_matmul_precision("highest")

    rank = torch.distributed.get_rank()

    for i, arg in enumerate(sys.argv):
        if arg.startswith("--xml_output_file="):
            base, ext = os.path.splitext(arg)

            # Attach rank to the output file name
            sys.argv[i] = f"{base}_rank{rank}{ext}"
            break

    absltest.main()

    torch.distributed.destroy_process_group()
