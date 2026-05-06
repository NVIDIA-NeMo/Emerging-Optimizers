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

import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.soap.rekls import REKLS, TpRekls
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


def tearDownModule() -> None:
    torch.distributed.destroy_process_group()


class TpReklsCpuTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.tp_group = torch.distributed.group.WORLD
        self.world_size = get_pg_size(self.tp_group)
        self.rank = get_pg_rank(self.tp_group)

    def test_5steps_matches_non_distributed_rekls(self):
        """Multiple TpRekls steps over multiple params (mixed partition_dim) must produce
        bit-identical updates to non-distributed REKLS for every param at every step.
        """
        # ``partition_dim=None`` exercises the replicated fallback path: param has no partition_dim
        # attribute, no all-gather, full-size L/R, full update applied directly.
        params_config = [
            {"shape": (16, 32), "partition_dim": 0},
            {"shape": (32, 16), "partition_dim": 1},
            {"shape": (96, 200), "partition_dim": 0},
            {"shape": (96, 200), "partition_dim": 1},
            {"shape": (24, 40), "partition_dim": None},
        ]
        for cfg in params_config:
            if cfg["partition_dim"] is None:
                continue
            m, n = cfg["shape"]
            assert m % self.world_size == 0 and n % self.world_size == 0, (
                f"shape {cfg['shape']} must be divisible by world size {self.world_size}"
            )

        # Initial param data — all-reduce so every rank starts from the same tensors.
        full_params_data = []
        for cfg in params_config:
            d = torch.randn(cfg["shape"])
            torch.distributed.all_reduce(d, group=self.tp_group)
            full_params_data.append(d)

        ref_params = [torch.nn.Parameter(d.clone()) for d in full_params_data]
        ref_optimizer = REKLS(ref_params, lr=1e-3)

        tp_params = []
        for cfg, d in zip(params_config, full_params_data):
            pd = cfg["partition_dim"]
            if pd is None:
                local_data = d.clone()
            else:
                local_data = d.chunk(self.world_size, dim=pd)[self.rank].contiguous()
            local_param = torch.nn.Parameter(local_data)
            if pd is not None:
                local_param.partition_dim = pd
            tp_params.append(local_param)
        tp_optimizer = TpRekls(tp_params, lr=1e-3, tp_group=self.tp_group)

        num_steps = 5
        for _ in range(num_steps):
            full_grads = []
            for cfg in params_config:
                g = torch.randn(cfg["shape"])
                torch.distributed.all_reduce(g, group=self.tp_group)
                full_grads.append(g)

            for ref_p, full_g in zip(ref_params, full_grads):
                ref_p.grad = full_g.clone()
            for tp_p, cfg, full_g in zip(tp_params, params_config, full_grads):
                pd = cfg["partition_dim"]
                if pd is None:
                    tp_p.grad = full_g.clone()
                else:
                    tp_p.grad = full_g.chunk(self.world_size, dim=pd)[self.rank].contiguous()

            ref_optimizer.step()
            tp_optimizer.step()

            for ref_p, tp_p, cfg in zip(ref_params, tp_params, params_config):
                pd = cfg["partition_dim"]
                if pd is None:
                    ref_local = ref_p.detach()
                else:
                    ref_local = ref_p.detach().chunk(self.world_size, dim=pd)[self.rank]
                torch.testing.assert_close(
                    tp_p.detach(),
                    ref_local,
                    atol=0,
                    rtol=0,
                )


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="gloo")
    torch.set_float32_matmul_precision("highest")

    rank = get_pg_rank(torch.distributed.group.WORLD)

    for i, arg in enumerate(sys.argv):
        if arg.startswith("--xml_output_file="):
            base, ext = os.path.splitext(arg)
            sys.argv[i] = f"{base}_rank{rank}{ext}"
            break

    absltest.main()
