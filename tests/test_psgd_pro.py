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


if __name__ == "__main__":
    absltest.main()
