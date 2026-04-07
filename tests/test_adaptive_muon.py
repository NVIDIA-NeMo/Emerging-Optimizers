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
import torch.nn as nn
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers.adaptive_muon import (
    AdaptiveMuon,
)


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")

FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        moment2_method=["adamuon", "normuon"],
        nesterov=[True, False],
    )
    def test_smoke(self, shape, moment2_method, nesterov) -> None:
        """Smoke test AdaptiveMuon with both second moment methods."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        adaptive_opt = AdaptiveMuon(
            [test_param],
            lr=0.01,
            momentum=0.9,
            weight_decay=0.01,
            nesterov=nesterov,
            moment2_method=moment2_method,
        )
        adaptive_opt.step()

    @parameterized.parameters(
        {"shape": (8, 16), "moment2_method": "adamuon"},
        {"shape": (16, 8), "moment2_method": "normuon"},
    )
    def test_second_moment_matches_shapes(self, shape, moment2_method) -> None:
        """Test that second moment buffers are properly initialized."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        adaptive_opt = AdaptiveMuon(
            [test_param],
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            moment2_method=moment2_method,
        )

        # Run one step to initialize buffers
        adaptive_opt.step()

        # Check that second moment buffer was created
        state = adaptive_opt.state[test_param]
        self.assertIn("moment2_buffer", state)
        self.assertIn("momentum_buffer", state)

        # Check second moment buffer shape
        moment2 = state["moment2_buffer"]
        if moment2_method == "adamuon":
            # Full elementwise buffer
            self.assertEqual(moment2.shape, test_param.shape)
        elif moment2_method == "normuon":
            # Reduced shape buffer
            avg_dim = -1 if shape[-2] >= shape[-1] else -2
            expected_shape = list(shape)
            expected_shape[avg_dim] = 1
            self.assertEqual(list(moment2.shape), expected_shape)

    @parameterized.parameters(
        {"moment2_method": "adamuon"},
        {"moment2_method": "normuon"},
    )
    def test_non_2d_param_raises_value_error_in_adamuon_step(self, moment2_method) -> None:
        """Test that AdaptiveMuon raises ValueError for non-2D parameters during step."""
        test_param = nn.Parameter(torch.randn(8, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randn_like(test_param)

        adaptive_opt = AdaptiveMuon(
            [test_param],
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            moment2_method=moment2_method,
        )

        with self.assertRaisesRegex(ValueError, "only supports 2D"):
            adaptive_opt.step()

    def test_unknown_moment2_method_raise_type_error(self) -> None:
        """Test that AdaptiveMuon raises TypeError for unknown moment2_method."""
        test_param = nn.Parameter(torch.randint(-5, 5, (8, 16), dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        adaptive_opt = AdaptiveMuon(
            [test_param],
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            moment2_method="unknown",
        )

        with self.assertRaises(TypeError):
            adaptive_opt.step()


if __name__ == "__main__":
    absltest.main()
