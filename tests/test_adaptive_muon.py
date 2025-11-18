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
from absl import flags
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers.adaptive_muon import (
    AdaptiveMuon,
)


flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS


class AdaptiveMuonTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        second_moment_method=["adamuon", "normuon"],
        use_nesterov=[True, False],
    )
    def test_smoke(self, shape, second_moment_method, use_nesterov) -> None:
        """Smoke test AdaptiveMuon with both second moment methods."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        adaptive_opt = AdaptiveMuon(
            [test_param],
            lr=0.01,
            momentum_beta=0.9,
            weight_decay=0.01,
            use_nesterov=use_nesterov,
            moment2_method=second_moment_method,
            beta2=0.999,
            eps=1e-8,
            weight_decay_method="decoupled",
            fp32_matmul_prec="highest",
        )
        adaptive_opt.step()

    @parameterized.parameters(
        {"shape": (8, 16), "second_moment_method": "adamuon"},
        {"shape": (16, 8), "second_moment_method": "normuon"},
    )
    def test_second_moment_matches_shapes(self, shape, second_moment_method) -> None:
        """Test that second moment buffers are properly initialized."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        adaptive_opt = AdaptiveMuon(
            [test_param],
            lr=0.01,
            momentum_beta=0.9,
            weight_decay=0.0,
            use_nesterov=False,
            moment2_method=second_moment_method,
            beta2=0.999,
            eps=1e-8,
            weight_decay_method="decoupled",
            fp32_matmul_prec="highest",
        )

        # Run one step to initialize buffers
        adaptive_opt.step()

        # Check that second moment buffer was created
        state = adaptive_opt.state[test_param]
        self.assertIn("moment2_buffer", state)
        self.assertIn("momentum_buffer", state)

        # Check second moment buffer shape
        second_moment = state["moment2_buffer"]
        if second_moment_method == "adamuon":
            # Full elementwise buffer
            self.assertEqual(second_moment.shape, test_param.shape)
        elif second_moment_method == "normuon":
            # Reduced shape buffer
            avg_dim = -1 if shape[-2] >= shape[-1] else -2
            expected_shape = list(shape)
            expected_shape[avg_dim] = 1
            self.assertEqual(list(second_moment.shape), expected_shape)

    def test_unknown_moment2_method_raise_type_error(self) -> None:
        """Test that AdaptiveMuon raises TypeError for unknown moment2_method."""
        test_param = nn.Parameter(torch.randint(-5, 5, (8, 16), dtype=torch.float32, device=FLAGS.device))

        with self.assertRaises(TypeError):
            AdaptiveMuon(
                [test_param],
                lr=0.01,
                momentum_beta=0.9,
                weight_decay=0.0,
                use_nesterov=False,
                moment2_method=None,
                beta2=0.999,
                eps=1e-8,
                weight_decay_method="decoupled",
                fp32_matmul_prec="highest",
            )


if __name__ == "__main__":
    absltest.main()
