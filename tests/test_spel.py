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
import torch.nn as nn
from absl import flags
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers import spel


flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS


class SpelTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        weight_decay_method=["decoupled", "independent", "l2"],
        use_nesterov=[True, False],
    )
    def test_smoke(self, shape, weight_decay_method, use_nesterov) -> None:
        """Smoke test Spel optimizer with various shapes, weight decay methods, and Nesterov."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        spel_opt = spel.Spel(
            [test_param],
            weight_decay_method=weight_decay_method,
            use_nesterov=use_nesterov,
        )
        spel_opt.step()

    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
    )
    def test_post_update_produces_approximately_orthogonal_weights(self, shape) -> None:
        """Test that post_weight_update_fn_inplace produces approximately orthogonal matrices.

        After each optimizer step, the weight matrix W should satisfy W @ W^T ≈ I (up to scale)
        for the smaller dimension, which is the defining property of SPEL.
        """
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))

        spel_opt = spel.Spel(
            [test_param],
            lr=0.01,
            momentum_beta=0.0,
            weight_decay=0.0,
        )

        for _ in range(5):
            test_param.grad = torch.randn_like(test_param)
            spel_opt.step()

            # After post_weight_update_fn_inplace, W should be approximately orthogonal.
            # For an m x n matrix with m <= n: W @ W^T ≈ I_m
            # For an m x n matrix with m > n:  W^T @ W ≈ I_n
            W = test_param.data
            m, n = W.shape
            if m <= n:
                WWT = W @ W.mT
                eye = torch.eye(m, device=FLAGS.device)
            else:
                WWT = W.mT @ W
                eye = torch.eye(n, device=FLAGS.device)

            # Newton-Schulz normalizes the spectral norm to ~1, so WWT ≈ I
            torch.testing.assert_close(
                WWT,
                eye,
                atol=0.1,
                rtol=0.1,
                msg=f"Weight matrix of shape {shape} is not approximately orthogonal after step",
            )


if __name__ == "__main__":
    absltest.main()
