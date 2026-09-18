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
from _comparison import assert_close_to_identity
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers import spel


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")

FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class SpelTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        weight_decay_method=["decoupled", "independent", "l2"],
        nesterov=[True, False],
    )
    def test_smoke(self, shape, weight_decay_method, nesterov) -> None:
        """Smoke test Spel optimizer with various shapes, weight decay methods, and Nesterov."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        spel_opt = spel.Spel(
            [test_param],
            weight_decay_method=weight_decay_method,
            nesterov=nesterov,
        )
        spel_opt.step()

    @parameterized.product(
        shape=[(16, 16), (32, 16), (64, 32)],
        scale_mode=["unit_spectral_norm", "unit_rms_to_rms_norm"],
    )
    def test_tangent_space_projection(self, shape, scale_mode) -> None:
        """Test that the gradient is properly projected onto the (scaled) tangent space.

        A vector V is in the tangent space at W if W^T V is skew-symmetric,
        meaning its symmetric part W^T V + (W^T V)^T = 0.
        """
        m, n = shape
        expected_scale = 1.0 if scale_mode == "unit_spectral_norm" else (m / n) ** 0.5

        # 1. Create a parameter matrix W on the appropriately scaled manifold
        w_data = torch.randn(shape, dtype=torch.float32, device=FLAGS.device)
        q, _ = torch.linalg.qr(w_data)
        test_param = nn.Parameter(q * expected_scale)

        # 2. Create a random Euclidean gradient G
        grad = torch.randn_like(test_param)

        spel_opt = spel.Spel([test_param], scale_mode=scale_mode)

        # 3. Bypass the LMO (Newton-Schulz) so we can inspect the raw tangent projection
        spel_opt.scaled_orthogonalize_fn = torch.nn.Identity()

        # 4. Extract the projected gradient V
        projected_grad = spel_opt.orthogonalize(test_param, grad)

        # 5. Verify skew-symmetry of W^T V
        wt_v = torch.matmul(test_param.transpose(-2, -1), projected_grad)
        sym_wt_v = 0.5 * (wt_v + wt_v.transpose(-2, -1))

        # The symmetric part should be practically zero
        torch.testing.assert_close(sym_wt_v, torch.zeros_like(sym_wt_v), atol=1e-5, rtol=1e-5)

    @parameterized.product(
        shape=[(16, 32), (33, 65), (127, 257)],
        scale_mode=["unit_spectral_norm", "unit_rms_to_rms_norm"],
    )
    def test_post_update_produces_appropriately_scaled_orthogonal_weights(self, shape, scale_mode) -> None:
        """Test that post_weight_update_fn_inplace enforces the correct manifold scale constraint."""
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))

        spel_opt = spel.Spel(
            [test_param],
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            scale_mode=scale_mode,
        )

        for _ in range(5):
            test_param.grad = torch.randn_like(test_param)
            spel_opt.step()

        W = test_param.data
        m, n = W.shape

        expected_scale_sq = 1.0 if scale_mode == "unit_spectral_norm" else (m / n)

        if m <= n:
            gram = W @ W.mT
        else:
            gram = W.mT @ W

        # Normalize the Gram matrix by the expected squared scale before asserting identity
        normalized_gram = gram / expected_scale_sq
        assert_close_to_identity(normalized_gram, diag_atol=0.06, off_diag_atol=0.06)


if __name__ == "__main__":
    absltest.main()
