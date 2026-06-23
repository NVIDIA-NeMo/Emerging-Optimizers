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

from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer
from emerging_optimizers.weight_update_hooks import Hyperball, NoOpWeightUpdateHook, RadialBrake


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class WeightUpdateHooksTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = FLAGS.device

    def test_no_op_hook_leaves_update_and_param_unchanged(self) -> None:
        hook = NoOpWeightUpdateHook()
        param = torch.tensor([3.0, 4.0], device=self.device)
        update = torch.tensor([1.0, -2.0], device=self.device)
        param_before = param.clone()
        update_before = update.clone()

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(param, param_before, atol=0.0, rtol=0.0)
        torch.testing.assert_close(update, update_before, atol=0.0, rtol=0.0)

    def test_radial_brake_dampens_outward_norm_change(self) -> None:
        hook = RadialBrake(outward_scale_factor=0.5, inward_scale_factor=1.0)
        param = torch.tensor([3.0, 4.0], device=self.device)
        update = torch.tensor([3.0, 4.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(7.5, device=self.device))

    def test_radial_brake_dampens_inward_norm_change(self) -> None:
        hook = RadialBrake(outward_scale_factor=1.0, inward_scale_factor=0.2)
        param = torch.tensor([6.0, 8.0], device=self.device)
        update = torch.tensor([-3.0, -4.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(9.0, device=self.device))

    def test_radial_brake_rejects_amplifying_scale_factors(self) -> None:
        with self.assertRaisesRegex(ValueError, "outward_scale_factor"):
            RadialBrake(outward_scale_factor=1.1)
        with self.assertRaisesRegex(ValueError, "inward_scale_factor"):
            RadialBrake(inward_scale_factor=1.1)

    def test_hyperball_normalizes_update_and_final_weight_norm(self) -> None:
        hook = Hyperball()
        param = torch.tensor([3.0, 4.0], device=self.device)
        update = torch.tensor([0.0, 10.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        torch.testing.assert_close(torch.linalg.vector_norm(update), torch.tensor(5.0, device=self.device))

        param.add_(update, alpha=-1.0)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(5.0, device=self.device))

    def test_hyperball_fixed_radius_allows_zero_norm_param(self) -> None:
        hook = Hyperball(radius=2.0)
        param = torch.zeros(2, device=self.device)
        update = torch.tensor([0.0, 3.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update, alpha=-1.0)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(2.0, device=self.device))

    def test_hyperball_dynamic_radius_rejects_zero_norm_param(self) -> None:
        hook = Hyperball()
        param = torch.zeros(2, device=self.device)
        update = torch.tensor([0.0, 3.0], device=self.device)

        with self.assertRaisesRegex(ValueError, "when radius is not fixed"):
            hook.pre_weight_update_inplace(param, update)

    def test_orthogonalized_optimizer_applies_weight_update_hook(self) -> None:
        param = torch.tensor([[3.0, 4.0]], device=self.device)
        param.grad = torch.tensor([[3.0, 4.0]], device=self.device)
        optimizer = OrthogonalizedOptimizer(
            [param],
            lr=-1.0,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            weight_decay_method="l2",
            fp32_matmul_prec="highest",
            scaled_orthogonalize_fn=torch.nn.Identity(),
            weight_update_hook=RadialBrake(outward_scale_factor=0.5),
        )

        optimizer.step()

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(7.5, device=self.device))


if __name__ == "__main__":
    absltest.main()
