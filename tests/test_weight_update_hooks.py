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
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest

from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer
from emerging_optimizers.weight_update_hooks import Hyperball, NoOpWeightUpdateHook, RadialBrake, RelativeUpdate


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

        assert_equal(param, param_before)
        assert_equal(update, update_before)

    def test_radial_brake_halves_outward_norm_increase(self) -> None:
        hook = RadialBrake(outward_scale=0.5, inward_scale=1.0)
        param = torch.tensor([3.0, 4.0], device=self.device)
        update = torch.tensor([3.0, 4.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(7.5, device=self.device))

    def test_radial_brake_retains_twenty_percent_of_inward_norm_decrease(self) -> None:
        hook = RadialBrake(outward_scale=1.0, inward_scale=0.2)
        param = torch.tensor([6.0, 8.0], device=self.device)
        update = torch.tensor([-3.0, -4.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(9.0, device=self.device))

    def test_radial_brake_ignores_norm_delta_below_eps(self) -> None:
        hook = RadialBrake(outward_scale=1.0, inward_scale=1.0, eps=1e-3)
        param = torch.tensor([1.0, 0.0], device=self.device)
        update = torch.tensor([5e-4, 0.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(1.0, device=self.device))

    def test_radial_brake_applies_outward_scale_to_norm_delta_equal_to_eps(self) -> None:
        hook = RadialBrake(outward_scale=0.5, inward_scale=1.0, eps=0.25)
        param = torch.tensor([1.0, 0.0], device=self.device)
        update = torch.tensor([0.25, 0.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(1.125, device=self.device))

    def test_radial_brake_rejects_amplifying_scales(self) -> None:
        with self.assertRaisesRegex(ValueError, "outward_scale"):
            RadialBrake(outward_scale=1.1)
        with self.assertRaisesRegex(ValueError, "inward_scale"):
            RadialBrake(inward_scale=1.1)

    def test_hyperball_scales_update_and_weight_to_radius(self) -> None:
        hook = Hyperball(radius=5.0)
        param = torch.tensor([3.0, 4.0], device=self.device)
        update = torch.tensor([0.0, 10.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        torch.testing.assert_close(torch.linalg.vector_norm(update), torch.tensor(5.0, device=self.device))

        param.add_(update, alpha=-1.0)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(5.0, device=self.device))

    def test_hyperball_projects_zero_weight_after_nonzero_update(self) -> None:
        hook = Hyperball(radius=2.0)
        param = torch.zeros(2, device=self.device)
        update = torch.tensor([0.0, 3.0], device=self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update, alpha=-1.0)
        hook.post_weight_update_inplace(param, pre_update_state)

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(2.0, device=self.device))

    def test_hyperball_sets_update_below_eps_to_zero(self) -> None:
        hook = Hyperball(radius=2.0, eps=1e-3)
        param = torch.tensor([2.0, 0.0], device=self.device)
        update = torch.tensor([5e-4, 0.0], device=self.device)

        hook.pre_weight_update_inplace(param, update)

        assert_equal(update, torch.zeros_like(update))

    def test_hyperball_scales_update_equal_to_eps_to_radius(self) -> None:
        hook = Hyperball(radius=2.0, eps=0.25)
        param = torch.tensor([2.0, 0.0], device=self.device)
        update = torch.tensor([0.25, 0.0], device=self.device)

        hook.pre_weight_update_inplace(param, update)

        torch.testing.assert_close(torch.linalg.vector_norm(update), torch.tensor(2.0, device=self.device))

    def test_hyperball_sets_weight_below_eps_to_zero(self) -> None:
        hook = Hyperball(radius=2.0, eps=1e-3)
        param = torch.tensor([5e-4, 0.0], device=self.device)

        hook.post_weight_update_inplace(param, None)

        assert_equal(param, torch.zeros_like(param))

    def test_hooks_reject_nonpositive_eps(self) -> None:
        with self.assertRaisesRegex(ValueError, "eps must be finite and positive"):
            Hyperball(radius=1.0, eps=0.0)
        with self.assertRaisesRegex(ValueError, "eps must be finite and positive"):
            RadialBrake(eps=0.0)

    def test_hyperball_rejects_radius_below_eps(self) -> None:
        with self.assertRaisesRegex(ValueError, "radius must be finite and at least eps"):
            Hyperball(radius=1e-4, eps=1e-3)

    def test_relative_update_scales_update_to_weight_norm_without_post_projection(self) -> None:
        hook = RelativeUpdate()
        param = torch.tensor([3.0, 4.0], device=self.device)
        update = torch.tensor([0.0, 10.0], device=self.device)
        param_before = param.clone()

        pre_update_state = hook.pre_weight_update_inplace(param, update)

        assert_equal(param, param_before)
        torch.testing.assert_close(torch.linalg.vector_norm(update), torch.tensor(5.0, device=self.device))

        param.add_(update, alpha=-0.1)
        param_after_update = param.clone()
        hook.post_weight_update_inplace(param, pre_update_state)

        assert_equal(param, param_after_update)

    def test_relative_update_sets_update_below_eps_to_zero(self) -> None:
        hook = RelativeUpdate(eps=0.25)
        param = torch.tensor([2.0, 0.0], device=self.device)
        update = torch.tensor([0.125, 0.0], device=self.device)

        hook.pre_weight_update_inplace(param, update)

        assert_equal(update, torch.zeros_like(update))

    def test_relative_update_sets_update_to_zero_when_weight_is_below_eps(self) -> None:
        hook = RelativeUpdate(eps=0.25)
        param = torch.tensor([0.125, 0.0], device=self.device)
        update = torch.tensor([2.0, 0.0], device=self.device)

        hook.pre_weight_update_inplace(param, update)

        assert_equal(update, torch.zeros_like(update))

    def test_relative_update_scales_update_equal_to_eps(self) -> None:
        hook = RelativeUpdate(eps=0.25)
        param = torch.tensor([2.0, 0.0], device=self.device)
        update = torch.tensor([0.25, 0.0], device=self.device)

        hook.pre_weight_update_inplace(param, update)

        torch.testing.assert_close(torch.linalg.vector_norm(update), torch.tensor(2.0, device=self.device))

    def test_relative_update_rejects_nonpositive_eps(self) -> None:
        with self.assertRaisesRegex(ValueError, "eps must be finite and positive"):
            RelativeUpdate(eps=0.0)

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
            weight_update_hook=RadialBrake(outward_scale=0.5),
        )

        optimizer.step()

        torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(7.5, device=self.device))


if __name__ == "__main__":
    absltest.main()
