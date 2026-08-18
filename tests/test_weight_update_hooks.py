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
import math

import torch
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer
from emerging_optimizers.weight_update_hooks import (
    HyperballHook,
    NoOpWeightUpdateHook,
    RadialBrakeHook,
    RelativeUpdateHook,
)


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def _single_nonzero_random_location(shape: tuple[int, ...], value: float, device: str) -> tuple[torch.Tensor, int]:
    """Build a tensor with ``value`` at one random flat index, returning the tensor and that index."""
    index = int(torch.randint(math.prod(shape), (1,)).item())
    tensor = torch.zeros(shape, device=device)
    tensor.flatten()[index] = value
    return tensor, index


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class WeightUpdateHooksTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = FLAGS.device

    @parameterized.parameters((1,), (2, 3), (3, 4, 5))
    def test_no_op_hook_leaves_update_and_param_unchanged(self, *shape: int) -> None:
        hook = NoOpWeightUpdateHook()
        param, index = _single_nonzero_random_location(shape, 8.0, self.device)
        update = torch.zeros_like(param)
        update.flatten()[index] = -4.0
        param_before = param.clone()
        update_before = update.clone()

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        hook.post_weight_update_inplace(param, pre_update_state)

        assert_equal(param, param_before)
        assert_equal(update, update_before)

    @parameterized.product(
        (
            dict(update_value=8.0, outward_scale=0.5, inward_scale=1.0, expected_value=8.0 + 0.5 * (16.0 - 8.0)),
            dict(update_value=-4.0, outward_scale=1.0, inward_scale=0.5, expected_value=8.0 + 0.5 * (4.0 - 8.0)),
        ),
        shape=((1,), (2, 3), (2, 2, 2)),
    )
    def test_radial_brake_retains_configured_fraction_for_multiple_shapes(
        self,
        shape: tuple[int, ...],
        update_value: float,
        outward_scale: float,
        inward_scale: float,
        expected_value: float,
    ) -> None:
        hook = RadialBrakeHook(outward_scale=outward_scale, inward_scale=inward_scale)
        param, index = _single_nonzero_random_location(shape, 8.0, self.device)
        update = torch.zeros_like(param)
        update.flatten()[index] = update_value
        expected = torch.zeros_like(param)
        expected.flatten()[index] = expected_value

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        assert_equal(param, expected)

    @parameterized.parameters(
        dict(update_value=0.125, expected_value=1.0),
        dict(update_value=0.25, expected_value=1.0 + 0.5 * 0.25),
    )
    def test_radial_brake_eps_boundary(self, update_value: float, expected_value: float) -> None:
        hook = RadialBrakeHook(outward_scale=0.5, eps=0.25)
        param, index = _single_nonzero_random_location((2, 3), 1.0, self.device)
        update = torch.zeros_like(param)
        update.flatten()[index] = update_value
        expected = torch.zeros_like(param)
        expected.flatten()[index] = expected_value

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update)
        hook.post_weight_update_inplace(param, pre_update_state)

        assert_equal(param, expected)

    @parameterized.parameters(
        dict(
            hook_cls=HyperballHook,
            hook_kwargs=dict(radius=1.0, eps=0.0),
            error_regex="eps must be finite and positive",
        ),
        dict(hook_cls=RadialBrakeHook, hook_kwargs=dict(eps=0.0), error_regex="eps must be finite and positive"),
        dict(hook_cls=RelativeUpdateHook, hook_kwargs=dict(eps=0.0), error_regex="eps must be positive"),
        dict(
            hook_cls=HyperballHook,
            hook_kwargs=dict(radius=1e-4, eps=1e-3),
            error_regex="radius must be finite and at least eps",
        ),
        dict(hook_cls=RadialBrakeHook, hook_kwargs=dict(outward_scale=1.1), error_regex="outward_scale"),
        dict(hook_cls=RadialBrakeHook, hook_kwargs=dict(inward_scale=1.1), error_regex="inward_scale"),
    )
    def test_constructor_validation(self, hook_cls: type, hook_kwargs: dict[str, float], error_regex: str) -> None:
        with self.assertRaisesRegex(ValueError, error_regex):
            hook_cls(**hook_kwargs)

    @parameterized.parameters((1,), (2, 3), (2, 2, 2))
    def test_hyperball_scales_update_and_weight_exactly_for_multiple_shapes(self, *shape: int) -> None:
        hook = HyperballHook(radius=4.0)
        param, index = _single_nonzero_random_location(shape, 8.0, self.device)
        update = param.clone()
        expected = torch.zeros_like(param)
        expected.flatten()[index] = 4.0

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        assert_equal(update, expected)

        param.add_(update, alpha=-1.0)
        hook.post_weight_update_inplace(param, pre_update_state)
        assert_equal(param, expected)

    @parameterized.parameters(
        dict(stage="pre", value=0.125, expected_value=0.0),
        dict(stage="pre", value=0.25, expected_value=2.0),
        dict(stage="post", value=0.125, expected_value=0.0),
    )
    def test_hyperball_handles_eps_boundary_for_pre_and_post_updates(
        self, stage: str, value: float, expected_value: float
    ) -> None:
        hook = HyperballHook(radius=2.0, eps=0.25)
        tensor, index = _single_nonzero_random_location((2, 2, 2), value, self.device)
        expected = torch.zeros_like(tensor)
        expected.flatten()[index] = expected_value

        if stage == "pre":
            hook.pre_weight_update_inplace(torch.empty_like(tensor), tensor)
        else:
            hook.post_weight_update_inplace(tensor, None)

        assert_equal(tensor, expected)

    def test_hyperball_accepts_zero_weight_with_fixed_radius(self) -> None:
        hook = HyperballHook(radius=4.0)
        update, index = _single_nonzero_random_location((2, 3), 8.0, self.device)
        param = torch.zeros_like(update)
        expected = torch.zeros_like(update)
        expected.flatten()[index] = -4.0

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update, alpha=-1.0)
        hook.post_weight_update_inplace(param, pre_update_state)

        assert_equal(param, expected)

    @parameterized.parameters((1,), (2, 3), (2, 2, 2))
    def test_relative_update_scales_exactly_without_post_projection(self, *shape: int) -> None:
        hook = RelativeUpdateHook()
        param, index = _single_nonzero_random_location(shape, 4.0, self.device)
        update = torch.zeros_like(param)
        update.flatten()[index] = 8.0
        expected_update = param.clone()
        param_before = param.clone()

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        assert_equal(param, param_before)
        assert_equal(update, expected_update)

        param.add_(update, alpha=-0.5)
        param_after_update = param.clone()
        hook.post_weight_update_inplace(param, pre_update_state)
        assert_equal(param, param_after_update)

    @parameterized.parameters(
        dict(weight_value=0.125, update_value=2.0, expected_value=0.0),
        dict(weight_value=2.0, update_value=0.125, expected_value=0.0),
        dict(weight_value=2.0, update_value=0.25, expected_value=2.0),
    )
    def test_relative_update_eps_boundary(
        self, weight_value: float, update_value: float, expected_value: float
    ) -> None:
        hook = RelativeUpdateHook(eps=0.25)
        param, index = _single_nonzero_random_location((2, 2, 2), weight_value, self.device)
        update = torch.zeros_like(param)
        update.flatten()[index] = update_value
        expected = torch.zeros_like(param)
        expected.flatten()[index] = expected_value

        hook.pre_weight_update_inplace(param, update)

        assert_equal(update, expected)

    @parameterized.parameters((1, 1), (2, 3), (3, 2))
    def test_orthogonalized_optimizer_applies_weight_update_hook(self, *shape: int) -> None:
        param, index = _single_nonzero_random_location(shape, 8.0, self.device)
        param.grad = param.clone()
        expected = torch.zeros_like(param)
        expected.flatten()[index] = 12.0
        optimizer = OrthogonalizedOptimizer(
            [param],
            lr=-1.0,
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            weight_decay_method="l2",
            fp32_matmul_prec="highest",
            scaled_orthogonalize_fn=torch.nn.Identity(),
            weight_update_hook=RadialBrakeHook(outward_scale=0.5),
        )

        optimizer.step()

        # The raw update doubles the norm from 8 to 16; the hook retains half of that increase.
        assert_equal(param, expected)


if __name__ == "__main__":
    absltest.main()
