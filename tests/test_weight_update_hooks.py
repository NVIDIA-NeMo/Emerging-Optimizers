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
from emerging_optimizers.weight_update_hooks import (
    HyperballHook,
    NoOpWeightUpdateHook,
    RadialBrakeHook,
    RelativeUpdateHook,
)


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS
_SHAPES = ((1,), (2, 3), (2, 2, 2))


def _single_nonzero(shape: tuple[int, ...], value: float, device: str) -> torch.Tensor:
    tensor = torch.zeros(shape, device=device)
    tensor.flatten()[0] = value
    return tensor


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
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                hook = NoOpWeightUpdateHook()
                param = _single_nonzero(shape, 8.0, self.device)
                update = _single_nonzero(shape, -4.0, self.device)
                param_before = param.clone()
                update_before = update.clone()

                pre_update_state = hook.pre_weight_update_inplace(param, update)
                hook.post_weight_update_inplace(param, pre_update_state)

                assert_equal(param, param_before)
                assert_equal(update, update_before)

    def test_radial_brake_retains_configured_fraction_for_multiple_shapes(self) -> None:
        cases = (
            ("outward", 8.0, 0.5, 1.0, 8.0 + 0.5 * (16.0 - 8.0)),
            ("inward", -4.0, 1.0, 0.5, 8.0 + 0.5 * (4.0 - 8.0)),
        )
        for shape in _SHAPES:
            for name, update_value, outward_scale, inward_scale, expected_value in cases:
                with self.subTest(shape=shape, direction=name):
                    hook = RadialBrakeHook(outward_scale=outward_scale, inward_scale=inward_scale)
                    param = _single_nonzero(shape, 8.0, self.device)
                    update = _single_nonzero(shape, update_value, self.device)

                    pre_update_state = hook.pre_weight_update_inplace(param, update)
                    param.add_(update)
                    hook.post_weight_update_inplace(param, pre_update_state)

                    assert_equal(param, _single_nonzero(shape, expected_value, self.device))

    def test_radial_brake_eps_boundary(self) -> None:
        cases = (("below", 0.125, 1.0), ("equal", 0.25, 1.0 + 0.5 * 0.25))
        for name, update_value, expected_value in cases:
            with self.subTest(position=name):
                hook = RadialBrakeHook(outward_scale=0.5, eps=0.25)
                param = _single_nonzero((2, 3), 1.0, self.device)
                update = _single_nonzero((2, 3), update_value, self.device)

                pre_update_state = hook.pre_weight_update_inplace(param, update)
                param.add_(update)
                hook.post_weight_update_inplace(param, pre_update_state)

                assert_equal(param, _single_nonzero((2, 3), expected_value, self.device))

    def test_constructor_validation(self) -> None:
        cases = (
            ("hyperball eps", lambda: HyperballHook(radius=1.0, eps=0.0), "eps must be finite and positive"),
            ("radial brake eps", lambda: RadialBrakeHook(eps=0.0), "eps must be finite and positive"),
            ("relative update eps", lambda: RelativeUpdateHook(eps=0.0), "eps must be positive"),
            (
                "hyperball radius",
                lambda: HyperballHook(radius=1e-4, eps=1e-3),
                "radius must be finite and at least eps",
            ),
            ("outward scale", lambda: RadialBrakeHook(outward_scale=1.1), "outward_scale"),
            ("inward scale", lambda: RadialBrakeHook(inward_scale=1.1), "inward_scale"),
        )
        for name, constructor, error_regex in cases:
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, error_regex):
                constructor()

    def test_hyperball_scales_update_and_weight_exactly_for_multiple_shapes(self) -> None:
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                hook = HyperballHook(radius=4.0)
                param = _single_nonzero(shape, 8.0, self.device)
                update = _single_nonzero(shape, 8.0, self.device)
                expected = _single_nonzero(shape, 4.0, self.device)

                pre_update_state = hook.pre_weight_update_inplace(param, update)
                assert_equal(update, expected)

                param.add_(update, alpha=-1.0)
                hook.post_weight_update_inplace(param, pre_update_state)
                assert_equal(param, expected)

    def test_hyperball_handles_eps_boundary_for_pre_and_post_updates(self) -> None:
        cases = (
            ("pre below", "pre", 0.125, 0.0),
            ("pre equal", "pre", 0.25, 2.0),
            ("post below", "post", 0.125, 0.0),
        )
        for name, stage, value, expected_value in cases:
            with self.subTest(case=name):
                hook = HyperballHook(radius=2.0, eps=0.25)
                tensor = _single_nonzero((2, 2, 2), value, self.device)
                if stage == "pre":
                    hook.pre_weight_update_inplace(torch.empty_like(tensor), tensor)
                else:
                    hook.post_weight_update_inplace(tensor, None)

                assert_equal(tensor, _single_nonzero((2, 2, 2), expected_value, self.device))

    def test_hyperball_accepts_zero_weight_with_fixed_radius(self) -> None:
        hook = HyperballHook(radius=4.0)
        param = torch.zeros((2, 3), device=self.device)
        update = _single_nonzero((2, 3), 8.0, self.device)

        pre_update_state = hook.pre_weight_update_inplace(param, update)
        param.add_(update, alpha=-1.0)
        hook.post_weight_update_inplace(param, pre_update_state)

        assert_equal(param, _single_nonzero((2, 3), -4.0, self.device))

    def test_relative_update_scales_exactly_without_post_projection(self) -> None:
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                hook = RelativeUpdateHook()
                param = _single_nonzero(shape, 4.0, self.device)
                update = _single_nonzero(shape, 8.0, self.device)
                param_before = param.clone()

                pre_update_state = hook.pre_weight_update_inplace(param, update)
                assert_equal(param, param_before)
                assert_equal(update, _single_nonzero(shape, 4.0, self.device))

                param.add_(update, alpha=-0.5)
                param_after_update = param.clone()
                hook.post_weight_update_inplace(param, pre_update_state)
                assert_equal(param, param_after_update)

    def test_relative_update_eps_boundary(self) -> None:
        cases = (
            ("weight below", 0.125, 2.0, 0.0),
            ("update below", 2.0, 0.125, 0.0),
            ("update equal", 2.0, 0.25, 2.0),
        )
        for name, weight_value, update_value, expected_value in cases:
            with self.subTest(case=name):
                hook = RelativeUpdateHook(eps=0.25)
                param = _single_nonzero((2, 2, 2), weight_value, self.device)
                update = _single_nonzero((2, 2, 2), update_value, self.device)

                hook.pre_weight_update_inplace(param, update)

                assert_equal(update, _single_nonzero((2, 2, 2), expected_value, self.device))

    def test_orthogonalized_optimizer_applies_weight_update_hook(self) -> None:
        for shape in ((1, 1), (2, 3), (3, 2)):
            with self.subTest(shape=shape):
                param = _single_nonzero(shape, 8.0, self.device)
                param.grad = _single_nonzero(shape, 8.0, self.device)
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
                assert_equal(param, _single_nonzero(shape, 12.0, self.device))


if __name__ == "__main__":
    absltest.main()
