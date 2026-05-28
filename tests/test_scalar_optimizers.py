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
from absl import flags, logging, testing
from absl.testing import parameterized

from emerging_optimizers.scalar_optimizers import (
    LaProp,
    Lion,
    Signum,
    SimplifiedAdEMAMix,
    update_functions,
)


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")

FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class UpdateFunctionTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
    )
    def test_calculate_adam_update_simple(self, shape) -> None:
        exp_avg_initial = torch.full(shape, 1.0, device=self.device)
        exp_avg_sq_initial = torch.full(shape, 2.0, device=self.device)
        grad = torch.full(shape, 0.5, device=self.device)

        betas = (0.9, 0.99)
        eps = 1e-8
        step = 10
        nesterov = False
        lr = 0.25

        exp_avg_for_manual_calc = exp_avg_initial.clone()
        exp_avg_sq_for_manual_calc = exp_avg_sq_initial.clone()

        manual_update_value = update_functions.calculate_adam_update(
            grad,
            exp_avg_for_manual_calc,
            exp_avg_sq_for_manual_calc,
            betas=betas,
            eps=eps,
            correct_bias=True,
            nesterov=nesterov,
            step=step,
        )

        initial_param_val_tensor = torch.full(shape, 10.0, device=self.device)
        param = torch.nn.Parameter(initial_param_val_tensor.clone())
        param.grad = grad.clone()

        adam_optimizer = torch.optim.Adam(
            [param],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0,
            amsgrad=False,
        )

        # Manually set Adam's internal state to match conditions *before* the current update
        adam_optimizer.state[param]["step"] = torch.tensor(float(step - 1), device=self.device)
        adam_optimizer.state[param]["exp_avg"] = exp_avg_initial.clone()
        adam_optimizer.state[param]["exp_avg_sq"] = exp_avg_sq_initial.clone()

        adam_optimizer.step()

        # exp_avg_for_manual_calc and exp_avg_sq_for_manual_calc are the expected m_t and v_t from the torch Adam optimizer
        torch.testing.assert_close(
            adam_optimizer.state[param]["exp_avg"], exp_avg_for_manual_calc, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            adam_optimizer.state[param]["exp_avg_sq"], exp_avg_sq_for_manual_calc, atol=1e-6, rtol=1e-6
        )

        # manual_update_value is the calculated update term (= m_hat_t / (sqrt(v_hat_t) + eps))
        # With lr=lr, new_param = old_param - lr * manual_update_value
        expected_param_val_after_step = initial_param_val_tensor - lr * manual_update_value
        torch.testing.assert_close(param.data, expected_param_val_after_step, atol=1e-6, rtol=1e-6)

    def test_calculate_laprop_update_with_zero_momentum_equals_rmsprop(self) -> None:
        # LaProp with momentum (beta1) = 0 should be equivalent to RMSProp.
        exp_avg_initial = torch.tensor([[0.0]], device=self.device)  # Momentum is 0, so exp_avg starts at 0
        exp_avg_sq_initial = torch.tensor([[2.0]], device=self.device)
        grad = torch.tensor([[0.5]], device=self.device)
        betas = (0.0, 0.99)  # beta1=0 for momentum
        eps = 1e-8
        step = 10
        correct_bias = False
        lr = 0.25
        exp_avg_for_laprop = exp_avg_initial.clone()
        exp_avg_sq_for_laprop = exp_avg_sq_initial.clone()

        # Calculate LaProp update
        laprop_update = update_functions.calculate_laprop_update(
            grad,
            exp_avg_for_laprop,
            exp_avg_sq_for_laprop,
            betas=betas,
            eps=eps,
            correct_bias=correct_bias,
            step=step,
        )

        # Manually verify with RMSProp logic
        initial_param_val_tensor = torch.tensor([[10.0]], device=self.device)
        param = torch.nn.Parameter(initial_param_val_tensor.clone())
        param.grad = grad.clone()

        rmsprop_optimizer = torch.optim.RMSprop(
            [param],
            lr=lr,
            alpha=betas[1],
            eps=eps,
            weight_decay=0,
            momentum=0,
            centered=False,
        )

        # Manually set RMSProp's internal state
        rmsprop_optimizer.state[param]["step"] = torch.tensor(float(step), device=self.device)
        rmsprop_optimizer.state[param]["square_avg"] = exp_avg_sq_initial.clone()
        rmsprop_optimizer.state[param]["momentum_buffer"] = exp_avg_initial.clone()

        rmsprop_optimizer.step()

        # The LaProp update should be equivalent to the change in param calculated by RMSProp
        # Note: RMSProp internally calculates the update and applies it.
        # update = grad / (sqrt(square_avg) + eps)
        # new_param = old_param - lr * update
        # With lr=lr, the change is just the update value * lr.
        expected_param_val_after_step = initial_param_val_tensor - lr * laprop_update
        torch.testing.assert_close(param.data, expected_param_val_after_step, atol=1e-6, rtol=1e-6)

    @parameterized.parameters(
        {"correct_bias": True, "num_beta_slow_warmup_steps": None},
        {"correct_bias": False, "num_beta_slow_warmup_steps": 2},
    )
    def test_calculate_ademamix_update_with_alpha_zero_equals_adam(
        self, correct_bias: bool, num_beta_slow_warmup_steps: int | None
    ) -> None:
        # AdEMAMix with alpha=0 and no beta scheduling should be equivalent to Adam.
        exp_avg_fast_initial = torch.tensor([[1.0]], device=self.device)
        exp_avg_slow_initial = torch.tensor([[1.0]], device=self.device)
        exp_avg_sq_initial = torch.tensor([[2.0]], device=self.device)
        grad = torch.tensor([[0.5]], device=self.device)
        betas = (0.9, 0.99, 0.999)
        eps = 1e-8
        step = 10

        # Calculate AdEMAMix update
        exp_avg_fast_for_ademamix = exp_avg_fast_initial.clone()
        exp_avg_slow_for_ademamix = exp_avg_slow_initial.clone()
        exp_avg_sq_for_ademamix = exp_avg_sq_initial.clone()
        ademamix_update = update_functions.calculate_ademamix_update(
            grad,
            exp_avg_fast_for_ademamix,
            exp_avg_slow_for_ademamix,
            exp_avg_sq_for_ademamix,
            betas=betas,
            eps=eps,
            correct_bias=correct_bias,
            step=step,
            num_beta_slow_warmup_steps=num_beta_slow_warmup_steps,
            num_alpha_warmup_steps=None,
            alpha=0.0,
        )

        # Calculate Adam update
        exp_avg_for_adam = exp_avg_fast_initial.clone()
        exp_avg_sq_for_adam = exp_avg_sq_initial.clone()
        adam_update = update_functions.calculate_adam_update(
            grad,
            exp_avg_for_adam,
            exp_avg_sq_for_adam,
            betas=(betas[0], betas[1]),
            eps=eps,
            correct_bias=correct_bias,
            nesterov=False,
            step=step,
        )

        torch.testing.assert_close(ademamix_update, adam_update, atol=1e-6, rtol=1e-6)

    def test_calculate_sim_ademamix_update_with_zero_momentum_and_alpha_equals_rmsprop(self) -> None:
        # sim_ademamix with momentum (beta_fast) = 0 and alpha = 0 should be equivalent to RMSProp.
        exp_avg_initial = torch.tensor([[0.0]], device=self.device)  # Momentum is 0, so exp_avg starts at 0
        exp_avg_sq_initial = torch.tensor([[2.0]], device=self.device)
        grad = torch.tensor([[0.5]], device=self.device)
        betas = (0.0, 0.99)  # beta1=0 for momentum
        eps = 1e-8
        correct_bias = False
        step = 10
        lr = 0.25
        exp_avg_for_sim_ademamix = exp_avg_initial.clone()
        exp_avg_sq_for_sim_ademamix = exp_avg_sq_initial.clone()

        # Calculate LaProp update
        sim_ademamix_update = update_functions.calculate_sim_ademamix_update(
            grad,
            exp_avg_for_sim_ademamix,
            exp_avg_sq_for_sim_ademamix,
            betas=betas,
            eps=eps,
            correct_bias=correct_bias,
            step=step,
            num_beta_fast_warmup_steps=None,
            min_beta_fast=0.0,
            alpha=0.0,
        )

        # Manually verify with RMSProp logic
        initial_param_val_tensor = torch.tensor([[10.0]], device=self.device)
        param = torch.nn.Parameter(initial_param_val_tensor.clone())
        param.grad = grad.clone()

        rmsprop_optimizer = torch.optim.RMSprop(
            [param],
            lr=lr,
            alpha=betas[1],
            eps=eps,
            weight_decay=0,
            momentum=0,
            centered=correct_bias,
        )

        # Manually set RMSProp's internal state
        rmsprop_optimizer.state[param]["step"] = torch.tensor(float(step), device=self.device)
        rmsprop_optimizer.state[param]["square_avg"] = exp_avg_sq_initial.clone()

        rmsprop_optimizer.step()

        # The sim_ademamix update should be equivalent to the change in param calculated by RMSProp
        # Note: RMSProp internally calculates the update and applies it.
        # update = grad / (sqrt(square_avg) + eps)
        # new_param = old_param - lr * update
        # With lr=lr, the change is just the update value * lr.
        expected_param_val_after_step = initial_param_val_tensor - lr * sim_ademamix_update
        torch.testing.assert_close(param.data, expected_param_val_after_step, atol=1e-6, rtol=1e-6)

    @parameterized.product(
        shape=[(3, 3), (15, 31)],
        momentum=[0.9, 0.99],
        correct_bias=[True, False],
        nesterov=[True, False],
        step=[1, 5],
    )
    def test_calculate_signum_update_returns_sign(self, shape, momentum, correct_bias, nesterov, step) -> None:
        """Signum output should be +1 or -1 everywhere (the sign of the momentum)."""
        # generate random numbers that are not 0 centered
        grad = torch.rand(shape, device=self.device) - 0.3
        exp_avg = torch.lerp(torch.randn(shape, device=self.device), grad, 1 - momentum)

        update = update_functions.calculate_signum_update(
            grad, exp_avg, momentum=momentum, correct_bias=correct_bias, nesterov=nesterov, step=step
        )

        torch.testing.assert_close(update.abs(), torch.ones(shape, device=self.device), atol=0, rtol=0)

    def test_calculate_signum_with_shape_scaling_returns_sign(self) -> None:
        shape = (8, 12)
        momentum = 0.5
        grad = torch.rand(shape, device=self.device) - 0.3
        exp_avg = torch.lerp(torch.randn(shape, device=self.device), grad, 1 - momentum)

        update_abs = update_functions.calculate_signum_update(
            grad,
            exp_avg,
            momentum=momentum,
            correct_bias=False,
            nesterov=False,
            step=1,
            use_shape_scaling=True,
        ).abs()
        expected_update = torch.sign(exp_avg).abs() * (2 / (shape[0] + shape[1]))
        torch.testing.assert_close(update_abs, expected_update, atol=0, rtol=0)

    def test_calculate_lion_update_returns_sign(self) -> None:
        """Tests that Lion update returns sign of interpolated momentum."""
        shape = (8, 12)
        beta = 0.9
        grad = torch.randn(shape, device=self.device)
        exp_avg = torch.randn(shape, device=self.device)
        exp_avg_clone = exp_avg.clone()

        update = update_functions.calculate_lion_update(grad, exp_avg, betas=(beta, beta), step=1)

        # Update should be sign(beta * m + (1 - beta) * g)
        expected_update = torch.sign(beta * exp_avg_clone + (1 - beta) * grad)
        torch.testing.assert_close(update, expected_update, atol=0, rtol=0)

        # exp_avg should be updated in-place: lerp_(grad, 1 - beta)
        expected_exp_avg = torch.lerp(exp_avg_clone, grad, 1 - beta)
        torch.testing.assert_close(exp_avg, expected_exp_avg, atol=1e-6, rtol=1e-6)

    def test_calculate_lion_update_with_separate_betas(self) -> None:
        """Tests Lion with different beta1 and beta2."""
        shape = (4, 6)
        beta1, beta2 = 0.9, 0.99
        grad = torch.randn(shape, device=self.device)
        exp_avg = torch.randn(shape, device=self.device)
        exp_avg_clone = exp_avg.clone()

        update = update_functions.calculate_lion_update(grad, exp_avg, betas=(beta1, beta2), step=1)

        expected_update = torch.sign(beta1 * exp_avg_clone + (1 - beta1) * grad)
        torch.testing.assert_close(update, expected_update, atol=0, rtol=0)

        # With separate beta2, momentum uses beta2
        expected_exp_avg = torch.lerp(exp_avg_clone, grad, 1 - beta2)
        torch.testing.assert_close(exp_avg, expected_exp_avg, atol=1e-6, rtol=1e-6)


class MAdamUpdateTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    def test_calculate_madam_update_smoke(self) -> None:
        """Single MAdam step runs, mutates state in place, and returns a finite update."""
        shape = (4, 8)
        grad = torch.randn(shape, device=self.device)
        exp_avg = torch.zeros(shape, device=self.device)
        exp_avg_sq_scaled = torch.zeros(shape, device=self.device)

        update = update_functions.calculate_madam_update(
            grad,
            exp_avg,
            exp_avg_sq_scaled,
            betas=(0.9, 0.99),
            correct_bias=True,
            step=1,
            scale_log2=8,
        )

        self.assertEqual(update.shape, grad.shape)
        self.assertEqual(update.dtype, grad.dtype)
        self.assertTrue(torch.isfinite(update).all())
        # State should have been mutated in place.
        self.assertFalse(torch.all(exp_avg == 0))
        self.assertFalse(torch.all(exp_avg_sq_scaled == 0))

    @parameterized.product(
        scale_log2=[0, 8, 14],
        correct_bias=[True, False],
    )
    def test_calculate_madam_update_matches_adam_eps_zero(self, scale_log2: int, correct_bias: bool) -> None:
        shape = (5, 7)
        # Normal-magnitude gradient — no underflow path is exercised.
        grad = torch.randn(shape, device=self.device)
        betas = (0.9, 0.99)
        step = 3

        exp_avg_madam = torch.zeros(shape, device=self.device)
        exp_avg_sq_scaled = torch.zeros(shape, device=self.device)
        madam_update = update_functions.calculate_madam_update(
            grad,
            exp_avg_madam,
            exp_avg_sq_scaled,
            betas=betas,
            correct_bias=correct_bias,
            step=step,
            scale_log2=scale_log2,
        )

        exp_avg_adam = torch.zeros(shape, device=self.device)
        exp_avg_sq_adam = torch.zeros(shape, device=self.device)
        adam_update = update_functions.calculate_adam_update(
            grad,
            exp_avg_adam,
            exp_avg_sq_adam,
            betas=betas,
            eps=0.0,
            correct_bias=correct_bias,
            nesterov=False,
            step=step,
        )

        case = f"scale_log2={scale_log2}, correct_bias={correct_bias}"
        torch.testing.assert_close(
            madam_update,
            adam_update,
            atol=0,
            rtol=0,
            msg=lambda msg: f"MAdam vs Adam(eps=0) mismatch at {case}:\n\n{msg}",
        )
        # First-moment EMA is unaffected by scaling.
        torch.testing.assert_close(exp_avg_madam, exp_avg_adam, atol=0, rtol=0)
        # Second-moment storage differs by exactly the (power-of-two) scale.
        torch.testing.assert_close(
            exp_avg_sq_scaled,
            exp_avg_sq_adam * (2**scale_log2),
            atol=0,
            rtol=0,
            msg=lambda msg: f"exp_avg_sq_scaled != s * exp_avg_sq at {case}:\n\n{msg}",
        )

    def test_calculate_madam_update_5steps_zero_masked_is_finite(self) -> None:
        """Entries whose gradient has been exactly zero get a zero update, not NaN."""
        shape = (4, 8)
        grad = torch.randn(shape, device=self.device)
        # Force one column to have been zero from t=1 onward — for those entries
        # exp_avg and exp_avg_sq_scaled both stay 0, so the unmasked division
        # would be 0/0 = NaN.
        zero_col = 3
        grad[:, zero_col] = 0.0

        exp_avg = torch.zeros(shape, device=self.device)
        exp_avg_sq_scaled = torch.zeros(shape, device=self.device)

        for step in range(5):
            update = update_functions.calculate_madam_update(
                grad,
                exp_avg,
                exp_avg_sq_scaled,
                betas=(0.9, 0.99),
                correct_bias=True,
                step=step + 1,
                scale_log2=8,
            )

        # Masked column: exactly zero (not NaN/Inf).
        torch.testing.assert_close(
            update[:, zero_col],
            torch.zeros(shape[0], device=self.device, dtype=update.dtype),
            atol=0,
            rtol=0,
        )
        # Non-masked columns: finite and non-zero.
        nonzero_col_idx = [c for c in range(shape[1]) if c != zero_col]
        nonzero_cols = update[:, nonzero_col_idx]
        self.assertTrue(torch.isfinite(nonzero_cols).all())
        self.assertTrue((nonzero_cols.abs() > 0).all())


class _CommonScalarOptimizerTests:
    """Tests applied via mixin to every scalar optimizer test class.

    Concrete subclasses must additionally inherit from ``parameterized.TestCase``
    and set:

    - ``OPTIMIZER_CLS``: the optimizer class under test
    - ``STATE_KEYS``: tuple of per-parameter state keys (must include ``"step"``)
    """

    OPTIMIZER_CLS: type
    STATE_KEYS: tuple[str, ...]

    def setUp(self) -> None:
        self.device = FLAGS.device

    def test_smoke(self) -> None:
        param = torch.nn.Parameter(torch.randn(15, 31, device=self.device))
        opt = self.OPTIMIZER_CLS([param], lr=1e-4)
        param.grad = torch.randn_like(param)
        opt.step()

    def test_no_grad_no_update_params_unchanged(self) -> None:
        """Parameters without gradients are not updated."""
        param = torch.nn.Parameter(torch.randn(15, 31, device=self.device))
        original = param.detach().clone()
        opt = self.OPTIMIZER_CLS([param], lr=1e-4)
        opt.step()
        torch.testing.assert_close(param, original, atol=0, rtol=0)

    def test_state_keys_after_first_step(self) -> None:
        """First step populates exactly the expected state keys, with step==1 and matching-shape buffers."""
        param = torch.nn.Parameter(torch.randn(5, 7, device=self.device))
        param.grad = torch.randn_like(param)
        opt = self.OPTIMIZER_CLS([param], lr=1e-4)
        opt.step()
        self.assertEqual(set(opt.state[param].keys()), set(self.STATE_KEYS))
        self.assertEqual(opt.state[param]["step"], 1)
        for key in self.STATE_KEYS:
            if key == "step":
                continue
            self.assertEqual(opt.state[param][key].shape, param.data.shape)

    def test_init_group_skip_non_grad_params(self) -> None:
        """``_init_group(..., skip_non_grad_params=...)`` honors the flag."""
        with_grad = torch.nn.Parameter(torch.randn(5, 7, device=self.device))
        without_grad = torch.nn.Parameter(torch.randn(5, 7, device=self.device))
        with_grad.grad = torch.randn_like(with_grad)

        for skip in (True, False):
            with self.subTest(skip_non_grad_params=skip):
                opt = self.OPTIMIZER_CLS([with_grad, without_grad], lr=1e-4)
                opt._init_group(opt.param_groups[0], skip_non_grad_params=skip)
                for key in self.STATE_KEYS:
                    self.assertIn(key, opt.state[with_grad])
                self.assertEqual(opt.state[with_grad]["step"], 0)
                self.assertEqual("exp_avg" in opt.state[without_grad], not skip)

    def test_param_groups_large_lr_moves_more(self) -> None:
        """A param group with larger lr moves farther after one step."""
        shape = (15, 31)
        p1 = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        p2 = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        opt = self.OPTIMIZER_CLS(
            [{"params": [p1], "lr": 0.01}, {"params": [p2], "lr": 0.001}],
            weight_decay=0.0,
        )
        p1_orig = p1.detach().clone()
        p2_orig = p2.detach().clone()
        grad = torch.randint(1, 5, shape, device=self.device, dtype=torch.float32)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt.step()
        self.assertGreater(
            (p1.data - p1_orig).abs().mean().item(),
            (p2.data - p2_orig).abs().mean().item(),
        )

    def test_closure_unsupported(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        param.grad = torch.randn_like(param)
        opt = self.OPTIMIZER_CLS([param], lr=1e-4)
        with self.assertRaisesRegex(ValueError, "closure is not supported"):
            opt.step(closure=lambda: 0.0)

    def test_negative_lr_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid learning rate"):
            self.OPTIMIZER_CLS([param], lr=-1.0)

    def test_negative_weight_decay_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay"):
            self.OPTIMIZER_CLS([param], weight_decay=-0.1)


class _HasBetasTests:
    """Mixed into optimizer test classes whose ``__init__`` accepts ``betas=(b1, b2)``."""

    OPTIMIZER_CLS: type

    def test_beta0_out_of_range_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid beta at index 0"):
            self.OPTIMIZER_CLS([param], betas=(1.0, 0.99))

    def test_beta1_out_of_range_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid beta at index 1"):
            self.OPTIMIZER_CLS([param], betas=(0.9, 1.0))


class _HasEpsTests:
    """Mixed into optimizer test classes whose ``__init__`` accepts ``eps``."""

    OPTIMIZER_CLS: type

    def test_negative_eps_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid epsilon"):
            self.OPTIMIZER_CLS([param], eps=-1e-8)


class LionOptimizerTest(_CommonScalarOptimizerTests, _HasBetasTests, parameterized.TestCase):
    OPTIMIZER_CLS = Lion
    STATE_KEYS = ("exp_avg", "step")

    @parameterized.product(
        betas=[(0.9, 0.99), (0.95, 0.98)],
        shape=[(3, 3), (15, 31), (127, 255)],
    )
    def test_update_is_sign_based(self, betas, shape) -> None:
        """Lion updates should be +/- lr (sign-based)."""
        param = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        opt = Lion([param], lr=0.25, betas=betas, weight_decay=0.0)
        param.grad = torch.randint(1, 5, shape, device=self.device, dtype=torch.float32)
        old_param = param.data.clone()
        opt.step()
        diff = old_param - param.data
        torch.testing.assert_close(diff.abs(), torch.full_like(diff, 0.25), atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_exp_avg_evolves_correctly(self, shape) -> None:
        """``exp_avg`` matches the analytical EMA after three deterministic steps."""
        beta1, beta2 = 0.9, 0.99
        param = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        opt = Lion([param], lr=0.01, betas=(beta1, beta2), weight_decay=0.0)
        grads = [
            torch.randint(-3, 3, shape, device=self.device, dtype=torch.float32),
            torch.randint(-3, 3, shape, device=self.device, dtype=torch.float32),
            torch.randint(-3, 3, shape, device=self.device, dtype=torch.float32),
        ]
        expected_exp_avg = torch.zeros(*shape, device=self.device)
        for grad in grads:
            param.grad = grad.clone()
            opt.step()
            expected_exp_avg = beta2 * expected_exp_avg + (1 - beta2) * grad
        torch.testing.assert_close(opt.state[param]["exp_avg"], expected_exp_avg, atol=1e-6, rtol=1e-6)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
    )
    def test_weight_decay_decoupled_matches_analytical(self, shape) -> None:
        lr, wd = 0.25, 0.5
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        opt = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="decoupled")
        param.grad = torch.zeros(*shape, device=self.device)
        old_param = param.data.clone()
        opt.step()
        torch.testing.assert_close(param.data, old_param * (1 - lr * wd), atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
    )
    def test_weight_decay_independent_matches_analytical(self, shape) -> None:
        lr, wd = 0.25, 0.5
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        opt = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="independent")
        param.grad = torch.zeros(*shape, device=self.device)
        old_param = param.data.clone()
        opt.step()
        torch.testing.assert_close(param.data, old_param * (1 - wd), atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
    )
    def test_weight_decay_l2(self, shape) -> None:
        """L2 weight decay folds into the gradient before sign(); with zero grad it shrinks via WD."""
        lr, wd = 0.25, 0.5
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        opt = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="l2")
        param.grad = torch.zeros(*shape, device=self.device)
        old_param = param.data.clone()
        opt.step()
        torch.testing.assert_close(param.data, old_param - lr, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
    )
    def test_weight_decay_l2_masked_by_gradient(self, shape) -> None:
        """A large negative grad dominates the sign so L2 cannot guarantee shrinkage."""
        lr, wd = 0.25, 0.125
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        opt = Lion([param], lr=lr, weight_decay=wd, weight_decay_method="l2")
        param.grad = torch.randint(-10, -5, shape, device=self.device, dtype=torch.float32)
        old_param = param.data.clone()
        opt.step()
        torch.testing.assert_close(param.data, old_param + lr, atol=0, rtol=0)


class SignumOptimizerTest(_CommonScalarOptimizerTests, parameterized.TestCase):
    OPTIMIZER_CLS = Signum
    STATE_KEYS = ("exp_avg", "step")

    def test_invalid_momentum_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        for invalid in (-0.1, 1.0, 1.5):
            with self.subTest(momentum=invalid):
                with self.assertRaisesRegex(ValueError, "Invalid momentum"):
                    Signum([param], momentum=invalid)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
    )
    def test_update_is_sign_based(self, shape) -> None:
        """With ``use_shape_scaling=False`` and a positive gradient, Signum updates are +/- lr."""
        param = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        opt = Signum(
            [param],
            lr=0.25,
            momentum=0.9,
            weight_decay=0.0,
            correct_bias=True,
            nesterov=False,
            use_shape_scaling=False,
        )
        param.grad = torch.randint(1, 5, shape, device=self.device, dtype=torch.float32)
        old_param = param.data.clone()
        opt.step()
        # bias_correction at step 1 cancels the (1-momentum) factor, so sign(corrected) = sign(grad) = +1.
        torch.testing.assert_close(old_param - param.data, torch.full(shape, 0.25, device=self.device), atol=0, rtol=0)


class LaPropOptimizerTest(_CommonScalarOptimizerTests, _HasBetasTests, _HasEpsTests, parameterized.TestCase):
    OPTIMIZER_CLS = LaProp
    STATE_KEYS = ("exp_avg", "exp_avg_sq", "step")

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_state_evolves_correctly(self, shape) -> None:
        """After one step, ``exp_avg`` and ``exp_avg_sq`` match LaProp's analytical values."""
        beta1, beta2 = 0.5, 0.75
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        opt = LaProp([param], lr=0.25, betas=(beta1, beta2), weight_decay=0.0, correct_bias=True)
        grad = torch.randint_like(param, 1, 5)
        param.grad = grad.clone()
        opt.step()
        expected_exp_avg_sq = (1 - beta2) * grad.square()
        normalized_grad = grad / (grad.abs() + opt.param_groups[0]["eps"])
        expected_exp_avg = (1 - beta1) * normalized_grad
        torch.testing.assert_close(opt.state[param]["exp_avg_sq"], expected_exp_avg_sq, atol=0, rtol=0)
        torch.testing.assert_close(opt.state[param]["exp_avg"], expected_exp_avg, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_optimizer_step_matches_update_function(self, shape) -> None:
        """LaProp optimizer delegates update math to ``calculate_laprop_update``."""
        lr = 0.25
        betas = (0.5, 0.75)
        eps = 1e-8
        param = torch.nn.Parameter(torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32))
        grad = torch.randint(-5, 5, shape, device=self.device, dtype=torch.float32)
        opt = LaProp([param], lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        old_param = param.detach().clone()
        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        expected_update = update_functions.calculate_laprop_update(
            grad, exp_avg, exp_avg_sq, betas=betas, eps=eps, correct_bias=True, step=1
        )
        param.grad = grad.clone()
        opt.step()
        torch.testing.assert_close(param, old_param - lr * expected_update, atol=0, rtol=0)
        torch.testing.assert_close(opt.state[param]["exp_avg"], exp_avg, atol=0, rtol=0)
        torch.testing.assert_close(opt.state[param]["exp_avg_sq"], exp_avg_sq, atol=0, rtol=0)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
        {"shape": (127, 255)},
    )
    def test_frob_normalize_preserves_parameter_norm(self, shape) -> None:
        """LaProp with ``frob_normalize=True`` restores the pre-step Frobenius norm."""
        param = torch.nn.Parameter(torch.randint(1, 5, shape, device=self.device, dtype=torch.float32))
        opt = LaProp([param], lr=0.25, weight_decay=0.0, frob_normalize=True)
        param.grad = torch.randint(1, 5, shape, device=self.device, dtype=torch.float32)
        original_norm = param.norm()
        opt.step()
        torch.testing.assert_close(param.norm(), original_norm)


class SimplifiedAdEMAMixOptimizerTest(
    _CommonScalarOptimizerTests, _HasBetasTests, _HasEpsTests, parameterized.TestCase
):
    OPTIMIZER_CLS = SimplifiedAdEMAMix
    STATE_KEYS = ("exp_avg", "exp_avg_sq", "step")

    def test_invalid_min_beta_fast_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid min_beta_fast"):
            SimplifiedAdEMAMix([param], min_beta_fast=1.0)

    def test_invalid_num_beta_fast_warmup_steps_raises_value_error(self) -> None:
        param = torch.nn.Parameter(torch.randn(3, 3, device=self.device))
        with self.assertRaisesRegex(ValueError, "Invalid num_beta_fast_warmup_steps"):
            SimplifiedAdEMAMix([param], num_beta_fast_warmup_steps=-1)

    @parameterized.parameters(
        {"shape": (3, 3)},
        {"shape": (15, 31)},
    )
    def test_optimizer_step_matches_update_function(self, shape) -> None:
        """SimplifiedAdEMAMix optimizer delegates update math to ``calculate_sim_ademamix_update``."""
        lr = 0.25
        betas = (0.9999, 0.999)
        eps = 1e-8
        min_beta_fast = 0.9
        alpha = 2.0
        param = torch.nn.Parameter(torch.randn(*shape, device=self.device))
        grad = torch.randn_like(param)
        opt = SimplifiedAdEMAMix(
            [param],
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            min_beta_fast=min_beta_fast,
            alpha=alpha,
        )
        old_param = param.detach().clone()
        exp_avg = torch.zeros_like(param)
        exp_avg_sq = torch.zeros_like(param)
        expected_update = update_functions.calculate_sim_ademamix_update(
            grad,
            exp_avg,
            exp_avg_sq,
            betas=betas,
            eps=eps,
            correct_bias=True,
            step=1,
            num_beta_fast_warmup_steps=None,
            min_beta_fast=min_beta_fast,
            alpha=alpha,
        )
        param.grad = grad.clone()
        opt.step()
        torch.testing.assert_close(param, old_param - lr * expected_update)
        torch.testing.assert_close(opt.state[param]["exp_avg"], exp_avg)
        torch.testing.assert_close(opt.state[param]["exp_avg_sq"], exp_avg_sq)


if __name__ == "__main__":
    testing.absltest.main()
