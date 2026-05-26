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

from emerging_optimizers.scalar_optimizers import update_functions


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

        update = update_functions.calculate_lion_update(grad, exp_avg, betas=(beta, beta))

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

        update = update_functions.calculate_lion_update(grad, exp_avg, betas=(beta1, beta2))

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


if __name__ == "__main__":
    testing.absltest.main()
