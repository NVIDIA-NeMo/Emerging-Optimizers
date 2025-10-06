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
from absl import flags
from absl.testing import absltest, parameterized

from emerging_optimizers.scalar_optimizers import (
    calculate_adam_update,
    calculate_ademamix_update,
    calculate_laprop_update,
    calculate_sim_ademamix_update,
)


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu', 'cuda', or 'auto'")
flags.DEFINE_integer("seed", 42, "Random seed for reproducible tests")
flags.DEFINE_boolean("skip_gpu_tests", False, "Skip GPU tests even if CUDA is available")

FLAGS = flags.FLAGS


# Base class for tests requiring seeding for determinism
class BaseTestCase(parameterized.TestCase):
    def setUp(self):
        """Set random seed and device before each test."""
        # Set seed for PyTorch (using seed from flags)
        torch.manual_seed(FLAGS.seed)
        # Set seed for CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)

        # Set up device based on flags
        self.device = self._get_test_device()

    def _get_test_device(self):
        """Get the device to use for testing based on flags."""
        if FLAGS.device == "auto":
            return "cuda" if torch.cuda.is_available() and not FLAGS.skip_gpu_tests else "cpu"
        elif FLAGS.device == "cuda":
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            if FLAGS.skip_gpu_tests:
                self.skipTest("GPU tests skipped by flag")
            return "cuda"
        else:
            return "cpu"

    def _move_to_device(self, *tensors):
        """Helper method to move tensors to the test device."""
        return tuple(tensor.to(self.device) for tensor in tensors)


class ScalarOptimizerTest(BaseTestCase):
    def test_calculate_adam_update_simple(self) -> None:
        exp_avg_initial = torch.tensor([[1.0]])
        exp_avg_sq_initial = torch.tensor([[2.0]])
        grad = torch.tensor([[0.5]])

        # Move tensors to the test device
        exp_avg_initial, exp_avg_sq_initial, grad = self._move_to_device(exp_avg_initial, exp_avg_sq_initial, grad)

        betas = (0.9, 0.99)
        eps = 1e-8
        step = 10
        use_nesterov = False
        lr = 0.25

        exp_avg_for_manual_calc = exp_avg_initial.clone()
        exp_avg_sq_for_manual_calc = exp_avg_sq_initial.clone()

        manual_update_value = calculate_adam_update(
            grad,
            exp_avg_for_manual_calc,
            exp_avg_sq_for_manual_calc,
            betas,
            correct_bias=True,
            use_nesterov=use_nesterov,
            step=step,
            eps=eps,
        )

        initial_param_val_tensor = torch.tensor([[10.0]]).to(self.device)
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
        adam_optimizer.state[param]["step"] = torch.tensor(float(step - 1))
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

        self.assertEqual(manual_update_value.shape, (1, 1))

    def test_calculate_laprop_update_with_zero_momentum_equals_rmsprop(self) -> None:
        # LaProp with momentum (beta1) = 0 should be equivalent to RMSProp.
        exp_avg_initial = torch.tensor([[0.0]])  # Momentum is 0, so exp_avg starts at 0
        exp_avg_sq_initial = torch.tensor([[2.0]])
        grad = torch.tensor([[0.5]])
        betas = (0.0, 0.99)  # beta1=0 for momentum
        eps = 1e-8
        step = 10
        correct_bias = False
        lr = 0.25
        exp_avg_for_laprop = exp_avg_initial.clone()
        exp_avg_sq_for_laprop = exp_avg_sq_initial.clone()

        # Calculate LaProp update
        laprop_update = calculate_laprop_update(
            grad,
            exp_avg_for_laprop,
            exp_avg_sq_for_laprop,
            correct_bias,
            betas,
            step,
            eps,
        )

        # Manually verify with RMSProp logic
        initial_param_val_tensor = torch.tensor([[10.0]])
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
        rmsprop_optimizer.state[param]["step"] = torch.tensor(float(step))
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

    def test_calculate_ademamix_update_with_alpha_zero_equals_adam(self) -> None:
        # AdEMAMix with alpha=0 and no beta scheduling should be equivalent to Adam.
        exp_avg_fast_initial = torch.tensor([[1.0]])
        exp_avg_slow_initial = torch.tensor([[1.0]])
        exp_avg_sq_initial = torch.tensor([[2.0]])
        grad = torch.tensor([[0.5]])
        betas = (0.9, 0.99, 0.999)
        eps = 1e-8
        step = 10
        correct_bias_manual = True

        # Calculate AdEMAMix update
        exp_avg_fast_for_ademamix = exp_avg_fast_initial.clone()
        exp_avg_slow_for_ademamix = exp_avg_slow_initial.clone()
        exp_avg_sq_for_ademamix = exp_avg_sq_initial.clone()
        ademamix_update = calculate_ademamix_update(
            grad,
            exp_avg_fast_for_ademamix,
            exp_avg_slow_for_ademamix,
            exp_avg_sq_for_ademamix,
            num_beta_slow_warmup_steps=None,
            num_alpha_warmup_steps=None,
            betas=betas,
            step=step,
            eps=eps,
            correct_bias=correct_bias_manual,
            alpha=0.0,
        )

        # Calculate Adam update
        exp_avg_for_adam = exp_avg_fast_initial.clone()
        exp_avg_sq_for_adam = exp_avg_sq_initial.clone()
        adam_update = calculate_adam_update(
            grad,
            exp_avg_for_adam,
            exp_avg_sq_for_adam,
            (betas[0], betas[1]),
            correct_bias=correct_bias_manual,
            use_nesterov=False,
            step=step,
            eps=eps,
        )

        torch.testing.assert_close(ademamix_update, adam_update, atol=1e-6, rtol=1e-6)

    def test_calculate_sim_ademamix_update_with_zero_momentum_and_alpha_equals_rmsprop(self) -> None:
        # sim_ademamix with momentum (beta_fast) = 0 and alpha = 0 should be equivalent to RMSProp.
        exp_avg_initial = torch.tensor([[0.0]])  # Momentum is 0, so exp_avg starts at 0
        exp_avg_sq_initial = torch.tensor([[2.0]])
        grad = torch.tensor([[0.5]])
        betas = (0.0, 0.99)  # beta1=0 for momentum
        eps = 1e-8
        step = 10
        correct_bias = False
        lr = 0.25
        exp_avg_for_sim_ademamix = exp_avg_initial.clone()
        exp_avg_sq_for_sim_ademamix = exp_avg_sq_initial.clone()

        # Calculate LaProp update
        sim_ademamix_update = calculate_sim_ademamix_update(
            grad,
            exp_avg_for_sim_ademamix,
            exp_avg_sq_for_sim_ademamix,
            num_beta_fast_warmup_steps=None,
            min_beta_fast=0.0,
            betas=betas,
            step=step,
            eps=eps,
            correct_bias=correct_bias,
            alpha=0.0,
        )

        # Manually verify with RMSProp logic
        initial_param_val_tensor = torch.tensor([[10.0]])
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
        rmsprop_optimizer.state[param]["step"] = torch.tensor(float(step))
        rmsprop_optimizer.state[param]["square_avg"] = exp_avg_sq_initial.clone()

        rmsprop_optimizer.step()

        # The sim_ademamix update should be equivalent to the change in param calculated by RMSProp
        # Note: RMSProp internally calculates the update and applies it.
        # update = grad / (sqrt(square_avg) + eps)
        # new_param = old_param - lr * update
        # With lr=lr, the change is just the update value * lr.
        expected_param_val_after_step = initial_param_val_tensor - lr * sim_ademamix_update
        torch.testing.assert_close(param.data, expected_param_val_after_step, atol=1e-6, rtol=1e-6)

    def test_device_functionality(self) -> None:
        """Test that tensors are correctly moved to the specified device."""
        # Create test tensors
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([[1.0], [2.0]])

        # Move to test device
        tensor1_device, tensor2_device = self._move_to_device(tensor1, tensor2)

        # Verify they are on the correct device
        self.assertEqual(str(tensor1_device.device), self.device)
        self.assertEqual(str(tensor2_device.device), self.device)

        # Verify values are preserved
        torch.testing.assert_close(tensor1_device.cpu(), tensor1, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(tensor2_device.cpu(), tensor2, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    absltest.main()
