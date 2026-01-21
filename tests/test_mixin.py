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
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.mixin import WeightUpdateMixin


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS


class DummyOptimizer(WeightUpdateMixin):
    """Dummy optimizer class to test WeightUpdateMixin functionality."""

    def __init__(self, weight_update_method="sgd"):
        self.weight_update_method = weight_update_method


class TestWeightUpdateMixin(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device
        logging.info(f"Using device: {self.device}")
        torch.manual_seed(42)

    @parameterized.product(
        shape=[(10, 10), (50, 20), (100,), (5, 5, 5)],
        lr=[0.01, 0.1, 1.0],
    )
    def test_sgd_update(self, shape, lr):
        """Test standard SGD update: W_{t+1} = W_t - lr * update."""

        optimizer = DummyOptimizer(weight_update_method="sgd")

        # Create parameter and update tensors
        p = torch.randn(shape, device=self.device, dtype=torch.float32)
        update = torch.randn(shape, device=self.device, dtype=torch.float32)

        # Store original values
        p_original = p.clone()
        update_original = update.clone()

        # Apply update
        optimizer._apply_weight_update_inplace(p, update, lr)

        # Verify SGD formula: p_new = p_old - lr * update
        expected = p_original - lr * update_original
        torch.testing.assert_close(p, expected, rtol=1e-5, atol=1e-5)

        logging.info(f"SGD update test passed for shape {shape} with lr={lr}")

    @parameterized.product(
        shape=[(10, 10), (50, 20), (100,), (5, 5, 5)],
        lr=[0.01, 0.1, 0.5],
    )
    def test_hyperball_preserves_norm(self, shape, lr):
        """Test that hyperball update preserves the Frobenius norm of parameters."""

        optimizer = DummyOptimizer(weight_update_method="hyperball")

        # Create parameter and update tensors
        p = torch.randn(shape, device=self.device, dtype=torch.float32)
        update = torch.randn(shape, device=self.device, dtype=torch.float32)

        # Store original norm
        original_norm = p.norm().item()

        # Apply update
        optimizer._apply_weight_update_inplace(p, update, lr)

        # Verify norm is preserved
        new_norm = p.norm().item()

        logging.info(f"Hyperball update - Original norm: {original_norm:.6f}, New norm: {new_norm:.6f}")

        # Allow small numerical tolerance
        self.assertAlmostEqual(
            original_norm,
            new_norm,
            places=4,
            msg=f"Hyperball update should preserve norm, but got {original_norm:.6f} -> {new_norm:.6f}",
        )

    @parameterized.product(
        shape=[(10, 10), (50, 20), (100,)],
    )
    def test_hyperball_zero_update(self, shape):
        """Test hyperball update with zero update tensor."""

        optimizer = DummyOptimizer(weight_update_method="hyperball")

        # Create parameter tensor
        p = torch.randn(shape, device=self.device, dtype=torch.float32)
        p_original = p.clone()

        # Zero update
        update = torch.zeros(shape, device=self.device, dtype=torch.float32)

        # Apply update
        optimizer._apply_weight_update_inplace(p, update, lr=0.1)

        # With zero update, the parameter should remain unchanged (or very close)
        # since normalized_update will be 0 / eps and the update step will be minimal
        original_norm = p_original.norm().item()
        new_norm = p.norm().item()

        # Norm should be preserved
        self.assertAlmostEqual(original_norm, new_norm, places=4)

        logging.info(f"Hyperball zero update test passed for shape {shape}")

    @parameterized.product(
        shape=[(10, 10), (50, 20)],
        lr=[0.01, 0.1],
    )
    def test_hyperball_moves_on_sphere(self, shape, lr):
        """Test that hyperball update actually moves the parameter on the sphere."""

        optimizer = DummyOptimizer(weight_update_method="hyperball")

        # Create parameter and non-zero update tensors
        p = torch.randn(shape, device=self.device, dtype=torch.float32)
        update = torch.randn(shape, device=self.device, dtype=torch.float32)

        p_original = p.clone()

        # Apply update
        optimizer._apply_weight_update_inplace(p, update, lr)

        # Verify that parameter has changed (movement on the sphere)
        diff = torch.norm(p - p_original).item()

        logging.info(f"Hyperball update movement: {diff:.6f} for lr={lr}")

        # Should have moved (diff > 0)
        self.assertGreater(diff, 0.0, msg="Hyperball update should move the parameter")

        # But the norm should be preserved
        self.assertAlmostEqual(p_original.norm().item(), p.norm().item(), places=4)

    def test_sgd_with_zero_lr(self):
        """Test that SGD update with lr=0 doesn't change parameters."""

        optimizer = DummyOptimizer(weight_update_method="sgd")

        p = torch.randn(10, 10, device=self.device, dtype=torch.float32)
        update = torch.randn(10, 10, device=self.device, dtype=torch.float32)

        p_original = p.clone()

        # Apply update with lr=0
        optimizer._apply_weight_update_inplace(p, update, lr=0.0)

        # Parameters should be unchanged
        torch.testing.assert_close(p, p_original, rtol=1e-7, atol=1e-7)

        logging.info("SGD with lr=0 test passed")

    def test_hyperball_with_small_eps(self):
        """Test hyperball update with custom epsilon value."""

        optimizer = DummyOptimizer(weight_update_method="hyperball")

        p = torch.randn(10, 10, device=self.device, dtype=torch.float32)
        update = torch.randn(10, 10, device=self.device, dtype=torch.float32)

        original_norm = p.norm().item()

        # Apply update with custom eps
        optimizer._apply_weight_update_inplace(p, update, lr=0.1, eps=1e-10)

        new_norm = p.norm().item()

        # Norm should still be preserved
        self.assertAlmostEqual(original_norm, new_norm, places=4)

        logging.info("Hyperball with custom eps test passed")

    def test_invalid_weight_update_method(self):
        """Test that invalid weight update method raises ValueError."""

        optimizer = DummyOptimizer(weight_update_method="invalid_method")

        p = torch.randn(10, 10, device=self.device, dtype=torch.float32)
        update = torch.randn(10, 10, device=self.device, dtype=torch.float32)

        with self.assertRaises(ValueError) as context:
            optimizer._apply_weight_update_inplace(p, update, lr=0.1)

        self.assertIn("Invalid weight update method", str(context.exception))

        logging.info("Invalid weight update method test passed")

    @parameterized.product(
        weight_update_method=["sgd", "hyperball"],
    )
    def test_default_weight_update_method(self, weight_update_method):
        """Test that weight update method defaults work correctly."""

        optimizer = DummyOptimizer(weight_update_method=weight_update_method)

        p = torch.randn(10, 10, device=self.device, dtype=torch.float32)
        update = torch.randn(10, 10, device=self.device, dtype=torch.float32)

        # Should not raise any errors
        optimizer._apply_weight_update_inplace(p, update, lr=0.1)

        logging.info(f"Default method test passed for {weight_update_method}")

    @parameterized.product(
        shape=[(64, 128), (32, 32, 32)],
    )
    def test_hyperball_normalized_update_computation(self, shape):
        """Test that hyperball correctly normalizes the update."""

        optimizer = DummyOptimizer(weight_update_method="hyperball")

        # Create tensors with known magnitudes
        p = torch.ones(shape, device=self.device, dtype=torch.float32) * 2.0
        update = torch.ones(shape, device=self.device, dtype=torch.float32) * 5.0

        original_norm = p.norm().item()
        update_norm = update.norm().item()

        logging.info(f"Original param norm: {original_norm:.6f}")
        logging.info(f"Update norm: {update_norm:.6f}")

        # Apply hyperball update
        optimizer._apply_weight_update_inplace(p, update, lr=0.1)

        new_norm = p.norm().item()

        logging.info(f"New param norm: {new_norm:.6f}")

        # Verify norm preservation
        self.assertAlmostEqual(original_norm, new_norm, places=4)

        # Verify the parameter has changed
        self.assertGreater(
            torch.norm(p - torch.ones(shape, device=self.device) * 2.0).item(),
            0.0,
            msg="Parameter should have changed",
        )


if __name__ == "__main__":
    absltest.main()
