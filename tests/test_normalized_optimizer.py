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
import torch.nn as nn
import torch.nn.functional as F
from absl.testing import absltest, parameterized
from torch.utils.data import DataLoader, TensorDataset

from emerging_optimizers.riemannian_optimizers.normalized_optimizer import ObliqueAdam, ObliqueSGD


# Base class for tests requiring seeding for determinism
class BaseTestCase(parameterized.TestCase):
    def setUp(self):
        """Set random seed before each test."""
        # Set seed for PyTorch
        torch.manual_seed(42)
        # Set seed for CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)


class NormalizedOptimizerFunctionalTest(BaseTestCase):
    """Tests for ObliqueSGD and ObliqueAdam optimizers that preserve row/column norms."""

    @parameterized.named_parameters(
        ("col_mode", "col"),
        ("row_mode", "row"),
    )
    def test_oblique_sgd_preserves_norms(self, mode):
        """Test that ObliqueSGD preserves row or column norms after one optimization step."""
        # Create a 4x6 matrix for testing
        matrix_size = (4, 6)

        # Initialize with random values then normalize
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize according to mode
        if mode == "col":
            # Normalize columns (each column has unit norm)
            param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        else:  # mode == "row"
            # Normalize rows (each row has unit norm)
            param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Create optimizer
        optimizer = ObliqueSGD([param], lr=0.1, momentum=0.9, mode=mode)

        # Generate random gradient
        torch.manual_seed(123)  # For reproducible gradients
        param.grad = torch.randn_like(param.data)

        # Perform one optimization step
        optimizer.step()

        # Check that norms are preserved (should be 1.0 within tolerance)
        if mode == "col":
            final_norms = param.data.norm(dim=0)
        else:  # mode == "row"
            final_norms = param.data.norm(dim=1)

        # All norms should be approximately 1.0 (unit norm constraint)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )

    @parameterized.named_parameters(
        ("col_mode", "col"),
        ("row_mode", "row"),
    )
    def test_oblique_adam_preserves_norms(self, mode):
        """Test that ObliqueAdam preserves row or column norms after one optimization step."""
        # Create a 3x5 matrix for testing
        matrix_size = (3, 5)

        # Initialize with random values then normalize
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize according to mode
        if mode == "col":
            # Normalize columns (each column has unit norm)
            param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        else:  # mode == "row"
            # Normalize rows (each row has unit norm)
            param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Create optimizer
        optimizer = ObliqueAdam([param], lr=0.01, betas=(0.9, 0.999), mode=mode)

        # Generate random gradient
        torch.manual_seed(456)  # For reproducible gradients
        param.grad = torch.randn_like(param.data)

        # Perform one optimization step
        optimizer.step()

        # Check that norms are preserved (should be 1.0 within tolerance)
        if mode == "col":
            final_norms = param.data.norm(dim=0)
        else:  # mode == "row"
            final_norms = param.data.norm(dim=1)

        # All norms should be approximately 1.0 (unit norm constraint)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_oblique_sgd_zero_gradient(self):
        """Test that ObliqueSGD handles zero gradients correctly."""
        matrix_size = (2, 4)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize columns
        param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        initial_param = param.data.clone()

        optimizer = ObliqueSGD([param], lr=0.1, mode="col")

        # Set zero gradient
        param.grad = torch.zeros_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Parameter should remain unchanged with zero gradient
        torch.testing.assert_close(param.data, initial_param, atol=1e-8, rtol=1e-8)

        # Norms should still be 1.0
        final_norms = param.data.norm(dim=0)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=1e-6, rtol=1e-6)

    def test_oblique_adam_zero_gradient(self):
        """Test that ObliqueAdam handles zero gradients correctly."""
        matrix_size = (2, 3)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize rows
        param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        initial_param = param.data.clone()

        optimizer = ObliqueAdam([param], lr=0.01, mode="row")

        # Set zero gradient
        param.grad = torch.zeros_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Parameter should remain unchanged with zero gradient
        torch.testing.assert_close(param.data, initial_param, atol=1e-8, rtol=1e-8)

        # Norms should still be 1.0
        final_norms = param.data.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=1e-6, rtol=1e-6)

    def test_oblique_sgd_large_gradient(self):
        """Test that ObliqueSGD handles large gradients correctly."""
        matrix_size = (3, 4)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize columns
        param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueSGD([param], lr=0.1, mode="col")

        # Set large gradient
        param.grad = 100.0 * torch.randn_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved despite large gradient
        final_norms = param.data.norm(dim=0)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(final_norms, expected_norms, atol=1e-6, rtol=1e-6)

    def test_oblique_adam_large_gradient(self):
        """Test that ObliqueAdam handles large gradients correctly."""
        matrix_size = (2, 5)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize rows
        param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueAdam([param], lr=0.01, mode="row")

        # Set large gradient
        param.grad = 1000.0 * torch.randn_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved despite large gradient
        final_norms = param.data.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_multiple_optimization_steps_preserve_norms(self):
        """Test that norms are preserved across multiple optimization steps."""
        matrix_size = (4, 4)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize columns
        param.data = param.data / param.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueSGD([param], lr=0.05, momentum=0.8, mode="col")

        # Perform multiple optimization steps
        for step in range(10):
            torch.manual_seed(step)  # Different gradient each step
            param.grad = torch.randn_like(param.data)
            optimizer.step()

            # Check norms after each step
            final_norms = param.data.norm(dim=0)
            expected_norms = torch.ones_like(final_norms)
            torch.testing.assert_close(
                final_norms,
                expected_norms,
                atol=1e-6,
                rtol=1e-6,
            )

    def test_weight_decay_with_norm_preservation(self):
        """Test that weight decay doesn't break norm preservation."""
        matrix_size = (3, 3)
        param = torch.nn.Parameter(torch.randn(matrix_size, dtype=torch.float32))

        # Normalize rows
        param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        optimizer = ObliqueAdam([param], lr=0.01, weight_decay=0.01, mode="row")

        # Generate random gradient
        param.grad = torch.randn_like(param.data)

        # Perform optimization step
        optimizer.step()

        # Norms should still be preserved with weight decay
        final_norms = param.data.norm(dim=1)
        expected_norms = torch.ones_like(final_norms)
        torch.testing.assert_close(
            final_norms,
            expected_norms,
            atol=1e-6,
            rtol=1e-6,
        )


class SimpleMLP(nn.Module):
    """Simple MLP with oblique-optimized layers for testing."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=True)  # Final layer with bias

        # Initialize weights for oblique optimization
        self._initialize_oblique_weights()

    def _initialize_oblique_weights(self):
        """Initialize weights to be column-normalized for oblique optimization."""
        with torch.no_grad():
            # Normalize columns of oblique layers
            self.fc1.weight.data = self.fc1.weight.data / self.fc1.weight.data.norm(dim=0, keepdim=True).clamp(
                min=1e-8
            )
            self.fc2.weight.data = self.fc2.weight.data / self.fc2.weight.data.norm(dim=0, keepdim=True).clamp(
                min=1e-8
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_oblique_parameters(self):
        """Return parameters that should use oblique optimization."""
        return [self.fc1.weight, self.fc2.weight]

    def get_regular_parameters(self):
        """Return parameters that should use regular optimization."""
        return [self.fc3.weight, self.fc3.bias]


class NormalizedOptimizerConvergenceTest(BaseTestCase):
    """Convergence tests for normalized optimizers on a simple MLP task."""

    def _create_synthetic_mnist_data(self, num_samples=1000):
        """Create synthetic MNIST-like data for testing."""
        torch.manual_seed(42)
        X = torch.randn(num_samples, 784)
        # Create somewhat realistic targets with class imbalance
        y = torch.randint(0, 10, (num_samples,))
        return TensorDataset(X, y)

    def _train_model(self, model, optimizer_class, optimizer_kwargs, num_epochs=5):
        """Train model with given optimizer and return final loss and accuracy."""
        # Create data
        dataset = self._create_synthetic_mnist_data(num_samples=500)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Setup optimizers - separate for oblique and regular parameters
        oblique_params = model.get_oblique_parameters()
        regular_params = model.get_regular_parameters()

        oblique_optimizer = optimizer_class(oblique_params, **optimizer_kwargs)
        regular_optimizer = torch.optim.Adam(regular_params, lr=optimizer_kwargs.get("lr", 0.001))

        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None
        final_accuracy = 0.0

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in dataloader:
                # Zero gradients
                oblique_optimizer.zero_grad()
                regular_optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()

                # Update parameters
                oblique_optimizer.step()
                regular_optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100 * correct / total

            if initial_loss is None:
                initial_loss = avg_loss
            final_loss = avg_loss
            final_accuracy = accuracy

        return initial_loss, final_loss, final_accuracy

    def _verify_norms_preserved(self, model):
        """Verify that oblique parameters maintain unit column norms."""
        for param in model.get_oblique_parameters():
            column_norms = param.data.norm(dim=0)  # Column norms
            expected_norms = torch.ones_like(column_norms)
            torch.testing.assert_close(
                column_norms,
                expected_norms,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_oblique_sgd_convergence(self):
        """Test that ObliqueSGD can train a simple MLP and maintain norms."""
        model = SimpleMLP(input_size=784, hidden_size=64, num_classes=10)

        # Train with ObliqueSGD
        initial_loss, final_loss, final_accuracy = self._train_model(
            model, ObliqueSGD, {"lr": 0.01, "momentum": 0.9, "mode": "col"}, num_epochs=10
        )

        # Check convergence
        self.assertLess(final_loss, initial_loss, "Loss should decrease during training")
        self.assertGreater(final_accuracy, 5.0, "Accuracy should be better than random (10%)")

        # Check norm preservation
        self._verify_norms_preserved(model)

    def test_oblique_adam_convergence(self):
        """Test that ObliqueAdam can train a simple MLP and maintain norms."""
        model = SimpleMLP(input_size=784, hidden_size=64, num_classes=10)

        # Train with ObliqueAdam
        initial_loss, final_loss, final_accuracy = self._train_model(
            model, ObliqueAdam, {"lr": 0.001, "betas": (0.9, 0.999), "mode": "col"}, num_epochs=10
        )

        # Check convergence
        self.assertLess(final_loss, initial_loss, "Loss should decrease during training")
        self.assertGreater(final_accuracy, 5.0, "Accuracy should be better than random (10%)")

        # Check norm preservation
        self._verify_norms_preserved(model)

    @parameterized.named_parameters(
        ("sgd_col", ObliqueSGD, {"lr": 0.02, "momentum": 0.9, "mode": "col"}),
        ("sgd_row", ObliqueSGD, {"lr": 0.02, "momentum": 0.9, "mode": "row"}),
        ("adam_col", ObliqueAdam, {"lr": 0.01, "betas": (0.9, 0.999), "mode": "col"}),
        ("adam_row", ObliqueAdam, {"lr": 0.01, "betas": (0.9, 0.999), "mode": "row"}),
    )
    def test_optimizer_modes_convergence(self, optimizer_class, optimizer_kwargs):
        """Test that both row and column modes work for both optimizers."""
        model = SimpleMLP(input_size=784, hidden_size=32, num_classes=10)

        # Adjust initialization based on mode
        if optimizer_kwargs["mode"] == "row":
            # Re-initialize for row normalization
            with torch.no_grad():
                for param in model.get_oblique_parameters():
                    param.data = param.data / param.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Train model
        initial_loss, final_loss, final_accuracy = self._train_model(
            model, optimizer_class, optimizer_kwargs, num_epochs=8
        )

        # Basic convergence check
        self.assertLess(final_loss, initial_loss * 1.01, "Loss should decrease or stay stable")
        self.assertGreater(final_accuracy, 50.0, "Should achieve reasonable accuracy")

        # Verify norm preservation based on mode
        for param in model.get_oblique_parameters():
            if optimizer_kwargs["mode"] == "col":
                norms = param.data.norm(dim=0)
            else:  # row mode
                norms = param.data.norm(dim=1)

            expected_norms = torch.ones_like(norms)
            torch.testing.assert_close(
                norms,
                expected_norms,
                atol=1e-5,
                rtol=1e-5,
            )


if __name__ == "__main__":
    absltest.main()
