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
from absl import flags, logging
from absl.testing import absltest, parameterized
from torch.utils.data import DataLoader, TensorDataset

from emerging_optimizers.psgd.psgd import PSGDPro


# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")

FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class SimpleMLP(nn.Module):
    """Simple MLP for testing PSGD convergence."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PSGDConvergenceTest(parameterized.TestCase):
    """Convergence tests for PSGD optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = FLAGS.device

    def test_quadratic_function_convergence(self):
        """Test PSGD convergence on a simple quadratic function: f(x) = (x - target)^2."""
        # Create a parameter to optimize
        target = torch.tensor([2.0, -1.5, 3.2], device=self.device)
        x = torch.nn.Parameter(torch.zeros(3, device=self.device))

        # Create PSGD optimizer
        optimizer = PSGDPro([x], lr=0.1, precond_lr=0.1, beta_lip=0.9, damping_noise_scale=0.1)

        initial_loss = None
        final_loss = None

        # Optimization loop
        for _ in range(100):
            optimizer.zero_grad()

            # Compute quadratic loss
            loss = torch.sum((x - target) ** 2)
            loss.backward()

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.step()
            final_loss = loss.item()

        # Check convergence
        self.assertLess(final_loss, initial_loss, "Loss should decrease during optimization")
        self.assertLess(final_loss, 0.01, "Should converge reasonably close to minimum")

        # Check that x is close to target
        torch.testing.assert_close(x, target, atol=1e-1, rtol=1e-1)

    def test_matrix_optimization_convergence(self):
        """Test PSGD convergence on matrix optimization problem."""
        # Create target matrix and parameter matrix
        target = torch.randn(4, 6, device=self.device)
        A = torch.nn.Parameter(torch.randn(4, 6, device=self.device))
        per_element_eps = 1e-3
        # Create PSGD optimizer
        optimizer = PSGDPro([A], lr=0.05)

        initial_loss = None
        final_loss = None

        # Optimization loop
        for iteration in range(100):
            optimizer.zero_grad()

            # Frobenius norm loss
            loss = torch.norm(A - target, p="fro") ** 2
            loss.backward()

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.step()
            final_loss = loss.item()

        # Check per-element MSE for convergence
        per_element_mse = final_loss / A.numel()
        self.assertLess(
            per_element_mse,
            per_element_eps,
            f"Per-element MSE should be < {per_element_eps}, got {per_element_mse:.6f}",
        )

    def _create_synthetic_mnist_data(self, num_samples: int = 1000) -> TensorDataset:
        """Create synthetic MNIST-like data for testing."""
        torch.manual_seed(1234)
        X = torch.randn(num_samples, 784, device=self.device)
        # Create somewhat realistic targets with class distribution
        y = torch.randint(0, 10, (num_samples,), device=self.device)
        return TensorDataset(X, y)

    def _train_model(
        self, model: SimpleMLP, optimizer: torch.optim.Optimizer, num_epochs: int = 5
    ) -> tuple[float, float, float]:
        """Train model with given optimizer and return initial loss, final loss, and final accuracy."""
        # Create data
        dataset = self._create_synthetic_mnist_data(num_samples=500)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None
        final_accuracy = 0.0

        model.train()
        for _ in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in dataloader:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

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

    def test_mnist_convergence(self):
        """Test PSGD convergence on a simple neural network classification task."""
        model = SimpleMLP(input_size=784, hidden_size=64, num_classes=10).to(self.device)

        # Create PSGD optimizer
        optimizer = PSGDPro(model.parameters(), lr=0.01, weight_decay=0.001)

        # Train model
        initial_loss, final_loss, final_accuracy = self._train_model(model, optimizer, num_epochs=10)

        # Check convergence
        self.assertLess(final_loss, initial_loss, "Loss should decrease during training")
        self.assertGreater(final_accuracy, 80.0, "Accuracy should be better than random (10%)")


if __name__ == "__main__":
    absltest.main()
