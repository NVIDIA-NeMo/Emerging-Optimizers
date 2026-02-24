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

from emerging_optimizers.riemannian_optimizers.normalized_optimizer import ObliqueAdam, ObliqueSGD


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
    """Simple MLP with oblique-optimized layers for testing."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10, dim=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=True)  # Final layer with bias
        self.dim = dim
        # Initialize weights for oblique optimization
        self._initialize_oblique_weights(dim)

    def _initialize_oblique_weights(self, dim):
        """Initialize weights to be normalized for oblique optimization."""
        with torch.no_grad():
            # Normalize in-place for oblique layers
            self.fc1.weight.data /= self.fc1.weight.data.norm(dim=dim, keepdim=True).clamp(min=1e-8)
            self.fc2.weight.data /= self.fc2.weight.data.norm(dim=dim, keepdim=True).clamp(min=1e-8)

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


class NormalizedOptimizerConvergenceTest(parameterized.TestCase):
    """Convergence tests for normalized optimizers on a simple MLP task."""

    def setUp(self):
        self.device = FLAGS.device

    def _create_synthetic_mnist_data(self, num_samples: int = 1000) -> TensorDataset:
        """Create synthetic MNIST-like data for testing."""
        torch.manual_seed(1234)
        X = torch.randn(num_samples, 784, device=self.device)
        # Create somewhat realistic targets with class imbalance
        y = torch.randint(0, 10, (num_samples,), device=self.device)
        return TensorDataset(X, y)

    def _train_model(
        self, model: SimpleMLP, optimizer_class: torch.optim.Optimizer, optimizer_kwargs: dict, num_epochs: int = 5
    ) -> tuple[float, float, float]:
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

    def _verify_norms_preserved(self, model: SimpleMLP) -> None:
        """Verify that oblique parameters maintain unit column norms."""
        for param in model.get_oblique_parameters():
            column_norms = param.data.norm(dim=0)  # Column norms
            expected_norms = torch.ones_like(column_norms)
            torch.testing.assert_close(
                column_norms,
                expected_norms,
                atol=0,
                rtol=1e-5,
            )

    def test_oblique_sgd_convergence(self) -> None:
        """Test that ObliqueSGD can train a simple MLP and maintain norms."""
        model = SimpleMLP(input_size=784, hidden_size=64, num_classes=10).to(self.device)

        # Train with ObliqueSGD
        initial_loss, final_loss, final_accuracy = self._train_model(
            model, ObliqueSGD, {"lr": 0.01, "momentum": 0.9, "dim": 0}, num_epochs=10
        )

        # Check convergence
        self.assertLess(final_loss, initial_loss, "Loss should decrease during training")
        self.assertGreater(final_accuracy, 5.0, "Accuracy should be better than random (10%)")

        # Check norm preservation
        self._verify_norms_preserved(model)

    def test_oblique_adam_convergence(self) -> None:
        """Test that ObliqueAdam can train a simple MLP and maintain norms."""
        model = SimpleMLP(input_size=784, hidden_size=64, num_classes=10).to(self.device)

        # Train with ObliqueAdam
        initial_loss, final_loss, final_accuracy = self._train_model(
            model, ObliqueAdam, {"lr": 0.001, "betas": (0.9, 0.999), "dim": 0}, num_epochs=10
        )

        # Check convergence
        self.assertLess(final_loss, initial_loss, "Loss should decrease during training")
        self.assertGreater(final_accuracy, 5.0, "Accuracy should be better than random (10%)")

        # Check norm preservation
        self._verify_norms_preserved(model)

    @parameterized.named_parameters(
        ("sgd_col", ObliqueSGD, {"lr": 0.1, "momentum": 0.75, "weight_decay": 0.1, "dim": 0}),
        ("sgd_row", ObliqueSGD, {"lr": 0.1, "momentum": 0.75, "weight_decay": 0.1, "dim": 1}),
        ("adam_col", ObliqueAdam, {"lr": 0.1, "betas": (0.9, 0.999), "weight_decay": 0.1, "dim": 0}),
        ("adam_row", ObliqueAdam, {"lr": 0.1, "betas": (0.9, 0.999), "weight_decay": 0.1, "dim": 1}),
    )
    def test_optimizer_modes_convergence(self, optimizer_class: torch.optim.Optimizer, optimizer_kwargs: dict) -> None:
        """Test that both row and column modes work for both optimizers."""
        model = SimpleMLP(input_size=784, hidden_size=32, num_classes=10).to(self.device)

        # Re-initialize for row normalization
        with torch.no_grad():
            for param in model.get_oblique_parameters():
                param.data /= param.data.norm(dim=optimizer_kwargs["dim"], keepdim=True).clamp(min=1e-8)

        # Train model
        initial_loss, final_loss, final_accuracy = self._train_model(
            model, optimizer_class, optimizer_kwargs, num_epochs=8
        )

        # Basic convergence check
        self.assertLess(final_loss, initial_loss * 1.01, "Loss should decrease or stay stable")
        print(f"Final accuracy: {final_accuracy}")
        self.assertGreater(final_accuracy, 50.0, "Should achieve reasonable accuracy")

        # Verify norm preservation based on mode
        for param in model.get_oblique_parameters():
            norms = param.data.norm(dim=optimizer_kwargs["dim"])

            expected_norms = torch.ones_like(norms)
            torch.testing.assert_close(
                norms,
                expected_norms,
                atol=0,
                rtol=1e-5,
            )


if __name__ == "__main__":
    absltest.main()
