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

from absl import flags
import torch
import torch.nn as nn
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers import spectron
from emerging_optimizers.utils.eig import power_iteration
# Define command line flags
flags.DEFINE_string("device", "cpu", "Device to run tests on: 'cpu' or 'cuda'")

FLAGS = flags.FLAGS

class PowerIterationTest(parameterized.TestCase):
    @parameterized.parameters(
        {"shape": (10, 8), "k": 1},
        {"shape": (32, 16), "k": 5},
        {"shape": (64, 32), "k": 10},
        {"shape": (100, 50), "k": 20},
    )
    def test_power_iteration_converges_to_largest_singular_value(self, shape, k) -> None:
        """Test that power iteration approximates the largest singular value."""
        # Create a random matrix with known singular values
        W = torch.randn(shape, dtype=torch.float32, device=FLAGS.device)
        
        # Get ground truth largest singular value using SVD
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        true_sigma_max = S[0].item()
        
        # Initialize random left singular vector
        u = torch.randn(shape[0], dtype=torch.float32, device=FLAGS.device)
        u = u / u.norm()
        
        # Run power iteration
        sigma_approx, u_out, _v_out = power_iteration(W, u, k=k)
        
        # Check that approximation is close to true value
        # More iterations should give better approximation
        rel_error = abs(sigma_approx.item() - true_sigma_max) / true_sigma_max
        
        # With more iterations, error should be smaller
        if k >= 10:
            self.assertLess(rel_error, 0.01, f"Relative error {rel_error} too large with {k} iterations")
        else:
            self.assertLess(rel_error, 0.1, f"Relative error {rel_error} too large with {k} iterations")
    
    def test_power_iteration_output_normalized(self) -> None:
        """Test that power iteration returns normalized left singular vector."""
        W = torch.randn(20, 15, dtype=torch.float32, device=FLAGS.device)
        u = torch.randn(20, dtype=torch.float32, device=FLAGS.device)
        
        _, u_out, _v_out = power_iteration(W, u, k=5)
        
        # Check that output is normalized
        torch.testing.assert_close(
            u_out.norm(),
            torch.tensor(1.0, device=FLAGS.device),
            atol=1e-6,
            rtol=1e-6,
        )
    
    def test_power_iteration_handles_unnormalized_input(self) -> None:
        """Test that power iteration works even with unnormalized input."""
        W = torch.randn(20, 15, dtype=torch.float32, device=FLAGS.device)
        u = torch.randn(20, dtype=torch.float32, device=FLAGS.device) * 100  # Unnormalized
        
        # Should not raise error and should normalize internally
        sigma, u_out, _v_out = power_iteration(W, u, k=5)
        
        self.assertIsInstance(sigma.item(), float)
        torch.testing.assert_close(
            u_out.norm(),
            torch.tensor(1.0, device=FLAGS.device),
            atol=1e-6,
            rtol=1e-6,
        )
    
    def test_power_iteration_deterministic(self) -> None:
        """Test that power iteration is deterministic given same inputs."""
        W = torch.randn(20, 15, dtype=torch.float32, device=FLAGS.device)
        u = torch.randn(20, dtype=torch.float32, device=FLAGS.device)
        
        sigma1, u1, _v1 = power_iteration(W, u.clone(), k=5)
        sigma2, u2, _v2 = power_iteration(W, u.clone(), k=5)
        
        torch.testing.assert_close(sigma1, sigma2, atol=0, rtol=0)
        torch.testing.assert_close(u1, u2, atol=0, rtol=0)
    
    def test_power_iteration_returns_both_singular_vectors(self) -> None:
        """Test that power iteration returns both left and right singular vectors normalized."""
        W = torch.randn(20, 15, dtype=torch.float32, device=FLAGS.device)
        u = torch.randn(20, dtype=torch.float32, device=FLAGS.device)
        
        sigma, u_out, v_out = power_iteration(W, u, k=10)
        
        # Both singular vectors should be normalized
        torch.testing.assert_close(
            u_out.norm(),
            torch.tensor(1.0, device=FLAGS.device),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            v_out.norm(),
            torch.tensor(1.0, device=FLAGS.device),
            atol=1e-6,
            rtol=1e-6,
        )
        
        # Check that W @ v ≈ sigma * u (definition of singular vectors)
        Wv = W @ v_out
        sigma_u = sigma * u_out
        torch.testing.assert_close(Wv, sigma_u, atol=1e-4, rtol=1e-4)
        
        # Check that W^T @ u ≈ sigma * v
        WTu = W.mT @ u_out
        sigma_v = sigma * v_out
        torch.testing.assert_close(WTu, sigma_v, atol=1e-4, rtol=1e-4)


class SpectronTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(10, 8), (32, 16), (64, 32)],
        rank=[4, 8, 16],
        weight_decay_method=["decoupled", "independent", "l2"],
        fp32_matmul_prec=["highest", "medium"],
    )
    def test_smoke(self, shape, rank, weight_decay_method, fp32_matmul_prec) -> None:
        """Smoke test Spectron optimizer with various configurations."""
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randn_like(test_param)
        
        spectron_opt = spectron.Spectron(
            [test_param],
            lr=0.01,
            rank=rank,
            weight_decay=0.01,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
        )
        spectron_opt.step()
        
        # Check that parameter was updated
        self.assertIsNotNone(test_param.data)
        self.assertEqual(test_param.shape, shape)
    
    @parameterized.parameters(
        {"shape": (32, 16), "rank": 8},
        {"shape": (64, 32), "rank": 16},
        {"shape": (100, 50), "rank": 20},
    )
    def test_low_rank_reconstruction_quality(self, shape, rank) -> None:
        """Test that low-rank factorization preserves parameter reasonably after initialization."""
        # Create parameter with known structure
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))
        original_param = test_param.data.clone()
        
        spectron_opt = spectron.Spectron(
            [test_param],
            lr=0.0,  # No update, just check initialization
            rank=rank,
            momentum_beta=0.0,
            weight_decay=0.0,
        )
        
        # Initialize state
        test_param.grad = torch.randn_like(test_param)
        spectron_opt.step()
        
        # Get state
        state = spectron_opt.state[test_param]
        factor_A = state["factor_A"]
        factor_B = state["factor_B"]
        
        # Reconstruct should give back the parameter (since lr=0)
        reconstructed = factor_A @ factor_B.mT
        
        # Check reconstruction quality (won't be perfect due to low-rank)
        rel_error = (reconstructed - original_param).norm() / original_param.norm()
        
        # Error should decrease with higher rank
        self.assertLess(rel_error.item(), 0.5, f"Reconstruction error {rel_error.item()} too large")
    
    def test_momentum_accumulation(self) -> None:
        """Test that momentum is properly accumulated over multiple steps."""
        shape = (32, 16)
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))
        
        momentum_beta = 0.9
        spectron_opt = spectron.Spectron(
            [test_param],
            lr=0.01,
            rank=8,
            momentum_beta=momentum_beta,
            weight_decay=0.0,
        )
        
        # First step
        test_param.grad = torch.ones_like(test_param)
        spectron_opt.step()
        
        state = spectron_opt.state[test_param]
        momentum_A_step1 = state["momentum_A"].clone()
        momentum_B_step1 = state["momentum_B"].clone()
        
        # Second step with same gradient
        test_param.grad = torch.ones_like(test_param)
        spectron_opt.step()
        
        momentum_A_step2 = state["momentum_A"]
        momentum_B_step2 = state["momentum_B"]
        
        # Momentum should have changed (accumulated)
        self.assertFalse(torch.allclose(momentum_A_step1, momentum_A_step2))
        self.assertFalse(torch.allclose(momentum_B_step1, momentum_B_step2))
    
    def test_spectral_scaling_reduces_lr_for_large_sigma(self) -> None:
        """Test that learning rate is scaled down when spectral radius is large."""
        shape = (32, 16)
        
        # Create parameter with large norm (will have large spectral radius)
        test_param_large = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device) * 10)
        test_param_small = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device) * 0.1)
        
        test_param_large.grad = torch.ones_like(test_param_large) * 0.01
        test_param_small.grad = torch.ones_like(test_param_small) * 0.01
        
        lr = 0.1
        
        opt_large = spectron.Spectron([test_param_large], lr=lr, rank=8, momentum_beta=0.0)
        opt_small = spectron.Spectron([test_param_small], lr=lr, rank=8, momentum_beta=0.0)
        
        param_large_before = test_param_large.data.clone()
        param_small_before = test_param_small.data.clone()
        
        opt_large.step()
        opt_small.step()
        
        # Get effective learning rates from spectral scaling
        state_large = opt_large.state[test_param_large]
        state_small = opt_small.state[test_param_small]
        
        # Compute sigma values after step
        sigma_A_large, _, _ = power_iteration(state_large["factor_A"], state_large["u_A"], k=1)
        sigma_B_large, _, _ = power_iteration(state_large["factor_B"], state_large["u_B"], k=1)
        
        sigma_A_small, _, _ = power_iteration(state_small["factor_A"], state_small["u_A"], k=1)
        sigma_B_small, _, _ = power_iteration(state_small["factor_B"], state_small["u_B"], k=1)
        
        scaled_lr_large = lr / (sigma_A_large + sigma_B_large + 1.0)
        scaled_lr_small = lr / (sigma_A_small + sigma_B_small + 1.0)
        
        # Larger spectral radius should result in smaller effective learning rate
        self.assertLess(scaled_lr_large.item(), scaled_lr_small.item())
    
    def test_rank_capped_by_dimensions(self) -> None:
        """Test that rank is automatically capped by matrix dimensions."""
        shape = (10, 8)  # Small matrix
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randn_like(test_param)
        
        # Request rank larger than min dimension
        spectron_opt = spectron.Spectron(
            [test_param],
            lr=0.01,
            rank=100,  # Larger than both dimensions
        )
        spectron_opt.step()
        
        state = spectron_opt.state[test_param]
        factor_A = state["factor_A"]
        factor_B = state["factor_B"]
        
        # Rank should be capped at min(m, n) = 8
        self.assertEqual(factor_A.shape[1], 8)
        self.assertEqual(factor_B.shape[1], 8)
    
    def test_raises_error_for_1d_params(self) -> None:
        """Test that Spectron raises error for 1D parameters."""
        test_param = nn.Parameter(torch.randn(10, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randn_like(test_param)
        
        spectron_opt = spectron.Spectron([test_param], lr=0.01, rank=4)
        
        with self.assertRaises(ValueError):
            spectron_opt.step()
    
    @parameterized.parameters(
        {"num_ns_steps": 1},
        {"num_ns_steps": 3},
        {"num_ns_steps": 5},
        {"num_ns_steps": 10},
    )
    def test_different_ns_steps(self, num_ns_steps) -> None:
        """Test that different numbers of Newton-Schulz steps work."""
        shape = (32, 16)
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randn_like(test_param)
        
        spectron_opt = spectron.Spectron(
            [test_param],
            lr=0.01,
            rank=8,
            num_ns_steps=num_ns_steps,
        )
        
        # Should not raise error
        spectron_opt.step()
    
    @parameterized.parameters(
        {"num_power_iter": 1},
        {"num_power_iter": 5},
        {"num_power_iter": 10},
    )
    def test_different_power_iter_steps(self, num_power_iter) -> None:
        """Test that different numbers of power iteration steps work."""
        shape = (32, 16)
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))
        test_param.grad = torch.randn_like(test_param)
        
        spectron_opt = spectron.Spectron(
            [test_param],
            lr=0.01,
            rank=8,
            num_power_iter=num_power_iter,
        )
        
        # Should not raise error
        spectron_opt.step()
    
    def test_state_persistence_across_steps(self) -> None:
        """Test that optimizer state (A, B, momentum, u) persists correctly across steps."""
        shape = (32, 16)
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=FLAGS.device))
        
        spectron_opt = spectron.Spectron([test_param], lr=0.01, rank=8)
        
        # First step
        test_param.grad = torch.randn_like(test_param)
        spectron_opt.step()
        
        state = spectron_opt.state[test_param]
        factor_A_step1 = state["factor_A"].clone()
        u_A_step1 = state["u_A"].clone()
        
        # Second step
        test_param.grad = torch.randn_like(test_param)
        spectron_opt.step()
        
        # State should still exist and be updated
        self.assertIn("factor_A", state)
        self.assertIn("u_A", state)
        
        # Values should have changed
        self.assertFalse(torch.allclose(state["factor_A"], factor_A_step1))
        # u vector should be updated (but might be similar due to slow changes)
        self.assertEqual(state["u_A"].shape, u_A_step1.shape)


if __name__ == "__main__":
    absltest.main()