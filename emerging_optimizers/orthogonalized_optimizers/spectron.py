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

from typing import Callable, overload, override

import torch
import torch.optim as optim
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry, utils
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.utils import FP32MatmulPrecT
from emerging_optimizers.utils.eig import power_iteration


__all__ = ["Spectron"]


@registry.register_optimizer("spectron")
class Spectron(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Spectron: Low-rank spectral optimizer with orthogonalized momentum.

    Spectron maintains each 2D weight matrix W as a low-rank factorization W = A @ B^T,
    where A ∈ R^(m×r) and B ∈ R^(n×r). It applies momentum, orthogonalizes the updates
    using Newton-Schulz iteration, and scales the learning rate by the spectral radii
    of both factors.

    The algorithm:
    1. Compute gradients with respect to A and B from parameter gradients
    2. Apply momentum to both factors
    3. Orthogonalize momentum buffers using Newton-Schulz iteration
    4. Estimate spectral radius of A and B using power iteration
    5. Update with scaled learning rate: η / (σ_A + σ_B + 1)
    6. Reconstruct full weight matrix W = A @ B^T

    References:
        - Algorithm 1 (Spectron) and Algorithm 3 (PowerIter) from the Spectron paper (https://arxiv.org/abs/2602.12429).
          Low-rank spectral optimization with orthogonalized momentum.

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - Low-rank factorization may not be suitable for all parameter types.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate (η in the algorithm). Default: 3e-4
        rank: The rank of the low-rank factorization. Default: 64
        momentum_beta: The momentum decay coefficient (β). Default: 0.9
        weight_decay: The weight decay coefficient. Default: 0.01
        weight_decay_method: Method to apply weight decay. Default: "decoupled"
        fp32_matmul_prec: Precision of matmul operations. Default: "medium"
        num_ns_steps: Number of Newton-Schulz iteration steps. Default: 5
        num_power_iter: Number of power iteration steps for spectral radius. Default: 1
        coefficient_type: Type of coefficient set for Newton-Schulz. Default: "quintic"
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        rank: int = 64,
        momentum_beta: float = 0.9,
        weight_decay: float = 0.01,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        num_ns_steps: int = 5,
        num_power_iter: int = 1,
        coefficient_type: NSCoeffT = "quintic",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if not 0.0 <= momentum_beta < 1.0:
            raise ValueError(f"Invalid momentum_beta: {momentum_beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if num_power_iter < 1:
            raise ValueError(f"num_power_iter must be at least 1, got {num_power_iter}")

        self.fp32_matmul_prec = fp32_matmul_prec
        self.weight_decay_method = weight_decay_method
        self.rank = rank
        self.num_power_iter = num_power_iter

        # Create orthogonalization function following OrthogonalizedOptimizer pattern
        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            logging.debug(f"Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient")
            return muon_utils.newton_schulz(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
            )

        self.scaled_orthogonalize_fn = scaled_orthogonalize_fn

        defaults = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim != 2:
                    raise ValueError(f"Spectron only supports 2D parameters, got shape {p.shape}")

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                if state["step"] == 0:
                    assert all(
                        key not in state
                        for key in ["factor_A", "factor_B", "momentum_A", "momentum_B", "u_A", "u_B"]
                    ), (
                        "factor_A, factor_B, momentum_A, momentum_B, u_A, u_B should not be initialized at step 0. "
                        "Some mismatch has been created likely in checkpointing"
                    )
                    self._initialize_state(p, state)

                state["step"] += 1

                # Get state variables
                factor_A = state["factor_A"]
                factor_B = state["factor_B"]
                momentum_A = state["momentum_A"]
                momentum_B = state["momentum_B"]
                u_A = state["u_A"]
                u_B = state["u_B"]

                # Compute gradients for A and B from parameter gradient
                # Using chain rule: ∂L/∂A = ∂L/∂W @ B, ∂L/∂B = ∂L/∂W^T @ A
                grad_A = grad @ factor_B  # shape: (m, r)
                grad_B = grad.mT @ factor_A  # shape: (n, r)

                # Apply weight decay
                self._apply_weight_decay_inplace(factor_A, grad_A, group["lr"], group["weight_decay"])
                self._apply_weight_decay_inplace(factor_B, grad_B, group["lr"], group["weight_decay"])

                # Update momentum buffers (EMA of gradients)
                momentum_A.lerp_(grad_A, 1 - group["momentum_beta"])
                momentum_B.lerp_(grad_B, 1 - group["momentum_beta"])

                # Orthogonalize momentum using Newton-Schulz
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    orth_momentum_A = self.scaled_orthogonalize_fn(momentum_A)
                    orth_momentum_B = self.scaled_orthogonalize_fn(momentum_B)

                # Estimate spectral radius using power iteration (Algorithm 3)
                sigma_A, u_A = self._power_iteration(factor_A, u_A, self.num_power_iter)
                sigma_B, u_B = self._power_iteration(factor_B, u_B, self.num_power_iter)

                # Update power iteration vectors
                state["u_A"] = u_A
                state["u_B"] = u_B

                # Compute scaled learning rate
                scaled_lr = group["lr"] / (sigma_A + sigma_B + 1.0)

                # Update low-rank factors
                factor_A.add_(orth_momentum_A, alpha=-scaled_lr)
                factor_B.add_(orth_momentum_B, alpha=-scaled_lr)

                # Reconstruct full weight matrix: W = A @ B^T
                p.copy_(factor_A @ factor_B.mT)

        return loss

    def _initialize_state(self, p: torch.Tensor, state: dict[str, torch.Tensor]) -> None:
        """Initialize low-rank factors and state for a parameter.

        Args:
            p: The parameter tensor (shape: m × n)
            state: The state dictionary for this parameter
        """
        m, n = p.shape
        r = min(self.rank, m, n)  # Ensure rank doesn't exceed dimensions

        # Initialize A and B using SVD of the parameter
        # This provides a good initialization close to the original weights
        # Low-rank factors are stored in fp32 for numerical stability
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(p.float(), full_matrices=False)
            # Keep only top r singular values/vectors
            sqrt_S = torch.sqrt(S[:r])
            factor_A = U[:, :r] * sqrt_S
            factor_B = Vh[:r, :].mT * sqrt_S

        state["factor_A"] = factor_A
        state["factor_B"] = factor_B
        # Momentum buffers are always stored in fp32 for numerical stability
        state["momentum_A"] = torch.zeros_like(factor_A, dtype=torch.float32)
        state["momentum_B"] = torch.zeros_like(factor_B, dtype=torch.float32)

        # Initialize power iteration vectors (normalized random vectors in fp32)
        u_A = torch.randn(m, dtype=torch.float32, device=p.device)
        u_A = u_A / u_A.norm()
        u_B = torch.randn(n, dtype=torch.float32, device=p.device)
        u_B = u_B / u_B.norm()

        state["u_A"] = u_A
        state["u_B"] = u_B

    def _power_iteration(
        self, X: torch.Tensor, u: torch.Tensor, num_iters: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate the largest singular value using power iteration.

        Args:
            X: The matrix to estimate largest singular value for
            u: The current approximation of the dominant left singular vector
            num_iters: Number of power iteration steps

        Returns:
            Tuple of (largest singular value, updated_u)
        """
        # power_iteration returns (sigma, u, v) but Spectron only needs sigma and u (left singular vector)
        sigma, u, _v = power_iteration(X, u, k=num_iters)
        return sigma, u
