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
import math
from typing import Callable, Iterable, List, Tuple, override

import torch
from absl import logging

from emerging_optimizers.psgd.procrustes_step import procrustes_step
from emerging_optimizers.psgd.psgd_kron_contractions import apply_preconditioner, partial_contraction
from emerging_optimizers.psgd.psgd_utils import norm_lower_bound_spd, uniformize_q_in_place
from emerging_optimizers.soap.soap import _clip_update_rms_in_place


__all__ = [
    "PSGDPro",
]


class PSGDPro(torch.optim.Optimizer):
    """Implements a variant of the PSGD optimization algorithm (PSGD-Kron-Whiten with Procrustes step for preconditioner update).

    PSGD ()

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate to use
        weight_decay: Weight decay coefficient
        use_decoupled_weight_decay: Whether to use decoupled weight decay, see Decoupled Weight Decay Regularization:
            https://arxiv.org/abs/1711.05101.
        momentum: Momentum coefficient for exponential moving average of gradient.
        betaL: Inner learning rate for the Lipschitz constants.
        precond_lr: Inner learning rate for the preconditioner.
        precond_init_scale: scale of initial preconditioner values.
        min_precond_lr: Minimum learning rate for preconditioner learning rate schedule.
        warmup_steps: Warmup steps for preconditioner learning rate schedule.
        damping_noise_scale: scale of dampening noise added to gradients.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        momentum: float = 0.9,
        betaL: float = 0.9,
        precond_lr: float = 0.1,
        precond_init_scale: float = 1.0,
        damping_noise_scale: float = 0.1,
        min_precond_lr: float = 0.3,
        warmup_steps: int = 10000,
        max_update_rms: float = 0.0,
    ) -> None:
        if betaL is None:
            betaL = 0.9
            logging.debug(f"betaL not provided. Setting betaL equal to betaL = {betaL} by default.")

        if precond_lr is None:
            precond_lr = 0.95
            logging.debug(
                f"precond_lr not provided. Setting precond_lr equal to precond_lr = {precond_lr} by default."
            )

        defaults = {
            "lr": lr,
            "betaL": betaL,
            "weight_decay": weight_decay,
            "use_decoupled_weight_decay": use_decoupled_weight_decay,
            "momentum": momentum,
            "precond_lr": precond_lr,
            "precond_init_scale": precond_init_scale,
            "max_update_rms": max_update_rms,
            "min_precond_lr": min_precond_lr,
            "warmup_steps": warmup_steps,
            "damping_noise_scale": damping_noise_scale,
        }
        super().__init__(params, defaults)

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
                grad = p.grad
                state = self.state[p]

                # Optimizer state initialization
                if "step" not in state:
                    state["step"] = 0
                # Momentum buffer
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                # PSGD kronecker factor matrices and Lipschitz constants initialization
                if "Q" not in state or "L" not in state:
                    state["Q"], state["L"] = _init_psgd_kron_states(
                        grad,
                        precond_init_scale=group["precond_init_scale"],
                    )

                # weight decay
                if group["weight_decay"] > 0.0:
                    if group["use_decoupled_weight_decay"]:
                        # Apply decoupled weight decay
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    else:
                        # add l2 regularization before preconditioning (i.e. adding a squared loss term)
                        grad += group["weight_decay"] * p

                # update momentum buffer with EMA of gradient
                exp_avg = state["exp_avg"]
                exp_avg.lerp_(grad, 1 - group["momentum"])

                # Get hyperparameters for preconditioner update
                damping_noise_scale = group["damping_noise_scale"]
                betaL = group["betaL"]
                precond_lr = _get_precond_lr(
                    group["precond_lr"], state["step"], group["min_precond_lr"], group["warmup_steps"]
                )
                betaL = group["betaL"]

                # Preconditioner update
                state["Q"], state["L"] = _update_precond_procrustes(
                    state["Q"], state["L"], exp_avg, damping_noise_scale, precond_lr, betaL
                )
                uniformize_q_in_place(state["Q"])

                # Get weight update by preconditioning the momentum
                update = apply_preconditioner(state["Q"], exp_avg)
                _clip_update_rms_in_place(update, group["max_update_rms"])

                # Apply weight update
                p.add_(update, alpha=-group["lr"])

        return loss


def _init_psgd_kron_states(
    grad: torch.Tensor,
    precond_init_scale: float = 1.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Initialize the Kronecker factor matrices and Lipschitz constants.

    Args:
        grad: Gradient tensor.
        precond_init_scale: Scale of preconditioner initialization.

    Returns:
        Q: List of Kronecker factors.
        L: List of Lipschitz constants for the Kronecker factors.
    """
    Q: List[torch.Tensor] = []
    L: List[torch.Tensor] = []

    # Create identity matrices scaled by precond_init_scale for each dimension
    for size in grad.shape:
        Q.append(torch.eye(size, device=grad.device) * precond_init_scale)
        L.append(torch.tensor(1.0, device=grad.device))

    return Q, L


def _update_precond_procrustes(
    Q: List[torch.Tensor],
    L: List[torch.Tensor],
    exp_avg: torch.Tensor,
    damping_noise_scale: float,
    precond_lr: float = 0.1,
    betaL: float = 0.9,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""Update the Kron preconditioner Q using procrustes step and uniformization.

    Args:
        Q: List of Kronecker factors.
        L: List of Lipschitz constants for the Kronecker factors.
        exp_avg: Exponential moving average of gradient.
        damping_noise_scale: Scale of noise added to gradient.
        precond_lr: Learning rate.
        betaL: Beta for the lower bound.

    Returns:
        Q: List of Kronecker factors.
        L: List of Lipschitz constants for the Kronecker factors.
    """
    Pg = apply_preconditioner(Q, _dampen_tensor(exp_avg, damping_noise_scale))
    total_numel = Pg.numel()
    updated_Q: List[torch.Tensor] = []
    updated_L: List[torch.Tensor] = []
    for dim, q in enumerate(Q):
        # compute gradient covariance
        precond_grad_cov = partial_contraction(Pg, Pg, dim)
        if q.dim() < 2:
            # diagonal or scalar-structured preconditioner
            q, l_updated = _update_1d_preconditioner(q, L[dim], precond_grad_cov, total_numel, precond_lr, betaL)
        else:
            # matrix-structured preconditioner
            q, l_updated = _update_matrix_preconditioner(q, L[dim], precond_grad_cov, total_numel, precond_lr, betaL)
        updated_Q.append(q)
        updated_L.append(l_updated)

    return updated_Q, updated_L


def _update_matrix_preconditioner(
    q: torch.Tensor,
    L: torch.Tensor,
    precond_grad_cov: torch.Tensor,
    total_numel: int,
    precond_lr: float,
    betaL: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Update matrix-structured preconditioner with adaptive Lipschitz constant.

    Args:
        q: Kronecker factor matrix for this dimension to update.
        L: Lipschitz constant for this dimension.
        precond_grad_cov: Gradient covariance.
        total_numel: Total number of elements in the gradient.
        precond_lr: Learning rate.
        betaL: Beta for the Lipschitz constant exponential moving average.

    Returns:
        q: Updated Kronecker factor matrix for this dimension.
        L: Updated Lipschitz constant for this dimension.
    """
    normalization = total_numel / q.shape[0]
    ell = norm_lower_bound_spd(precond_grad_cov) + normalization
    L = torch.max(betaL * L + (1 - betaL) * ell, ell)
    q = q - precond_lr / L * (precond_grad_cov @ q - normalization * q)
    q = procrustes_step(q)
    return q, L


def _update_1d_preconditioner(
    q: torch.Tensor,
    L: torch.Tensor,
    precond_grad_cov: torch.Tensor,
    total_numel: int,
    precond_lr: float,
    betaL: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Update 1D preconditioner with adaptive Lipschitz constant.

    Args:
        q: Kronecker factor 1D tensor for this dimension to update.
        L: Lipschitz constant for this dimension.
        precond_grad_cov: Gradient covariance.
        total_numel: Total number of elements in the gradient.
        precond_lr: Learning rate.
        betaL: Beta for the Lipschitz constant exponential moving average.

    Returns:
        q: Updated Kronecker factor 1D tensor for this dimension.
        L: Updated Lipschitz constant for this dimension.
    """
    normalization = total_numel / q.numel()
    ell = torch.max(torch.real(precond_grad_cov)) + normalization
    L = torch.max(betaL * L + (1 - betaL) * ell, ell)
    q = q * (1 - precond_lr / L * (precond_grad_cov - normalization))
    return q, L


def _get_precond_lr(precond_lr: float, step: int, min_precond_lr: float = 0.3, warmup_steps: int = 10000) -> float:
    r"""Helper function to get preconditioner learning rate for this optimization step based on a square root schedule.

    Decaying from a higher lr down to min_precond_lr improves accuracy.

    Args:
        precond_lr: Learning rate.
        step: Current step.
        min_precond_lr: Minimum learning rate.
        warmup_steps: Warmup steps.

    Returns:
        The preconditioner learning rate.
    """

    scheduled_lr = precond_lr / math.sqrt(1.0 + step / warmup_steps)
    return max(scheduled_lr, min_precond_lr)


def _dampen_tensor(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Helper function to dampen the tensor by adding noise.

    Args:
        x: The tensor to dampen.
        scale: The scale of the noise.
    """
    return torch.add(x, torch.randn_like(x) * scale, alpha=1.0)
