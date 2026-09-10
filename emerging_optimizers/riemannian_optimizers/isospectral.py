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
from typing import TYPE_CHECKING, Callable, override

from emerging_optimizers.riemannian_optimizers.retractions.stiefel import (
    RetractionT,
    cayley_retraction,
    newton_schulz_retraction,
    polar_retraction,
    qr_retraction,
)


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from emerging_optimizers import registry, utils


__all__ = ["Iso"]


def _retract_factors(
    u: torch.Tensor,
    v: torch.Tensor,
    momentum_u: torch.Tensor,
    momentum_v: torch.Tensor,
    step_size: float,
    retraction: RetractionT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update and retract both Stiefel factors.

    Args:
        u: Left Stiefel factor.
        v: Right Stiefel factor.
        momentum_u: Momentum update for the left factor.
        momentum_v: Momentum update for the right factor.
        step_size: Optimizer step size.
        retraction: Retraction method applied to both factors.

    Returns:
        The updated and retracted ``(u, v)`` factors.

    Raises:
        ValueError: If the retraction method is unsupported.
    """
    if retraction == "qr":
        retract = qr_retraction
    elif retraction == "polar":
        retract = polar_retraction
    elif retraction == "cayley":
        retract = cayley_retraction
    elif retraction == "newton_schulz":
        retract = newton_schulz_retraction
    else:
        raise ValueError(f"Invalid retraction: {retraction}")

    return (
        retract(u, momentum_u, step_size),
        retract(v, momentum_v, step_size),
    )


@registry.register_optimizer("iso")
class Iso(Optimizer):
    """Isospectral optimizer for two-dimensional parameters.

    The optimizer factorizes each parameter as ``U @ diag(Sigma) @ V.T`` and
    updates both Stiefel factors while keeping ``Sigma`` fixed. It is designed
    for reinforcement learning with verifiable rewards (RLVR), particularly for
    LLM reasoning post-training, but its implementation does not depend on an
    RL-specific training interface.

    References:
        - *ISO: An RLVR-Native Optimization Stack.* arXiv:2607.19331 (2026).
          [`arXiv:2607.19331 <https://arxiv.org/abs/2607.19331>`_]

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient.
        retraction: Retraction used to restore the Stiefel constraints.
        fp32_matmul_prec: Precision used for FP32 matrix multiplications.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        retraction: RetractionT = "qr",
        *,
        fp32_matmul_prec: utils.FP32MatmulPrecT = "highest",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if retraction not in ("qr", "polar", "cayley", "newton_schulz"):
            raise ValueError(f"Invalid retraction: {retraction}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "retraction": retraction,
        }
        self.fp32_matmul_prec = fp32_matmul_prec
        super().__init__(params, defaults)

    @torch.no_grad()  # type: ignore[misc]
    def _init_state(self, param: torch.Tensor) -> None:
        state = self.state[param]
        if len(state) != 0:
            return
        if param.ndim != 2:
            raise ValueError("Iso only supports 2D parameters")

        # Factor state must be FP32 because the required linalg kernels do not
        # support every lower-precision device and dtype combination.
        factor_param = param.float()
        u, sigma, vh = torch.linalg.svd(factor_param, full_matrices=False)
        state["step"] = 0
        state["u"] = u
        state["sigma"] = sigma
        state["v"] = vh.mT
        state["momentum_u"] = torch.zeros_like(u)
        state["momentum_v"] = torch.zeros_like(vh.mT)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            retraction = group["retraction"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise ValueError("Iso only supports 2D parameters")
                if param.grad.is_sparse:
                    raise ValueError("Iso does not support sparse gradients")

                state = self.state[param]
                self._init_state(param)

                u = state["u"]
                sigma = state["sigma"]
                v = state["v"]
                momentum_u = state["momentum_u"]
                momentum_v = state["momentum_v"]

                grad = param.grad.float()

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    grad_u = (grad @ v) * sigma.unsqueeze(0)
                    grad_v = (grad.mT @ u) * sigma.unsqueeze(0)
                    momentum_u.mul_(momentum).add_(grad_u)
                    momentum_v.mul_(momentum).add_(grad_v)

                    u, v = _retract_factors(
                        u,
                        v,
                        momentum_u,
                        momentum_v,
                        lr,
                        retraction,
                    )
                    scaled_u = u * sigma.unsqueeze(0)
                    if param.dtype == torch.float32:
                        torch.addmm(param, scaled_u, v.mT, beta=0.0, out=param)
                    else:
                        # Mixed-dtype addmm cannot write an FP32 result directly into param.
                        param.copy_(torch.mm(scaled_u, v.mT))

                state["u"] = u
                state["v"] = v
                state["step"] += 1

        return loss
