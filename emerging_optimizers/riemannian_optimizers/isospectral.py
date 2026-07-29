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
from typing import TYPE_CHECKING, Callable, Literal, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from emerging_optimizers import registry


__all__ = ["ISO"]

RetractionT = Literal["qr", "polar", "cayley"]


def _qr_retraction(matrix: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.diagonal(r).sign()
    signs.masked_fill_(signs == 0, 1)
    return q * signs


def _polar_retraction(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def _cayley_retraction(
    point: torch.Tensor,
    direction: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    skew = direction @ point.mT - point @ direction.mT
    identity = torch.eye(point.shape[0], dtype=point.dtype, device=point.device)
    lhs = identity - 0.5 * step_size * skew
    rhs = (identity + 0.5 * step_size * skew) @ point
    return torch.linalg.solve(lhs, rhs)


@registry.register_optimizer("iso")
class ISO(Optimizer):
    """Isospectral optimizer for two-dimensional parameters.

    The optimizer factorizes each parameter as ``U @ diag(Sigma) @ V.T`` and
    updates the two Stiefel factors while keeping ``Sigma`` fixed.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient.
        retraction: Retraction used to restore the Stiefel constraints.
        weight_decay: L2 penalty added to the parameter gradient.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        retraction: RetractionT = "qr",
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if retraction not in ("qr", "polar", "cayley"):
            raise ValueError(f"Invalid retraction: {retraction}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "retraction": retraction,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # type: ignore[misc]
    def _init_state(self, param: torch.Tensor) -> None:
        state = self.state[param]
        if len(state) != 0:
            return
        if param.ndim != 2:
            raise ValueError("ISO only supports 2D parameters")

        u, sigma, vh = torch.linalg.svd(param, full_matrices=False)
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
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise ValueError("ISO only supports 2D parameters")
                if param.grad.is_sparse:
                    raise ValueError("ISO does not support sparse gradients")

                state = self.state[param]
                self._init_state(param)

                u = state["u"]
                sigma = state["sigma"]
                v = state["v"]
                momentum_u = state["momentum_u"]
                momentum_v = state["momentum_v"]

                grad = param.grad
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                grad_u = (grad @ v) * sigma.unsqueeze(0)
                grad_v = (grad.mT @ u) * sigma.unsqueeze(0)
                momentum_u.mul_(momentum).add_(grad_u)
                momentum_v.mul_(momentum).add_(grad_v)

                if retraction == "cayley":
                    u = _cayley_retraction(u, -momentum_u, lr)
                    v = _cayley_retraction(v, -momentum_v, lr)
                else:
                    retract = _qr_retraction if retraction == "qr" else _polar_retraction
                    u = retract(u - lr * momentum_u)
                    v = retract(v - lr * momentum_v)

                state["u"] = u
                state["v"] = v
                state["step"] += 1
                param.copy_((u * sigma.unsqueeze(0)) @ v.mT)

        return loss
