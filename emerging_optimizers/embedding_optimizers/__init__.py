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
import math
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry


__all__ = ["Sinkhorn"]


def _sinkhorn_balance(
    update: torch.Tensor,
    *,
    eps: float,
    num_steps: int,
    zero_row_threshold: float,
) -> torch.Tensor:
    work = update.to(torch.float32)
    row_norms = torch.linalg.vector_norm(work, dim=1, keepdim=True)
    work.masked_fill_(row_norms <= zero_row_threshold * row_norms.mean(), 0.0)

    for step in range(num_steps):
        dim = 1 if step % 2 == 0 else 0
        norms = torch.linalg.vector_norm(work, dim=dim, keepdim=True)
        work.div_(norms.add_(eps))

    return work.mul_(math.sqrt(update.size(1))).to(update.dtype)


@registry.register_optimizer("sinkhorn")
class Sinkhorn(Optimizer):
    """Momentum optimizer with Sinkhorn-balanced updates for tall token-by-feature matrices.

    The optimizer maintains an EMA first moment, forms a Nesterov update, masks near-zero rows, and
    alternates row and column L2 normalization. The final update is scaled by the square root of the
    hidden dimension, while the effective learning rate is scaled by ``lr_correction``. Parameters must
    use ``torch.bfloat16``, ``torch.float16``, or ``torch.float32``; the balancing workspace uses
    ``torch.float32`` for all supported parameter dtypes.

    Args:
        params: Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr: Base learning rate.
        momentum: EMA momentum coefficient.
        eps: Numerical stability term added to row and column norms.
        num_steps: Positive odd number of alternating normalization steps.
        zero_row_threshold: Rows whose norm is at most this factor times the mean row norm are masked.
        lr_correction: Multiplier applied to the base learning rate.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        *,
        eps: float = 1e-20,
        num_steps: int = 11,
        zero_row_threshold: float = 1e-3,
        lr_correction: float = 0.18,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if num_steps < 1 or num_steps % 2 == 0:
            raise ValueError(f"num_steps must be a positive odd integer, got {num_steps}")
        if zero_row_threshold < 0.0:
            raise ValueError(f"Invalid zero_row_threshold: {zero_row_threshold}")
        if lr_correction < 0.0:
            raise ValueError(f"Invalid lr_correction: {lr_correction}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            eps=eps,
            num_steps=num_steps,
            zero_row_threshold=zero_row_threshold,
            lr_correction=lr_correction,
        )
        super().__init__(params, defaults)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is not None:
            raise ValueError("closure is not supported")

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise ValueError("Sinkhorn only supports 2D parameters")
                if param.size(0) < param.size(1):
                    raise ValueError("Sinkhorn expects rows to be the larger matrix dimension")
                if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(
                        f"Sinkhorn only supports bfloat16, float16, and float32 parameters, got {param.dtype}"
                    )
                if param.grad.is_sparse:
                    raise ValueError("Sinkhorn does not support sparse gradients")

                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                momentum_buffer = state["momentum_buffer"]
                momentum = group["momentum"]

                momentum_buffer.lerp_(grad, 1.0 - momentum)
                nesterov_update = grad.lerp(momentum_buffer, momentum)
                balanced_update = _sinkhorn_balance(
                    nesterov_update,
                    eps=group["eps"],
                    num_steps=group["num_steps"],
                    zero_row_threshold=group["zero_row_threshold"],
                )
                param.add_(balanced_update, alpha=-group["lr"] * group["lr_correction"])

        return None
