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
    r"""Balance a signed 2D update along its row and column axes.

    Given an update ``G`` with row norms ``rho_i = ||G[i, :]||_2``, rows satisfying
    ``rho_i <= zero_row_threshold * mean(rho)`` are first set to zero. Starting from the masked update
    ``U``, an odd number ``K = num_steps`` of alternating normalizations applies

    ``U[i, :] <- U[i, :] / (||U[i, :]||_2 + eps)`` for odd steps, and
    ``U[:, j] <- U[:, j] / (||U[:, j]||_2 + eps)`` for even steps.

    The returned update is ``Delta = sqrt(n) * U``, where ``n`` is the number of columns. This
    approximately targets ``(1 / n) * sum_j Delta[i, j]^2 = 1`` for every row and
    ``(1 / m) * sum_i Delta[i, j]^2 = 1`` for every column, where ``m`` is the number of rows.

    Args:
        update: Signed Nesterov update with shape ``(num_rows, num_columns)``.
        eps: Numerical stability term added to each row or column norm.
        num_steps: Total positive odd number of individual axis-normalization steps.
        zero_row_threshold: Masking threshold relative to the mean pre-balancing row norm.

    Returns:
        Sinkhorn-balanced update in the input dtype.
    """
    balanced_update = update.to(torch.float32)

    # DeepSeek Algorithm 1 measures each row norm before balancing and masks rows whose norm is at most
    # zero_row_threshold times the mean row norm. This keeps inactive or near-zero token rows at zero
    # instead of amplifying their numerical noise during normalization.
    row_norms = torch.linalg.vector_norm(balanced_update, dim=1, keepdim=True)
    balanced_update.masked_fill_(row_norms <= zero_row_threshold * row_norms.mean(), 0.0)

    # Apply the first row step separately, then express each remaining iteration as a column/row pair.
    balanced_update.div_(row_norms.add_(eps))
    for _ in range(num_steps // 2):
        # Normalize columns along the row/token dimension.
        column_norms = torch.linalg.vector_norm(balanced_update, dim=0, keepdim=True)
        balanced_update.div_(column_norms.add_(eps))

        # Normalize rows along the column/feature dimension.
        row_norms = torch.linalg.vector_norm(balanced_update, dim=1, keepdim=True)
        balanced_update.div_(row_norms.add_(eps))

    # A unit-L2 row has RMS 1 / sqrt(n); this scale converts the balanced update to unit row-wise RMS.
    balanced_update.mul_(math.sqrt(update.size(1)))
    return balanced_update.to(update.dtype)


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
        num_steps: Total positive odd number of individual row or column normalization steps. The first
            step normalizes rows, and each subsequent iteration normalizes columns then rows.
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
