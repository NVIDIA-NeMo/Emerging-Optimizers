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
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.soap.soap import SOAP


__all__ = ["StackedSoap"]


def _stack_2d(x: torch.Tensor) -> torch.Tensor:
    """Flattens a 2D or 3D tensor to 2D, merging the batch dim into the smaller matrix edge.

    A 2D tensor is returned unchanged. A 3D tensor ``(b, m, n)`` is merged into the smaller of its two
    matrix edges: ``(m, b * n)`` when ``n <= m``, otherwise ``(b * m, n)``.

    Args:
        x: A 2D matrix ``(m, n)`` or a 3D batched matrix ``(b, m, n)``.

    Returns:
        The 2D stacking of ``x``.
    """
    if x.ndim == 2:
        return x
    b, m, n = x.shape
    if n <= m:
        # -> (m, b*n): move the batch next to the smaller edge, then merge.
        out = x.permute(1, 0, 2).reshape(m, b * n)
    else:
        # -> (b*m, n): contiguous merge into rows.
        out = x.reshape(b * m, n)
    return out.contiguous()


def _unstack(u: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Inverse of :func:`_stack_2d`, restoring the original ``shape``."""
    if len(shape) == 2:
        return u
    b, m, n = shape
    if n <= m:
        return u.reshape(m, b, n).permute(1, 0, 2).reshape(shape)
    return u.reshape(shape)


@registry.register_optimizer("stacked_soap")
class StackedSoap(SOAP):
    """Limited-memory SOAP for batched / 3D parameters via transient 2D stacking.

    Optimizes the real parameters directly: ``self.param_groups``, ``self.state``, and gradients are all
    keyed by the user's parameters, so learning-rate schedulers, gradient clipping, and ``state_dict``
    behave exactly as for plain :class:`SOAP`. Each 3D parameter is flattened to 2D by merging its batch
    dim into the smaller matrix edge (see :func:`_stack_2d`) only for the duration of :meth:`step`: the
    parameter's ``data`` and ``grad`` are swapped to their 2D views, the inherited SOAP step runs, and the
    2D update is unstacked back into the original storage. Because the swap happens before the inherited
    step, its lazy state initialization sizes the optimizer state to the stacked 2D shape automatically.

    Stacking on the smaller edge keeps both Kronecker factors small (the larger edge becomes a single
    shared factor) while reusing the full, unmodified SOAP machinery (KL-Shampoo + QR eigenbasis). The
    stacking is a storage-sharing view except for the permute branch (``q <= p``), which allocates one
    transient 2D buffer per step. A plain 2D parameter is stacked as itself, so this is exactly stock SOAP.

    SOAP is configured with the fixed settings appropriate for this use: decoupled weight decay, no
    Nesterov, bias correction on, the QR eigenbasis path with 1 power-iteration step, KL-Shampoo on, and
    the default matmul precision.

    Args:
        params: Iterable of 2D or 3D parameters to optimize or dicts defining parameter groups.
        lr: The learning rate.
        betas: Inner Adam betas ``(b1, b2)``.
        shampoo_beta: Beta for the kronecker factor moving average.
        eps: Inner Adam epsilon.
        weight_decay: Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_method="decoupled",
            nesterov=False,
            correct_bias=True,
            use_eigh=False,
            power_iter_steps=1,
            use_kl_shampoo=True,
        )

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

        # Swap each parameter's data/grad to their 2D stacking, run the inherited SOAP step on the 2D
        # views (state is keyed by the real parameter and sized for the stacked shape), then unstack the
        # update back into the original storage. The restore runs in a finally so that an exception inside
        # super().step() (e.g. OOM, a NaN check) cannot leave parameters stuck in their 2D stacked shape.
        saved: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        try:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue  # pragma: no cover
                    data, grad = p.data, p.grad
                    saved.append((p, data, grad))
                    p.data = _stack_2d(data)
                    p.grad = _stack_2d(grad)

            super().step()
        finally:
            for p, data, grad in saved:
                stacked = p.data
                p.data = data
                p.grad = grad
                # Copy back only when stacking allocated an independent buffer (permute branch); the view
                # branches already wrote the update through to the original storage.
                if stacked.data_ptr() != data.data_ptr():
                    data.copy_(_unstack(stacked, data.shape))
        return None
