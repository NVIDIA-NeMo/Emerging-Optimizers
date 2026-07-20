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


if TYPE_CHECKING:
    from typing import overload

import torch

from emerging_optimizers import registry
from emerging_optimizers.soap.soap import SOAP


__all__ = ["BlockedSoap"]


def _to_blocks(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise TypeError(f"BlockedSoap is only supported for 2D tensors. Got {x.dim()}D")
    rows, cols = x.shape
    if rows >= cols:
        num_full_blocks, remainder = divmod(rows, cols)
        if remainder == 0:
            return x.view(num_full_blocks, cols, cols)
        padded = x.new_zeros((num_full_blocks + 1) * cols, cols)
        padded[:rows] = x
        return padded.view(num_full_blocks + 1, cols, cols)
    num_full_blocks, remainder = divmod(cols, rows)
    if remainder == 0:
        return x.view(rows, num_full_blocks, rows).transpose(0, 1)
    padded = x.new_zeros(rows, (num_full_blocks + 1) * rows)
    padded[:, :cols] = x
    return padded.view(rows, num_full_blocks + 1, rows).transpose(0, 1)


def _from_blocks(blocked: torch.Tensor, x: torch.Tensor) -> None:
    rows, cols = x.shape
    if rows >= cols:
        x.copy_(blocked.reshape(-1, cols)[:rows])
    else:
        x.copy_(blocked.transpose(0, 1).reshape(rows, -1)[:, :cols])


@registry.register_optimizer("blocked_soap")
class BlockedSoap(SOAP):
    """SOAP that preconditions each 2D parameter as independent square blocks along its long dimension.

    A parameter of shape ``(m, n)`` with ``m >= n`` is split along the rows into blocks of shape
    ``(n, n)`` (column blocks when ``n > m``), and the full SOAP pipeline runs batched over the
    blocks: each block keeps its own kronecker factors, eigenbases, and eigenvalues, stored stacked
    as ``(ceil(m / n), n, n)`` (eigenvalues as ``(ceil(m / n), n)``). When ``m`` is not divisible by
    ``n``, the last block covers the remainder rows and is zero-padded to ``(n, n)`` internally; the
    padded slots are exact null directions of that block's kronecker factor, so they receive zero
    statistics and zero updates and the tail behaves as a smaller independent block. Compared to
    full SOAP this replaces the ``(m, m)`` factor and its eigendecomposition with batched ``(n, n)``
    ones, cutting preconditioner memory and the cubic decomposition cost at the price of dropping
    cross-block correlations (a block-diagonal approximation of the long-side preconditioner).

    The parameter's ``data`` and ``grad`` are swapped to their blocked 3D views only for the
    duration of :meth:`step`. For divisible shapes both views share storage with the original
    tensors, so the update writes through with no copy-back; non-divisible shapes go through a
    transient zero-padded buffer that is copied back after the step. ``param_groups``, ``state``,
    and ``state_dict`` are keyed by the real parameters, with state tensors sized to the blocked
    shape.

    Accepts the same arguments as :class:`SOAP`.
    """

    _supports_batched_params = True

    @torch.no_grad()  # type: ignore[misc]
    @override
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        saved: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            for p in group["params"]:
                if p.dim() == 2:
                    saved.append((p, p.data))
                    p.data = _to_blocks(p.data)
            super()._init_group(group, skip_non_grad_params=skip_non_grad_params)
        finally:
            for p, data in saved:
                p.data = data

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

        saved: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        try:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue  # pragma: no cover
                    saved.append((p, p.data, p.grad))
                    p.data = _to_blocks(p.data)
                    p.grad = _to_blocks(p.grad)
            super().step()
        finally:
            for p, data, grad in saved:
                blocked = p.data
                p.data = data
                p.grad = grad
                if blocked.data_ptr() != data.data_ptr():
                    _from_blocks(blocked, data)
        return None
