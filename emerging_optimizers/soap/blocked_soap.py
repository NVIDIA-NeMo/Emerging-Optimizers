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
    if max(rows, cols) % min(rows, cols) != 0:
        raise ValueError(f"BlockedSoap requires the long dim to be divisible by the short dim. Got {tuple(x.shape)}")
    if rows >= cols:
        return x.view(rows // cols, cols, cols)
    return x.view(rows, cols // rows, rows).transpose(0, 1)


@registry.register_optimizer("blocked_soap")
class BlockedSoap(SOAP):
    """SOAP that preconditions each 2D parameter as independent square blocks along its long dimension.

    A parameter of shape ``(m, n)`` with ``m >= n`` is viewed as ``m // n`` row blocks of shape
    ``(n, n)`` (column blocks when ``n > m``), and the full SOAP pipeline runs batched over the
    blocks: each block keeps its own kronecker factors, eigenbases, and eigenvalues, stored stacked
    as ``(m // n, n, n)`` (eigenvalues as ``(m // n, n)``). Compared to full SOAP this replaces the
    ``(m, m)`` factor and its eigendecomposition with ``m // n`` batched ``(n, n)`` ones, cutting
    preconditioner memory and the cubic decomposition cost at the price of dropping cross-block
    correlations (a block-diagonal approximation of the long-side preconditioner). The long
    dimension must be divisible by the short dimension.

    The parameter's ``data`` and ``grad`` are swapped to their blocked 3D views only for the
    duration of :meth:`step`. Both views share storage with the original tensors, so the update
    writes through and no copy-back is needed; ``param_groups``, ``state``, and ``state_dict`` are
    keyed by the real parameters, with state tensors sized to the blocked shape.

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
                p.data = data
                p.grad = grad
        return None
