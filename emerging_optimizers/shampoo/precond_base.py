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
import dataclasses
from collections.abc import Iterator
from typing import Any, Protocol

import torch


__all__ = [
    "ShampooPreconditionerProtocol",
    "SoapPreconditionerProtocol",
    "TensorPair",
]


@dataclasses.dataclass
class TensorPair:
    """A pair of tensors"""

    L: torch.Tensor
    R: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterates over the pair as ``L`` then ``R``."""
        return iter((self.L, self.R))


class _PreconditionerProtocol(Protocol):
    """Interface every preconditioner in the family must provide, for one parameter.

    Preconditioner is designed to be created and used in side each step() function call of torch optimizer
    """

    def __init__(self, state: dict, /, *args: Any, **kwargs: Any) -> None:
        """Binds the preconditioner to one parameter's state."""

    @staticmethod
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Creates the state entries this preconditioner owns for a parameter of the given shape.

        Called through ``PreconditionerCls`` so that an optimizer's ``_init_group`` allocates the state
        layout of whichever preconditioner is selected.

        Args:
            shape: Shape of the 2D parameter the preconditioner will be attached to.
            device: Device to allocate the state tensors on.

        Returns:
            The state entries owned by this preconditioner, keyed as :meth:`rebind_state` expects them.
        """

    def init_step(self, grad: torch.Tensor, shampoo_beta: float, /) -> None:
        """Performs the first step's update, before any history exists to correct with."""

    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float, /) -> None:
        """Accumulates ``grad`` into the Kronecker factors.

        KL correction or any other correction should be implemented in this function of a preconditioner class.
        """

    def step(self, grad: torch.Tensor, shampoo_beta: float, /) -> None:
        """Updates the preconditioner internal with latest grad"""

    def rebind_state(self, state: dict, /) -> None:
        """Binds the current preconditioner tensors back into the optimizer state dict."""


class SoapPreconditionerProtocol(_PreconditionerProtocol, Protocol):
    """Soap preconditioner which projects update from/to eigen bases"""

    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    def project_in(self, x: torch.Tensor, /) -> torch.Tensor:
        """Projects a tensor from the parameter basis into the eigenbasis.

        Args:
            x: Tensor in the parameter basis.

        Returns:
            The tensor expressed in the eigenbasis.
        """

    def project_out(self, x: torch.Tensor, /) -> torch.Tensor:
        """Projects a tensor from the eigenbasis back to the parameter basis.

        Args:
            x: Tensor in the eigenbasis.

        Returns:
            The tensor expressed in the parameter basis.
        """


class ShampooPreconditionerProtocol(_PreconditionerProtocol, Protocol):
    """Shampoo preconditioner"""

    def precondition(self, x: torch.Tensor, /) -> torch.Tensor:
        """Applies the two-sided preconditioner to a matrix in the parameter basis.

        Args:
            x: Matrix in the parameter basis.

        Returns:
            The preconditioned matrix, in the parameter basis.
        """
