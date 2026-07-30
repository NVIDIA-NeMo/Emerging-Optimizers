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

    A preconditioner owns the covariance factors of a single parameter and whatever it derives from them.
    Implementations are constructed from the per-parameter state dict and must write their tensors back
    with :meth:`rebind_state`, since the updates are partly out-of-place.

    An optimizer's ``PreconditionerCls`` is annotated ``ClassVar[type[...]]`` of one of the subclasses
    below, which is what checks a swapped-in preconditioner and lets the step loop be written against the
    interface rather than a concrete class. Note that mypy excludes ``__init__`` from protocol member
    checks, so the constructors declared here type the construction site but do not verify implementations.

    Positional-only parameters let implementations name the driving tensor after what it actually is -- a
    gradient for :class:`~emerging_optimizers.experimental.soap_v3.KlSoapPreconditioner`, a momentum for a
    Muon-style variant.
    """

    def __init__(self, state: dict, /, *args: Any, **kwargs: Any) -> None:
        """Binds the preconditioner to one parameter's state.

        Only ``state`` is fixed. The rest are whatever hyperparameters the preconditioner needs, passed by
        the optimizer that selects it, so implementations are free to differ there. Declaring the
        constructor at all is what makes construction through ``PreconditionerCls`` type-check; mypy
        excludes ``__init__`` from protocol member checks, so it does not verify implementations.

        Args:
            state: Per-parameter optimizer state to bind to.
            *args: Preconditioner-specific positional hyperparameters.
            **kwargs: Preconditioner-specific keyword hyperparameters.
        """

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

    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float, /) -> None:
        """Accumulates ``grad`` into the covariance factors.

        Exposed separately from :meth:`step` so that an optimizer can drive the factor update itself --
        for instance to run a different accumulation on the first step, before any eigenbasis or inverse
        root exists to correct with.

        Args:
            grad: Tensor driving the covariance update, in the parameter basis.
            shampoo_beta: EMA coefficient for the covariance factor update.
        """

    def step(self, grad: torch.Tensor, shampoo_beta: float, /) -> None:
        """Updates the covariance factors and everything derived from them.

        Equivalent to :meth:`update_kronecker_factors` followed by refreshing whatever the preconditioner
        derives from the factors -- an eigenbasis, or their inverse square roots.

        Args:
            grad: Tensor driving the covariance update, in the parameter basis.
            shampoo_beta: EMA coefficient for the covariance factor update.
        """

    def rebind_state(self, state: dict, /) -> None:
        """Writes the current preconditioner tensors back into the optimizer state dict.

        Args:
            state: Per-parameter optimizer state, updated in place.
        """


class SoapPreconditionerProtocol(_PreconditionerProtocol, Protocol):
    """A preconditioner that maintains an eigenbasis and the moments of an inner scalar optimizer.

    The moments live in the eigenbasis and are re-projected whenever the eigenbasis rotates, which is why
    they belong to the preconditioner rather than to the optimizer.
    """

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
    """A preconditioner that keeps inverse square roots of the covariance factors instead of an eigenbasis.

    It applies the roots directly, so it exposes a single :meth:`precondition` rather than a
    ``project_in`` / ``project_out`` pair, and it owns no moments -- the momentum lives in the parameter
    basis and is never re-projected, so the optimizer keeps it.
    """

    def precondition(self, x: torch.Tensor, /) -> torch.Tensor:
        """Applies the two-sided preconditioner to a matrix in the parameter basis.

        Args:
            x: Matrix in the parameter basis.

        Returns:
            The preconditioned matrix, in the parameter basis.
        """
