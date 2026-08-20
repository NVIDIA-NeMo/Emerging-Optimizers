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
from typing import TYPE_CHECKING, Callable, ClassVar, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry
from emerging_optimizers.shampoo import precond_base
from emerging_optimizers.utils import eig as eig_utils


__all__ = [
    "Shampoo",
    "ShampooBase",
    "ShampooPreconditioner",
]


class ShampooPreconditioner:
    """Per-parameter Shampoo preconditioner holding the Kronecker factors of one 2D parameter.

    Args:
        state: Per-parameter optimizer state holding ``L`` and ``R``.
        p_root_inv: Inverse root order; each factor is applied as ``A^(-1/p_root_inv)``.
        eps: Floor on the eigenvalue magnitudes before inversion.
    """

    def __init__(
        self,
        state: dict,
        p_root_inv: float,
        eps: float,
    ) -> None:
        self.kronecker_factor_pair = precond_base.TensorPair(state["L"], state["R"])
        self.p_root_inv = p_root_inv
        self.eps = eps

    @staticmethod
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Creates the Kronecker factors for a parameter shape.

        Args:
            shape: Shape of the 2D parameter the preconditioner will be attached to.
            device: Device to allocate the state tensors on.

        Returns:
            The state entries owned by this preconditioner, keyed as :meth:`rebind_state` expects them.

        Raises:
            TypeError: If ``shape`` is not 2D.
        """
        if len(shape) != 2:
            raise TypeError(f"ShampooPreconditioner is only supported for 2D tensors, got shape {tuple(shape)}")
        m, n = shape
        return {
            "L": torch.zeros(m, m, device=device),
            "R": torch.zeros(n, n, device=device),
        }

    def rebind_state(self, state: dict) -> None:
        """Binds the current preconditioner tensors back into the optimizer state dict.

        Args:
            state: Per-parameter optimizer state, updated in place.

        Raises:
            KeyError: If ``state`` is missing any of the preconditioner keys.
        """
        updates = {
            "L": self.kronecker_factor_pair.L,
            "R": self.kronecker_factor_pair.R,
        }
        missing = updates.keys() - state.keys()
        if missing:
            raise KeyError(f"rebind_state: state missing keys {sorted(missing)}")
        state.update(updates)

    def init_step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        """Performs the first step's factor update, before any history exists.

        Args:
            grad: Gradient of the parameter on the first step.
            shampoo_beta: EMA coefficient for the Kronecker factor update.
        """
        # Changing initial kronecker factors to be epsilon along the diagonal to match the paper
        self.kronecker_factor_pair.L += self.eps * torch.eye(grad.shape[-2], device=grad.device)
        self.kronecker_factor_pair.R += self.eps * torch.eye(grad.shape[-1], device=grad.device)
        self.update_kronecker_factors(grad, shampoo_beta)

    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        """Accumulates the gradient outer products into the Kronecker factors.

        Args:
            grad: Gradient of the parameter.
            shampoo_beta: EMA coefficient for the Kronecker factor update.
        """
        self.kronecker_factor_pair.L.lerp_(grad @ grad.T, 1 - shampoo_beta)
        self.kronecker_factor_pair.R.lerp_(grad.T @ grad, 1 - shampoo_beta)

    def step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        """Updates the Kronecker factors with the latest gradient.

        Args:
            grad: Gradient of the parameter.
            shampoo_beta: EMA coefficient for the Kronecker factor update.
        """
        self.update_kronecker_factors(grad, shampoo_beta)

    def _get_root_inverse(self, kronecker_factor: torch.Tensor) -> torch.Tensor:
        """Computes ``kronecker_factor^(-1/p_root_inv)`` from its eigendecomposition.

        Args:
            kronecker_factor: left or right kronecker factor

        Returns:
            The inverse root of the factor.
        """
        eigvals, eigvecs = eig_utils.eigh_with_fallback(kronecker_factor)

        # Eigh can sometime return negative values for numerical 0; clamping to 0 removes them
        eigvals = eigvals.clamp_min(0)

        # Tikhonov regularization
        exp = 1.0 / self.p_root_inv
        inv_root_scale = eigvals**exp / (eigvals ** (2 * exp) + self.eps ** (2 * exp))
        return (eigvecs * inv_root_scale) @ eigvecs.mT

    def precondition(self, x: torch.Tensor) -> torch.Tensor:
        """Applies both root inverse to a matrix in the parameter basis.

        Args:
            x: Matrix in the parameter basis.

        Returns:
            The preconditioned matrix, in the parameter basis.
        """
        root_inv_L = self._get_root_inverse(self.kronecker_factor_pair.L)
        root_inv_R = self._get_root_inverse(self.kronecker_factor_pair.R)

        return root_inv_L @ x @ root_inv_R


class ShampooBase(optim.Optimizer, opt_mixin.WeightDecayMixin):
    """Canonical Shampoo step loop, shared by the Shampoo-family optimizers.

    :meth:`step` is the whole algorithm: update the preconditioner from the gradient, run an inner scalar
    optimizer in the parameter basis, and precondition its update on both sides. Subclasses customize the
    two pieces that vary between Shampoo variants and leave the loop alone:

    - :attr:`PreconditionerCls` -- how the Kronecker factors and their inverse roots are maintained.
    - :meth:`_scalar_update` -- which scalar optimizer produces the update being preconditioned.

    Args:
        params: Iterable of 2D CUDA parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: Momentum EMA coefficient.
        shampoo_beta: Kronecker factor EMA coefficient.
        eps: Numerical epsilon
        weight_decay: Decoupled weight decay coefficient.
        p_root_inv: Inverse root order applied to each Kronecker factor.

    Attributes:
        PreconditionerCls: Preconditioner used for every parameter, and the source of the state layout
            allocated by :meth:`_init_group`. Subclasses set it to change how the factors are maintained.
    """

    PreconditionerCls: ClassVar[type[precond_base.ShampooPreconditionerProtocol]]

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float = 0.9,
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        p_root_inv: float = 4,
    ) -> None:
        self.eps = eps
        self.weight_decay_method = "decoupled"
        self.p_root_inv = p_root_inv

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if p_root_inv <= 0 or round(p_root_inv) != p_root_inv:
            raise ValueError(f"p_root_inv must be positive integer, got {p_root_inv}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "shampoo_beta": shampoo_beta,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        *,
        momentum: float,
    ) -> torch.Tensor:
        """Applies the inner scalar optimizer to the gradient, in the parameter basis.

        Override this to run a different scalar update ahead of the preconditioner.

        Args:
            grad: Gradient of the parameter.
            exp_avg: Momentum buffer, updated in place.
            momentum: Momentum EMA coefficient.

        Returns:
            The scalar update, in the parameter basis.

        Raises:
            NotImplementedError: Always; subclasses must provide the inner update.
        """
        raise NotImplementedError

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        """Performs lazy state initialization for parameters with gradients.

        Args:
            group: Parameter group dictionary.
            skip_non_grad_params: Whether to skip parameters with no gradients.

        Raises:
            TypeError: If the parameter is not a 2D tensor.
        """
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue

            if p.dim() != 2:
                raise TypeError(f"{type(self).__name__} is only supported for 2D tensors")

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)

                state.update(self.PreconditionerCls.init_state(p.shape, p.device))

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: Unsupported; must be ``None``.

        Raises:f
            ValueError: If ``closure`` is not ``None``.
        """
        if closure is not None:
            raise ValueError("closure is not supported")

        for group in self.param_groups:
            self._init_group(group)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue  # pragma: no cover

                grad = p.grad.to(torch.float32)
                state = self.state[p]

                preconditioner = self.PreconditionerCls(state, self.p_root_inv, self.eps)

                scalar_update = self._scalar_update(grad, state["exp_avg"], momentum=group["momentum"])

                # bias correction on shampoo beta
                curr_iter_1_based = state["step"] + 1
                shampoo_beta = group["shampoo_beta"]
                shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**curr_iter_1_based)

                if state["step"] == 0:
                    preconditioner.init_step(grad, shampoo_beta)
                else:
                    preconditioner.step(grad, shampoo_beta)
                preconditioned_update = preconditioner.precondition(scalar_update)

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )
                p.add_(preconditioned_update.to(p.dtype), alpha=-group["lr"])

                preconditioner.rebind_state(state)
                state["step"] += 1

        return None


@registry.register_optimizer("shampoo")
class Shampoo(ShampooBase):
    """Shampoo with EMA momentum as the inner scalar optimizer."""

    PreconditionerCls: ClassVar[type[precond_base.ShampooPreconditionerProtocol]] = ShampooPreconditioner

    @torch.compile
    @override
    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        *,
        momentum: float,
    ) -> torch.Tensor:
        """Applies EMA momentum to the gradient, in the parameter basis.

        Args:
            grad: Gradient of the parameter.
            exp_avg: Momentum buffer, updated in place.
            momentum: Momentum EMA coefficient.

        Returns:
            The momentum update, in the parameter basis.
        """
        exp_avg.lerp_(grad, 1 - momentum)
        return exp_avg
