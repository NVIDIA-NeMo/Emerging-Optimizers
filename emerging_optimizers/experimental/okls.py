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
from typing import TYPE_CHECKING, Callable, ClassVar, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry, utils
from emerging_optimizers.experimental import shampoo_base
from emerging_optimizers.soap import matrix_root_inverse_utils


__all__ = [
    "OKLS",
    "OklsPreconditioner",
]


class OklsPreconditioner:
    """Per-parameter online KL-Shampoo preconditioner holding the factors and their inverse roots.

    Each factor is updated from a gradient that has already been preconditioned by the *other* side's
    current inverse root, which is what makes the scheme online: the factors and the roots they are
    derived from never lag behind each other by more than the current step.

    Args:
        state: Per-parameter optimizer state holding ``L``/``R`` and ``P_L``/``P_R``.
        ridge_eps: Diagonal stability offset added to the covariance factors.
        fp32_matmul_prec: Precision used for the FP32 matmuls of the inverse-root iteration.
    """

    def __init__(
        self,
        state: dict,
        ridge_eps: float,
        fp32_matmul_prec: utils.FP32MatmulPrecT,
    ) -> None:
        self.kronecker_factor_pair = shampoo_base.TensorPair(state["L"], state["R"])
        self.inverse_root_pair = shampoo_base.TensorPair(state["P_L"], state["P_R"])
        self.ridge_eps = ridge_eps
        self.fp32_matmul_prec = fp32_matmul_prec

    @staticmethod
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Creates the covariance factors and their inverse roots for a parameter shape.

        Args:
            shape: Shape of the 2D parameter the preconditioner will be attached to.
            device: Device to allocate the state tensors on.

        Returns:
            The state entries owned by this preconditioner, keyed as :meth:`rebind_state` expects them.

        Raises:
            ValueError: If ``shape`` is not 2D.
        """
        if len(shape) != 2:
            raise ValueError(f"OklsPreconditioner is only supported for 2D tensors, got shape {tuple(shape)}")
        m, n = shape
        return {
            "L": torch.zeros(m, m, device=device, dtype=torch.float32),
            "R": torch.zeros(n, n, device=device, dtype=torch.float32),
            # Identity roots make the first factor update degenerate to the plain Gram products
            # ``G G^T`` and ``G^T G``, so the first step needs no separate seeding path.
            "P_L": torch.eye(m, device=device, dtype=torch.float32),
            "P_R": torch.eye(n, device=device, dtype=torch.float32),
        }

    def rebind_state(self, state: dict) -> None:
        """Writes the current preconditioner tensors back into the optimizer state dict.

        All updates here are in place, so this only re-binds the same tensors; it is kept for parity with
        the other experimental preconditioners and for the missing-key check.

        Args:
            state: Per-parameter optimizer state, updated in place.

        Raises:
            KeyError: If ``state`` is missing any of the preconditioner keys.
        """
        updates = {
            "L": self.kronecker_factor_pair.L,
            "R": self.kronecker_factor_pair.R,
            "P_L": self.inverse_root_pair.L,
            "P_R": self.inverse_root_pair.R,
        }
        missing = updates.keys() - state.keys()
        if missing:
            raise KeyError(f"rebind_state: state missing keys {sorted(missing)}")
        state.update(updates)

    def _refresh_inverse_roots(self) -> None:
        """Recomputes both inverse square roots from the current covariance factors."""
        self.inverse_root_pair = shampoo_base.TensorPair(
            *(
                matrix_root_inverse_utils.mat_root_inv_via_scaled_cans(
                    kronecker_factor,
                    eps=self.ridge_eps,
                    fp32_matmul_prec=self.fp32_matmul_prec,
                )
                for kronecker_factor in self.kronecker_factor_pair
            )
        )

    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        """Accumulates the cross-preconditioned gradient into both covariance factors.

        This is the KL-Shampoo factor update of
        :func:`~emerging_optimizers.soap.soap.update_kronecker_factors_kl_shampoo` at ``eigval_exp=-1``,
        reached through the inverse roots rather than an eigendecomposition: with ``P = A^{-1/2}``
        symmetric, ``(G P)(G P)^T`` is ``G A^{-1} G^T``. Both factors are driven by the roots from the
        *previous* refresh, so as in the reference each update sees the other factor as it was before this
        step rather than after.

        Kept separate from :meth:`_refresh_inverse_roots` so that ``precond_grad_pair`` dies with this
        frame instead of staying live across the inverse-root iteration, which allocates several square
        temporaries of its own.

        Args:
            grad: Gradient of the parameter, in the parameter basis.
            shampoo_beta: EMA coefficient for the covariance factor update.
        """
        # Paired by the factor each one feeds, so the side it was preconditioned on is the opposite one:
        # ``L`` is the gradient preconditioned on the right and drives the left factor, and vice versa. The
        # right entry is transposed so that both sides are the same ``X X^T`` contraction, which lets the
        # per-side dimensions fall out of ``X.shape`` instead of being spelled out.
        precond_grad_pair = shampoo_base.TensorPair(
            grad @ self.inverse_root_pair.R, (self.inverse_root_pair.L @ grad).T
        )

        updated_factors = []
        for kronecker_factor, precond_grad in zip(self.kronecker_factor_pair, precond_grad_pair, strict=True):
            # A <- shampoo_beta * A + (1 - shampoo_beta) / dim * X X^T, in one call. ``beta`` decays the
            # running factor and ``alpha`` scales the update; folding the 1/dim into lerp_'s single weight
            # instead would decay the factor by 1 - (1 - shampoo_beta) / dim rather than by shampoo_beta.
            kronecker_factor.addmm_(
                precond_grad,
                precond_grad.T,
                beta=shampoo_beta,
                alpha=(1 - shampoo_beta) / precond_grad.shape[1],
            )
            kronecker_factor = (kronecker_factor + kronecker_factor.T) * 0.5
            updated_factors.append(kronecker_factor)

        self.kronecker_factor_pair = shampoo_base.TensorPair(*updated_factors)

    def step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        """Updates the covariance factors from the cross-preconditioned gradient, then refreshes the roots.

        Args:
            grad: Gradient of the parameter, in the parameter basis.
            shampoo_beta: EMA coefficient for the covariance factor update.
        """
        self.update_kronecker_factors(grad, shampoo_beta)
        self._refresh_inverse_roots()

    def precondition(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the two-sided preconditioner to a matrix in the parameter basis.

        Args:
            x: Matrix in the parameter basis.

        Returns:
            The preconditioned matrix, in the parameter basis.
        """
        return self.inverse_root_pair.L @ x @ self.inverse_root_pair.R


@registry.register_optimizer("okls_v3")
class OKLS(optim.Optimizer, opt_mixin.WeightDecayMixin):
    """Online KL-Shampoo with scaled CANS inverse roots and zero-staleness preconditioning.

    The preconditioner is refreshed inside every step, so the inverse roots applied to the update are
    derived from factors that already saw the current gradient. The update is Nesterov momentum in the
    parameter basis, preconditioned on both sides and rescaled to unit RMS: ``momentum_scale`` undoes the
    variance the Nesterov blend introduces, and ``shape_scale`` is the usual Shampoo aspect-ratio factor.

    Args:
        params: Iterable of 2D CUDA parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        beta1: Nesterov momentum EMA coefficient.
        beta2: KL-Shampoo factor EMA coefficient.
        ridge_eps: Numerical stability offset added to the KL-Shampoo factors.
        weight_decay: Decoupled weight-decay coefficient.
        cans_fp32_matmul_prec: Precision used for FP32 matrix multiplications in CANS: ``"medium"`` for
            BF16, ``"high"`` for TF32, or ``"highest"`` for FP32.

    Attributes:
        PreconditionerCls: Preconditioner used for every parameter. Subclasses set it to change how the
            covariance factors and their inverse roots are maintained; it must satisfy
            :class:`~emerging_optimizers.experimental.shampoo_base.OklsPreconditionerProtocol`. It is
            also what :meth:`_init_group` allocates state from, so a subclass that swaps it gets that
            preconditioner's state layout.
    """

    PreconditionerCls: ClassVar[type[shampoo_base.ShampooPreconditionerProtocol]] = OklsPreconditioner

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float,
        beta1: float = 0.9684,
        beta2: float = 0.9482,
        ridge_eps: float = 1e-9,
        weight_decay: float = 0.0,
        cans_fp32_matmul_prec: utils.FP32MatmulPrecT = "high",
    ) -> None:
        self.weight_decay_method = "decoupled"
        self.cans_fp32_matmul_prec = cans_fp32_matmul_prec

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if ridge_eps < 0.0:
            raise ValueError(f"Invalid ridge epsilon: {ridge_eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "ridge_eps": ridge_eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.compile
    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        *,
        beta1: float,
    ) -> torch.Tensor:
        """Applies the inner scalar optimizer to the gradient, in the parameter basis.

        Override this to run a different scalar update ahead of the preconditioner. The result is expected
        to be normalized to unit RMS -- the Nesterov blend below divides by its own standard deviation --
        so an override has to supply whatever correction its own update needs rather than inherit this one.

        Args:
            grad: Gradient of the parameter.
            exp_avg: Momentum buffer, in the parameter basis and updated in place.
            beta1: Nesterov momentum EMA coefficient.

        Returns:
            The scalar update, in the parameter basis.
        """
        exp_avg.lerp_(grad, 1 - beta1)
        nesterov_variance = ((1 - beta1) / (1 + beta1)) * (1 + 2 * beta1 - 2 * beta1**3)
        return torch.lerp(grad, exp_avg, beta1) * nesterov_variance**-0.5

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
            TypeError: If the parameter is not a 2D CUDA tensor.
        """
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue

            if p.dim() != 2:
                raise TypeError(f"{type(self).__name__} is only supported for 2D tensors")
            if not p.is_cuda:
                raise TypeError(f"{type(self).__name__} only supports CUDA tensors")

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)

                # Use shape of p instead of grad for initialization because of the introduction of
                # skip_non_grad_params for megatron-lm distributed checkpointing use. _init_group can be
                # called without grad.
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

                preconditioner = self.PreconditionerCls(state, group["ridge_eps"], self.cans_fp32_matmul_prec)

                scalar_update = self._scalar_update(grad, state["exp_avg"], beta1=group["beta1"])

                preconditioner.step(grad, group["beta2"])
                preconditioned_update = preconditioner.precondition(scalar_update)

                # Aspect-ratio factor, a property of the preconditioned matrix rather than of the inner
                # scalar optimizer, so it stays out of _scalar_update.
                m, n = grad.shape
                shape_scale = math.sqrt(m / n) / (math.sqrt(m) + math.sqrt(n))

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )
                p.add_(preconditioned_update.to(p.dtype), alpha=-group["lr"] * shape_scale)

                preconditioner.rebind_state(state)
                state["step"] += 1

        return None
