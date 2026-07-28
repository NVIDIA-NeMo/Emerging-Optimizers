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
from functools import partial
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry, utils
from emerging_optimizers.experimental.soap_v3 import PreconditionerProtocol, _clip_update_rms_in_place
from emerging_optimizers.scalar_optimizers import update_functions
from emerging_optimizers.utils import eig as eig_utils


__all__ = [
    "MosoPreconditioner",
    "MosoV3",
]


class MosoPreconditioner:
    """Per-parameter one-sided preconditioner holding the momentum covariance and its eigenbasis.

    The covariance is accumulated on the smaller matrix side, so ``M`` is always
    ``min(rows, cols) x min(rows, cols)``. ``precond_dim`` is the parameter dimension that side corresponds
    to -- ``0`` for rows (:math:`M M^T`, projecting as :math:`Q^T x`) and ``1`` for columns
    (:math:`M^T M`, projecting as :math:`x Q`) -- and therefore which way :meth:`project_in` and
    :meth:`project_out` multiply.

    Args:
        state: Per-parameter optimizer state holding ``M``, ``Q_M``, and ``exp_avg_sq``.
        use_eigh: Whether to use eigh (else orthogonal iteration) to update the eigenbasis.
    """

    def __init__(
        self,
        state: dict,
        use_eigh: bool,
    ) -> None:
        self.momentum_factor = state["M"]
        self.eigenbasis = state["Q_M"]
        self.exp_avg_sq = state["exp_avg_sq"]
        self.use_eigh = use_eigh
        self.precond_dim = 0 if self.exp_avg_sq.shape[0] <= self.exp_avg_sq.shape[1] else 1

    @staticmethod
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Creates the one-sided covariance, its eigenbasis, and the second moment for a parameter shape.

        Args:
            shape: Shape of the 2D parameter the preconditioner will be attached to.
            device: Device to allocate the state tensors on.
            dtype: Dtype of the state tensors.

        Returns:
            The state entries owned by this preconditioner, keyed as :meth:`rebind_state` expects them.

        Raises:
            ValueError: If ``shape`` is not 2D.
        """
        if len(shape) != 2:
            raise ValueError(f"MosoPreconditioner is only supported for 2D tensors, got shape {tuple(shape)}")
        m, n = shape
        preconditioner_size = min(m, n)
        return {
            "exp_avg_sq": torch.zeros(m, n, device=device, dtype=dtype),
            "M": torch.zeros(preconditioner_size, preconditioner_size, device=device, dtype=dtype),
            "Q_M": torch.eye(preconditioner_size, device=device, dtype=dtype),
        }

    def rebind_state(self, state: dict) -> None:
        """Writes the current preconditioner tensors back into the optimizer state dict.

        Args:
            state: Per-parameter optimizer state, updated in place.

        Raises:
            KeyError: If ``state`` is missing any of the preconditioner keys.
        """
        updates = {
            "M": self.momentum_factor,
            "Q_M": self.eigenbasis,
            "exp_avg_sq": self.exp_avg_sq,
        }
        missing = updates.keys() - state.keys()
        if missing:
            raise KeyError(f"rebind_state: state missing keys {sorted(missing)}")
        state.update(updates)

    def step(
        self,
        momentum: torch.Tensor,
        shampoo_beta: float,
    ) -> None:
        """Updates the one-sided momentum covariance and its eigenbasis.

        Args:
            momentum: Momentum of the parameter, in the parameter basis.
            shampoo_beta: EMA coefficient for the covariance update.
        """
        with utils.fp32_matmul_precision("highest"):
            maybe_transposed_momentum = momentum if self.precond_dim == 0 else momentum.mT
            self.momentum_factor.lerp_(maybe_transposed_momentum @ maybe_transposed_momentum.mT, 1 - shampoo_beta)

            if self.use_eigh:
                _, self.eigenbasis = eig_utils.eigh_with_fallback(self.momentum_factor)
            else:
                _, self.eigenbasis = eig_utils.orthogonal_iteration(
                    self.momentum_factor, self.eigenbasis, power_iter_steps=1
                )

    def project_in(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a tensor into the covariance eigenbasis.

        Args:
            x: Tensor to project.

        Returns:
            The tensor projected into the eigenbasis.
        """
        if self.precond_dim == 0:
            return self.eigenbasis.mT @ x
        return x @ self.eigenbasis

    def project_out(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a tensor out of the covariance eigenbasis, back to the original basis.

        Args:
            x: Tensor to project back.

        Returns:
            The tensor in the original basis.
        """
        if self.precond_dim == 0:
            return self.eigenbasis @ x
        return x @ self.eigenbasis.mT


if TYPE_CHECKING:
    # Static assertion only: mypy rejects the assignment if MosoPreconditioner stops satisfying the protocol.
    _moso_preconditioner_implements_protocol: type[PreconditionerProtocol] = MosoPreconditioner


@registry.register_optimizer("moso_v3")
class MosoV3(optim.Optimizer, opt_mixin.WeightDecayMixin):
    r"""Momentum One-Sided SOAP.

    MOSO tracks EMA momentum like Muon, accumulates a SOAP/Shampoo-style covariance of that momentum on the
    smaller matrix side, and applies an RMSProp update in the covariance eigenbasis:

    .. math::

        C_t = \beta_s C_{t-1} + (1 - \beta_s) M_t M_t^T,\quad C_t = Q_M \Lambda_M Q_M^T

        U_t = Q_M \operatorname{RMSprop}(Q_M^T M_t)

    for the left-preconditioned case where ``M_t.shape[0] <= M_t.shape[1]``; the right-preconditioned case
    uses ``C_t = M_t^T M_t`` and computes ``U_t = \operatorname{RMSprop}(M_t Q_M) Q_M^T``.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: EMA coefficient for the Muon-style momentum.
        rms_beta: EMA coefficient for the second-moment (RMS) normalization in the eigenbasis.
        shampoo_beta: EMA coefficient for the one-sided momentum covariance.
        eps: RMSProp epsilon for numerical stability.
        weight_decay: Weight decay coefficient.
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis. The first step uses eigh
            regardless, since there is no eigenbasis to refine yet.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        rms_beta: float = 0.95,
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        use_eigh: bool = False,
        max_update_rms: float = 0.0,
    ) -> None:
        self.weight_decay_method = "decoupled"
        self.use_eigh = use_eigh
        self.max_update_rms = max_update_rms

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "shampoo_beta": shampoo_beta,
            "weight_decay": weight_decay,
        }
        self.update_fun = partial(
            update_functions.calculate_rmsprop_update,
            alpha=rms_beta,
            eps=eps,
        )
        super().__init__(params, defaults)

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
                raise TypeError("MosoV3 is only supported for 2D tensors")

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p.data, dtype=torch.float32)

                # Use shape of p instead of grad for initialization because of the introduction of
                # skip_non_grad_params for megatron-lm distributed checkpointing use. _init_group can be
                # called without grad.
                state.update(MosoPreconditioner.init_state(p.shape, p.device))

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

                curr_iter_1_based = state["step"] + 1

                # Always use eigh for the first eigenbasis update
                use_eigh = self.use_eigh or state["step"] == 0

                # bias correction on shampoo beta
                shampoo_beta = group["shampoo_beta"]
                shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**curr_iter_1_based)

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                momentum = state["momentum_buffer"]

                preconditioner = MosoPreconditioner(state, use_eigh)
                preconditioner.step(momentum, shampoo_beta)

                with utils.fp32_matmul_precision("highest"):
                    # Project the momentum to the eigenbasis of the one-sided covariance
                    momentum_projected = preconditioner.project_in(momentum)

                    rmsprop_update = self.update_fun(
                        momentum_projected,
                        preconditioner.exp_avg_sq,
                        step=curr_iter_1_based,
                    )

                    # Projecting back the preconditioned (by RMSProp) momentum
                    update = preconditioner.project_out(rmsprop_update)

                _clip_update_rms_in_place(update, self.max_update_rms)
                p.add_(update, alpha=-group["lr"])

                # Preconditioner does both inplace and out-of-place changes, rebind state to make sure
                # everything in state is properly updated
                preconditioner.rebind_state(state)
                state["step"] += 1

        return None
