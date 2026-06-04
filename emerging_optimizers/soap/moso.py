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
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry, utils
from emerging_optimizers.scalar_optimizers import update_functions
from emerging_optimizers.soap import soap_utils
from emerging_optimizers.soap.soap import _clip_update_rms_in_place
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["MOSO"]


@registry.register_optimizer("moso")
class MOSO(opt_mixin.WeightDecayMixin, optim.Optimizer):
    r"""Momentum One-Sided SOAP.

    MOSO tracks EMA momentum like Muon, accumulates a SOAP/Shampoo-style covariance of that momentum on the
    smaller matrix side, and applies an Adam update in the covariance eigenbasis.
    Conceptually, this is one-sided SOAP where ``G_t G_t^T`` is replaced by ``M_t M_t^T`` (or ``M_t^T M_t`` for the
    right side), and the update is computed by projecting the momentum into the eigenbasis, applying Adam there, and
    projecting back:

    .. math::

        C_t = \beta_s C_{t-1} + (1 - \beta_s) M_t M_t^T,\quad C_t = Q_M \Lambda_M Q_M^T

        U_t = Q_M \operatorname{Adam}(Q_M^T M_t)

    for the left-preconditioned case where ``M_t.shape[0] <= M_t.shape[1]``; the right-preconditioned case uses
    ``C_t = M_t^T M_t`` and computes ``U_t = \operatorname{Adam}(M_t Q_M) Q_M^T``.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: EMA coefficient for the Muon-style momentum.
        betas: Inner Adam beta parameters ``(beta1, beta2)``.
        shampoo_beta: EMA coefficient for the one-sided momentum covariance.
        eps: Inner Adam epsilon for numerical stability.
        weight_decay: Weight decay coefficient.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        max_update_rms: float = 0.0,
    ) -> None:
        self.nesterov = False
        self.correct_bias = True
        self.weight_decay_method = "decoupled"
        self.correct_shampoo_beta_bias = True
        self.fp32_matmul_prec: FP32MatmulPrecT = "highest"
        self.use_eigh = False
        self.qr_fp32_matmul_prec: FP32MatmulPrecT = "highest"
        self.power_iter_steps = 1
        self.max_update_rms = max_update_rms

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        """Performs lazy state initialization for 2D parameters."""
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue

            if p.dim() != 2:
                raise TypeError("MOSO is only supported for 2D tensors")

            state = self.state[p]
            if len(state) == 0:
                rows, cols = p.shape
                preconditioner_size = min(rows, cols)
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p.data, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                state["M"] = torch.zeros(
                    preconditioner_size,
                    preconditioner_size,
                    device=p.device,
                    dtype=torch.float32,
                )
                state["Q_M"] = torch.eye(preconditioner_size, device=p.device, dtype=torch.float32)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step."""
        if closure is not None:
            raise ValueError("closure is not supported")

        for group in self.param_groups:
            self._init_group(group)

            for p in group["params"]:
                if p.grad is None:
                    continue  # pragma: no cover

                grad = p.grad.to(torch.float32)
                state = self.state[p]
                curr_iter_1_based = state["step"] + 1

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                if self.nesterov:
                    momentum = grad.lerp(state["momentum_buffer"], group["momentum"])
                else:
                    momentum = state["momentum_buffer"]

                shampoo_beta = group["shampoo_beta"]
                if self.correct_shampoo_beta_bias:
                    shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**curr_iter_1_based)

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    _update_one_sided_momentum_factor(
                        momentum_factor=state["M"],
                        momentum=momentum,
                        shampoo_beta=shampoo_beta,
                    )

                left_preconditioned = momentum.shape[0] <= momentum.shape[1]
                use_eigh = self.use_eigh if state["step"] != 0 else True
                with utils.fp32_matmul_precision(self.qr_fp32_matmul_prec):
                    state["Q_M"], state["exp_avg"], state["exp_avg_sq"] = _update_eigenbasis_and_adam_exp_avgs(
                        momentum_factor=state["M"],
                        eigenbasis=state["Q_M"],
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                        left_preconditioned=left_preconditioned,
                        use_eigh=use_eigh,
                        power_iter_steps=self.power_iter_steps,
                    )

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    momentum_projected = _project_to_one_sided_eigenbasis(
                        x=momentum,
                        eigenbasis=state["Q_M"],
                        left_preconditioned=left_preconditioned,
                    )
                    adam_update = update_functions.calculate_adam_update(
                        momentum_projected,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        betas=group["betas"],
                        eps=group["eps"],
                        correct_bias=self.correct_bias,
                        nesterov=False,
                        step=curr_iter_1_based,
                    )
                    update = _project_from_one_sided_eigenbasis(
                        x=adam_update,
                        eigenbasis=state["Q_M"],
                        left_preconditioned=left_preconditioned,
                    )

                _clip_update_rms_in_place(update, self.max_update_rms)
                p.add_(update, alpha=-group["lr"])

                state["step"] += 1

        return None


@torch.no_grad()  # type: ignore[misc]
def _update_one_sided_momentum_factor(
    momentum_factor: torch.Tensor,
    momentum: torch.Tensor,
    shampoo_beta: float,
) -> None:
    """Update the smaller-side covariance of the Muon momentum."""
    left_preconditioned = momentum.shape[0] <= momentum.shape[1]
    maybe_transposed_momentum = momentum if left_preconditioned else momentum.T
    momentum_factor.lerp_(maybe_transposed_momentum @ maybe_transposed_momentum.T, 1 - shampoo_beta)


@torch.no_grad()  # type: ignore[misc]
def _update_eigenbasis_and_adam_exp_avgs(
    momentum_factor: torch.Tensor,
    eigenbasis: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    left_preconditioned: bool,
    *,
    use_eigh: bool,
    power_iter_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update one eigenbasis and keep Adam state aligned with that basis."""
    exp_avg = _project_from_one_sided_eigenbasis(
        x=exp_avg,
        eigenbasis=eigenbasis,
        left_preconditioned=left_preconditioned,
    )

    eigenbasis, exp_avg_sq = _sort_one_sided_eigenbasis_and_exp_avg_sq(
        momentum_factor=momentum_factor,
        eigenbasis=eigenbasis,
        exp_avg_sq=exp_avg_sq,
        left_preconditioned=left_preconditioned,
    )

    if use_eigh:
        (updated_eigenbasis,) = soap_utils.get_eigenbasis_eigh([momentum_factor])
    else:
        (updated_eigenbasis,) = soap_utils.get_eigenbasis_qr(
            [momentum_factor],
            [eigenbasis],
            power_iter_steps=power_iter_steps,
        )

    exp_avg = _project_to_one_sided_eigenbasis(
        x=exp_avg,
        eigenbasis=updated_eigenbasis,
        left_preconditioned=left_preconditioned,
    )
    return updated_eigenbasis, exp_avg, exp_avg_sq


@torch.no_grad()  # type: ignore[misc]
def _sort_one_sided_eigenbasis_and_exp_avg_sq(
    momentum_factor: torch.Tensor,
    eigenbasis: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    left_preconditioned: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort eigenbasis slots by approximate eigenvalue and permute Adam second moments."""
    approx_eigvals = utils.eig.conjugate(momentum_factor, eigenbasis, diag=True)
    sort_idx = torch.argsort(approx_eigvals, descending=True, stable=True)
    sorted_eigenbasis = eigenbasis[:, sort_idx]
    exp_avg_sq_dim = 0 if left_preconditioned else 1
    return sorted_eigenbasis, exp_avg_sq.index_select(exp_avg_sq_dim, sort_idx)


@torch.no_grad()  # type: ignore[misc]
def _project_to_one_sided_eigenbasis(
    x: torch.Tensor,
    eigenbasis: torch.Tensor,
    left_preconditioned: bool,
) -> torch.Tensor:
    """Project a matrix into the smaller-side covariance eigenbasis."""
    if left_preconditioned:
        return eigenbasis.T @ x
    return x @ eigenbasis


@torch.no_grad()  # type: ignore[misc]
def _project_from_one_sided_eigenbasis(
    x: torch.Tensor,
    eigenbasis: torch.Tensor,
    left_preconditioned: bool,
) -> torch.Tensor:
    """Project a matrix from the smaller-side covariance eigenbasis."""
    if left_preconditioned:
        return eigenbasis @ x
    return x @ eigenbasis.T
