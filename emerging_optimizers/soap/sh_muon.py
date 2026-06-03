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
from emerging_optimizers.orthogonalized_optimizers.muon import MuonScaleT, get_muon_scale_factor
from emerging_optimizers.soap import soap_utils
from emerging_optimizers.soap.soap import _clip_update_rms_in_place
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["ShMuon"]


@registry.register_optimizer("shmuon")
class ShMuon(opt_mixin.WeightDecayMixin, optim.Optimizer):
    r"""One-sided SOAP on Muon momentum.

    ShMuon tracks EMA momentum like Muon, accumulates a SOAP/Shampoo-style covariance of that momentum on the
    smaller matrix side, and applies a one-sided inverse-square-root update in the covariance eigenbasis.
    Conceptually, this is one-sided SOAP where ``G_t G_t^T`` is replaced by ``M_t M_t^T`` (or ``M_t^T M_t`` for the
    right side), and the update is computed by projecting the momentum into the eigenbasis, applying Adam-like
    normalization there, and projecting back:

    .. math::

        C_t = \beta_s C_{t-1} + (1 - \beta_s) M_t M_t^T,\quad C_t = Q_M \Lambda_M Q_M^T

        U_t = Q_M \left(\frac{Q_M^T M_t}{\sqrt{\operatorname{diag}(Q_M^T C_t Q_M)} + \epsilon}\right)

    for the left-preconditioned case where ``M_t.shape[0] <= M_t.shape[1]``; the right-preconditioned case uses
    ``C_t = M_t^T M_t`` and applies the same operation on the right. When ``momentum=0`` and ``shampoo_beta=0``,
    the preconditioner is the current momentum covariance and the update is the exact one-sided polar/Muon update
    up to ``eps``.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: EMA coefficient for the Muon-style momentum.
        shampoo_beta: EMA coefficient for the one-sided momentum covariance.
        eps: Small constant for numerical stability in the inverse square root.
        weight_decay: Weight decay coefficient.
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`.
        nesterov: Whether to use Nesterov momentum.
        correct_shampoo_beta_bias: Whether to bias-correct the covariance EMA.
        fp32_matmul_prec: Precision of the matmul operations in optimizer state GEMMs.
        use_eigh: Whether to use full symmetric eigendecomposition for eigenbasis updates after the first step.
        qr_fp32_matmul_prec: Precision of the matmul operations in QR decomposition.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
        scale_mode: Muon update scale mode.
        extra_scale_factor: Additional update scale factor.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        nesterov: bool = False,
        correct_shampoo_beta_bias: bool = True,
        fp32_matmul_prec: FP32MatmulPrecT = "highest",
        use_eigh: bool = False,
        qr_fp32_matmul_prec: FP32MatmulPrecT = "high",
        power_iter_steps: int = 1,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 1.0,
        max_update_rms: float = 0.0,
    ) -> None:
        self.nesterov = nesterov
        self.weight_decay_method = weight_decay_method
        self.correct_shampoo_beta_bias = correct_shampoo_beta_bias
        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_eigh = use_eigh
        self.qr_fp32_matmul_prec = qr_fp32_matmul_prec
        self.power_iter_steps = power_iter_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor
        self.max_update_rms = max_update_rms

        defaults = {
            "lr": lr,
            "momentum": momentum,
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
                raise TypeError("ShMuon is only supported for 2D tensors")

            state = self.state[p]
            if len(state) == 0:
                rows, cols = p.shape
                preconditioner_size = min(rows, cols)
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p.data, dtype=torch.float32)
                state["M"] = torch.zeros(preconditioner_size, preconditioner_size, device=p.device)
                state["Q_M"] = torch.eye(preconditioner_size, device=p.device)

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

                use_eigh = self.use_eigh if state["step"] != 0 else True
                with utils.fp32_matmul_precision(self.qr_fp32_matmul_prec):
                    state["Q_M"] = _update_one_sided_eigenbasis(
                        momentum_factor=state["M"],
                        eigenbasis=state["Q_M"],
                        use_eigh=use_eigh,
                        power_iter_steps=self.power_iter_steps,
                    )

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    update = _one_sided_eigenbasis_update(
                        momentum=momentum,
                        momentum_factor=state["M"],
                        eigenbasis=state["Q_M"],
                        eps=group["eps"],
                    )

                scale_factor = get_muon_scale_factor(momentum.shape[0], momentum.shape[1], mode=self.scale_mode)
                update = update * scale_factor * self.extra_scale_factor
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
    if momentum.shape[0] <= momentum.shape[1]:
        momentum_factor.lerp_(momentum @ momentum.T, 1 - shampoo_beta)
    else:
        momentum_factor.lerp_(momentum.T @ momentum, 1 - shampoo_beta)


@torch.no_grad()  # type: ignore[misc]
def _update_one_sided_eigenbasis(
    momentum_factor: torch.Tensor,
    eigenbasis: torch.Tensor,
    *,
    use_eigh: bool,
    power_iter_steps: int,
) -> torch.Tensor:
    """Update one eigenbasis for the smaller-side momentum covariance."""
    if use_eigh:
        return soap_utils.get_eigenbasis_eigh([momentum_factor])[0]

    approx_eigvals = utils.eig.conjugate(momentum_factor, eigenbasis, diag=True)
    sort_idx = torch.argsort(approx_eigvals, descending=True)
    sorted_eigenbasis = eigenbasis[:, sort_idx]
    return soap_utils.get_eigenbasis_qr(
        [momentum_factor],
        [sorted_eigenbasis],
        power_iter_steps=power_iter_steps,
    )[0]


@torch.no_grad()  # type: ignore[misc]
def _one_sided_eigenbasis_update(
    momentum: torch.Tensor,
    momentum_factor: torch.Tensor,
    eigenbasis: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize momentum by the smaller-side covariance in its eigenbasis."""
    eigvals = utils.eig.conjugate(momentum_factor, eigenbasis, diag=True).clamp_min(eps)
    if momentum.shape[0] <= momentum.shape[1]:
        projected = eigenbasis.T @ momentum
        return eigenbasis @ (projected * eigvals.rsqrt()[:, None])

    projected = momentum @ eigenbasis
    return (projected * eigvals.rsqrt()[None, :]) @ eigenbasis.T
