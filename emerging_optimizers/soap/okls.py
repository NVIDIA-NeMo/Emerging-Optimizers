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
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry
from emerging_optimizers.soap.matrix_root_inverse_utils import scaled_cans_coupled_ns
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["OKLS", "update_kronecker_factors_okls"]


def _update_inverse_roots(
    kronecker_factor_list: list[torch.Tensor],
    inverse_root_list: list[torch.Tensor],
    ridge_eps: float,
    cans_fp32_matmul_prec: FP32MatmulPrecT,
) -> None:
    for kronecker_factor, inverse_root in zip(kronecker_factor_list, inverse_root_list, strict=True):
        inverse_root.copy_(
            scaled_cans_coupled_ns(
                kronecker_factor,
                eps=ridge_eps,
                fp32_matmul_prec=cans_fp32_matmul_prec,
            )
        )


def _initialize_preconditioners(
    kronecker_factor_list: list[torch.Tensor],
    inverse_root_list: list[torch.Tensor],
    grad: torch.Tensor,
    ridge_eps: float,
    cans_fp32_matmul_prec: FP32MatmulPrecT,
) -> None:
    rows, cols = grad.shape
    grad_norm_sq = grad.square().sum()
    factor_left, factor_right = kronecker_factor_list

    factor_left.copy_(grad @ grad.T)
    factor_left.mul_(torch.sqrt(rows / (cols * grad_norm_sq + ridge_eps)))
    factor_left.copy_((factor_left + factor_left.T) / 2.0)
    diagonal_shift_left = torch.linalg.norm(factor_left) / math.sqrt(rows)
    factor_left.diagonal().add_(diagonal_shift_left + ridge_eps)

    factor_right.copy_(grad.T @ grad)
    factor_right.mul_(torch.sqrt(cols / (rows * grad_norm_sq + ridge_eps)))
    factor_right.copy_((factor_right + factor_right.T) / 2.0)
    diagonal_shift_right = torch.linalg.norm(factor_right) / math.sqrt(cols)
    factor_right.diagonal().add_(diagonal_shift_right + ridge_eps)

    _update_inverse_roots(
        kronecker_factor_list,
        inverse_root_list,
        ridge_eps,
        cans_fp32_matmul_prec,
    )


@torch.no_grad()  # type: ignore[misc]
def update_kronecker_factors_okls(
    kronecker_factor_list: list[torch.Tensor],
    inverse_root_list: list[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
    ridge_eps: float,
) -> None:
    """Update KL-Shampoo factors using the previous inverse-square-root preconditioners.

    Args:
        kronecker_factor_list: Left and right covariance factors.
        inverse_root_list: Previous inverse square roots of the left and right factors.
        grad: Matrix gradient.
        shampoo_beta: EMA coefficient for the factors.
        ridge_eps: Diagonal stability offset.
    """
    if grad.dim() != 2:
        raise TypeError("OKLS is only supported for 2D tensors")

    factor_left, factor_right = kronecker_factor_list
    inverse_root_left, inverse_root_right = inverse_root_list
    rows, cols = grad.shape

    grad_right_preconditioned = grad @ inverse_root_right
    factor_left.lerp_(grad_right_preconditioned @ grad_right_preconditioned.T / cols, 1 - shampoo_beta)
    factor_left.copy_((factor_left + factor_left.T) / 2.0)
    factor_left.diagonal().add_(ridge_eps)

    grad_left_preconditioned = inverse_root_left @ grad
    factor_right.lerp_(grad_left_preconditioned.T @ grad_left_preconditioned / rows, 1 - shampoo_beta)
    factor_right.copy_((factor_right + factor_right.T) / 2.0)
    factor_right.diagonal().add_(ridge_eps)


@registry.register_optimizer("okls")
class OKLS(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Online KL-Shampoo with scaled CANS inverse roots and zero-staleness preconditioning.

    Args:
        params: Iterable of 2D CUDA parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        beta1: Nesterov momentum EMA coefficient.
        beta2: KL-Shampoo factor EMA coefficient.
        ridge_eps: Numerical stability offset added to the KL-Shampoo factors.
        weight_decay: PaLM weight-decay coefficient.
        cans_fp32_matmul_prec: Precision used for FP32 matrix multiplications in CANS: ``"medium"`` for BF16,
            ``"high"`` for TF32, or ``"highest"`` for FP32.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float,
        beta1: float = 0.9684,
        beta2: float = 0.9482,
        ridge_eps: float = 1e-9,
        weight_decay: float = 0.0,
        cans_fp32_matmul_prec: FP32MatmulPrecT = "high",
    ) -> None:
        self.weight_decay_method = "palm"
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

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue

            if p.dim() != 2:
                raise TypeError("OKLS is only supported for 2D tensors")
            if not p.is_cuda:
                raise TypeError("OKLS only supports CUDA tensors")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["L"] = p.new_zeros((p.shape[0], p.shape[0]), dtype=torch.float32)
                state["R"] = p.new_zeros((p.shape[1], p.shape[1]), dtype=torch.float32)
                state["P_L"] = p.new_zeros((p.shape[0], p.shape[0]), dtype=torch.float32)
                state["P_R"] = p.new_zeros((p.shape[1], p.shape[1]), dtype=torch.float32)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

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
                kronecker_factor_list = [state["L"], state["R"]]
                inverse_root_list = [state["P_L"], state["P_R"]]
                ridge_eps = group["ridge_eps"]

                if state["step"] == 0:
                    _initialize_preconditioners(
                        kronecker_factor_list,
                        inverse_root_list,
                        grad,
                        ridge_eps,
                        self.cans_fp32_matmul_prec,
                    )

                beta1 = group["beta1"]
                state["exp_avg"].lerp_(grad, 1 - beta1)
                nesterov_momentum = torch.lerp(grad, state["exp_avg"], beta1)

                update_kronecker_factors_okls(
                    kronecker_factor_list=kronecker_factor_list,
                    inverse_root_list=inverse_root_list,
                    grad=grad,
                    shampoo_beta=group["beta2"],
                    ridge_eps=ridge_eps,
                )
                _update_inverse_roots(
                    kronecker_factor_list,
                    inverse_root_list,
                    ridge_eps,
                    self.cans_fp32_matmul_prec,
                )

                preconditioned_update = inverse_root_list[0] @ nesterov_momentum @ inverse_root_list[1]
                rows, cols = grad.shape
                nesterov_variance = ((1 - beta1) / (1 + beta1)) * (1 + 2 * beta1 - 2 * beta1**3)
                momentum_scale = nesterov_variance**-0.5
                shape_scale = math.sqrt(rows / cols) / (math.sqrt(rows) + math.sqrt(cols))

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )
                p.add_(
                    preconditioned_update.to(p.dtype),
                    alpha=-group["lr"] * momentum_scale * shape_scale,
                )
                state["step"] += 1

        return None
