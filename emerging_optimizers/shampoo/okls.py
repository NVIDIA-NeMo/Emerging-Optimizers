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
from typing import ClassVar, Literal, override

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry, utils
from emerging_optimizers.legacy_soap import matrix_root_inverse_utils
from emerging_optimizers.shampoo import precond_base
from emerging_optimizers.shampoo.shampoo import ShampooBase


__all__ = ["OKLS", "OklsPreconditioner"]


class OklsPreconditioner:
    """Online KL-Shampoo preconditioner with CANS root inverses."""

    def __init__(
        self,
        state: dict,
        p_root_inv: float,
        eps: float,
        fp32_matmul_prec: Literal["high", "highest"],
    ) -> None:
        if p_root_inv != 2:
            raise ValueError(f"OKLS only supports p_root_inv=2, got {p_root_inv}")
        self.p_root_inv = 2
        self.kronecker_factor_pair = precond_base.TensorPair(state["L"], state["R"])
        self.root_inverse_pair = precond_base.TensorPair(state["P_L"], state["P_R"])
        self.eps = eps
        self.fp32_matmul_prec = fp32_matmul_prec

    @staticmethod
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if len(shape) != 2:
            raise TypeError(f"OklsPreconditioner is only supported for 2D tensors, got shape {tuple(shape)}")
        m, n = shape
        return {
            "L": torch.zeros(m, m, device=device, dtype=torch.float32),
            "R": torch.zeros(n, n, device=device, dtype=torch.float32),
            "P_L": torch.zeros(m, m, device=device, dtype=torch.float32),
            "P_R": torch.zeros(n, n, device=device, dtype=torch.float32),
        }

    def rebind_state(self, state: dict) -> None:
        updates = {
            "L": self.kronecker_factor_pair.L,
            "R": self.kronecker_factor_pair.R,
            "P_L": self.root_inverse_pair.L,
            "P_R": self.root_inverse_pair.R,
        }
        missing = updates.keys() - state.keys()
        if missing:
            raise KeyError(f"rebind_state: state missing keys {sorted(missing)}")
        state.update(updates)

    def update_root_inverses(self) -> None:
        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            self.root_inverse_pair = precond_base.TensorPair(
                matrix_root_inverse_utils.mat_root_inv_via_scaled_cans(
                    self.kronecker_factor_pair.L,
                    eps=self.eps,
                ),
                matrix_root_inverse_utils.mat_root_inv_via_scaled_cans(
                    self.kronecker_factor_pair.R,
                    eps=self.eps,
                ),
            )

    def init_step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        m, n = grad.shape
        grad_norm = torch.linalg.vector_norm(grad, dtype=torch.float64)

        L = grad @ grad.T * math.sqrt(m / n) / grad_norm
        R = grad.T @ grad * math.sqrt(n / m) / grad_norm

        self.kronecker_factor_pair = precond_base.TensorPair((L + L.T) * 0.5, (R + R.T) * 0.5)
        self.update_root_inverses()

    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        m, n = grad.shape
        precond_grad_pair = precond_base.TensorPair(
            grad @ self.root_inverse_pair.R,
            self.root_inverse_pair.L @ grad,
        )

        L = torch.addmm(
            self.kronecker_factor_pair.L,
            precond_grad_pair.L,
            precond_grad_pair.L.T,
            beta=shampoo_beta,
            alpha=(1 - shampoo_beta) / n,
        )
        R = torch.addmm(
            self.kronecker_factor_pair.R,
            precond_grad_pair.R.T,
            precond_grad_pair.R,
            beta=shampoo_beta,
            alpha=(1 - shampoo_beta) / m,
        )
        self.kronecker_factor_pair = precond_base.TensorPair((L + L.T) * 0.5, (R + R.T) * 0.5)

    def step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        self.update_kronecker_factors(grad, shampoo_beta)
        self.update_root_inverses()

    def precondition(self, x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        shape_scale = math.sqrt(m / n) / (math.sqrt(m) + math.sqrt(n))
        return (self.root_inverse_pair.L @ x @ self.root_inverse_pair.R) * shape_scale


@registry.register_optimizer("okls")
class OKLS(ShampooBase):
    """Online KL-Shampoo with scaled CANS root inverses."""

    PreconditionerCls: ClassVar[type[precond_base.ShampooPreconditionerProtocol]] = OklsPreconditioner

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float = 0.9684,
        shampoo_beta: float = 0.9482,
        eps: float = 1e-9,
        weight_decay: float = 0.0,
        *,
        cans_fp32_matmul_prec: Literal["high", "highest"] = "high",
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= shampoo_beta < 1.0:
            raise ValueError(f"Invalid shampoo_beta: {shampoo_beta}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.cans_fp32_matmul_prec = cans_fp32_matmul_prec
        super().__init__(
            params,
            lr,
            momentum,
            shampoo_beta,
            eps,
            weight_decay,
            p_root_inv=2,
        )

    @torch.compile
    @override
    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        *,
        momentum: float,
    ) -> torch.Tensor:
        exp_avg.lerp_(grad, 1 - momentum)
        nesterov_variance = ((1 - momentum) / (1 + momentum)) * (1 + 2 * momentum - 2 * momentum**3)
        return torch.lerp(grad, exp_avg, momentum) * nesterov_variance**-0.5
