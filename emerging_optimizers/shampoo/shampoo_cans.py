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
from typing import Callable, ClassVar, override

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry, utils
from emerging_optimizers.shampoo import matrix_root_inverse_utils, precond_base
from emerging_optimizers.shampoo.shampoo import KlShampoo, ShampooPreconditioner, ShampooScaleT
from emerging_optimizers.utils import eig as eig_utils


__all__ = [
    "KlShampooCans",
    "KlShampooCansPreconditioner",
]


class KlShampooCansPreconditioner(ShampooPreconditioner):
    """KL-corrected Shampoo preconditioner computing root inverses with CANS instead of eigh.

    Each Kronecker factor is applied as ``A^(-1/p_root_inv)`` via the fixed-schedule CANS routines.
    No explicit regularization is added: eigenvalues below the schedules' design floor (``1e-7``
    relative to the factor's infinity norm) saturate at a bounded value instead of diverging. The
    KL-corrected factor update needs the eigendecomposition only through the inverse
    ``Q diag(lambda^-1) Q^T``, which is recomputed from the current factors by squaring the CANS
    root inverses at each :meth:`update_kronecker_factors` call, so no state is kept beyond ``L``
    and ``R``.

    Args:
        state: Per-parameter optimizer state holding ``L`` and ``R``.
        p_root_inv: Inverse root order; each factor is applied as ``A^(-1/p_root_inv)``.
        eps: Unused; accepted to match the preconditioner construction interface.

    Raises:
        ValueError: If ``p_root_inv`` has no CANS schedule.
    """

    kl_corrected: ClassVar[bool] = True

    def __init__(
        self,
        state: dict,
        p_root_inv: float,
        eps: float,
    ) -> None:
        self._root_inv_fn: Callable[[torch.Tensor], torch.Tensor]
        if p_root_inv == 2:
            self._root_inv_fn = matrix_root_inverse_utils.mat_root_inv_via_scaled_cans
        elif p_root_inv == 4:
            self._root_inv_fn = matrix_root_inverse_utils.mat_4th_root_inv_via_scaled_cans
        else:
            raise ValueError(f"p_root_inv={p_root_inv} is not supported by CANS, supported orders: [2, 4]")
        super().__init__(state, p_root_inv, eps)

    @override
    def init_step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        """Performs the first step's factor update with an identity KL correction, before any history exists.

        Args:
            grad: Gradient of the parameter on the first step.
            shampoo_beta: EMA coefficient for the Kronecker factor update.
        """
        with utils.fp32_matmul_precision("highest"):
            self.kronecker_factor_pair.L.lerp_(grad @ grad.mT / grad.shape[1], 1 - shampoo_beta)
            self.kronecker_factor_pair.R.lerp_(grad.mT @ grad / grad.shape[0], 1 - shampoo_beta)

    def _inverse(self, factor: torch.Tensor) -> torch.Tensor:
        root_inv = self._root_inv_fn(factor)
        inverse = root_inv @ root_inv
        if self.p_root_inv == 4:
            inverse = inverse @ inverse
        return inverse

    @override
    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        with utils.fp32_matmul_precision("highest"):
            inv_L = self._inverse(self.kronecker_factor_pair.L)
            inv_R = self._inverse(self.kronecker_factor_pair.R)
            update_L = eig_utils.conjugate(inv_R / grad.shape[1], grad.mT)
            update_R = eig_utils.conjugate(inv_L / grad.shape[0], grad)
            self.kronecker_factor_pair.L.lerp_(update_L, 1 - shampoo_beta)
            self.kronecker_factor_pair.R.lerp_(update_R, 1 - shampoo_beta)

    @override
    def precondition(self, x: torch.Tensor) -> torch.Tensor:
        with utils.fp32_matmul_precision("highest"):
            root_inv_L = self._root_inv_fn(self.kronecker_factor_pair.L)
            root_inv_R = self._root_inv_fn(self.kronecker_factor_pair.R)
            return root_inv_L @ x @ root_inv_R


@registry.register_optimizer("kl_shampoo_cans")
class KlShampooCans(KlShampoo):
    """KL-Shampoo computing root inverses with CANS instead of eigh; supports ``p_root_inv`` in {2, 4}."""

    PreconditionerCls: ClassVar[type[precond_base.ShampooPreconditionerProtocol]] = KlShampooCansPreconditioner

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float = 0.9,
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        p_root_inv: float = 2,
        scale_mode: ShampooScaleT | None = "spectral",
    ) -> None:
        if p_root_inv not in (2, 4):
            raise ValueError(f"p_root_inv={p_root_inv} is not supported by CANS, supported orders: [2, 4]")
        super().__init__(
            params,
            lr,
            momentum,
            shampoo_beta,
            eps,
            weight_decay,
            p_root_inv=p_root_inv,
            scale_mode=scale_mode,
        )
