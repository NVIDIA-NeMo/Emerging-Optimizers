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

    Each Kronecker factor is applied as ``(A + ridge * I)^(-1/p_root_inv)`` via the fixed-schedule
    CANS routines, with ``ridge = eps + 1e-7 * ||A||_inf``; the relative term pins the regularized
    spectrum inside the schedules' ``[1e-7, 1]`` design range, which is normalized by the same
    infinity norm. The KL-corrected factor update needs the
    eigendecomposition only through the regularized inverse ``Q diag(lambda^-1) Q^T``, so the dense
    inverses obtained by squaring the root inverses are stored as ``inv_L``/``inv_R`` state,
    refreshed at each :meth:`precondition` call and consumed by the next step's
    :meth:`update_kronecker_factors` -- the same one-step staleness as
    :class:`~emerging_optimizers.shampoo.shampoo.KlShampooPreconditioner`'s eigenbasis.

    Args:
        state: Per-parameter optimizer state holding ``L``, ``R``, ``inv_L``, and ``inv_R``.
        p_root_inv: Inverse root order; each factor is applied as ``A^(-1/p_root_inv)``.
        eps: Absolute term of the ridge added to the factors before inversion.

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
        self.inverse_pair = precond_base.TensorPair(state["inv_L"], state["inv_R"])

    @staticmethod
    @override
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        state = ShampooPreconditioner.init_state(shape, device)
        m, n = shape
        state.update(
            {
                # Initialize the inverses to identity to skip KL correction for the first step.
                # They will be overwritten immediately afterward.
                "inv_L": torch.eye(m, device=device),
                "inv_R": torch.eye(n, device=device),
            }
        )
        return state

    @override
    def rebind_state(self, state: dict) -> None:
        updates = {
            "L": self.kronecker_factor_pair.L,
            "R": self.kronecker_factor_pair.R,
            "inv_L": self.inverse_pair.L,
            "inv_R": self.inverse_pair.R,
        }
        missing = updates.keys() - state.keys()
        if missing:
            raise KeyError(f"rebind_state: state missing keys {sorted(missing)}")
        state.update(updates)

    @override
    def init_step(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        """Performs the first step's factor update, before any history exists.

        Args:
            grad: Gradient of the parameter on the first step.
            shampoo_beta: EMA coefficient for the Kronecker factor update.
        """
        self.update_kronecker_factors(grad, shampoo_beta)

    @override
    def update_kronecker_factors(self, grad: torch.Tensor, shampoo_beta: float) -> None:
        with utils.fp32_matmul_precision("highest"):
            update_L = eig_utils.conjugate(self.inverse_pair.R / grad.shape[1], grad.mT)
            update_R = eig_utils.conjugate(self.inverse_pair.L / grad.shape[0], grad)
            self.kronecker_factor_pair.L.lerp_(update_L, 1 - shampoo_beta)
            self.kronecker_factor_pair.R.lerp_(update_R, 1 - shampoo_beta)

    @override
    def precondition(self, x: torch.Tensor) -> torch.Tensor:
        with utils.fp32_matmul_precision("highest"):
            root_invs = []
            invs = []
            for factor in self.kronecker_factor_pair:
                ridge = self.eps + 1e-7 * torch.linalg.matrix_norm(factor, ord=float("inf"))
                regularized = factor + ridge * torch.eye(factor.shape[-1], device=factor.device)
                root_inv = self._root_inv_fn(regularized)
                inv = root_inv @ root_inv
                if self.p_root_inv == 4:
                    inv = inv @ inv
                root_invs.append(root_inv)
                invs.append(inv)

            self.inverse_pair = precond_base.TensorPair(invs[0], invs[1])
            return root_invs[0] @ x @ root_invs[1]


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
