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
from typing import Callable, ClassVar, override

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry, utils
from emerging_optimizers.shampoo import matrix_root_inverse_utils, precond_base
from emerging_optimizers.shampoo.shampoo import KlShampoo, ShampooPreconditioner, ShampooScaleT
from emerging_optimizers.utils import eig as eig_utils


__all__ = [
    "LKlShampoo",
    "LKlShampooPreconditioner",
]


@torch.compile
def _pack_triu(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    k = torch.arange(dim * (dim + 1) // 2, device=x.device)
    # Invert the triangular numbering: row i of packed entry k, via float64 sqrt (verified exact
    # for dim up to 8192; the where-guard below never fired in profiling but stays as a safety net).
    i = ((2 * dim + 1 - torch.sqrt(((2 * dim + 1) ** 2 - 8 * k).double())) // 2).long()
    i = torch.where(k - i * (2 * dim - i + 1) // 2 < 0, i - 1, i)
    j = k - i * (2 * dim - i + 1) // 2 + i
    return x.reshape(-1)[i * dim + j]


@torch.compile
def _unpack_triu(packed: torch.Tensor) -> torch.Tensor:
    dim = (math.isqrt(8 * packed.shape[-1] + 1) - 1) // 2
    r = torch.arange(dim * dim, device=packed.device)
    i, j = r // dim, r % dim
    lo, hi = torch.minimum(i, j), torch.maximum(i, j)
    return packed[lo * (2 * dim - lo + 1) // 2 + (hi - lo)].view(dim, dim)


class LKlShampooPreconditioner(ShampooPreconditioner):
    """Limited-memory KL-corrected Shampoo preconditioner using CANS root inverses instead of eigh.

    Optimizer state holds only the packed upper triangles of the Kronecker factors ``L`` and ``R``
    (1-D tensors of length ``n * (n + 1) / 2``). Construction unpacks them into dense symmetric
    matrices for the duration of the step, and :meth:`rebind_state` repacks them, which also
    resymmetrizes by discarding the lower triangle. Each factor is applied as
    ``A^(-1/p_root_inv)`` via the fixed-schedule CANS routines. No explicit regularization is
    added: eigenvalues below the schedules' design floor (``1e-7`` relative to the factor's
    infinity norm) saturate at a bounded value instead of diverging. The KL-corrected factor
    update needs the eigendecomposition only through the inverse ``Q diag(lambda^-1) Q^T``, which
    is recomputed from the current factors by squaring the CANS root inverses at each
    :meth:`update_kronecker_factors` call, so no state is kept beyond the packed ``L`` and ``R``.

    Args:
        state: Per-parameter optimizer state holding the packed ``L`` and ``R``.
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
        self.kronecker_factor_pair = precond_base.TensorPair(_unpack_triu(state["L"]), _unpack_triu(state["R"]))
        state["L"] = self.kronecker_factor_pair.L
        state["R"] = self.kronecker_factor_pair.R

    @staticmethod
    @override
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if len(shape) != 2:
            raise TypeError(f"LKlShampooPreconditioner is only supported for 2D tensors, got shape {tuple(shape)}")
        m, n = shape
        return {
            "L": torch.zeros(m * (m + 1) // 2, device=device),
            "R": torch.zeros(n * (n + 1) // 2, device=device),
        }

    @override
    def rebind_state(self, state: dict) -> None:
        updates = {
            "L": _pack_triu(self.kronecker_factor_pair.L),
            "R": _pack_triu(self.kronecker_factor_pair.R),
        }
        missing = updates.keys() - state.keys()
        if missing:
            raise KeyError(f"rebind_state: state missing keys {sorted(missing)}")
        state.update(updates)

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


@registry.register_optimizer("l_kl_shampoo")
class LKlShampoo(KlShampoo):
    """Limited-memory KL-Shampoo: packed-triangle factor state and CANS root inverses, no eigh.

    Supports ``p_root_inv`` in {2, 4}.
    """

    PreconditionerCls: ClassVar[type[precond_base.ShampooPreconditionerProtocol]] = LKlShampooPreconditioner

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
