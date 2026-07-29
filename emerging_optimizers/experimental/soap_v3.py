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
from typing import ClassVar, override

import torch

from emerging_optimizers import registry, utils
from emerging_optimizers.experimental.shampoo import SoapBase, SoapPreconditionerFactory, TensorPair
from emerging_optimizers.scalar_optimizers import update_functions
from emerging_optimizers.soap import soap
from emerging_optimizers.utils import eig as eig_utils


__all__ = [
    "KlMSoap",
    "KlSoapPreconditioner",
    "KlSoapV3",
]


class KlSoapPreconditioner:
    """Per-parameter SOAP preconditioner holding the Kronecker factors, eigenbases, and eigenvalues.

    Args:
        state: Per-parameter optimizer state holding L/R, Q_L/R, eigvals_L/R, etc.
        eps: Epsilon for the KL-Shampoo Kronecker factor update.
        use_eigh: Whether to use eigh (else orthogonal iteration) to update the eigenbases.
    """

    def __init__(
        self,
        state: dict,
        eps: float,
        use_eigh: bool,
    ) -> None:
        self.kronecker_factor_pair = TensorPair(state["L"], state["R"])
        self.eigenbasis_pair = TensorPair(state["Q_L"], state["Q_R"])
        self.eigvals_pair = TensorPair(state["eigvals_L"], state["eigvals_R"])
        self.exp_avg, self.exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        self.eps = eps
        self.use_eigh = use_eigh

    @staticmethod
    def init_state(
        shape: tuple[int, ...],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Creates the Kronecker factors, eigenbases, eigenvalues, and moments for a parameter shape.

        Args:
            shape: Shape of the 2D parameter the preconditioner will be attached to.
            device: Device to allocate the state tensors on.

        Returns:
            The state entries owned by this preconditioner, keyed as :meth:`rebind_state` expects them.

        Raises:
            ValueError: If ``shape`` is not 2D.
        """
        if len(shape) != 2:
            raise ValueError(f"KlSoapPreconditioner is only supported for 2D tensors, got shape {tuple(shape)}")
        m, n = shape
        return {
            "exp_avg": torch.zeros(m, n, device=device),
            "exp_avg_sq": torch.zeros(m, n, device=device),
            "L": torch.zeros(m, m, device=device),
            "R": torch.zeros(n, n, device=device),
            "Q_L": torch.eye(m, device=device),
            "Q_R": torch.eye(n, device=device),
            "eigvals_L": torch.zeros(m, device=device),
            "eigvals_R": torch.zeros(n, device=device),
        }

    def rebind_state(self, state: dict) -> None:
        """Writes the current preconditioner tensors back into the optimizer state dict.

        Args:
            state: Per-parameter optimizer state, updated in place.

        Raises:
            KeyError: If ``state`` is missing any of the preconditioner keys.
        """
        updates = {
            "L": self.kronecker_factor_pair.L,
            "R": self.kronecker_factor_pair.R,
            "Q_L": self.eigenbasis_pair.L,
            "Q_R": self.eigenbasis_pair.R,
            "eigvals_L": self.eigvals_pair.L,
            "eigvals_R": self.eigvals_pair.R,
            "exp_avg": self.exp_avg,
            "exp_avg_sq": self.exp_avg_sq,
        }
        missing = updates.keys() - state.keys()
        if missing:
            raise KeyError(f"rebind_state: state missing keys {sorted(missing)}")
        state.update(updates)

    def step(
        self,
        grad: torch.Tensor,
        shampoo_beta: float,
    ) -> None:
        """Updates the kronecker factors and eigenbases, re-projecting exp_avg into the new eigenbasis.

        Args:
            grad: Gradient of the parameter.
            shampoo_beta: EMA coefficient for the kronecker factor update.
        """
        with utils.fp32_matmul_precision("highest"):
            soap.update_kronecker_factors_kl_shampoo(
                self.kronecker_factor_pair,
                grad,
                shampoo_beta,
                self.eigenbasis_pair,
                self.eigvals_pair,
                self.eps,
            )

        with utils.fp32_matmul_precision("high"):
            # Project exp_avg back to the original basis
            exp_avg = self.project_out(self.exp_avg)

            # Update eigenbases
            if self.use_eigh:
                eigvals_L, Q_L = eig_utils.eigh_with_fallback(self.kronecker_factor_pair.L)
                eigvals_R, Q_R = eig_utils.eigh_with_fallback(self.kronecker_factor_pair.R)
            else:
                eigvals_L, Q_L = eig_utils.orthogonal_iteration(
                    self.kronecker_factor_pair.L, self.eigenbasis_pair.L, power_iter_steps=1
                )
                eigvals_R, Q_R = eig_utils.orthogonal_iteration(
                    self.kronecker_factor_pair.R, self.eigenbasis_pair.R, power_iter_steps=1
                )

            self.eigenbasis_pair = TensorPair(Q_L, Q_R)
            self.eigvals_pair = TensorPair(eigvals_L, eigvals_R)

            # Project exp_avg to the new eigenbasis using the updated eigenbases
            self.exp_avg = self.project_in(exp_avg)

    def project_in(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a tensor into the eigenbasis.

        Args:
            x: Tensor to project.

        Returns:
            The tensor projected into the eigenbasis.
        """
        return self.eigenbasis_pair.L.mT @ x @ self.eigenbasis_pair.R

    def project_out(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a tensor out of the eigenbasis, back to the original basis.

        Args:
            x: Tensor to project back.

        Returns:
            The tensor in the original basis.
        """
        return self.eigenbasis_pair.L @ x @ self.eigenbasis_pair.R.mT


@registry.register_optimizer("kl_soap_v3")
class KlSoapV3(SoapBase):
    """Implements a variant of SOAP algorithm.

    Pairs the KL-Shampoo kronecker factor update with Adam as the inner scalar optimizer. Takes
    :class:`~emerging_optimizers.experimental.shampoo.SoapBase`'s constructor unchanged, where ``betas``
    are inner Adam's and ``eps`` serves both inner Adam's denominator and the kronecker factor update.
    """

    PreconditionerCls: ClassVar[SoapPreconditionerFactory] = KlSoapPreconditioner

    @override
    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        """Applies Adam to the projected gradient, in the eigenbasis.

        Args:
            grad: Gradient projected into the eigenbasis.
            exp_avg: Inner Adam's first moment, in the eigenbasis and updated in place.
            exp_avg_sq: Inner Adam's second moment, in the eigenbasis and updated in place.
            step: Current optimizer step (1-based), used for bias correction.

        Returns:
            The Adam update, in the eigenbasis.
        """
        return update_functions.calculate_adam_update(
            grad,
            exp_avg,
            exp_avg_sq,
            betas=self.betas,
            eps=self.eps,
            correct_bias=True,
            nesterov=False,
            step=step,
        )


@registry.register_optimizer("kl_m_soap")
class KlMSoap(SoapBase):
    """SOAP with the KL-Shampoo kronecker factor update and MAdam as the inner scalar optimizer."""

    PreconditionerCls: ClassVar[SoapPreconditionerFactory] = KlSoapPreconditioner

    @override
    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        """Applies MAdam to the projected gradient, in the eigenbasis.

        Args:
            grad: Gradient projected into the eigenbasis.
            exp_avg: Inner MAdam's first moment, in the eigenbasis and updated in place.
            exp_avg_sq: Inner MAdam's scaled second moment, in the eigenbasis and updated in place.
            step: Current optimizer step (1-based), used for bias correction.

        Returns:
            The MAdam update, in the eigenbasis.
        """
        return update_functions.calculate_madam_update(
            grad,
            exp_avg,
            exp_avg_sq,
            betas=self.betas,
            correct_bias=True,
            step=step,
            scale_log2=16.0,
        )
