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
import contextlib
from typing import TYPE_CHECKING, Callable, ClassVar, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry, utils
from emerging_optimizers.experimental import shampoo_base
from emerging_optimizers.scalar_optimizers import update_functions
from emerging_optimizers.soap import soap
from emerging_optimizers.utils import eig as eig_utils


__all__ = [
    "KlMSoap",
    "KlSoapPreconditioner",
    "KlSoapV3",
    "SoapBase",
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
        self.kronecker_factor_pair = shampoo_base.TensorPair(state["L"], state["R"])
        self.eigenbasis_pair = shampoo_base.TensorPair(state["Q_L"], state["Q_R"])
        self.eigvals_pair = shampoo_base.TensorPair(state["eigvals_L"], state["eigvals_R"])
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

    def update_kronecker_factors(
        self,
        grad: torch.Tensor,
        shampoo_beta: float,
    ) -> None:
        """Accumulates the gradient into the kronecker factors with the KL-Shampoo correction.

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
        self.update_kronecker_factors(grad, shampoo_beta)

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

            self.eigenbasis_pair = shampoo_base.TensorPair(Q_L, Q_R)
            self.eigvals_pair = shampoo_base.TensorPair(eigvals_L, eigvals_R)

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


class SoapBase(optim.Optimizer, opt_mixin.WeightDecayMixin):
    """Canonical SOAP step loop, shared by the SOAP-family optimizers.

    :meth:`step` is the whole algorithm: update the preconditioner from the gradient, project the gradient
    into the eigenbasis, run an inner scalar optimizer there, and project the update back out. Subclasses
    customize the two pieces that actually vary between SOAP variants and leave the loop alone:

    - :attr:`PreconditionerCls` -- how the covariance factors and eigenbases are maintained.
    - :meth:`_scalar_update` -- which scalar optimizer runs inside the eigenbasis.

    ``betas`` lives here because every inner update in the family takes a pair of EMA coefficients.
    Constants specific to one inner update (MAdam's ``scale_log2``, for instance) belong on the subclass
    as class attributes, so that variants do not have to redefine ``__init__`` at all.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate to use
        betas: Inner scalar optimizer's betas parameters (b1, b2)
        shampoo_beta: Beta for the kronecker factor matrices (L and R in paper) moving average
        eps: Epsilon for the Kronecker factor update, passed to the preconditioner. Whether the inner
            update also uses it is up to the subclass.
        weight_decay: Weight decay coefficient
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis. The first step uses eigh
            regardless, since there is no eigenbasis to refine yet.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
        stream_list: Optional list of CUDA streams. When provided, each parameter in the inner loop uses a
            stream from this list in round-robin fashion.

    Attributes:
        PreconditionerCls: Preconditioner used for every parameter. Subclasses set it to change how the
            covariance factors and eigenbases are maintained; it must satisfy
            :class:`~emerging_optimizers.experimental.shampoo_base.SoapPreconditionerProtocol`. It is
            also what :meth:`_init_group` allocates state from, so a subclass that swaps it gets that
            preconditioner's state layout.
    """

    PreconditionerCls: ClassVar[type[shampoo_base.SoapPreconditionerProtocol]]

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        use_eigh: bool = False,
        max_update_rms: float = 0.0,
        stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        self.weight_decay_method = "decoupled"
        self.use_eigh = use_eigh
        self.betas = betas
        self.eps = eps
        self.max_update_rms = max_update_rms
        self.stream_list = stream_list

        defaults = {
            "lr": lr,
            "shampoo_beta": shampoo_beta,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def _scalar_update(
        self,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        """Applies the inner scalar optimizer to the projected gradient, in the eigenbasis.

        Override this to run a different scalar update inside the eigenbasis. The moment buffers are owned
        by the preconditioner, so an override may only use the buffers that ``PreconditionerCls.init_state``
        allocates -- an update needing a different set of buffers (the slow EMA of AdEMAMix, say) is a
        preconditioner change too, since every buffer living in the eigenbasis has to be re-projected when
        the eigenbasis rotates.

        Args:
            grad: Gradient projected into the eigenbasis.
            exp_avg: Inner optimizer's first moment, in the eigenbasis and updated in place.
            exp_avg_sq: Inner optimizer's second moment, in the eigenbasis and updated in place.
            step: Current optimizer step (1-based), used for bias correction.

        Returns:
            The scalar update, in the eigenbasis.

        Raises:
            NotImplementedError: Always; subclasses must provide the inner update.
        """
        raise NotImplementedError

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
                raise TypeError(f"{type(self).__name__} is only supported for 2D tensors")

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0

                # Use shape of p instead of grad for initialization because of the introduction of skip_non_grad_params
                # for megatron-lm distributed checkpointing use. _init_group can be called without grad.
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

        current_stream = torch.cuda.current_stream() if torch.cuda.is_available() else None

        if self.stream_list is not None and current_stream is not None:
            for stream in self.stream_list:
                stream.wait_stream(current_stream)

        for group in self.param_groups:
            for param_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue  # pragma: no cover

                stream_ctx: torch.cuda.StreamContext | contextlib.nullcontext[None] = contextlib.nullcontext()
                if self.stream_list is not None and current_stream is not None:
                    stream = self.stream_list[param_idx % len(self.stream_list)]
                    stream_ctx = torch.cuda.stream(stream)

                with stream_ctx:
                    grad = p.grad.to(torch.float32)
                    state = self.state[p]

                    curr_iter_1_based = state["step"] + 1

                    # Always use eigh for the first eigenbasis update
                    use_eigh = self.use_eigh or state["step"] == 0

                    # bias correction on shampoo beta
                    shampoo_beta = group["shampoo_beta"]
                    shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**curr_iter_1_based)

                    preconditioner = self.PreconditionerCls(state, self.eps, use_eigh)
                    preconditioner.step(grad, shampoo_beta)

                    self._apply_weight_decay_inplace(
                        p,
                        grad,
                        group["lr"],
                        group["weight_decay"],
                    )

                    # No matmul in adam update, put it under matmul precision context to make code for code simplicity
                    with utils.fp32_matmul_precision("highest"):
                        # Project gradients to the eigenbases of Shampoo's preconditioner
                        grad_projected = preconditioner.project_in(grad)

                        # Calculate the inner scalar update for the projected gradient tensor
                        scalar_update = self._scalar_update(
                            grad_projected,
                            preconditioner.exp_avg,
                            preconditioner.exp_avg_sq,
                            step=curr_iter_1_based,  # 1-based iteration index is used for bias correction
                        )

                        # Projecting back the preconditioned exponential moving average of gradients
                        precond_update = preconditioner.project_out(scalar_update)

                    soap._clip_update_rms_in_place(precond_update, self.max_update_rms)
                    p.add_(precond_update, alpha=-group["lr"])

                    # Preconditioner does both inplace and out-of-place changes, rebind state to make sure
                    # everything in state is properly updated
                    preconditioner.rebind_state(state)
                    state["step"] += 1

        if self.stream_list is not None and current_stream is not None:
            for stream in self.stream_list:
                current_stream.wait_stream(stream)

        return None


@registry.register_optimizer("kl_soap_v3")
class KlSoapV3(SoapBase):
    """Implements a variant of SOAP algorithm.

    Pairs the KL-Shampoo kronecker factor update with Adam as the inner scalar optimizer. Takes
    :class:`SoapBase`'s constructor unchanged, where ``betas``
    are inner Adam's and ``eps`` serves both inner Adam's denominator and the kronecker factor update.
    """

    PreconditionerCls: ClassVar[type[shampoo_base.SoapPreconditionerProtocol]] = KlSoapPreconditioner

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

    PreconditionerCls: ClassVar[type[shampoo_base.SoapPreconditionerProtocol]] = KlSoapPreconditioner

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
