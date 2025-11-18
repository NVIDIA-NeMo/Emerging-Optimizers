# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Callable, Literal


# TODO(@boxiangw): remove this once bump to python 3.12
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon


class AdaptiveMuon(Muon):
    """Adaptive Muon optimizer with adaptive second moment (AdaMuon/NorMuon variants).

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum_beta: The exponential decay rate for momentum.
        weight_decay: Weight decay coefficient.
        use_nesterov: Whether to use Nesterov momentum.
        weight_decay_method: The weight decay method to use.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        use_syrk: Whether to use the Triton kernel for the Newton-Schulz iteration.
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum_beta: float,
        weight_decay: float,
        *,
        use_nesterov: bool,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        fp32_matmul_prec: str,
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        use_syrk: bool = False,
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum_beta=momentum_beta,
            weight_decay=weight_decay,
            use_nesterov=use_nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            use_syrk=use_syrk,
        )
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    def _initialize_moment2(
        self,
        state: dict[str, torch.Tensor],
        grad: torch.Tensor,
    ) -> None:
        """Initialize the second moment buffer if it doesn't exist.

        The shape of the buffer depends on the moment2_method:
        - "adamuon": Full elementwise buffer with same shape as grad
        - "normuon": Reduced shape buffer (averaged along -1 if shape[-2] >= shape[-1], else -2)

        Args:
            state: The optimizer state dict for a parameter.
            grad: The gradient tensor (used for shape/dtype).
        """
        if "moment2_buffer" not in state:
            if self.moment2_method == "adamuon":
                # Full elementwise second moment
                moment2 = torch.zeros_like(grad)
            elif self.moment2_method == "normuon":
                # Row/column-wise second moment - reduced along one dimension
                # Determine which dimension to reduce based on parameter shape
                avg_dim = -1 if grad.shape[-2] >= grad.shape[-1] else -2
                # Specify the shape with reduced dimension
                moment2_shape = list(grad.shape)
                moment2_shape[avg_dim] = 1
                moment2 = torch.zeros(moment2_shape, dtype=grad.dtype, device=grad.device)
            else:
                raise TypeError(f"Invalid second moment method: {self.moment2_method}")

            state["moment2_buffer"] = moment2

    def _apply_moment2_normalization(
        self,
        orth_grad: torch.Tensor,
        moment2: torch.Tensor,
        beta2: float,
        eps: float,
    ) -> torch.Tensor:
        """Apply AdamW-style second moment accumulation and normalization.

        This method supports two variants:
        - "adamuon": Full elementwise second moment (like AdamW, https://arxiv.org/abs/2507.11005)
        - "normuon": Row or column-wise second moment (https://arxiv.org/abs/2510.05491)

        For both methods:
        1. Updates the second moment as an EMA of squared gradients
        2. Returns the adaptively scaled gradient

        Args:
            orth_grad: The orthogonalized gradient tensor.
            moment2: The second moment buffer from state.
            beta2: The exponential decay rate for second moment.
            eps: Small constant for numerical stability.

        Returns:
            The adaptively scaled weight update tensor.
        """
        if self.moment2_method == "adamuon":
            # AdamMuon: Full elementwise second moment like AdamW
            # Update second moment with EMA of squared orthogonalized gradient
            moment2.lerp_(orth_grad.square(), 1 - beta2)

            # AdamW-style division: grad / (sqrt(moment2) + eps)
            denom = moment2.sqrt() + eps
            return orth_grad / denom

        elif self.moment2_method == "normuon":
            # NorMuon: Row or column-wise second moment
            # Compute mean of squared gradients along one dimension based on shape
            # Average along the longer dimension to preserve structure along shorter dim
            avg_dim = -1 if orth_grad.shape[-2] >= orth_grad.shape[-1] else -2
            v_mean = orth_grad.square().mean(dim=avg_dim, keepdim=True)

            # Update second moment with EMA
            moment2.lerp_(v_mean, 1 - beta2)

            # NorMuon uses reciprocal square root with clamping
            step_size = moment2.clamp_min(eps).rsqrt_()
            return orth_grad * step_size

        else:
            raise TypeError(f"Invalid second moment method: {self.moment2_method}")

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError("AdaptiveMuon only supports 2D parameters")
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    self._initialize_moment2(state, grad)

                exp_avg = state["momentum_buffer"]

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                # update momentum buffer with EMA of gradient
                exp_avg.lerp_(grad, 1 - group["momentum_beta"])

                if self.use_nesterov:
                    grad = grad.lerp(exp_avg, group["momentum_beta"])
                else:
                    grad = exp_avg

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    orth_grad = self.scaled_orthogonalize_fn(grad)

                update = self._apply_moment2_normalization(
                    orth_grad=orth_grad,
                    moment2=state["moment2_buffer"],
                    beta2=group["beta2"],
                    eps=group["eps"],
                )

                # perform weight update
                # scale is applied to have update RMS == 1
                p.add_(update, alpha=-group["lr"])

        return loss
