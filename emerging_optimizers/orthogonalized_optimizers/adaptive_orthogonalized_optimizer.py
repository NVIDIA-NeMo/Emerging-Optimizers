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
from typing import Any, Callable, Literal


# TODO(@boxiangw): remove this once bump to python 3.12
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer


_adaptive_args_doc = """params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate used by the internal SGD.
        momentum_beta: The momentum used by the internal SGD.
        weight_decay: The weight decay used by the optimizer, default to be decoupled weight decay.
            See Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        use_nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        second_moment_method: Method to apply second moment. Options: "adamuon" (elementwise like AdamW), "normuon" (row/column-wise).
        beta2: The exponential decay rate for second moment (like AdamW's beta2).
        eps: Small constant for numerical stability in second moment normalization.
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`
            for more details.
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations.
"""


class AdaptiveOrthogonalizedOptimizer(OrthogonalizedOptimizer):
    """Orthogonalized optimizer with adaptive second moment (AdaMuon/NorMuon variants).

    This class extends OrthogonalizedOptimizer by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. The step() method is overridden to include second moment
    normalization logic.

    Args:
        {_adaptive_args_doc}
        scaled_orthogonalize_fn: Function to orthogonalize and scale the updates.
        **kwargs: Arguments passed through to the base optimizer.

    Note:
        Keyword arguments passed through are not checked here. Optimizer inherited from this class should check them.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum_beta: float,
        weight_decay: float,
        *,
        use_nesterov: bool,
        second_moment_method: Literal["adamuon", "normuon"],
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        fp32_matmul_prec: str,
        scaled_orthogonalize_fn: Callable | None = None,
        **kwargs: Any,
    ):
        self.second_moment_method = second_moment_method

        super().__init__(
            params=params,
            lr=lr,
            momentum_beta=momentum_beta,
            weight_decay=weight_decay,
            use_nesterov=use_nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
            **kwargs,
        )

        for group in self.param_groups:
            group["beta2"] = beta2
            group["eps"] = eps

    def _initialize_second_moment(
        self,
        state: dict[str, torch.Tensor],
        grad: torch.Tensor,
    ) -> None:
        """Initialize the second moment buffer if it doesn't exist.

        The shape of the buffer depends on the second_moment_method:
        - "adamuon": Full elementwise buffer with same shape as grad
        - "normuon": Reduced shape buffer (averaged along -1 if shape[-2] >= shape[-1], else -2)

        Args:
            state: The optimizer state dict for a parameter.
            grad: The gradient tensor (used for shape/dtype).
        """
        if "second_moment_buffer" not in state:
            if self.second_moment_method == "adamuon":
                # Full elementwise second moment
                second_moment = torch.zeros_like(grad)
            elif self.second_moment_method == "normuon":
                # Row/column-wise second moment - reduced along one dimension
                # Determine which dimension to reduce based on parameter shape
                avg_dim = -1 if grad.shape[-2] >= grad.shape[-1] else -2
                # Specify the shape with reduced dimension
                second_moment_shape = list(grad.shape)
                second_moment_shape[avg_dim] = 1
                second_moment = torch.zeros(second_moment_shape, dtype=grad.dtype, device=grad.device)
            else:
                raise ValueError(f"Invalid second moment method: {self.second_moment_method}")

            state["second_moment_buffer"] = second_moment

    def _apply_second_moment_normalization(
        self,
        orth_grad: torch.Tensor,
        second_moment: torch.Tensor,
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
            second_moment: The second moment buffer from state.
            beta2: The exponential decay rate for second moment.
            eps: Small constant for numerical stability.

        Returns:
            The adaptively scaled weight update tensor.
        """
        if self.second_moment_method == "adamuon":
            # AdamMuon: Full elementwise second moment like AdamW
            # Update second moment with EMA of squared gradient
            second_moment.lerp_(orth_grad.square(), 1 - beta2)

            # AdamW-style division: grad / (sqrt(second_moment) + eps)
            denom = second_moment.sqrt() + eps
            return orth_grad / denom

        elif self.second_moment_method == "normuon":
            # NorMuon: Row or column-wise second moment
            # Compute mean of squared gradients along one dimension based on shape
            # Average along the longer dimension to preserve structure along shorter dim
            avg_dim = -1 if orth_grad.shape[-2] >= orth_grad.shape[-1] else -2
            v_mean = orth_grad.square().mean(dim=avg_dim, keepdim=True)

            # Update second moment with EMA
            second_moment.lerp_(v_mean, 1 - beta2)

            # NorMuon uses reciprocal square root with clamping
            step_size = second_moment.clamp_min(eps).rsqrt_()
            return orth_grad * step_size

        else:
            raise ValueError(f"Invalid second moment method: {self.second_moment_method}")

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step with second moment normalization.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() == 1:
                    raise ValueError(f"{self.__class__.__name__} does not support 1D parameters")
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]

                # initialize momentum buffer and second moment buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    self._initialize_second_moment(state, grad)

                exp_avg = state["momentum_buffer"]

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                # update momentum buffer with EMA of gradient
                exp_avg.lerp_(grad, 1 - group["momentum_beta"])

                # include nesterov momentum
                if self.use_nesterov:
                    grad = grad.lerp(exp_avg, group["momentum_beta"])
                else:
                    grad = exp_avg

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    group_kwargs = {k: v for k, v in group.items() if k != "params"}
                    grad = self.orthogonalize(p, grad, **group_kwargs)

                # Apply second moment normalization
                grad = self._apply_second_moment_normalization(
                    orth_grad=grad,
                    second_moment=state["second_moment_buffer"],
                    beta2=group["beta2"],
                    eps=group["eps"],
                )

                # perform weight update
                # scale is applied to have update RMS == 1
                p.add_(grad, alpha=-group["lr"])

        return loss


AdaptiveOrthogonalizedOptimizer.__doc__ = AdaptiveOrthogonalizedOptimizer.__doc__.format(  # type: ignore[union-attr]
    _adaptive_args_doc=_adaptive_args_doc
)
