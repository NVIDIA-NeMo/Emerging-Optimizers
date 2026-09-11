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
from typing import TYPE_CHECKING, Callable, Literal, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import Optimizer

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry


__all__ = [
    "ObliqueAdam",
    "ObliqueScaleT",
    "ObliqueSGD",
    "ObliqueSteepestAdam",
    "ObliqueSteepestSGD",
]

ObliqueScaleT = Literal["unit_l2_norm", "unit_rms_norm"]


@registry.register_optimizer("oblique_sgd")
class ObliqueSGD(opt_mixin.WeightDecayMixin, Optimizer):
    """SGD optimizer for row- or column-normalized 2D parameters on oblique manifolds.

    This optimizer performs SGD on oblique manifolds, where parameters are constrained
    to have unit-norm rows or columns. It implements Riemannian SGD with manifold-aware
    gradient updates and retraction operations.

    References:
        - An Introduction to Optimization on Smooth Manifolds (Nicolas Boumal)
        - EDM2: https://arxiv.org/abs/2312.02696
        - Jianlin Su: https://kexue.fm/archives/11196
        - Raman et al.: https://arxiv.org/abs/1909.06463
        - Franz Cesista: https://leloykun.github.io/ponder/steepest-descent-stiefel/#6-bonus-a-muon-like-optimizer-for-the-embedding-and-unembedding-layers

    Args:
        lr: learning rate
        momentum: momentum coefficient
        weight_decay: weight decay coefficient
        weight_decay_method: Method to apply weight decay.
        dim: The dimension to normalize over
        eps: epsilon for numerical stability
        scale_mode: The type of scale factor to use for the Riemannian gradient and retraction computation. Defaults to "unit_l2_norm" scaling.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        dim: int = 0,
        eps: float = 1e-8,
        scale_mode: ObliqueScaleT = "unit_l2_norm",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dim=dim, eps=eps, scale_mode=scale_mode)
        self.weight_decay_method = weight_decay_method
        super().__init__(params, defaults)

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
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            dim = group["dim"]
            eps = group["eps"]
            scale_mode = group["scale_mode"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise ValueError("ObliqueSGD only supports 2D parameters")

                scale = _compute_scale_factor(param, dim, scale_mode)
                grad = param.grad

                # Initialize momentum buffer if needed
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)

                buf = state["momentum_buffer"]

                # theory style momentum
                torch.add(grad, buf, alpha=mom, out=buf)

                riem_grad = self._get_riem_grad(param, buf, dim, scale, eps)

                self._apply_weight_decay_inplace(param, riem_grad, lr, wd)
                param.add_(riem_grad, alpha=-lr)

                # Retraction back to the manifold, the hyper-sphere
                torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=eps, out=param)
                param.mul_(scale)

        return None

    @torch.no_grad()
    def _get_riem_grad(
        self, param: torch.Tensor, buf: torch.Tensor, dim: int, scale: float, eps: float
    ) -> torch.Tensor:
        """Compute the Riemannian update direction from an ambient update buffer.

        This method defines how momentum buffers are mapped
        to the tangent space of the oblique manifold across optimizer variants:

        Projects the momentum buffer onto the tangent space T_W(OB) via Euclidean projection:
        proj_{T_W}(G) = G - W * (<W, G> / ||W||^2). The step magnitude remains proportional
        to the raw momentum norm.
        """
        return _compute_riemannian_grad(param, buf, dim, eps=eps)


@registry.register_optimizer("oblique_steepest_sgd")
class ObliqueSteepestSGD(ObliqueSGD):
    """Muon-style steepest descent optimizer for parameters on Oblique manifolds.

    Unlike standard Riemannian SGD, this optimizer computes updates using a Linear Minimization
    Oracle (LMO) with respect to the $l_1 \to \text{RMS}$ norm constraint. Mathematically, this
    is equivalent to finding the Euclidean descent direction, projecting it onto the tangent space,
    and normalizing the resulting directional vector.

    This approach effectively strips the magnitude of the gradient, applying uniform step sizes
    across all dimensions, mirroring the behavior of the Muon optimizer for Stiefel manifolds when the input features are one-hot encoded.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        dim: int = 0,
        eps: float = 1e-8,
        scale_mode: ObliqueScaleT = "unit_rms_norm",
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            dim=dim,
            eps=eps,
            scale_mode=scale_mode,
        )

    @override
    @torch.no_grad()
    def _get_riem_grad(
        self, param: torch.Tensor, buf: torch.Tensor, dim: int, scale: float, eps: float
    ) -> torch.Tensor:
        """Compute the Riemannian update direction from an ambient update buffer.

        This method defines how momentum buffers are mapped
        to the tangent space of the oblique manifold across optimizer variants:

        Projects onto the tangent space and then normalizes each slice along `dim` to unit
        RMS (or L2) norm: normalize(proj_{T_W}(G)) * scale. This solves the L1 -> RMS Linear
        Minimization Oracle (LMO), stripping update magnitude variations across slices and
        enforcing uniform angular step sizes in the chosen dimension.
        """
        riem_grad = _compute_riemannian_grad(param, buf, dim, eps=eps)
        torch.nn.functional.normalize(riem_grad, p=2.0, dim=dim, eps=eps, out=riem_grad)
        riem_grad.mul_(scale)
        return riem_grad


@registry.register_optimizer("oblique_adam")
class ObliqueAdam(opt_mixin.WeightDecayMixin, Optimizer):
    """Adam optimizer for row- or column-normalized 2D parameters on oblique manifolds.

    This optimizer adapts an Adam-like algorithm to work on oblique manifolds, where
    parameters are constrained to have unit-norm rows or columns. It combines
    adaptive momentum estimation with Riemannian gradient computation and manifold retraction.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        dim: int = 0,
        eps: float = 1e-8,
        correct_bias: bool = True,
        scale_mode: ObliqueScaleT = "unit_l2_norm",
    ) -> None:
        """An Adam-like optimizer for Normalized 2d Parameters

        Args:
            lr: The learning rate.
            betas: The coefficients used for computing running averages of gradient and its square.
            weight_decay: The weight decay coefficient.
            weight_decay_method: Method to apply weight decay.
            dim: The dimension to normalize over.
            eps: The epsilon for numerical stability.
            correct_bias: Whether to correct bias in Adam-like computation.
            scale_mode: The mode for scaling the Riemannian gradients and manifold retractions.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            dim=dim,
            eps=eps,
            correct_bias=correct_bias,
            scale_mode=scale_mode,
        )
        self.weight_decay_method = weight_decay_method
        super().__init__(params, defaults)

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
            lr = group["lr"]
            betas = group["betas"]
            wd = group["weight_decay"]
            dim = group["dim"]
            eps = group["eps"]
            correct_bias = group["correct_bias"]
            scale_mode = group["scale_mode"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise ValueError("ObliqueAdam only supports 2D parameters")

                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0

                scale = _compute_scale_factor(param, dim, scale_mode)
                grad = param.grad

                # Initialize momentum buffer if needed
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param)
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Increment step counter
                state["step"] += 1
                step = state["step"]

                # Update biased first and second moment estimates
                exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                if correct_bias:
                    # step size correction for ADAM moments EMA
                    bias_correction1 = 1.0 - betas[0] ** step
                    bias_correction2_sqrt = (1.0 - betas[1] ** step) ** 0.5
                else:
                    bias_correction1 = 1.0
                    bias_correction2_sqrt = 1.0

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                norm_grad = (exp_avg / bias_correction1) / denom

                riem_grad = self._get_riem_grad(param, norm_grad, dim, scale, eps)

                self._apply_weight_decay_inplace(param, riem_grad, lr, wd)
                param.add_(riem_grad, alpha=-lr)

                # Retraction back to the manifold, i.e. the hyper-sphere
                torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=eps, out=param)
                param.mul_(scale)

        return None

    @torch.no_grad()
    def _get_riem_grad(
        self, param: torch.Tensor, buf: torch.Tensor, dim: int, scale: float, eps: float
    ) -> torch.Tensor:
        """Compute the Riemannian update direction from an ambient update buffer.

        This method defines how Adam moments are mapped
        to the tangent space of the oblique manifold across optimizer variants:

        Projects the Adam moments onto the tangent space T_W(OB) via Euclidean projection:
        proj_{T_W}(G) = G - W * (<W, G> / ||W||^2). The step magnitude remains proportional
        to the Adam update norm.
        """
        return _compute_riemannian_grad(param, buf, dim, eps)


@registry.register_optimizer("oblique_steepest_adam")
class ObliqueSteepestAdam(ObliqueAdam):
    """Steepest descent Adam optimizer for parameters on Oblique manifolds.

    This optimizer blends the adaptive momentum estimation of Adam with the Muon-style normalized
    update direction. It computes the ambient Adam step, projects it onto the tangent space, and
    then normalizes the resulting vector.

    This guarantees that the final step direction incorporates Adam's historical second-moment
    scaling, while the actual step size taken is uniformly bounded by the learning rate via the
    Linear Minimization Oracle (LMO) constraint.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        dim: int = 0,
        eps: float = 1e-8,
        correct_bias: bool = True,
        scale_mode: ObliqueScaleT = "unit_rms_norm",
    ) -> None:
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            dim=dim,
            eps=eps,
            correct_bias=correct_bias,
            scale_mode=scale_mode,
        )

    @override
    @torch.no_grad()
    def _get_riem_grad(
        self, param: torch.Tensor, buf: torch.Tensor, dim: int, scale: float, eps: float
    ) -> torch.Tensor:
        """Compute the Riemannian update direction from an ambient update buffer.

        This method defines how Adam moments are mapped
        to the tangent space of the oblique manifold across optimizer variants:

        Projects onto the tangent space and then normalizes each slice along `dim` to unit
        RMS (or L2) norm: normalize(proj_{T_W}(G)) * scale. This solves the L1 -> RMS Linear
        Minimization Oracle (LMO), stripping update magnitude variations across slices and
        enforcing uniform angular step sizes in the chosen dimension.
        """
        riem_grad = _compute_riemannian_grad(param, buf, dim, eps)
        torch.nn.functional.normalize(riem_grad, p=2.0, dim=dim, eps=eps, out=riem_grad)
        riem_grad.mul_(scale)
        return riem_grad


def _compute_scale_factor(param: torch.Tensor, dim: int, scale_mode: ObliqueScaleT) -> float:
    """Compute the scale factor for the Riemannian gradient and retraction.

    Args:
        param: The parameter tensor.
        dim: The dimension over which to normalize.
        scale_mode: The scaling mode, either "unit_l2_norm" or "unit_rms_norm".

    Returns:
        The Oblique manifold scale factor.
    """
    if scale_mode == "unit_rms_norm":
        m = param.size(dim)
        scale = m**0.5
    elif scale_mode == "unit_l2_norm":
        scale = 1.0
    else:
        raise ValueError(f"Invalid scale mode: {scale_mode}")
    return scale


def _compute_riemannian_grad(
    param: torch.Tensor, grad_like: torch.Tensor, dim: int, eps: float = 1e-8
) -> torch.Tensor:
    num = (param * grad_like).sum(dim=dim, keepdim=True)
    den = (param * param).sum(dim=dim, keepdim=True).clamp_min(eps)
    inner = num / den
    tangent_proj = torch.add(grad_like, param * inner, alpha=-1.0)
    return tangent_proj
