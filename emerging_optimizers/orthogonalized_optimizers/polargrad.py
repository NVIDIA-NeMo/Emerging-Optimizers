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

from typing import Literal

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc
from emerging_optimizers.utils import FP32MatmulPrecT
from emerging_optimizers.utils.eig import eigh_with_fallback


__all__ = ["PolarGrad", "left_polargrad_orth_fn", "one_sided_polargrad_orth_fn", "right_polargrad_orth_fn"]


@registry.register_optimizer("polargrad")
class PolarGrad(OrthogonalizedOptimizer):
    """PolarGrad: Polar Gradient Methods.

    Note that the update is scaled by the nuclear norm of the momentum term. This is
    equivalent to solving the steepest descent w.r.t. the spectral norm, as opposed to the LMO formulation
    of Scion and Muon. To efficiently orthogonalize each update, Newton-Schulz iteration is used, which has the
    advantage that it may be stably run on tensor cores on GPUs.

    This implementation incorporates decoupled weight decay.

    References:
        - *PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective.* arXiv:2505.21799 (2025).
          [`arXiv:2505.21799 <https://arxiv.org/abs/2505.21799>`_]
        - Lau, T. T.-K. *PolarGrad Optimizer Implementation.*
          [`polar_grad.py <https://github.com/timlautk/polargrad/blob/main/polar_grad.py>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer (with the Newton-Schulz iteration), or any 1-D
          parameters; those can all be optimized by a standard method (e.g., AdamW).

    Args:
        {_args_doc}
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express", "cans"].
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        extra_scale_factor: The additional scale factor to use for the update. Setting it to 0.2 can closely match
            the update RMS norm of AdamW as suggested by https://arxiv.org/abs/2502.16982.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        *,
        nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "highest",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        extra_scale_factor: float = 1.0,
    ) -> None:
        if num_ns_steps < 0:
            # 0 NS steps is allowed for some tests to bypass Newton-Schulz iterations and have exact match.
            raise ValueError(f"num_ns_steps must be positive, got {num_ns_steps}")

        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            logging.debug(
                f"Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, "
                f"multiplied with the nuclear norm of grad, extra_scale_factor={extra_scale_factor}"
            )
            orth_grad = muon_utils.newton_schulz(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
            )
            scale_factor = (orth_grad * grad).sum()
            return orth_grad * scale_factor * extra_scale_factor

        super().__init__(
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )


PolarGrad.__doc__ = PolarGrad.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]


def one_sided_polargrad_orth_fn(
    grad: torch.Tensor,
    *,
    side: Literal["left", "right"],
    alpha: float = 1.0,
    center_rows: bool = False,
    eps: float = 1e-15,
    extra_scale_factor: float = 1.0,
) -> torch.Tensor:
    r"""One-sided (spectral) polar orthogonalization of a matrix.

    Orthogonalizes a single polar factor of ``G``, selected by ``side``:

    .. math::
        u_\text{left} = (G \, G^\top)^{-1/2} \, G, \qquad
        u_\text{right} = G \, (G^\top G)^{-1/2}, \qquad
        \text{update} = \lVert G \rVert_*^{\,\alpha} \, u

    ``side="right"`` suits tall matrices (e.g. an embedding or LM-head weight, ``vocab x hidden``);
    ``side="left"`` suits wide matrices (e.g. an MoE router weight, ``num_experts x hidden``).

    Args:
        grad: The (momentum) tensor to orthogonalize.
        side: Which polar factor to orthogonalize, ``"left"`` or ``"right"``.
        alpha: Exponent applied to the nuclear-norm scale factor.
        center_rows: If True, subtract the per-column mean (the average over the row axis, ``dim=0``)
            before and after the update, so each column is zero-mean.
        eps: Floor on the Gram eigenvalues for the nuclear-norm computation.
        extra_scale_factor: Extra multiplier on the update.

    Returns:
        The scaled one-sided polar update, same shape and dtype as ``grad``.

    Raises:
        ValueError: If ``side`` is not ``"left"`` or ``"right"``.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    m = grad.to(torch.float32)
    if center_rows:
        m = m - m.mean(dim=0, keepdim=True)

    gram = m @ m.transpose(-1, -2) if side == "left" else m.transpose(-1, -2) @ m
    eigvals, eigvecs = eigh_with_fallback(gram)
    eigvals.clamp_min_(eps)
    cutoff = eigvals.amax() * gram.shape[-1] * torch.finfo(m.dtype).eps
    inv_sqrt_eigvals = torch.where(eigvals > cutoff, eigvals.rsqrt(), torch.zeros_like(eigvals))
    gram_inv_sqrt = (eigvecs * inv_sqrt_eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)

    u = gram_inv_sqrt @ m if side == "left" else m @ gram_inv_sqrt
    nuclear_norm = eigvals.sqrt().sum()
    update = u * nuclear_norm.pow(alpha) * extra_scale_factor

    if center_rows:
        update = update - update.mean(dim=0, keepdim=True)
    return update.to(grad.dtype)


def right_polargrad_orth_fn(
    grad: torch.Tensor,
    *,
    alpha: float = 1.0,
    center_rows: bool = False,
    eps: float = 1e-15,
    extra_scale_factor: float = 1.0,
) -> torch.Tensor:
    r"""Right-spectral orthogonalization for tall matrices (e.g. embedding or LM-head weights).

    Equivalent to :func:`one_sided_polargrad_orth_fn` with ``side="right"``.
    """
    return one_sided_polargrad_orth_fn(
        grad, side="right", alpha=alpha, center_rows=center_rows, eps=eps, extra_scale_factor=extra_scale_factor
    )


def left_polargrad_orth_fn(
    grad: torch.Tensor,
    *,
    alpha: float = 1.0,
    center_rows: bool = False,
    eps: float = 1e-15,
    extra_scale_factor: float = 1.0,
) -> torch.Tensor:
    r"""Left-spectral orthogonalization for wide matrices (e.g. MoE router weights).

    Equivalent to :func:`one_sided_polargrad_orth_fn` with ``side="left"``.
    """
    return one_sided_polargrad_orth_fn(
        grad, side="left", alpha=alpha, center_rows=center_rows, eps=eps, extra_scale_factor=extra_scale_factor
    )
