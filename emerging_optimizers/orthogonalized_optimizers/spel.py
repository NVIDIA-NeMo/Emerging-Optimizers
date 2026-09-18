from typing import Any, Literal, override

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry, utils
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["Spel", "SpelScaleT", "get_spel_scale_factor"]

SpelScaleT = Literal["unit_spectral_norm", "unit_rms_to_rms_norm"]


def get_spel_scale_factor(size_out: int, size_in: int, mode: SpelScaleT = "unit_spectral_norm") -> float:
    """Calculates the target scale factor for the Stiefel manifold constraint.

    Args:
        size_out: The output dimension (rows) of the weight matrix.
        size_in: The input dimension (columns) of the weight matrix.
        mode: The scaling mode.
            - "unit_spectral_norm": Unscaled Stiefel manifold where W^T W = I.
            - "unit_rms_to_rms_norm": Scaled Stiefel manifold preserving the RMS norm
              of activations across the layer. For tall matrices (size_out >= size_in),
              W^T W = (size_out / size_in) * I. For wide matrices (size_out < size_in),
              set the scale factor to 1.0 to avoid losing the RMS norm of activations.

    Returns:
        The scalar multiplier for the matrix sign function output.
    """
    if mode == "unit_spectral_norm":
        return 1.0
    elif mode == "unit_rms_to_rms_norm":
        return max(1.0, (size_out / size_in) ** 0.5)
    else:
        raise ValueError(f"Invalid mode for Spel update scale factor: {mode}")


@registry.register_optimizer("spel")
class Spel(OrthogonalizedOptimizer):
    r"""SPEL: SPectral steepest descent on the stiefEL manifold

    SPEL is the spectral-norm specialization of Manifold Constrained Steepest Descent (MCSD)
    on the Stiefel manifold. It selects a norm-induced steepest-descent direction via the matrix
    sign function applied to the momentum, then projects back onto the manifold via Newton-Schulz iteration:

    .. math::

        x_{{t+1}} = \sigma \text{{msign}}\!\left(x_t - \alpha_t \sigma \, \text{{msign}}\!\left(\nabla_{{\mathcal{{M}}}} f(x_t)\right)\right)

    The inner :math:`\text{{msign}}` orthogonalizes the gradient via Newton-Schulz iteration.
    The outer :math:`\text{{msign}}` re-projects the updated weights onto the Stiefel manifold,
    keeping parameters strictly semi-orthogonal.

    By setting `scale_mode="unit_rms_to_rms_norm"`, SPEL optimizes on a scaled Stiefel manifold,
    adjusting the retraction and tangent projections by :math:`\sigma = \max(1.0, \sqrt{{m/n}})` to inherently
    preserve the RMS norm of activations without requiring separate activation normalization layers.

    Note:
        Weight decay acts as a convex-like linear combination before the outer projection, controlling
        the relative influence of the old parameters versus the update direction while ensuring the
        final weights remain bound to the manifold constraint.

    References:
        - *Manifold Constrained Steepest Descent.* arXiv:2601.21487 (2026).
          [`arXiv:2601.21487 <https://arxiv.org/abs/2601.21487>`_]
        - *Modular Duality in Deep Learning.* arXiv:2410.21265 (2024).
          [`arXiv:2410.21265 <https://arxiv.org/abs/2410.21265>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for 1-D parameters (biases, layernorm weights) or the final
          classification head; those should be optimized by a standard method (e.g., AdamW).

    Args:
        {_args_doc}
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express"].
        num_ns_steps: The number of Newton-Schulz iteration steps for both gradient orthogonalization
            and the post-update weight projection.
        scale_mode: The constraint scaling mode. Defaults to "unit_spectral_norm".
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        *,
        nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        scale_mode: SpelScaleT = "unit_spectral_norm",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(X: torch.Tensor) -> torch.Tensor:
            logging.debug("Orthogonalizing with %s steps, %s coefficient", num_ns_steps, coefficient_type)
            return muon_utils.newton_schulz(
                X,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                use_syrk=False,
            )

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )
        self._scale_mode = scale_mode

    @override
    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Projects the gradient onto the tangent space before applying the LMO.

        Computes the orthogonal projection of the Euclidean gradient onto the tangent space of
        the (potentially scaled) Stiefel manifold at the current weights :math:`p`.

        The tangent space projection accounts for the metric via:
        .. math::
            P_{T_p}(G) = G - \frac{1}{\sigma^2} p \, \text{sym}(p^\top G)

        Args:
            p: The current parameter tensor.
            grad: The Euclidean gradient or momentum tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            The orthogonalized update direction scaled to match the manifold constraint.
        """
        if grad.ndim != 2:
            raise ValueError("Only 2D parameters are supported.")

        m, n = p.size(-2), p.size(-1)
        scale = get_spel_scale_factor(m, n, mode=self._scale_mode)
        scale_sq = scale * scale

        # Transpose wide matrices so we only ever solve the tall column-Stiefel projection
        is_wide = m < n
        if is_wide:
            p = p.mT
            grad = grad.mT

        # Tall column-Stiefel projection: contracts over the larger dimension,
        # yielding a small (min(m, n) x min(m, n)) symmetric matrix.
        pt_grad = torch.matmul(p.mT, grad)
        sym = 0.5 * (pt_grad + pt_grad.mT)
        reim_grad = grad - (1.0 / scale_sq) * torch.matmul(p, sym)

        if is_wide:
            reim_grad = reim_grad.mT

        return scale * self.scaled_orthogonalize_fn(reim_grad)

    @override
    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Re-projects the weight matrix onto the (scaled) Stiefel manifold after the update.

        Args:
            p: The updated parameter tensor.
        """
        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            orth_p = self.scaled_orthogonalize_fn(p)

        scale = get_spel_scale_factor(p.size(-2), p.size(-1), mode=self._scale_mode)
        p.copy_(scale * orth_p)


Spel.__doc__ = Spel.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
