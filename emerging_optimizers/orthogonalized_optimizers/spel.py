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

from typing import override

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry, triton_kernels, utils
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["Spel"]


@registry.register_optimizer("spel")
class Spel(OrthogonalizedOptimizer):
    r"""SPEL: SPectral steepest descent on the stiefEL manifold

    SPEL is the spectral-norm specialization of Manifold Constrained Steepest Descent (MCSD)
    on the Stiefel manifold. It selects a norm-induced steepest-descent direction via the matrix
    sign function applied to the momentum, then projects back onto the manifold via Newton-Schulz iteration:

    .. math::

        x_{{t+1}} = \text{{msign}}\!\left(x_t - \alpha_t \, \text{{msign}}\!\left(\nabla_M f(x_t)\right)\right)

    The inner :math:`\text{{msign}}` orthogonalizes the gradient via Newton-Schulz iteration
    (identical to Muon, without scale factors). The outer :math:`\text{{msign}}` re-projects the
    updated weights onto the Stiefel manifold, keeping parameters on (or near) the orthogonal
    manifold. Both operations admit scalable implementations via fast matrix sign computations.

    Note:
        Weight decay still has an effect despite the re-orthogonalization. Before projection,
        :math:`(1 - \eta_t*\lambda) \, x_t - \eta_t \, \text{{msign}}(\cdot)` acts as a convex-like
        linear combination that rebalances the proportion of the previous weight and the new update.
        The outer :math:`\text{{msign}}` then re-orthogonalizes this mixture, so weight decay controls
        the relative influence of the old parameters versus the update direction.

    References:
        - *Manifold Constrained Steepest Descent.* arXiv:2601.21487 (2026).
          [`arXiv:2601.21487 <https://arxiv.org/abs/2601.21487>`_]
        - Jordan, K. *Muon Optimizer Implementation.*
          [`GitHub <https://github.com/KellerJordan/Muon/blob/master/muon.py>`_]
        - *Modular Duality in Deep Learning.* arXiv:2410.21265 (2024).
          [`arXiv:2410.21265 <https://arxiv.org/abs/2410.21265>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer, or any 1-D
          parameters; those can all be optimized by a standard method (e.g., AdamW).

    Args:
        {{_args_doc}}
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express"].
        num_ns_steps: The number of Newton-Schulz iteration steps for both gradient orthogonalization
            and the post-update weight projection.
        use_syrk: Whether to use the Triton kernel for the Newton-Schulz iteration.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        weight_decay: float = 0.1,
        *,
        use_nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        use_syrk: bool = False,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        if use_syrk:
            if torch.cuda.is_available():
                sm_version = torch.cuda.get_device_capability()
            else:
                sm_version = (0, 0)
            if not triton_kernels.HAS_TRITON_340:  # type: ignore[attr-defined]
                logging.error("Triton 3.4.0 or higher is required for use_syrk to be True.")
                use_syrk = False
            elif sm_version not in ((8, 0), (9, 0), (10, 0), (10, 3)):
                logging.error(
                    f"Correctness of Triton kernel on SM {sm_version} cannot be guaranteed. Setting use_syrk to False."
                )
                use_syrk = False

        def scaled_orthogonalize_fn(X: torch.Tensor) -> torch.Tensor:
            logging.debug(f"Orthogonalizing with {num_ns_steps} steps, {coefficient_type} coefficient")
            return muon_utils.newton_schulz(
                X,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                use_syrk=use_syrk,
            )

        super().__init__(
            params,
            lr,
            momentum_beta,
            weight_decay,
            use_nesterov=use_nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    @override
    def post_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Re-orthogonalize the weight matrix after the update via Newton-Schulz.

        This projects the updated weight matrix back onto (or near) the orthogonal manifold,
        implementing the outer msign in: x_{t+1} = msign(x_t - lr * msign(momentum)).

        Args:
            p: The updated parameter tensor.
            update: The orthogonalized gradient tensor that was applied.
        """
        with utils.fp32_matmul_precision(self.fp32_matmul_prec):
            orth_p = self.scaled_orthogonalize_fn(p)
        p.copy_(orth_p)


Spel.__doc__ = Spel.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
