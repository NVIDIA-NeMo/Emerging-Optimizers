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

import torch
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc


class Scion(OrthogonalizedOptimizer):
    """Scion: Stochastic CondItional descent with Operator Norms

    Scion runs standard SGD-momentum and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, Newton-Schulz iteration is used, which has the
    advantage that it may be stably run on tensor cores on GPUs.

    This implementation incorporates `step_size` and `spectral_radius`, refer to Scion which views weight decay as constrained
    optimization via Frank-Wolfe.

    References:
        - *Training Deep Learning Models with Norm-Constrained LMOs.* arXiv:2502.07529 (2025).
          [`arXiv:2502.07529 <https://arxiv.org/abs/2502.07529>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer, or any 1-D
          parameters; those should all be optimized by the appropriate LMO for that layer. For example,
          for 1d params, it is scaled by the `ell_inf` radius.

    Args:
        {_args_doc}
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express"].
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        spectral_radius: The spectral radius to use for the update, we are scaling the LMO by this spectral radius.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        spectral_radius: float = 1.0,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        # Add checks for weight decay arguments to enable Franke-Wolfe step.
        logging.info("Scion does not use weight decay. Setting weight_decay to 1.")
        weight_decay = 1

        logging.info("Scion does not use weight decay. Setting use_decoupled_wd to True to allow Franke-Wolfe.")
        use_decoupled_wd = True

        logging.info("Scion does not use weight decay. Setting use_independent_wd to False to allow Franke-Wolfe.")
        use_independent_wd = False

        # Scion does not use Nesterov momentum.
        logging.info("Scion does not use Nesterov momentum. Setting use_nesterov to False.")
        use_nesterov = False

        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            logging.debug(
                f"Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, spectral_radius={spectral_radius}"
            )
            orth_grad = newton_schulz(grad, steps=num_ns_steps, coefficient_type=coefficient_type, use_syrk=False)
            width_factor = get_muon_scale_factor(grad.size(-2), grad.size(-1), mode="unit_rms_norm")
            return orth_grad * width_factor * spectral_radius

        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov,
            weight_decay,
            use_decoupled_wd,
            use_independent_wd,
            fp32_matmul_prec,
            scaled_orthogonalize_fn,
        )


Scion.__doc__ = Scion.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
