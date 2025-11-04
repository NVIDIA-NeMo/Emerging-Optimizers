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

from emerging_optimizers import triton_kernels
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc


class Scion(OrthogonalizedOptimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz

    Scion runs standard SGD-momentum and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, Newton-Schulz iteration is used, which has the
    advantage that it may be stably run on tensor cores on GPUs.

    This implementation incorporates `step_size` and `spectral_radius` refer to Scion which views weight decay as constrained
    optimization via Frank-Wolfe.

    References:
        - *Training Deep Learning Models with Norm-Constrained LMOs.* arXiv:2502.07529 (2025).
          [`arXiv:2502.07529 <https://arxiv.org/abs/2502.07529>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer, or any 1-D
          parameters; those should all be optimized by a standard method (e.g., AdamW).

    Args:
        {_args_doc}
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express"].
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update. Defaults to "spectral" style scaling.
        spectral_radius: The spectral radius to use for the update, we are scaling the LMO by this spectral radius.
        use_syrk: Whether to use the Triton kernel for the Newton-Schulz iteration.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = False,
        weight_decay: float = 1,
        use_decoupled_wd: bool = True,
        use_independent_wd: bool = False,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        spectral_radius: float = 1.0,
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

        # Add checks for weight decay arguments to enable Franke-Wolfe step.
        if weight_decay != 1:
            logging.warning("Scion does not use weight decay. Setting weight_decay to 1.")
            weight_decay = 1

        if not use_decoupled_wd:
            logging.warning("Scion does not use weight decay. Setting use_decoupled_wd to True to allow Franke-Wolfe.")
            use_decoupled_wd = True

        if use_independent_wd:
            logging.warning(
                "Scion does not use weight decay. Setting use_independent_wd to False to allow Franke-Wolfe."
            )
            use_independent_wd = False

        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            logging.debug(
                f"Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, spectral_radius={spectral_radius}"
            )
            orth_grad = newton_schulz(grad, steps=num_ns_steps, coefficient_type=coefficient_type, use_syrk=use_syrk)
            width_factor = (grad.size(-2) / grad.size(-1)) ** 0.5
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
            spectral_radius=spectral_radius,
        )


Scion.__doc__ = Scion.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
