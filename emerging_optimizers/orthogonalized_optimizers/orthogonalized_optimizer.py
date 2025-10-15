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
from typing import Any, Callable


# TODO(@boxiangw): remove this once bump to python 3.12
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch
import torch.optim as optim
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import utils


_args_doc = """params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate used by the internal SGD.
        momentum_beta: The momentum used by the internal SGD.
        use_nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        weight_decay: The weight decay used by the optimizer, default to be decoupled weight decay.
            See Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        use_decoupled_weight_decay: Whether to use decoupled weight decay, default to be True.
        split_fused: Whether to split fused parameters (QKV, GQA, etc.) for preconditioning, default to be False.
        is_fused_fn: Function to check if a parameter is fused parameters (QKV, GQA, etc.).
            If multiple types of parameters are fused, the function should return True for all of which needs to be
            split for preconditioning.
        split_fn: Function to split the fused parameters (QKV, GQA, etc.) into a list of parameters.
            It should support all the types of parameters that is_fused_fn returns True for.
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations.
"""


class OrthogonalizedOptimizer(optim.Optimizer):
    """Base class for orthogonalized optimizers.

    This class is a wrapper around a base optimizer that performs orthogonalization on the updates.
    The theoretical foundation of orthogonalization for stochastic gradient descent was developed by the following papers:

    - Carlson, D., Cevher, V., and Carin, L. *Stochastic spectral descent for Restricted Boltzmann Machines.*
      In International Conference on Artificial Intelligence and Statistics (2015a).
    - Carlson, D., Hsieh, Y.-P., Collins, E., Carin, L., and Cevher, V.
      *Stochastic Spectral Descent for Discrete Graphical Models.*
      In IEEE Journal of Selected Topics in Signal Processing, vol. 10, no. 2, pp. 296-311 (2016).
    - Carlson, D., Collins, E., Hsieh, Y.-P., Carin, L., and Cevher, V.
      *Preconditioned spectral descent for deep learning.*
      In Neural Information Processing Systems (2015b).
    - Flynn, T. *The duality structure gradient descent algorithm: analysis and applications to neural networks.*
      arXiv preprint arXiv:1708.00523 (2017). [`arXiv:1708.00523 <https://arxiv.org/abs/1708.00523>`_]

    Note:
        Orthogonalizing fused parameters separately is supported but with limitations. User must provide
        a function to check if a weight tensor is fused parameters (QKV, GQA, etc.) as well as the
        split function to split the fused parameters into a list of parameters.

    Args:
        {_args_doc}
        orthogonalize_fn: Function to orthogonalize the updates.
        scale_factor_fn: Function to compute the scale factor for the update.
        **kwargs: Arguments passed through to the base optimizer.

    Note:
        Keyword arguments passed through are not checked here. Optimizer inherited from this class should check them.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum_beta: float,
        use_nesterov: bool,
        weight_decay: float,
        use_decoupled_weight_decay: bool,
        fp32_matmul_prec: str,
        scaled_orthogonalize_fn: Callable | None = None,
        **kwargs: Any,
    ):
        if scaled_orthogonalize_fn is None:
            logging.warning("scaled_orthogonalize_fn not provided. Using noop")
            scaled_orthogonalize_fn = torch.nn.Identity()

        self.fp32_matmul_prec = fp32_matmul_prec
        default_args_dict = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            **kwargs,
        )

        super().__init__(params, default_args_dict)
        self.scaled_orthogonalize_fn = scaled_orthogonalize_fn

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

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

                # initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                # Subsequent update to exp_avg are all inplace, so it is not assigned back to state.
                exp_avg = state["momentum_buffer"]

                # Apply weight decay
                if group["weight_decay"] > 0.0:
                    if group["use_decoupled_weight_decay"]:
                        # Apply decoupled weight decay
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                    else:
                        # add l2 regularization before preconditioning (i.e. adding a squared loss term)
                        grad += group["weight_decay"] * p

                # update momentum buffer with EMA of gradient
                exp_avg.lerp_(grad, 1 - group["momentum_beta"])

                # include nesterov momentum
                if group["use_nesterov"]:
                    grad = grad.lerp(exp_avg, group["momentum_beta"])
                else:
                    grad = exp_avg

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    grad = self.orthogonalize(p, grad)

                # perform weight update
                # scale is applied to have update RMS == 1
                p.add_(grad, alpha=-group["lr"])

        return loss

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum because a lot of
                information is only available in the param tensor, attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        grad = self.scaled_orthogonalize_fn(grad)
        return grad


OrthogonalizedOptimizer.__doc__ = OrthogonalizedOptimizer.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
