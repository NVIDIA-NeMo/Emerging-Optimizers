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
from typing import Any, Callable, override

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
        split_qkv: Whether parameter is fused attention parameters (QKV, GQA, etc.), default to be False.
        is_qkv_fn: Function to check if a parameter is fused attention parameters (QKV, GQA, etc.).
        qkv_split_shapes: For grouped attention parameters (QKV, GQA, etc.), specify the shapes as a tuple of 3 integers
            representing the sizes of Q, K, V components along the first dimension.
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations.
"""


class OrthogonalizedOptimizer(optim.Optimizer):
    """Base class for orthogonalized optimizers.

    This class is a wrapper around a base optimizer that performs orthogonalization on the updates.
    The theoretical foundation of orthogonalization for stochastic gradient descent was developed by the following papers:

    - Carlson, D., Cevher, V., and Carin, L. *Stochastic spectral descent for Restricted Boltzmann Machines.*
      In International Conference on Artificial Intelligence and Statistics (2015a).
    - Carlson, D., Collins, E., Hsieh, Y.-P., Carin, L., and Cevher, V. *Preconditioned spectral descent for deep learning.*
      In Neural Information Processing Systems (2015b).
    - Flynn, T. *The duality structure gradient descent algorithm: analysis and applications to neural networks.*
      arXiv preprint arXiv:1708.00523 (2017). [`arXiv:1708.00523 <https://arxiv.org/abs/1708.00523>`_]

    Note:
        Orthogonalizing QKV sperately when they are fused is supported but with limitations. User must provide
        a function to check if a weight tensor is fused attention parameters (QKV, GQA, etc.) as well as the
        leading dimension of Q, K, V components. Only one split size is supported, i.e. all attention layers across
        the network must have the same size.

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
        split_qkv: bool,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None,
        qkv_split_shapes: tuple[int, int, int] | None,
        fp32_matmul_prec: str,
        orthogonalize_fn: Callable | None = None,
        scale_factor_fn: Callable | None = None,
        **kwargs: Any,
    ):
        if orthogonalize_fn is None:
            logging.warning("orthogonalize_fn not provided. Using noop")
            orthogonalize_fn = torch.nn.Identity()

        if scale_factor_fn is None:
            logging.warning("scale_factor_fn not provided. Using default scale_factor_fn.")

            def return_one(*args, **kwargs):  # type: ignore[no-untyped-def]
                return 1.0

            scale_factor_fn = return_one

        if split_qkv:
            assert is_qkv_fn is not None, "is_qkv_fn must be provided when split_qkv is True"
            assert qkv_split_shapes is not None, "qkv_split_shapes must be provided when split_qkv is True"
            if len(qkv_split_shapes) != 3:
                raise ValueError(
                    f"qkv_split_shapes must be a tuple of 3 integers, got {len(qkv_split_shapes)} elements"
                )
            if not all(isinstance(s, int) for s in qkv_split_shapes):
                raise ValueError(f"All elements in qkv_split_shapes must be integers, got {qkv_split_shapes}")
            if any(s <= 0 for s in qkv_split_shapes):
                raise ValueError(f"All elements in qkv_split_shapes must be positive, got {qkv_split_shapes}")
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

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
        self.orthogonalize_fn = orthogonalize_fn
        self.scale_factor_fn = scale_factor_fn

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
        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            logging.log_first_n(logging.INFO, f"split qkv with {p.shape} to {self.qkv_split_shapes}", 1)
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            qkv_grads = torch.split(grad, self.qkv_split_shapes, dim=0)
            # Apply Newton-Schulz to each component
            qkv_whitened = [self.orthogonalize_fn(g) for g in qkv_grads]
            qkv_scales = [self.scale_factor_fn(g.size(0), g.size(1)) for g in qkv_grads]
            # Apply individual scales to each component and concatenate
            grad = torch.cat([whitened * scale for whitened, scale in zip(qkv_whitened, qkv_scales)])
        else:
            grad = self.orthogonalize_fn(grad) * self.scale_factor_fn(grad.size(0), grad.size(1))
        return grad


OrthogonalizedOptimizer.__doc__ = OrthogonalizedOptimizer.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
