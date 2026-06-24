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

from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry, utils
from emerging_optimizers.orthogonalized_optimizers.muon import Muon, MuonScaleT
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.scalar_optimizers import update_functions
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["Muown"]


@registry.register_optimizer("muown")
class Muown(Muon):
    """Muown: Muon with internal weight normalization (row-norm control).

    Muown (Lion et al., *Muown: Row-Norm Control for Muon Optimization*, arXiv:2605.10797) is a drop-in
    replacement for :class:`~emerging_optimizers.orthogonalized_optimizers.muon.Muon` that splits each 2D
    weight into a per-row magnitude and a direction, then optimizes them under their natural geometries:


    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate shared by the direction and magnitude updates.
        momentum: EMA momentum for the direction (Muon) update.
        betas: Adam ``(beta1, beta2)`` for the magnitude update.
        adam_eps: Adam epsilon for the magnitude update.
        weight_decay: Decoupled weight decay coefficient, applied to the magnitude ``g``.
        fp32_matmul_prec: Precision for the orthogonalization GEMM operations.
        coefficient_type: Newton-Schulz coefficient set (see :class:`Muon`).
        num_ns_steps: Number of Newton-Schulz iteration steps.
        scale_mode: Update scale mode (see :func:`~emerging_optimizers.orthogonalized_optimizers.muon.get_muon_scale_factor`).
        extra_scale_factor: Extra scale on the direction update; ``0.2`` matches Adam's update RMS norm.
        use_syrk: Whether to use the Triton SYRK kernel for Newton-Schulz.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        *,
        betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 1.0,
        use_syrk: bool = False,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        self.betas = betas
        self.adam_eps = adam_eps

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=False,
            weight_decay_method="decoupled",
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            use_syrk=use_syrk,
        )

    @torch.no_grad()  # type: ignore[misc]
    @override
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            if p.dim() != 2:
                raise TypeError("Muown is only supported for 2D parameters")
            state = self.state[p]
            if len(state) == 0:
                # Seed g, v from the current weight so Muown starts from the same point as Muon.
                row_norm = p.norm(dim=1, keepdim=True).to(torch.float32)
                state["step"] = 0
                state["g"] = row_norm.clone()
                state["v_norm"] = row_norm.clone()
                state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                state["m_g"] = torch.zeros_like(row_norm)
                state["v_g"] = torch.zeros_like(row_norm)

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
            self._init_group(group)

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue  # pragma: no cover

                state = self.state[p]
                state["step"] += 1
                g = state["g"]
                v_norm = state["v_norm"]

                grad = p.grad.to(torch.float32)
                weight = p.to(torch.float32)

                u = weight / g
                v = u * v_norm
                grad_g = (grad * u).sum(dim=1, keepdim=True)
                grad_v = (g / v_norm) * (grad - u * grad_g)

                state["momentum_buffer"].lerp_(grad_v, 1 - momentum)
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    direction_update = self.scaled_orthogonalize_fn(state["momentum_buffer"])
                v_new = v.add(direction_update, alpha=-lr)

                magnitude_update = update_functions.calculate_adam_update(
                    grad_g,
                    state["m_g"],
                    state["v_g"],
                    betas=self.betas,
                    eps=self.adam_eps,
                    correct_bias=True,
                    nesterov=False,
                    step=state["step"],
                )
                g.add_(magnitude_update, alpha=-lr)

                g.add_(g, alpha=-weight_decay * lr)

                v_norm_new = v_new.norm(dim=1, keepdim=True)
                p.copy_(g * (v_new / v_norm_new))
                state["v_norm"] = v_norm_new

        return None
