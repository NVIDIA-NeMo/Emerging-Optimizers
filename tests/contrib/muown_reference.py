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

# Reference Muown implementation, adapted from the authors' code for use as a test oracle:
#   https://github.com/kcc-lion/muown/blob/main/optim/muown.py
#   Lion et al., "Muown: Row-Norm Control for Muon Optimization", arXiv:2605.10797 (paper: CC BY 4.0).
# The repository did not declare a code license at the time of copying.
"""Reference Muown implementation used as a test oracle."""

from typing import Callable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _wn_pre_ns(W: Tensor, g: Tensor, v_norm: Tensor, grad_W: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct direction v from (W, g, v_norm) and split grad_W into (grad_g, grad_v)."""
    u = W / g
    v = u * v_norm
    grad_g = (grad_W * u).sum(dim=1, keepdim=True)
    grad_v = (g / v_norm) * (grad_W - u * grad_g)
    return v, grad_g, grad_v


def _wn_recompose(W: Tensor, g: Tensor, v_new: Tensor) -> Tensor:
    """Write W = g * v_new / ||v_new||_row in place and return the new row norms."""
    v_norm_new = v_new.norm(dim=1, keepdim=True)
    W.copy_(g * (v_new / v_norm_new))
    return v_norm_new


class MuownReference(Optimizer):
    """Single-process reference Muown with an injected orthogonalization callable."""

    def __init__(
        self,
        params,
        orthogonalize_fn: Callable[[Tensor], Tensor],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = False,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        adam_eps: float = 1e-8,
    ) -> None:
        self._orthogonalize_fn = orthogonalize_fn
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
        )
        super().__init__(params, defaults)

    def _init_state_2d(self, p: Tensor, state: dict) -> None:
        w_norm = p.data.norm(dim=1, keepdim=True)
        state["g"] = w_norm.clone()
        state["v_norm"] = w_norm.clone()
        state["m_v"] = torch.zeros_like(p.data)
        state["m_g"] = torch.zeros_like(w_norm)
        state["v_g"] = torch.zeros_like(w_norm)
        state["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            adam_eps = group["adam_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    self._init_state_2d(p, state)

                state["step"] += 1
                step = state["step"]

                g = state["g"]
                v_norm = state["v_norm"]
                m_v = state["m_v"]
                m_g = state["m_g"]
                v_g = state["v_g"]
                if weight_decay != 0.0:
                    W_old = p.data.clone()

                # Fused: reconstruct v + compute weight norm gradients
                v, grad_g, grad_v = _wn_pre_ns(p.data, g, v_norm, grad)

                # Muon update on v: momentum + orthogonalization
                m_v.mul_(momentum).add_(grad_v)
                if nesterov:
                    update = grad_v.add(m_v, alpha=momentum)
                else:
                    update = m_v.clone()

                # Injected orthogonalization (folds in the 0.2 * sqrt(max(m, n)) scaling).
                update = self._orthogonalize_fn(update)
                v_new = v.add(update, alpha=-lr)

                # Adam update on g (small [out_features, 1] vectors)
                beta1, beta2 = betas
                m_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                v_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                g.addcdiv_(m_g / bc1, (v_g / bc2).sqrt().add_(adam_eps), value=-lr)

                # Fused: recompose W = g * v_new / ||v_new||, writes directly into p.data
                state["v_norm"] = _wn_recompose(p.data, g, v_new)
                if weight_decay != 0.0:
                    p.data.add_(W_old, alpha=-lr * weight_decay)
                    g.copy_(p.data.norm(dim=1, keepdim=True))

        return loss
