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

from typing import Any, override

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import MuonScaleT, get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer
from emerging_optimizers.utils import FP32MatmulPrecT
from emerging_optimizers.weight_update_hooks import WeightUpdateHook
from emerging_optimizers.weight_update_hooks.base import HookStateT


__all__ = ["Dimuon"]


def _safety_scaled(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Coefficients of p(x / 1.01) for the odd quintic p(x) = a x + b x^3 + c x^5."""
    return a / 1.01, b / 1.01**3, c / 1.01**5


# Dimuon's orthogonalization: the first five Polar Express steps (https://arxiv.org/abs/2505.16932) with its 1.01
# safety factor. Dimuon also normalizes the input to X / (1.01 ||X||_F) where newton_schulz uses X / ||X||_F, so the
# first step folds in that extra 1 / 1.01.
_DIMUON_COEFFICIENTS = [
    _safety_scaled(*_safety_scaled(8.28721201814563, -23.595886519098837, 17.300387312530933)),
    _safety_scaled(4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    _safety_scaled(3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    _safety_scaled(3.3184196573706015, -2.488488024314874, 0.51004894012372),
    _safety_scaled(2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]


def _diagonal_conditioner(energy: torch.Tensor, damping: float) -> torch.Tensor:
    """Returns ``(e / max(e) + damping)^(-1/4)`` for row or column energies ``e``, or ones if all are zero."""
    peak = energy.amax()
    conditioner = (energy / peak.clamp_min(torch.finfo(energy.dtype).tiny) + damping) ** -0.25
    return torch.where(peak > 0, conditioner, torch.ones_like(conditioner))


class Dimuon(OrthogonalizedOptimizer[HookStateT]):
    """Dimuon: Muon with diagonal row and column conditioning around the orthogonalization.

    Each step builds diagonal conditioners ``L`` and ``R`` from the row and column energies (sums of squares) of the
    previous momentum, ``(e / max(e) + damping)^(-1/4)``, and orthogonalizes the momentum ``M`` as
    ``D = L polar(L M R) R`` with a five-step Polar Express. ``D`` is then rescaled to the Frobenius norm
    ``sqrt(min(m, n))`` of an exact orthogonalization, so Muon's scale factor sets the update size. The only extra
    state is the two energy vectors of each matrix.

    Ported from a reverse-engineered reconstruction of Dimuon, which targets LoRA factors. Differences from it:

    - Each matrix is rescaled to ``sqrt(min(m, n))`` and then by Muon's scale factor, instead of the
      ``sqrt(r) (r / 32)^0.175`` LoRA-rank step size.
    - Weight decay is applied (the reconstruction has none).
    - The momentum is kept in the parameter dtype, not BF16, and gradient clipping is left to the training loop.
    - The Polar Express runs at ``fp32_matmul_prec`` instead of FP16.

    References:
        - *Reconstruction of the Dimuon optimizer.*
          [`GitHub <https://github.com/anpaure/dimuon/blob/d6570bb18aec27cbd4bccf71dcbc814e20412741/dimuon.py>`_]
        - *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm.*
          arXiv:2505.16932 (2025). [`arXiv:2505.16932 <https://arxiv.org/abs/2505.16932>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer, or any 1-D
          parameters; those can all be optimized by a standard method (e.g., AdamW).

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: The learning rate.
        momentum: The EMA momentum coefficient. Dimuon conditions plain EMA momentum, without Nesterov.
        weight_decay: The weight decay coefficient.
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`.
        fp32_matmul_prec: Precision of the Polar Express matmuls.
        damping: Floor on the relative row and column energies, which bounds the conditioners by
            ``damping^(-1/4)``.
        scale_mode: The type of Muon scale factor applied to the fixed-norm direction.
        extra_scale_factor: The additional scale factor applied to the update.
        weight_update_hook: Optional hook that runs around the final in-place weight update.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.1,
        *,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        damping: float = 1e-4,
        scale_mode: MuonScaleT = "spectral",
        extra_scale_factor: float = 0.2,
        weight_update_hook: WeightUpdateHook[HookStateT] | None = None,
    ) -> None:
        if damping <= 0:
            raise ValueError(f"damping must be positive, got {damping}")
        self.damping = damping
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        def raw_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            return muon_utils.newton_schulz(
                grad,
                steps=len(_DIMUON_COEFFICIENTS),
                coefficient_type="custom",
                custom_coefficient_sets=_DIMUON_COEFFICIENTS,
            )

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=False,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=raw_orthogonalize_fn,
            weight_update_hook=weight_update_hook,
        )

    @torch.no_grad()  # type: ignore[misc]
    @override
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue  # pragma: no cover
            if p.dim() != 2:
                raise ValueError(f"{self.__class__.__name__} only supports 2D parameters")
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p.data)
                # Energies of the momentum before this step's EMA update; zero gives identity conditioners.
                state["row_energy"] = p.data.new_zeros(p.size(0))
                state["col_energy"] = p.data.new_zeros(p.size(1))

    @override
    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        state = self.state[p]
        left = _diagonal_conditioner(state["row_energy"], self.damping)[:, None]
        right = _diagonal_conditioner(state["col_energy"], self.damping)[None, :]
        direction = left * self.scaled_orthogonalize_fn(left * grad * right) * right
        norm = torch.linalg.vector_norm(direction).clamp_min(torch.finfo(direction.dtype).tiny)
        scale_factor = get_muon_scale_factor(grad.size(-2), grad.size(-1), mode=self.scale_mode)

        # grad is the momentum here, whose energies condition the next step.
        energy = grad.square()
        torch.sum(energy, dim=1, out=state["row_energy"])
        torch.sum(energy, dim=0, out=state["col_energy"])
        return direction * (min(grad.shape) ** 0.5 * scale_factor * self.extra_scale_factor / norm)
