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

import emerging_optimizers.utils as utils
from emerging_optimizers.psgd.psgd_utils import norm_lower_bound_skew


__all__ = [
    "procrustes_step",
]


@torch.compile  # type: ignore[misc]
def procrustes_step(Q: torch.Tensor, max_step_size: float = 0.125, eps: float = 1e-8) -> torch.Tensor:
    r"""One step of an online solver for the orthogonal Procrustes problem.

    The orthogonal Procrustes problem is :math:`\min_U \| U Q - I \|_F` s.t. :math:`U^H U = I`
    by rotating Q as :math:`\exp(a R) Q`, where :math:`R = Q^H - Q` is the generator and :math:`\|a R\| < 1`.

    `max_step_size` should be less than :math:`1/4` as we only expand :math:`\exp(a R)` to its 2nd order term.

    This method is a second order expansion of a Lie algebra parametrized rotation that
    uses a simple approximate line search to find the optimal step size, from Xi-Lin Li.

    Args:
        Q: Tensor of shape (n, n), general square matrix to orthogonalize.
        max_step_size: Maximum step size for the line search. Default is 1/8. (0.125)
        eps: Small number for numerical stability.
    """
    # Note: this function is written in fp32 to avoid numerical instability while computing the taylor expansion of the exponential map
    with utils.fp32_matmul_precision("highest"):
        R = Q.T - Q
        R /= torch.max(norm_lower_bound_skew(R), eps)
        RQ = R @ Q
        # trace of RQ is always positive,
        # since tr(RQ) = ⟨R, Q⟩_F = ⟨Q^T - Q, Q⟩_F = ||Q||_F^2 - ⟨Q, Q⟩_F = ||Q||_F^2 - tr(Q^T Q) ≥ 0
        tr_RQ = torch.trace(RQ)
        RRQ = R @ RQ
        tr_RRQ = torch.trace(RRQ)
        # clip step size to max_step_size, based on a 2nd order expansion.
        _step_size = torch.clamp(-tr_RQ / tr_RRQ, min=0, max=max_step_size)
        # If tr_RRQ >= 0, the quadratic approximation is not concave, we fallback to max_step_size.
        step_size = torch.where(tr_RRQ < 0, _step_size, max_step_size)
        # rotate Q as exp(a R) Q ~ (I + a R + a^2 R^2/2) Q with an optimal step size by line search
        # for 2nd order expansion, only expand exp(a R) to its 2nd term.
        # Q += step_size * (RQ + 0.5 * step_size * RRQ)
        Q = torch.add(Q, torch.add(RQ, RRQ, alpha=0.5 * step_size), alpha=step_size)

    return Q
