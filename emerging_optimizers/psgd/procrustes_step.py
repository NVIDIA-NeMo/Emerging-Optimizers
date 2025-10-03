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


def procrustes_step(Q, max_step_size=1 / 8):
    r"""One step of an in-place online solver for the orthogonal Procrustes problem.

    The orthogonal Procrustes problem is min_U || U Q - I ||_F,   s.t. U^H U = I
    by rotating Q as exp(a R) Q, where R = Q^H - Q is the generator and ||a R|| < 1.

    `max_step_size` should be less than 1/4 as we only expand exp(a R) to its 2nd order term.

    This method is a second order Taylor expansion of a Lie algebra parametrized rotation that
    uses a simple approximate line search to find the optimal step size, from Xi-Lin Li.

    Args:
        Q: Tensor of shape (n, n), general square matrix to orthogonalize.
        max_step_size: Maximum step size for the line search. Default is 1/8.
    """
    with utils.fp32_matmul_precision("highest"):
        R = Q.T - Q
        R /= norm_lower_bound_skew(R) + torch.finfo(R.dtype).smallest_normal
        RQ = R @ Q
        # trace of RQ is always positive, mathematically.
        tr_RQ = torch.trace(RQ)
        RRQ = R @ RQ
        tr_RRQ = torch.trace(RRQ)
        # clip step size to max_step_size, based on a 2nd order Taylor expansion.
        step_size = torch.clamp(-tr_RQ / tr_RRQ, min=0, max=max_step_size)
        # If tr_RRQ >= 0, the quadratic approximation is not concave, we fallback to max_step_size.
        a = torch.where(tr_RRQ < 0, step_size, max_step_size)
        # rotate Q as exp(a R) Q ~ (I + a R + a^2 R^2/2) Q with an optimal step size by line search
        # for 2nd order Taylor expansion, only expand exp(a R) to its 2nd term.
        Q.add_(a * (RQ + 0.5 * a * RRQ))
