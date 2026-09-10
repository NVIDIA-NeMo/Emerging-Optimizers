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
from typing import Literal

import torch

from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz


__all__ = [
    "RetractionT",
    "cayley_retraction",
    "newton_schulz_retraction",
    "polar_retraction",
    "qr_retraction",
]

RetractionT = Literal["qr", "polar", "cayley", "newton_schulz"]


def qr_retraction(
    point: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Retract a tangent step back to the Stiefel manifold via reduced QR decomposition.

    Forms the intermediate matrix $X = \text{point} - \text{step\\_size} \times \text{momentum}$,
    computes its reduced QR factorization, and resolves column phase ambiguity by ensuring
    positive diagonal elements in $R$.

    Args:
        point: Current matrix on the Stiefel manifold of shape ``(M, K)``.
        momentum: Tangent vector or momentum update of shape ``(M, K)``.
        step_size: Step size scaling the update direction.

    Returns:
        The retracted orthonormal matrix of shape ``(M, K)`` satisfying $Q^T Q = I_K$.
    """
    matrix = point - step_size * momentum
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.diagonal(r).sign()
    signs.masked_fill_(signs == 0, 1)
    return q * signs


def polar_retraction(
    point: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Retract a tangent step back to the Stiefel manifold via the analytical polar factor.

    Computes the closest orthonormal matrix in Frobenius norm to
    $X = \text{point} - \text{step\\_size} \times \text{momentum}$ using the singular value
    decomposition $X = U \Sigma V^T \implies \operatorname{polar}(X) = U V^T$.

    Args:
        point: Current matrix on the Stiefel manifold of shape ``(M, K)``.
        momentum: Tangent vector or momentum update of shape ``(M, K)``.
        step_size: Step size scaling the update direction.

    Returns:
        The retracted orthonormal matrix of shape ``(M, K)`` satisfying $Q^T Q = I_K$.
    """
    matrix = point - step_size * momentum
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def cayley_retraction(
    point: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Retract a tangent step back to the Stiefel manifold via the Cayley transform.

    Constructs a skew-symmetric matrix $A = D X^T - X D^T$ from the update direction
    $D = -\text{momentum}$ and applies the Padé approximation to the matrix exponential:
    $(I - \frac{\eta}{2} A)^{-1} (I + \frac{\eta}{2} A) X$.

    Args:
        point: Current matrix on the Stiefel manifold of shape ``(M, K)``.
        momentum: Tangent vector or momentum update of shape ``(M, K)``.
        step_size: Step size scaling the update direction ($\eta$).

    Returns:
        The retracted orthonormal matrix of shape ``(M, K)`` satisfying $Q^T Q = I_K$.
    """
    direction = -momentum
    skew = direction @ point.mT - point @ direction.mT
    identity = torch.eye(point.shape[0], dtype=point.dtype, device=point.device)
    lhs = identity - 0.5 * step_size * skew
    rhs = (identity + 0.5 * step_size * skew) @ point
    return torch.linalg.solve(lhs, rhs)


def newton_schulz_retraction(
    point: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
    coefficient_type: NSCoeffT = "polar_express",
    num_ns_steps: int = 8,
) -> torch.Tensor:
    """Retract a tangent step back to the Stiefel manifold via iterative Newton-Schulz iterations.

    Approximates the polar factor / matrix sign function of
    $X = \text{point} - \text{step\\_size} \times \text{momentum}$ using iterative matrix
    polynomial evaluations directly on Tensor Cores, avoiding host-device synchronization
    and LAPACK factorization overhead.

    Args:
        point: Current matrix on the Stiefel manifold of shape ``(M, K)``.
        momentum: Tangent vector or momentum update of shape ``(M, K)``.
        step_size: Step size scaling the update direction.
        coefficient_type: Polynomial coefficient scheme used for Newton-Schulz steps.
            Defaults to ``"polar_express"``.
        num_ns_steps: Number of iterative polynomial steps to perform. Defaults to 8.

    Returns:
        The retracted approximately orthonormal matrix of shape ``(M, K)`` satisfying
        $\|Q^T Q - I_K\|_\infty \le \epsilon$.
    """
    matrix = point - step_size * momentum
    return newton_schulz(
        matrix,
        steps=num_ns_steps,
        coefficient_type=coefficient_type,
        use_syrk=matrix.is_cuda,
    )
