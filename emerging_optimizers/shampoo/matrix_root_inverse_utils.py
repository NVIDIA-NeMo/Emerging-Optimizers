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
import torch
from torch import Tensor


__all__ = ["mat_4th_root_inv_via_scaled_cans", "mat_root_inv_via_scaled_cans"]


def mat_root_inv_via_scaled_cans(
    x: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Compute inverse square roots with scaled coupled CANS Newton-Schulz.

    CANS polynomial-based inverse-root computation from https://arxiv.org/abs/2506.10935.

    This implementation applies the CANS orthogonalization polynomials to the coupled
    Newton-Schulz iteration for a symmetric positive-definite matrix. It uses a fixed
    ten-step schedule and normalizes with the matrix infinity norm rather than the exact
    spectral norm. The infinity norm is an inexpensive upper bound, but can conservatively
    scale matrices whose rows contain substantial cancellation and consequently slow
    convergence for their smallest eigenvalues.

    The tabulated coefficients fold a 1% spectral safety margin into the polynomial. Starting
    from the unscaled CANS pairs ``(beta, alpha)``, steps zero through eight keep ``beta`` and
    use ``alpha / 1.01``. The final pair additionally absorbs the output normalization and is
    computed as ``(beta / sqrt(1.01), alpha / 1.01**1.5)``. This is algebraically equivalent
    in exact arithmetic to normalizing by ``1.01 * inf_norm`` and applying the original
    coefficients.

    Args:
        x: A 2D symmetric positive-definite FP32 matrix or 3D batch of matrices.
        eps: Lower bound used when normalizing the matrices.

    Returns:
        The approximate inverse square root as an FP32 tensor with the same shape as ``x``.

    Raises:
        TypeError: If ``x`` is not a square FP32 matrix or batch of square FP32 matrices.
        RuntimeError: If the active FP32 matmul precision is ``"medium"``.
    """
    # All constant arithmetic, including the 1.01 safety factor, is folded into these coefficients.
    _CANS_COEFFS = (
        (5.182503604966906, -5.126830178299687),
        (2.586120737395915, -0.641538812403133),
        (2.567364126726186, -0.6391058222170474),
        (2.520560084348265, -0.6330225823828756),
        (2.410759275435182, -0.6186815444268036),
        (2.1883348130094173, -0.5893091162177136),
        (1.8595760874873613, -0.5449991062102938),
        (1.589020160467417, -0.5075811685214573),
        (1.5051653981684994, -0.4957799077972079),
        (1.4925557853149838, -0.49259266842078675),
    )

    if x.dim() not in (2, 3) or x.shape[-2] != x.shape[-1]:
        raise TypeError(f"x must be a square matrix or batch of square matrices, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"x must be in float32, got {x.dtype}")
    if torch.get_float32_matmul_precision() == "medium":
        raise RuntimeError("`medium` (bf16 compute) is insufficient for CANS, use `high` or `highest`")

    is_batched = x.dim() == 3
    if not is_batched:
        x = x.unsqueeze(0)

    inf_norm = torch.linalg.matrix_norm(x, ord=float("inf"), dim=(-2, -1), keepdim=True).clamp_min_(eps)
    y = x / inf_norm

    z = torch.eye(x.shape[-1], device=x.device)
    z = z.expand(x.shape[0], -1, -1)

    p = y
    for beta, alpha in _CANS_COEFFS[:-1]:
        z = torch.baddbmm(z, p, z, beta=beta, alpha=alpha)
        y = torch.baddbmm(y, y, p, beta=beta, alpha=alpha)
        p = z @ y

    beta, alpha = _CANS_COEFFS[-1]
    z = torch.baddbmm(z, p, z, beta=beta, alpha=alpha)

    z.mul_(torch.rsqrt(inf_norm))
    result = (z + z.mT) * 0.5
    return result if is_batched else result.squeeze(0)


def mat_4th_root_inv_via_scaled_cans(
    x: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Compute inverse fourth roots with a scaled coupled CANS-style Newton-Schulz schedule.

    Runs the coupled iteration ``Z <- q(M) Z``, ``M <- q(M)^4 M`` with per-step degree-one
    polynomials ``q(m) = beta + alpha * m``, so ``Z`` converges to the inverse fourth root of the
    normalized input. The ten-step schedule was derived by a greedy per-step interval-ratio
    optimization for normalized eigenvalues in ``[1e-7, 1]`` (see
    ``benchmarks/fourth_root_inv_cans_schedule.py``, which derives and verifies the table).
    Like :func:`mat_root_inv_via_scaled_cans`, it normalizes with the matrix infinity norm rather
    than the exact spectral norm. The infinity norm is an inexpensive upper bound, but can
    conservatively scale matrices whose rows contain substantial cancellation and consequently
    slow convergence for their smallest eigenvalues.

    The tabulated coefficients fold a 1% spectral safety margin into the polynomial. Starting
    from the designed pairs ``(beta, alpha)``, steps zero through eight keep ``beta`` and use
    ``alpha / 1.01``. The final pair additionally absorbs the geometric recentering of the
    residual interval and the margin's share of the output normalization and is computed as
    ``(beta * rho**-0.125 / 1.01**0.25, alpha * rho**-0.125 / 1.01**1.25)`` with ``rho`` the
    final interval ratio. Only the data-dependent ``inf_norm**-0.25`` is applied explicitly.

    Args:
        x: A 2D symmetric positive-definite FP32 matrix or 3D batch of matrices.
        eps: Lower bound used when normalizing the matrices.

    Returns:
        The approximate inverse fourth root as an FP32 tensor with the same shape as ``x``.

    Raises:
        TypeError: If ``x`` is not a square FP32 matrix or batch of square FP32 matrices.
        RuntimeError: If the active FP32 matmul precision is ``"medium"``.
    """
    _CANS_4TH_ROOT_COEFFS = (
        (1.8599552, -1.8054311),
        (1.8535346, -1.7744838),
        (1.8398551, -1.709963),
        (1.8140982, -1.593575),
        (1.7653855, -1.3908052),
        (1.6739717, -1.0661273),
        (1.5188181, -0.65553737),
        (1.342475, -0.35367113),
        (1.2592417, -0.25681123),
        (1.2469802, -0.24699478),
    )

    if x.dim() not in (2, 3) or x.shape[-2] != x.shape[-1]:
        raise TypeError(f"x must be a square matrix or batch of square matrices, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"x must be in float32, got {x.dtype}")
    if torch.get_float32_matmul_precision() == "medium":
        raise RuntimeError("`medium` (bf16 compute) is insufficient for CANS, use `high` or `highest`")

    is_batched = x.dim() == 3
    if not is_batched:
        x = x.unsqueeze(0)

    inf_norm = torch.linalg.matrix_norm(x, ord=float("inf"), dim=(-2, -1), keepdim=True).clamp_min_(eps)
    m = x / inf_norm

    z = torch.eye(x.shape[-1], device=x.device)
    z = z.expand(x.shape[0], -1, -1)

    for beta, alpha in _CANS_4TH_ROOT_COEFFS[:-1]:
        z = torch.baddbmm(z, m, z, beta=beta, alpha=alpha)
        t = alpha * m
        t.diagonal(dim1=-2, dim2=-1).add_(beta)
        s = t @ t
        m = (s @ s) @ m

    beta, alpha = _CANS_4TH_ROOT_COEFFS[-1]
    z = torch.baddbmm(z, m, z, beta=beta, alpha=alpha)

    z = z * inf_norm.pow(-0.25)
    result = (z + z.mT) * 0.5
    return result if is_batched else result.squeeze(0)


def inv_root_via_eigh(x: Tensor, p_root_inv: float = 2, eps: float = 1e-12) -> Tensor:
    """Compute inverse roots ``x^(-1/p_root_inv)`` of a symmetric matrix by eigh"""
    w, V = torch.linalg.eigh(x)

    return (V * w.clamp_min(eps).pow(-1.0 / p_root_inv)) @ V.mT
