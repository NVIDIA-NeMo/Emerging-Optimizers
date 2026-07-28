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

from emerging_optimizers import utils
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["scaled_cans_coupled_ns"]

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


def scaled_cans_coupled_ns(
    x: Tensor,
    eps: float = 1e-12,
    fp32_matmul_prec: FP32MatmulPrecT = "highest",
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
    coefficients. The literals were generated with Python IEEE-754 binary64 arithmetic and
    rounded to the nearest representable binary64 value. In particular,
    ``1.5 / sqrt(1.01)`` evaluates to ``1.4925557853149838``; the decimal
    ``1.492555785314984`` is one binary64 ULP higher. All constant arithmetic is folded into
    these literals. At runtime they are applied to FP32 tensors, except that ``"medium"``
    explicitly casts the iteration to BF16, matching Muon's Newton-Schulz implementation.

    In practice, the result is approximate because the iteration is truncated after ten steps,
    the normalization may overestimate the largest eigenvalue, matrix multiplications may use
    reduced precision, and the returned inverse root is explicitly symmetrized. ``eps`` also
    clamps the normalization for degenerate inputs.

    Args:
        x: A 2D symmetric positive-definite FP32 matrix or 3D batch of matrices.
        eps: Lower bound used when normalizing the matrices.
        fp32_matmul_prec: Precision used for FP32 matrix multiplications: ``"medium"`` for BF16,
            ``"high"`` for TF32, or ``"highest"`` for FP32.

    Returns:
        The approximate inverse square root as an FP32 tensor with the same shape as ``x``.
    """
    if x.dim() not in (2, 3) or x.shape[-2] != x.shape[-1]:
        raise TypeError(f"x must be a square matrix or batch of square matrices, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"x must be in float32, got {x.dtype}")

    with utils.fp32_matmul_precision(fp32_matmul_prec):
        inf_norm = torch.linalg.matrix_norm(x, ord=float("inf"), dim=(-2, -1), keepdim=True).clamp_min_(eps)
        y = x / inf_norm
        if torch.get_float32_matmul_precision() == "medium":
            y = y.to(torch.bfloat16)

        z = torch.eye(x.shape[-1], device=x.device, dtype=y.dtype).expand_as(y)
        cans_addmm = torch.addmm if x.dim() == 2 else torch.baddbmm
        for beta, alpha in _CANS_COEFFS:
            p = z @ y
            z = cans_addmm(z, p, z, beta=beta, alpha=alpha)
            y = cans_addmm(y, y, p, beta=beta, alpha=alpha)

        z = z.to(torch.float32)
        z.mul_(torch.rsqrt(inf_norm))
        return (z + z.mT) / 2.0
