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
from absl import logging
from torch import Tensor


__all__ = [
    "eigh_with_fallback",
    "conjugate",
    "orthogonal_iteration",
]


def eigh_with_fallback(
    x: Tensor,
    force_double: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""torch.linalg.eigh() function with double precision fallback

    Unified wrapper over eigh() function with automatic fallback and force double precision options.
    Automatically falls back to double precision on failure and returns eigenvalues in descending order.
    Default 2nd argument of eigh UPLO is 'L'.

    Args:
        x: Tensor of shape (*, n, n) where "*" is zero or more batch dimensions consisting of symmetric or
            Hermitian matrices.
        force_double: Force double precision computation. Default False.

    Returns:
        Eigenvalues and eigenvectors tuple (eigenvalues in descending order).
    """
    input_dtype = x.dtype

    if force_double:
        logging.warning("Force double precision")
        x = x.to(torch.float64)

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(x)
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        if not force_double:
            logging.warning(f"Falling back to double precision: {e}")
            # Fallback to double precision if the default precision fails
            x = x.to(torch.float64)
            eigenvalues, eigenvectors = torch.linalg.eigh(x)
        else:
            raise e

    eigenvalues = eigenvalues.to(input_dtype)
    eigenvectors = eigenvectors.to(input_dtype)

    # Flip order to descending (`torch.linalg.eigh` returns ascending order by default)
    eigenvalues = torch.flip(eigenvalues, [-1])
    eigenvectors = torch.flip(eigenvectors, [-1])
    return (eigenvalues, eigenvectors)


def orthogonal_iteration(
    kronecker_factor: Tensor,
    eigenbasis: Tensor,
    power_iter_steps: int,
) -> Tensor:
    """Refines an eigenbasis via power iteration with QR re-orthogonalization.

    Performs ``power_iter_steps`` rounds of ``Q = QR(kronecker_factor @ Q)`` starting from
    ``eigenbasis``. The columns of ``eigenbasis`` are expected to already be aligned with the
    intended descending-eigenvalue ordering of ``kronecker_factor`` (see
    :func:`emerging_optimizers.soap.soap_utils.sort_eigenbasis_by_approx_eigvals`).

    Args:
        kronecker_factor: Kronecker factor matrix (symmetric, used as the projector).
        eigenbasis: Starting eigenbasis whose columns will be refined.
        power_iter_steps: Number of power-iteration / QR rounds to perform.

    Returns:
        The refined eigenbasis.
    """
    Q = eigenbasis
    for _ in range(power_iter_steps):
        # Project current eigenbases on kronecker factor
        Q = kronecker_factor @ Q
        # Perform QR to maintain orthogonality between iterations
        Q = torch.linalg.qr(Q).Q
    return Q


def conjugate(a: Tensor, p: Tensor, diag: bool = False) -> Tensor:
    """Calculate similarity transformation

    This function calculates :math:`B = P^T A P`. It assumes P is orthogonal so that :math:`P^{-1} = P^T` and
    the similarity transformation exists.

    Args:
        a: matrix to be transformed
        p: An orthogonal matrix.
        diag: If True, only return the diagonal of the similarity transformation

    Returns:
        b
    """
    if a.dim() != 2 or p.dim() != 2:
        raise TypeError("a and p must be 2D matrices")
    pta = p.T @ a
    if not diag:
        b = pta @ p
    else:
        # return the diagonal of the similarity transformation
        b = (pta * p.T).sum(dim=1)
    return b
