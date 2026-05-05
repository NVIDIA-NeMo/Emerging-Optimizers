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
    "met_approx_eigvals_criteria",
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


def met_approx_eigvals_criteria(
    kronecker_factor: torch.Tensor,
    approx_eigvals: torch.Tensor,
    tolerance: float,
) -> bool:
    """Determines whether the eigenbasis for a factor matrix met the desired criteria

    The approximated eigenvalues update criteria is then defined as
    :math:`||diag(Q^T K Q)||_F >= (1 - tolerance) * (Q^T K Q)_F`, where :math:`Q` is the approximated eigenvectors and
    :math:`K` is the kronecker factor (L or R).

    We use the kronecker factor and approximated eigenvalues directly to save compute because Frobenius norm of
    kronecker factor is the same as that of the approximated eigenvalues matrix.

    Args:
        kronecker_factor: Kronecker factor matrix.
        approx_eigvals: Approximated eigenvalues
        tolerance: Tolerance threshold for the normalized diagonal component of approximated eigenvalue matrix.

    Returns:
        Whether eigenbasis meet criteria and don't need to be updated
    """
    matrix_norm = torch.linalg.norm(kronecker_factor)
    diagonal_norm = torch.linalg.norm(approx_eigvals)

    return tolerance * matrix_norm >= (matrix_norm - diagonal_norm)


def orthogonal_iteration(
    approx_eigvals: torch.Tensor,
    kronecker_factor: torch.Tensor,
    eigenbasis: torch.Tensor,
    ind: int,
    exp_avg_sq: torch.Tensor,
    power_iter_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the eigenbases of the preconditioner using power iteration and QR decomposition.

    This function performs multiple rounds of power iteration followed by QR decomposition
    to recompute the eigenbases of the preconditioner kronecker factor. Generalizes Vyas et al.'s (SOAP) algorithm of 1 step of power iteration for updating the eigenbasis.

    Args:
        approx_eigvals : Projection of kronecker factor onto the eigenbasis, should be close to diagonal
        kronecker_factor : Kronecker factor matrix.
        eigenbasis : Kronecker factor eigenbasis matrix.
        ind : Index for selecting dimension in the exp_avg_sq matrix to apply the sorting order over.
        exp_avg_sq : inner Adam second moment (exp_avg_sq).
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Q: The updated eigenbasis
            - exp_avg_sq: The updated (sorted) inner Adam second moment
    """
    # Sort the approximated eigenvalues according to their magnitudes
    sort_idx = torch.argsort(approx_eigvals, descending=True)
    # re-order the inner adam second moment
    exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)

    # Initialize power iteration after sorting the columns of the eigenbasis matrix according to the descending eigenvalues
    Q = eigenbasis[:, sort_idx]

    # Perform multiple steps of power iteration
    for _ in range(power_iter_steps):
        # Project current eigenbases on kronecker factor
        Q = kronecker_factor @ Q
        # Perform QR to maintain orthogonality between iterations
        Q = torch.linalg.qr(Q).Q

    return Q, exp_avg_sq


def conjugate(a: torch.Tensor, p: torch.Tensor, diag: bool = False) -> torch.Tensor:
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
