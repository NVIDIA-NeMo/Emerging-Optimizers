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
from typing import Optional, Tuple

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
    eps: Optional[float] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> tuple[Tensor, Tensor]:
    r"""torch.linalg.eigh() function with double precision fallback

    Unified wrapper over eigh() function with automatic fallback and force double precision options.
    Automatically falls back to double precision on failure and returns eigenvalues in descending order.
    Default 2nd argument of eigh UPLO is 'L'.

    Args:
        x: Tensor of shape (*, n, n) where "*" is zero or more batch dimensions consisting of symmetric or
            Hermitian matrices.
        force_double: Force double precision computation. Default False.
        eps: Small offset for numerical stability. If None, uses dtype-appropriate values (1e-7 for float32,
            1e-15 for float64). Default None.
        output_dtype: Desired output dtype. If None, uses input dtype. Default None.

    Returns:
        Eigenvalues and eigenvectors tuple (eigenvalues in descending order).
    """
    input_dtype = x.dtype
    if output_dtype is None:
        output_dtype = input_dtype

    # Set precision-appropriate epsilon if not provided
    if eps is None:
        if x.dtype == torch.float64 or force_double:
            eps = 1e-15
        else:  # float32, float16
            eps = 1e-7

    # Check if x is already a diagonal matrix
    diag_result = _try_handle_diagonal_matrix(x)
    if diag_result is not None:
        L, Q = diag_result
        # Sort in descending order for diagonal case
        L_flipped, indices = L.sort(descending=True)
        Q_flipped = Q[:, indices]
        return (L_flipped.to(output_dtype), Q_flipped.to(output_dtype))

    # Add small identity for numerical stability
    eye = torch.eye(
        x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )
    stabilized_x = torch.addmm(x, eye, eye, alpha=eps)

    if force_double:
        logging.warning("Force double precision")
        stabilized_x = stabilized_x.to(torch.float64)

    try:
        L, Q = torch.linalg.eigh(stabilized_x)
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        if not force_double:
            logging.warning(f"Falling back to double precision: {e}")
            # Fallback to higher precision if the default precision fails
            stabilized_x_fp64 = stabilized_x.to(torch.float64)
            L, Q = torch.linalg.eigh(stabilized_x_fp64)
        else:
            raise e

    # Flip order to descending (`torch.linalg.eigh` returns ascending order by default)
    L_flipped = torch.flip(L, [-1])
    Q_flipped = torch.flip(Q, [-1])
    return (L_flipped.to(output_dtype), Q_flipped.to(output_dtype))


def eig_orthogonal_iteration(
    x: Tensor,
    approx_eigenvectors: Tensor,
    max_iterations: int = 1,
    tolerance: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Approximately compute the eigen decomposition

    [DEPRECATED] Use `orthogonal_iteration` instead.

    Orthogonal or subspace iteration uses iterative power iteration and QR decomposition to update the approximated
    eigenvectors. When the initial estimate is the zero matrix, the eigendecomposition is computed
    using `eigh_with_fallback`.

    Based on Purifying Shampoo (https://www.arxiv.org/abs/2506.03595), we use an early exit criteria to stop the
    QR iterations. This generalizes SOAP's algorithm of 1 step of power iteration for updating the eigenbasis.

    Args:
        x: tensor of shape (n, n) where x is a symmetric or Hermitian matrix.
        approx_eigenvectors: The current estimate of the eigenvectors of x. If None or a zero matrix,
            falls back to using `eigh_with_fallback`.
        max_iterations: The maximum number of iterations to perform.
        tolerance: The tolerance for determining convergence in terms of the norm of the off-diagonal elements
            of the approximated eigenvalues.

    Returns:
        A tuple containing the approximated eigenvalues and eigenvectors matrix of the input matrix A.
    """

    # Check if x is already a diagonal matrix
    diag_result = _try_handle_diagonal_matrix(x)
    if diag_result is not None:
        return diag_result

    if approx_eigenvectors is None or not approx_eigenvectors.any():
        return eigh_with_fallback(x, force_double=True)

    # Perform power iteration and QR decomposition iteratively.
    Q = approx_eigenvectors
    approx_eigvals = conjugate(x, Q, diag=True)
    iteration = 0
    sorted_approx_eigvals: Tensor = approx_eigvals
    while iteration < max_iterations and not met_approx_eigvals_criteria(x, approx_eigvals, tolerance):
        power_iteration = x @ Q
        Q = torch.linalg.qr(power_iteration).Q
        approx_eigvals = conjugate(x, Q, diag=True)
        iteration += 1
        # Sort eigenvalues in descending order and reorder eigenvectors accordingly
        # Sorting can help mitigate numerical instability since QR decompositions can mix the approximated eigenvectors
        sorted_approx_eigvals, indices = approx_eigvals.sort(stable=True, descending=True)
        Q = Q[:, indices]

    return sorted_approx_eigvals, Q


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
        perform_update: Whether to update eigenbasis this iteration
    """
    matrix_norm = torch.linalg.norm(kronecker_factor)
    diagonal_norm = torch.linalg.norm(approx_eigvals)

    return diagonal_norm >= (1 - tolerance) * matrix_norm


def orthogonal_iteration(
    approx_eigvals: torch.Tensor,
    kronecker_factor: torch.Tensor,
    eigenbasis: torch.Tensor,
    ind: int,
    exp_avg_sq: torch.Tensor,
    convert_to_float: bool,
    power_iter_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the eigenbases of the preconditioner using power iteration and QR decomposition.

    This function performs multiple rounds of power iteration followed by QR decomposition
    to recompute the eigenbases of the preconditioner kronecker factor. Generalizes Vyas et al.'s (SOAP) algorithm of 1 step of power iteration for updating the eigenbasis.

    Args:
        approx_eigenvalue_matrix : Projection of kronecker factor onto the eigenbasis, should be close to diagonal
        kronecker_factor : Kronecker factor matrix.
        eigenbasis : Kronecker factor eigenbasis matrix.
        ind : Index for selecting dimension in the exp_avg_sq matrix to apply the sorting order over.
        exp_avg_sq : inner Adam second moment (exp_avg_sq).
        convert_to_float : If True, preconditioner matrices and their corresponding
            orthonormal matrices will be cast to float. Otherwise, they are left in
            their original type. Defaults to False.
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

    #  By default, perform QR decomposition with power iteration with FP32 precision
    # Perform multiple steps of power iteration
    for _ in range(power_iter_steps):
        # Project current eigenbases on kronecker factor
        Q = kronecker_factor @ Q
        # Perform QR to maintain orthogonality between iterations
        Q = torch.linalg.qr(Q).Q

    # When not converting to float, ensure that Q is in the original dtype
    if not convert_to_float:
        Q = Q.to(kronecker_factor.dtype)

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


def _is_diagonal(x: Tensor) -> bool:
    r"""Checks if symmetric matrix is diagonal. Raises an error if the input is not a square matrix."""

    x_shape = x.shape
    if len(x_shape) != 2:
        raise ValueError(f"Matrix is not 2-dimensional! {x_shape=}")

    if x_shape[0] != x_shape[1]:
        raise ValueError(f"Matrix is not square! {x_shape=}")

    # Check both upper triangular part and lower triangular part are all zeros.
    return not x.triu(diagonal=1).any() and not x.tril(diagonal=-1).any()


def _try_handle_diagonal_matrix(x: Tensor) -> Optional[tuple[Tensor, Tensor]]:
    """Checks if matrix A is diagonal and returns its eigenvalues/vectors in ascending order if so.

    Args:
        x: Tensor of shape (n, n) where x is a symmetric or Hermitian matrix.

    Returns:
        Sorted eigenvalues and eigenvectors if A is diagonal, None otherwise.
    """
    input_dtype = x.dtype
    if _is_diagonal(x):
        # If x is diagonal, eigenvalues are the diagonal elements and eigenvectors are the identity matrix
        eigenvalues = torch.diag(x)
        eigenvectors = torch.eye(x.shape[0], device=x.device, dtype=input_dtype)
        # Sort eigenvalues in ascending order and reorder eigenvectors accordingly
        sorted_eigenvalues, indices = eigenvalues.sort()
        sorted_eigenvectors = eigenvectors[:, indices]
        return sorted_eigenvalues, sorted_eigenvectors
    return None
