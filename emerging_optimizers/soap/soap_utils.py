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
from typing import List, Optional, Tuple

import torch

from emerging_optimizers import utils


__all__ = [
    "get_eigenbasis_eigh",
    "get_eigenbasis_qr",
]


def get_eigenbasis_eigh(
    kronecker_factor_list: List[torch.Tensor],
    convert_to_float: bool = True,
    eigenbasis_list: Optional[List[torch.Tensor]] = None,
    use_adaptive_criteria: bool = False,
    adaptive_update_tolerance: Optional[float] = None,
    eps: Optional[float] = None,
) -> List[torch.Tensor]:
    """Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.

    Args:
        kronecker_factor_list: Matrix List to compute eigenbases of
        convert_to_float: If True, preconditioner matrices and their corresponding
            orthonormal matrices will be cast to float. Otherwise, they are left in
            their original type.
        eigenbasis_list: List of orthonormal eigenbases of the kronecker factor matrices
        use_adaptive_criteria: Whether to use update criteria strategy
        adaptive_update_tolerance: Tolerance threshold for the normalized diagonal component of approximated eigenvalue matrix.
            If None, defaults to 1e-7, which is appropriate for single precision computations.
        eps: Small offset for numerical stability. If None, uses dtype-appropriate values (1e-7 for float32, 1e-15 for float64)

    Returns:
        List[torch.Tensor]: List of orthonormal kronecker factor eigenbases matrices

    Example:
        .. code-block:: python

            # Create sample Kronecker factors (symmetric positive definite matrices)
            k_factor1 = torch.randn(4, 4)
            k_factor1 = k_factor1 @ k_factor1.T  # Make symmetric positive definite
            k_factor2 = torch.randn(5, 5)
            k_factor2 = k_factor2 @ k_factor2.T  # Make symmetric positive definite

            # Get orthogonal matrices for these factors
            ortho_matrices = get_eigenbasis_eigh([k_factor1, k_factor2])
            # ortho_matrices[0] has shape [4, 4] and ortho_matrices[1] has shape [5, 5]
    """
    if adaptive_update_tolerance is None:
        adaptive_update_tolerance = 1e-7

    # cast the kronecker factor matrices to float32 if convert_to_float is True
    casted_matrix_list: List[torch.Tensor] = []
    for kronecker_factor in kronecker_factor_list:
        if kronecker_factor.numel() == 0:
            casted_matrix_list.append(torch.empty(0, device=kronecker_factor.device))
            continue
        if convert_to_float:
            casted_matrix_list.append(kronecker_factor.to(torch.float))
        else:
            casted_matrix_list.append(kronecker_factor)

    updated_eigenbasis_list: List[torch.Tensor] = []

    # use adaptive early exit criteria
    if use_adaptive_criteria and eigenbasis_list is not None:
        for kronecker_factor, eigenbasis in zip(casted_matrix_list, eigenbasis_list, strict=True):
            if kronecker_factor.numel() == 0:
                # We use an empty tensor so that the `precondition` function will skip this factor.
                updated_eigenbasis_list.append(torch.empty(0, device=kronecker_factor.device))
                continue
            # Construct approximated eigenvalues using :math:`Q_L^T L Q_L` or :math:`Q_R^T R Q_R`.
            # The approximated eigenvalues should be close to diagonal if the eigenbasis is close to the true
            # eigenbasis of the kronecker factor (i.e. the approximated eigenvectors diagonalize the kronecker factor)
            approx_eigenvalue_matrix = eigenbasis.T @ kronecker_factor @ eigenbasis
            # Update eigenbasis when necessary. Update is skipped only when adaptive update criteria is met.
            if utils.eig.met_approx_eigvals_criteria(approx_eigenvalue_matrix, adaptive_update_tolerance):
                _, Q = utils.eig.eigh_with_fallback(
                    kronecker_factor,
                    force_double=False,
                    eps=eps,
                    output_dtype=torch.float if convert_to_float else None,
                )
                updated_eigenbasis_list.append(Q)
            else:
                # Do not update eigenbasis matrix since adaptive update criteria is not met
                updated_eigenbasis_list.append(eigenbasis)
    else:
        for kronecker_factor in casted_matrix_list:
            if kronecker_factor.numel() == 0:
                updated_eigenbasis_list.append(torch.empty(0, device=kronecker_factor.device))
                continue
            _, Q = utils.eig.eigh_with_fallback(
                kronecker_factor, force_double=False, eps=eps, output_dtype=torch.float if convert_to_float else None
            )
            updated_eigenbasis_list.append(Q)

    return updated_eigenbasis_list


def get_eigenbasis_qr(
    kronecker_factor_list: List[torch.Tensor],
    eigenbasis_list: List[torch.Tensor],
    exp_avg_sq: torch.Tensor,
    convert_to_float: bool = True,
    use_adaptive_criteria: bool = False,
    adaptive_update_tolerance: Optional[float] = None,
    power_iter_steps: int = 1,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Updates the eigenbases of the preconditioner using power iteration and QR.

    Computes using multiple rounds of power iteration followed by QR decomposition (orthogonal iteration).

    Args:
        kronecker_factor_list: List containing preconditioner (:math:`GG^T` and :math:`G^TG`)
        eigenbasis_list: List containing eigenbases (:math:`Q_L` and :math:`Q_R`)
        exp_avg_sq: inner adam second moment (exp_avg_sq). This tensor is modified in-place.
        convert_to_float: If True, preconditioner matrices and their corresponding
            orthonormal matrices will be cast to float. Otherwise, they are left in
            their original type.
        use_adaptive_criteria: Whether to use update criteria strategy
        adaptive_update_tolerance: Tolerance threshold for the normalized diagonal component of approximated eigenvalue matrix.
            If None, defaults to 1e-7, which is appropriate for single precision computations. This means adaptive update criteria will be used whenever there is a small change in the approximated eigenvalues
            matrix and QR will be used.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        List[torch.Tensor]: Updated list of orthonormal kronecker factor eigenbases matrices
        torch.Tensor: Updated (sorted) inner adam second moment

    Example:
        .. code-block:: python

            # Create sample Kronecker factors (symmetric positive definite matrices)
            n, m = 10, 20
            k_factor1 = torch.randn(n, n)
            k_factor1 = k_factor1 @ k_factor1.T  # Make symmetric positive definite
            k_factor2 = torch.randn(m, m)
            k_factor2 = k_factor2 @ k_factor2.T  # Make symmetric positive definite

            # Get orthogonal matrices for these kronecker factors
            kronecker_factor_list = [k_factor1, k_factor2]
            eigenbasis_list = get_eigenbasis_eigh(kronecker_factor_list)

            # Perturb the kronecker factor matrices, simulating the effect of gradient updates
            perturbation = 1e-2*torch.randn(n, m)
            perturbed_kronecker_factor_list = [None, None]
            perturbed_kronecker_factor_list[0] = k_factor1 + perturbation@perturbation.T
            perturbed_kronecker_factor_list[1] = k_factor2 + perturbation.T@perturbation

            # Initialize exp_avg_sq tensor
            exp_avg_sq = torch.randn(n, m).abs()

            # Refine the orthogonal matrices using QR
            updated_ortho_matrices, updated_exp_avg_sq = get_eigenbasis_qr(
                perturbed_kronecker_factor_list,
                eigenbasis_list,
                exp_avg_sq
            )
    """
    if adaptive_update_tolerance is None:
        adaptive_update_tolerance = 1e-7

    casted_matrix_list: List[torch.Tensor] = []
    casted_eigenbasis_list: List[torch.Tensor] = []
    for kronecker_factor, eigenbasis in zip(kronecker_factor_list, eigenbasis_list, strict=True):
        # If the tensor is empty, propagate an empty tensor to the output lists.
        if kronecker_factor.numel() == 0:
            casted_matrix_list.append(torch.empty(0, device=kronecker_factor.device))
            casted_eigenbasis_list.append(torch.empty(0, device=kronecker_factor.device))
            continue
        # Use the argument to decide whether to cast to float.
        if convert_to_float:
            casted_matrix_list.append(kronecker_factor.to(torch.float))
            casted_eigenbasis_list.append(eigenbasis.to(torch.float))
        else:
            casted_matrix_list.append(kronecker_factor)
            casted_eigenbasis_list.append(eigenbasis)

    # Cast exp_avg_sq to float in-place if needed
    if convert_to_float and exp_avg_sq.dtype != torch.float:
        exp_avg_sq = exp_avg_sq.to(torch.float)

    updated_eigenbasis_list: List[torch.Tensor] = []
    for ind, (kronecker_factor, eigenbasis) in enumerate(zip(casted_matrix_list, casted_eigenbasis_list, strict=True)):
        if kronecker_factor.numel() == 0:
            updated_eigenbasis_list.append(torch.empty(0, device=kronecker_factor.device))
            continue

        # Update eigenbasis when necessary. Update is skipped only when use_adaptive_criteria is True
        # but criteria is not met.
        if_update = True
        # construct approximated eigenvalues using :math:`Q_L^T L Q_L` or :math:`Q_R^T R Q_R`, which should be close to diagonal
        # if the eigenbasis is close to the true eigenbasis of the kronecker factor (i.e. diagonalizes it)
        if use_adaptive_criteria:
            approx_eigenvalue_matrix = _conjugate(kronecker_factor, eigenbasis)
            if_update = not utils.eig.met_approx_eigvals_criteria(approx_eigenvalue_matrix, adaptive_update_tolerance)
            if if_update:
                approx_eigvals = torch.diag(approx_eigenvalue_matrix)
        else:
            approx_eigvals = _conjugate(kronecker_factor, eigenbasis, diag=True)

        if if_update:
            Q, exp_avg_sq = _orthogonal_iteration(
                approx_eigvals=approx_eigvals,
                kronecker_factor=kronecker_factor,
                eigenbasis=eigenbasis,
                ind=ind,
                exp_avg_sq=exp_avg_sq,
                convert_to_float=convert_to_float,
                power_iter_steps=power_iter_steps,
            )
            updated_eigenbasis_list.append(Q)
        else:
            # Do not update eigenbasis matrix
            updated_eigenbasis_list.append(eigenbasis)

    return updated_eigenbasis_list, exp_avg_sq


def _orthogonal_iteration(
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


def _conjugate(a: torch.Tensor, p: torch.Tensor, diag: bool = False) -> torch.Tensor:
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
