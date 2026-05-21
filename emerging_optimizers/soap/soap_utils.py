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
from typing import TypeAlias

import torch

from emerging_optimizers.utils import eig as eig_utils


TensorList: TypeAlias = list[torch.Tensor]


__all__ = [
    "get_eigenbasis_eigh",
    "get_eigenbasis_qr",
    "sort_eigenbasis_by_approx_eigvals",
]


def sort_eigenbasis_by_approx_eigvals(
    kronecker_factor_list: TensorList,
    eigenbasis_list: TensorList,
    exp_avg_sq: torch.Tensor,
) -> tuple[TensorList, torch.Tensor]:
    """Permute each eigenbasis and matching ``exp_avg_sq`` axis by descending approximate eigenvalues.

    Both the eigh and QR eigenbasis-update paths consume the sorted output: the eigh path discards
    the permuted eigenbasis but uses the permuted ``exp_avg_sq`` (the sort_idx best approximates the
    permutation component of the eigh-vs-old basis change under small drift); the QR path power-
    iterates from the pre-sorted basis.

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R).
        eigenbasis_list: List of current eigenbases (QL and QR).
        exp_avg_sq: Inner Adam second moment tensor permuted along each Kronecker-factor
            axis to match the new descending-eigenvalue column ordering.

    Returns:
        ``(sorted_eigenbasis_list, sorted_exp_avg_sq)``.
    """
    sorted_eigenbasis_list: TensorList = []
    sorted_exp_avg_sq = exp_avg_sq
    for ind, (kronecker_factor, eigenbasis) in enumerate(zip(kronecker_factor_list, eigenbasis_list, strict=True)):
        approx_eigvals = eig_utils.conjugate(kronecker_factor, eigenbasis, diag=True)
        sort_idx = torch.argsort(approx_eigvals, descending=True)
        sorted_eigenbasis_list.append(eigenbasis[:, sort_idx])
        sorted_exp_avg_sq = sorted_exp_avg_sq.index_select(ind, sort_idx)
    return sorted_eigenbasis_list, sorted_exp_avg_sq


def get_eigenbasis_eigh(
    kronecker_factor_list: TensorList,
) -> TensorList:
    """Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.

    Args:
        kronecker_factor_list: Matrix List to compute eigenbases of

    Returns:
        List of orthonormal kronecker factor eigenbases matrices

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
    updated_eigenbasis_list: TensorList = []

    for kronecker_factor in kronecker_factor_list:
        _, eigenvectors = eig_utils.eigh_with_fallback(kronecker_factor, force_double=False)
        updated_eigenbasis_list.append(eigenvectors)

    return updated_eigenbasis_list


def get_eigenbasis_qr(
    kronecker_factor_list: TensorList,
    eigenbasis_list: TensorList,
    power_iter_steps: int = 1,
) -> TensorList:
    """Updates the eigenbases of the preconditioner using power iteration and QR.

    Computes using multiple rounds of power iteration followed by QR decomposition (orthogonal iteration).
    ``eigenbasis_list`` is expected to be already sorted by descending approximate eigenvalues (see
    :func:`sort_eigenbasis_by_approx_eigvals`).

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R).
        eigenbasis_list: List of current eigenbases (QL and QR).
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        Updated list of orthonormal eigenbases (QL and QR).

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

            # Refine the orthogonal matrices using QR (eigenbasis_list already sorted)
            updated_ortho_matrices = get_eigenbasis_qr(
                perturbed_kronecker_factor_list,
                eigenbasis_list,
            )
    """
    updated_eigenbasis_list: TensorList = []
    for kronecker_factor, eigenbasis in zip(kronecker_factor_list, eigenbasis_list, strict=True):
        Q = eig_utils.orthogonal_iteration(
            kronecker_factor=kronecker_factor,
            eigenbasis=eigenbasis,
            power_iter_steps=power_iter_steps,
        )
        updated_eigenbasis_list.append(Q)

    return updated_eigenbasis_list
