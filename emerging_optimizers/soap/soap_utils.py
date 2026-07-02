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

from emerging_optimizers import utils
from emerging_optimizers.utils import eig as eig_utils


TensorList: TypeAlias = list[torch.Tensor]


__all__ = [
    "get_eigenbasis_eigh",
    "get_eigenbasis_qr",
    "get_eigenbasis_svd",
]


def get_eigenbasis_eigh(
    kronecker_factor_list: TensorList,
) -> tuple[TensorList, TensorList]:
    """Computes the eigenvalues and eigenbases of the preconditioner using torch.linalg.eigh decomposition.

    Args:
        kronecker_factor_list: Matrix List to compute eigenbases of

    Returns:
        Tuple of (list of eigenvalues in descending order, list of orthonormal kronecker factor
        eigenbases matrices).
    """
    updated_eigenbasis_list: TensorList = []
    updated_eigvals_list: TensorList = []

    for kronecker_factor in kronecker_factor_list:
        eigvals, eigenvectors = eig_utils.eigh_with_fallback(kronecker_factor, force_double=False)
        updated_eigvals_list.append(eigvals)
        updated_eigenbasis_list.append(eigenvectors)

    return updated_eigvals_list, updated_eigenbasis_list


def get_eigenbasis_svd(
    kronecker_factor_list: TensorList,
) -> TensorList:
    """Computes the eigenbases of the preconditioner using torch.linalg.svd decomposition.

    The kronecker factors :math:`L = GG^\\top` and :math:`R = G^\\top G` are symmetric positive
    semi-definite, so the left and right singular vectors coincide (up to sign in the presence
    of repeated singular values); this function returns the left singular vectors :math:`U` as
    the eigenbasis. Singular values from ``torch.linalg.svd`` are returned in descending order.

    Args:
        kronecker_factor_list: Matrix List to compute eigenbases of

    Returns:
        List of orthonormal kronecker factor eigenbases matrices
    """
    updated_eigenbasis_list: TensorList = []

    for kronecker_factor in kronecker_factor_list:
        U, _, _ = torch.linalg.svd(kronecker_factor)
        updated_eigenbasis_list.append(U)

    return updated_eigenbasis_list


def get_eigenbasis_qr(
    kronecker_factor_list: TensorList,
    eigenbasis_list: TensorList,
    exp_avg_sq: torch.Tensor,
    power_iter_steps: int = 1,
) -> tuple[TensorList, TensorList, torch.Tensor]:
    """Updates the eigenbases of the preconditioner using power iteration and QR.

    Each eigenbasis is refined with multiple rounds of power iteration followed by QR decomposition
    (orthogonal iteration), then its columns are sorted by descending approximate eigenvalues of the
    kronecker factor (the Rayleigh quotients :math:`\\mathrm{diag}(Q^{\\top} K Q)`), so the returned
    eigenvalues and eigenbases are in descending order like :func:`get_eigenbasis_eigh`.
    ``exp_avg_sq`` is permuted along each kronecker-factor axis by the same sort, so its per-slot
    statistics stay aligned with the eigenbasis columns.

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R).
        eigenbasis_list: List of current eigenbases (QL and QR).
        exp_avg_sq: Inner Adam second moment tensor, permuted along each kronecker-factor axis to
            match the sorted eigenbasis columns.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        Tuple of (list of approximate eigenvalues of each kronecker factor in its updated eigenbasis,
        in descending order, updated list of orthonormal eigenbases (QL and QR) with columns ordered to
        match, ``exp_avg_sq`` permuted to match).
    """
    updated_eigenbasis_list: TensorList = []
    updated_eigvals_list: TensorList = []
    for ind, (kronecker_factor, eigenbasis) in enumerate(zip(kronecker_factor_list, eigenbasis_list, strict=True)):
        Q = eig_utils.orthogonal_iteration(
            kronecker_factor=kronecker_factor,
            eigenbasis=eigenbasis,
            power_iter_steps=power_iter_steps,
        )
        with utils.fp32_matmul_precision("highest"):
            approx_eigvals = eig_utils.conjugate(kronecker_factor, Q, diag=True)
        approx_eigvals, sort_idx = torch.sort(approx_eigvals, descending=True)
        updated_eigvals_list.append(approx_eigvals)
        updated_eigenbasis_list.append(Q[:, sort_idx])
        exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)

    return updated_eigvals_list, updated_eigenbasis_list, exp_avg_sq
