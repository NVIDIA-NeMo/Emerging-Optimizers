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
"""Comparison helpers for tests.

``assert_equal`` is :func:`torch.testing.assert_close` with ``atol=rtol=0``: it asserts that two
tensors are exactly (bitwise) equal. Use it for bit-identity checks instead of repeating
``atol=0, rtol=0``.
"""

import functools

import torch
from torch import testing as torch_testing


assert_equal = functools.partial(torch_testing.assert_close, rtol=0, atol=0)


def align_column_signs(actual, reference):
    signs = torch.sign((actual * reference).sum(dim=-2))
    return actual * signs.unsqueeze(-2), signs


def assert_close_to_identity(actual, *, off_diag_atol=0, diag_atol=0):
    r"""Assert that ``actual`` is close to the identity matrix.

    Checks the identity structure with separate tolerances for the diagonal (compared to ones) and the
    off-diagonal entries (compared to zeros). This is more informative, and allows looser tolerances on
    the off-diagonal, than comparing the whole matrix to ``torch.eye`` with a single tolerance.

    Args:
        actual: A square 2D tensor expected to be (approximately) the identity matrix.
        off_diag_atol: Absolute tolerance for the off-diagonal entries (compared to 0).
        diag_atol: Absolute tolerance for the diagonal entries (compared to 1).

    Raises:
        ValueError: If ``actual`` is not a square 2D matrix.
        AssertionError: If ``actual`` is not close to the identity matrix.
    """
    if actual.ndim != 2 or actual.shape[0] != actual.shape[1]:
        raise ValueError(f"actual must be a square 2D matrix, got shape {tuple(actual.shape)}")

    n = actual.shape[-1]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=actual.device)
    diag = torch.diagonal(actual)
    off_diag = actual[off_diag_mask]
    torch_testing.assert_close(
        diag,
        torch.ones_like(diag),
        atol=diag_atol,
        rtol=0,
        msg=lambda msg: f"Diagonal is not close to ones.\n\n{msg}",
    )
    torch_testing.assert_close(
        off_diag,
        torch.zeros_like(off_diag),
        atol=off_diag_atol,
        rtol=0,
        msg=lambda msg: f"Off-diagonal is not close to zeros.\n\n{msg}",
    )


def assert_close_to_orthogonal(actual, *, off_diag_atol=0, diag_atol=0):
    r"""Assert that a 2D matrix is (semi-)orthogonal.

    Builds the Gram matrix over the smaller dimension (``X @ Xᵀ`` when ``X`` has no more rows than
    columns, otherwise ``Xᵀ @ X``) and asserts it is close to the identity via
    :func:`assert_close_to_identity`.

    Args:
        actual: A 2D tensor expected to have (semi-)orthonormal rows or columns.
        off_diag_atol: Absolute tolerance for the off-diagonal entries of the Gram matrix.
        diag_atol: Absolute tolerance for the diagonal entries of the Gram matrix.

    Raises:
        ValueError: If ``actual`` is not a 2D matrix.
        AssertionError: If ``actual`` is not (semi-)orthogonal.
    """
    if actual.ndim != 2:
        raise ValueError(f"actual must be a 2D matrix, got shape {tuple(actual.shape)}")

    m, n = actual.shape
    gram = actual @ actual.mT if m <= n else actual.mT @ actual
    assert_close_to_identity(gram, off_diag_atol=off_diag_atol, diag_atol=diag_atol)
