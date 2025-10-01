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

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz


__all__ = ["spectral_hardcap", "spectral_clip"]


def spectral_hardcap(X: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    r"""Spectral hardcap function clips singular values from above to be less than beta.

    Based on https://leloykun.github.io/ponder/spectral-clipping/.

    Args:
        X: The input tensor.
        beta: The upper bound on the singular values.

    Returns:
        The spectral hardcapped tensor.

    """
    if needs_transpose := X.shape[0] > X.shape[1]:
        X = X.T
    OX = newton_schulz(X, steps=8, coefficient_type="polar_express")
    aX = beta * OX - X
    result = (1 / 2) * (beta * OX + X - aX @ newton_schulz(aX, steps=8, coefficient_type="polar_express").T @ OX)
    if needs_transpose:
        result = result.T
    return result


def spectral_clip(X: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0) -> torch.Tensor:
    r"""Applies spectral clipping to the input tensor.

    Based on https://leloykun.github.io/ponder/spectral-clipping/.

    Args:
        X: The input tensor.
        sigma_min: The minimum singular value.
        sigma_max: The maximum singular value.

    Returns:
        The spectral clipped tensor.
    """
    if needs_transpose := X.shape[0] > X.shape[1]:
        X = X.T
    OX = newton_schulz(X, steps=8, coefficient_type="polar_express")
    result = (1 / 2) * (
        (sigma_min + sigma_max) * OX
        + newton_schulz(sigma_min * torch.eye(X.shape[0]) - OX @ X.T, steps=8, coefficient_type="polar_express")
        @ (sigma_min * OX - X)
        - newton_schulz(sigma_max * torch.eye(X.shape[0]) - OX @ X.T, steps=8, coefficient_type="polar_express")
        @ (sigma_max * OX - X)
    )
    if needs_transpose:
        result = result.T
    return result


def spectral_clipped_weight_decay(X: torch.Tensor, beta: float = 1.0, c: float = 0.5) -> torch.Tensor:
    r"""Applies weight decay to the input tensor while applying spectral clipping.

    Based on https://leloykun.github.io/ponder/spectral-clipping/.

    Args:
        X: The input tensor.
        beta: The upper bound on the singular values.
        c: The coefficient parameter.

    Returns:
        The spectral clipped weight decay tensor.
    """
    return (1 - c) * X + c * spectral_hardcap(X, beta)
