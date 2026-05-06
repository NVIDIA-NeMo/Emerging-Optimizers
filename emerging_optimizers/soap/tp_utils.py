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
from torch import distributed as dist

from emerging_optimizers.utils import get_pg_size


@torch.no_grad()  # type: ignore[misc]
def all_gather_grad_and_kronecker_factors_tp(
    kronecker_factor_list: list[torch.Tensor],
    grad: torch.Tensor,
    partition_dim: int,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """All-gathers a sharded gradient and its kronecker factors across the tensor parallel group.

    This is a simple implementation to support tensor parallel. It assumes grad is sharded among tensor parallel domain
    with partition_dim indicating the dimension it was sharded. To save memory, kronecker factors are also sharded
    but always long dimension 0 to make gather operation easy.

    Gradient and kronecker factors are both all-gathered to all tensor parallel ranks and returned so the caller
    can pass them to ``update_kronecker_factors`` (and downstream eigenbasis computations) without further
    communication.

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R), each sharded along dimension 0 across
            the tensor parallel group.
        grad: Local shard of the gradient tensor, sharded along ``partition_dim`` across the tensor parallel group.
        partition_dim: Dimension along which ``grad`` is sharded across the tensor parallel group.
        tp_group: Tensor parallel process group used to all-gather ``grad`` and ``kronecker_factor_list``.

    Returns:
        full_grad: Full (un-sharded) gradient tensor on every rank.
        full_kronecker_factor_list: List of full (un-sharded) kronecker factor matrices ``[L, R]`` on every rank.
    """
    tp_size = get_pg_size(tp_group)

    grad_shards = [torch.empty_like(grad) for _ in range(tp_size)]
    dist.all_gather(grad_shards, grad, group=tp_group)
    full_grad = torch.cat(grad_shards, dim=partition_dim)

    full_kronecker_factor_list: list[torch.Tensor] = []
    for kronecker_factor in kronecker_factor_list:
        factor_shards = [torch.empty_like(kronecker_factor) for _ in range(tp_size)]
        dist.all_gather(factor_shards, kronecker_factor, group=tp_group)
        full_kronecker_factor_list.append(torch.cat(factor_shards, dim=0))

    return full_grad, full_kronecker_factor_list
