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
from contextlib import contextmanager
from typing import Generator

import torch


__all__ = ["fp32_matmul_precision", "get_pg_size", "get_pg_rank"]


@contextmanager
def fp32_matmul_precision(precision: str = "highest") -> Generator[None, None, None]:
    """Context manager for setting the precision of matmuls.

    Args:
        precision: Precision of matmuls (defaults to "highest")
    """
    prev_val = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_val)


def get_pg_size(group: torch.distributed.ProcessGroup | None = None) -> int:
    """Get world size for a distributed group with fallback"""
    if not torch.distributed.is_initialized() or group is None:
        return 1
    return group.size()


def get_pg_rank(group: torch.distributed.ProcessGroup | None = None) -> int:
    """Get rank for a distributed group with fallback"""
    if not torch.distributed.is_initialized() or group is None:
        return 0
    return group.rank()
