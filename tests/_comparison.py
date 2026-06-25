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

from torch import testing as torch_testing


assert_equal = functools.partial(torch_testing.assert_close, rtol=0, atol=0)
