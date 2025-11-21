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
import torch.nn as nn
from absl.testing import absltest, parameterized

from emerging_optimizers.utils.modules import Conv1dFlatWeights


class TestConv1dFlatWeights(parameterized.TestCase):
    @parameterized.product(
        in_channels=[3, 5, 7],
        out_channels=[4, 6, 8],
        kernel_size=[2, 3, 4],
        batch_size=[4, 5, 6],
        bias=[False, True],
    )
    def test_no_bias_matches_conv1d(self, in_channels, out_channels, kernel_size, batch_size, bias):
        torch.manual_seed(42)
        conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
        torch.manual_seed(42)
        conv_flat = Conv1dFlatWeights(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias
        )

        self.assertEqual(conv_flat.weight.dim(), 2)

        x = torch.randn(batch_size, in_channels, kernel_size)
        y_ref = conv(x)
        y_test = conv_flat(x)

        torch.testing.assert_close(y_ref, y_test, atol=0, rtol=0)

        y_ref.sum().backward()
        y_test.sum().backward()
        if bias:
            torch.testing.assert_close(
                conv.weight.grad.view(-1), conv_flat.weight.grad[:, :-1].reshape(-1), atol=0, rtol=0
            )
            torch.testing.assert_close(conv.bias.grad, conv_flat.weight.grad[:, -1], atol=0, rtol=0)
        else:
            torch.testing.assert_close(conv.weight.grad.view(-1), conv_flat.weight.grad.reshape(-1), atol=0, rtol=0)

    @parameterized.product(
        bias=[False, True],
    )
    def test_extra_repr(self, bias):
        conv_flat = Conv1dFlatWeights(in_channels=3, out_channels=4, kernel_size=2, bias=bias)
        print(conv_flat)


if __name__ == "__main__":
    absltest.main()
