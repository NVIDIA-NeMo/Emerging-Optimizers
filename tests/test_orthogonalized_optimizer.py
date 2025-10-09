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

from emerging_optimizers.orthogonalized_optimizers import muon
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer


class OrthogonalizedOptimizerTest(parameterized.TestCase):
    @parameterized.parameters(
        {"shape": (5, 7)},
        {"shape": (33, 65)},
        {"shape": (127, 257)},
    )
    def test_orthogonalized_optimizer_core_matches_sgd(self, shape) -> None:
        """Test that OrthogonalizedOptimizer matches SGD when orthogonalization is disabled."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        ref_param = nn.Parameter(torch.empty_like(test_param))
        ref_param.data.copy_(test_param.data)

        test_param.grad = torch.randint_like(test_param, -5, 5)
        ref_param.grad = test_param.grad.clone()

        orthogonalized_opt = OrthogonalizedOptimizer(
            [test_param],
            lr=2,
            momentum_beta=0,
            use_nesterov=False,
            weight_decay=0.5,
            use_decoupled_weight_decay=True,
            split_qkv=False,
            is_qkv_fn=None,
            qkv_split_shapes=None,
            fp32_matmul_prec="highest",
        )

        sgd_opt = torch.optim.SGD(
            [ref_param],
            lr=2,
            momentum=0,
            nesterov=False,
            weight_decay=0.5,
        )

        orthogonalized_opt.step()
        sgd_opt.step()

        torch.testing.assert_close(
            test_param.data,
            ref_param.data,
            atol=0,
            rtol=0,
        )

    @parameterized.parameters(
        {"shape": (5, 7)},
        {"shape": (33, 65)},
        {"shape": (127, 257)},
    )
    def test_orthogonalized_optimizer_core_matches_sgd_with_momentum(self, shape) -> None:
        """Test that OrthogonalizedOptimizer matches SGD with momentum over multiple steps."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        ref_param = nn.Parameter(torch.empty_like(test_param))
        ref_param.data.copy_(test_param.data)

        # Muon EMA momentum while torch SGD uses standard momentum. lr and momentum_beta values
        # are specially chosen for them to match.
        orthogonalized_opt = OrthogonalizedOptimizer(
            [test_param],
            lr=2.0,
            momentum_beta=0.5,
            use_nesterov=False,
            weight_decay=0.0,
            use_decoupled_weight_decay=False,
            split_qkv=False,
            is_qkv_fn=None,
            qkv_split_shapes=None,
            fp32_matmul_prec="highest",
        )

        sgd_opt = torch.optim.SGD(
            [ref_param],
            lr=1.0,
            momentum=0.5,
            nesterov=False,
            weight_decay=0.0,
        )

        for _ in range(5):
            test_param.grad = torch.randint_like(test_param, -5, 5)
            ref_param.grad = test_param.grad.clone()

            orthogonalized_opt.step()
            sgd_opt.step()

        torch.testing.assert_close(
            test_param.data,
            ref_param.data,
            atol=0,
            rtol=0,
        )

    def test_split_qkv_matches_ref(self) -> None:
        test_param = torch.randint(-5, 5, (6, 7), dtype=torch.float32, device="cuda")
        test_param.grad = torch.randint_like(test_param, -5, 5)
        split_shapes = (1, 2, 3)
        lr = 2.0

        def is_qkv_fn(x: torch.Tensor) -> bool:
            return x.shape == torch.Size([6, 7])

        def dummy_orth_fn(x: torch.Tensor) -> torch.Tensor:
            return x * x

        ref_orth_grads = []
        for g in torch.split(test_param.grad, split_shapes, dim=0):
            ref_orth_grads.append(dummy_orth_fn(g))
        ref_out = test_param - torch.cat(ref_orth_grads, dim=0) * lr

        orthogonalized_opt = OrthogonalizedOptimizer(
            [test_param],
            lr=lr,
            momentum_beta=0,
            use_nesterov=False,
            weight_decay=0.0,
            use_decoupled_weight_decay=False,
            split_qkv=True,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=(1, 2, 3),
            fp32_matmul_prec="highest",
            orthogonalize_fn=dummy_orth_fn,
        )
        orthogonalized_opt.step()

        torch.testing.assert_close(
            test_param.data,
            ref_out,
            atol=0,
            rtol=0,
        )


class MuonTest(parameterized.TestCase):
    @parameterized.parameters(
        {"shape": (5, 7)},
        {"shape": (33, 65)},
        {"shape": (127, 257)},
    )
    def test_smoke(self, shape) -> None:
        """Smoke test Muon optimizer.
        Most functionality of muon is tested in muon_utils. This test only entures everything run through
        the optimizer class.
        """
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        muon_opt = muon.Muon([test_param])
        muon_opt.step()

    def test_use_syrk(self) -> None:
        shape = (32, 32)
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        ref_param = test_param.clone()
        test_param.grad = torch.randint_like(test_param, -5, 5)

        muon_opt = muon.Muon([test_param], use_syrk=True)
        ref_muon_opt = muon.Muon([test_param], use_syrk=False)
        muon_opt.step()
        ref_muon_opt.step()

        torch.testing.assert_close(
            test_param.data,
            ref_param.data,
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    absltest.main()
