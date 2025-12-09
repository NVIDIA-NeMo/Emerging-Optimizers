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

from emerging_optimizers.orthogonalized_optimizers import mop, muon, scion
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer


class OrthogonalizedOptimizerTest(parameterized.TestCase):
    @parameterized.product(
        weight_decay_method=["decoupled", "independent", "l2"],
        shape=[(5, 7), (33, 65), (127, 257)],
        use_nesterov=[True, False],
        fp32_matmul_prec=["highest", "medium", "low"],
    )
    def test_smoke(self, weight_decay_method, shape, use_nesterov, fp32_matmul_prec) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        orthogonalized_opt = OrthogonalizedOptimizer(
            [test_param],
            lr=2,
            momentum_beta=0,
            weight_decay=0.5,
            use_nesterov=use_nesterov,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
        )
        orthogonalized_opt.step()

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
            weight_decay_method="decoupled",
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
            weight_decay_method="l2",
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

    def test_split_fn_interleaved(self) -> None:
        """Test a three way interleaved split function.

        With 0 weights and lr -1, returned param should match orthogonalized grads.
        """
        test_param = torch.zeros((6, 7), dtype=torch.float32, device="cuda")
        test_param.grad = torch.empty_like(test_param.data)

        for i in range(test_param.shape[0]):
            test_param.grad[i] = i + 1

        def dummy_interleaved_split_orth_fn(x: torch.Tensor) -> torch.Tensor:
            out_list = [[], [], []]
            for i in range(x.shape[0]):
                out_list[i % 3].append(x[i : i + 1])
            orth_grad_list = [torch.cat(t, dim=0) for t in out_list]
            return torch.cat([torch.empty_like(x).fill_(x.max()) for x in orth_grad_list], dim=0)

        orthogonalized_opt = OrthogonalizedOptimizer(
            [test_param],
            lr=-1,
            momentum_beta=0,
            use_nesterov=False,
            weight_decay=0.0,
            weight_decay_method="l2",
            fp32_matmul_prec="highest",
            scaled_orthogonalize_fn=dummy_interleaved_split_orth_fn,
        )
        orthogonalized_opt.step()

        assert not torch.allclose(test_param, test_param.grad)

        ref_out = dummy_interleaved_split_orth_fn(test_param.grad)
        torch.testing.assert_close(
            test_param,
            ref_out,
            atol=0,
            rtol=0,
        )


class MuonTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        weight_decay_method=["decoupled", "independent", "l2"],
        use_nesterov=[True, False],
    )
    def test_smoke(self, shape, weight_decay_method, use_nesterov) -> None:
        """Smoke test Muon optimizer.
        Most functionality of muon is tested in muon_utils. This test only entures everything run through
        the optimizer class.
        """
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        muon_opt = muon.Muon([test_param], weight_decay_method=weight_decay_method, use_nesterov=use_nesterov)
        muon_opt.step()

    def test_use_syrk_match_without_syrk(self) -> None:
        shape = (32, 32)
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        ref_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        ref_param.data.copy_(test_param.data)
        test_param.grad = torch.randint_like(test_param, -5, 5)
        ref_param.grad = test_param.grad.clone()

        muon_opt = muon.Muon([test_param], num_ns_steps=1, coefficient_type="simple", use_syrk=True)
        ref_muon_opt = muon.Muon([ref_param], num_ns_steps=1, coefficient_type="simple", use_syrk=False)
        muon_opt.step()
        ref_muon_opt.step()

        torch.testing.assert_close(
            test_param.data,
            ref_param.data,
        )

    def test_use_independent_wd(self) -> None:
        """Test that use_independent_wd properly decouples weight decay from learning rate."""
        shape = (32, 32)
        weight_decay = 0.25

        # Test with independent weight decay: with lr=0, weight decay should still be applied
        # With lr=0, no gradient update occurs, so param should be exactly (1-wd)*param
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)
        # With independent weight decay and lr=0, param should be exactly (1-wd)*param
        expected_param = (1 - weight_decay) * test_param.data

        muon_opt_indep = muon.Muon(
            [test_param],
            lr=0.0,  # Zero learning rate
            weight_decay=weight_decay,
            weight_decay_method="independent",
            momentum_beta=0.0,
        )
        muon_opt_indep.step()

        torch.testing.assert_close(
            test_param,
            expected_param,
            atol=0,
            rtol=0,
        )


class ScionTest(parameterized.TestCase):
    @parameterized.parameters(
        {"shape": (5, 7)},
        {"shape": (33, 65)},
        {"shape": (127, 257)},
    )
    def test_smoke(self, shape) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        scion_opt = scion.Scion([test_param])
        scion_opt.step()


class MopTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        weight_decay_method=["decoupled", "independent"],
        use_nesterov=[True, False],
        extra_scale_factor=[1.0, 2.0],
    )
    def test_smoke(self, shape, weight_decay_method, use_nesterov, extra_scale_factor) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        mop_opt = mop.MOP(
            [test_param],
            weight_decay_method=weight_decay_method,
            use_nesterov=use_nesterov,
            extra_scale_factor=extra_scale_factor,
        )
        mop_opt.step()


if __name__ == "__main__":
    absltest.main()
