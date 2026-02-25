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
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers import mop, muon, muon_hyperball, polargrad, scion
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class OrthogonalizedOptimizerTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        weight_decay_method=["decoupled", "independent", "l2"],
        shape=[(5, 7), (33, 65), (127, 257)],
        use_nesterov=[True, False],
        fp32_matmul_prec=["highest", "medium", "low"],
    )
    def test_smoke(self, weight_decay_method, shape, use_nesterov, fp32_matmul_prec) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
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
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
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
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
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
        test_param = torch.zeros((6, 7), dtype=torch.float32, device=self.device)
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
    def setUp(self):
        self.device = FLAGS.device

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
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        muon_opt = muon.Muon([test_param], weight_decay_method=weight_decay_method, use_nesterov=use_nesterov)
        muon_opt.step()

    def test_use_syrk_match_without_syrk(self) -> None:
        shape = (32, 32)
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        ref_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
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
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
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
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.parameters(
        {"shape": (5, 7)},
        {"shape": (33, 65)},
        {"shape": (127, 257)},
    )
    def test_smoke(self, shape) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        scion_opt = scion.Scion([test_param])
        scion_opt.step()


class MopTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        weight_decay_method=["decoupled", "independent"],
        use_nesterov=[True, False],
        scale_mode=["spectral", "nuclear_norm"],
        extra_scale_factor=[1.0, 0.2],
    )
    def test_smoke(self, shape, weight_decay_method, use_nesterov, scale_mode, extra_scale_factor) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        mop_opt = mop.MOP(
            [test_param],
            weight_decay_method=weight_decay_method,
            use_nesterov=use_nesterov,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
        )
        mop_opt.step()


class MuonHyperballTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
    )
    def test_norm_preservation(self, shape) -> None:
        """Test that MuonHyperball preserves parameter norm after optimizer steps."""
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=self.device))
        initial_norm = test_param.norm().item()

        opt = muon_hyperball.MuonHyperball(
            [test_param],
            lr=0.01,
            momentum_beta=0.0,
            weight_decay=0.0,
        )

        # Run multiple steps with random gradients
        for _ in range(5):
            test_param.grad = torch.randn_like(test_param)
            opt.step()

            # Norm should be preserved after each step
            torch.testing.assert_close(
                test_param.norm(),
                torch.tensor(initial_norm, device=self.device),
                atol=1e-5,
                rtol=1e-5,
            )

    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        hyperball_radius=[0.5, 1.0, 2.0],
    )
    def test_hyperball_radius_rescales_params(self, shape, hyperball_radius) -> None:
        """Test that hyperball_radius kwarg rescales parameters to specified radius."""
        test_param = nn.Parameter(torch.randn(shape, dtype=torch.float32, device=self.device))

        opt = muon_hyperball.MuonHyperball(
            [test_param],
            lr=0.01,
            hyperball_radius=hyperball_radius,
        )

        # After initialization, parameter should have the specified radius
        torch.testing.assert_close(
            test_param.norm(),
            torch.tensor(hyperball_radius, device=self.device),
            atol=1e-5,
            rtol=1e-5,
        )

        # Run multiple steps with random gradients
        for _ in range(5):
            test_param.grad = torch.randn_like(test_param)
            opt.step()

            # Norm should remain at hyperball_radius after each step
            torch.testing.assert_close(
                test_param.norm(),
                torch.tensor(hyperball_radius, device=self.device),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_zero_norm_raises_error(self) -> None:
        """Test that MuonHyperball raises ValueError for zero-norm parameters."""
        test_param = nn.Parameter(torch.zeros((5, 7), dtype=torch.float32, device=self.device))

        with self.assertRaises(ValueError):
            muon_hyperball.MuonHyperball([test_param], lr=0.01)


class PolarGradTest(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        extra_scale_factor=[1.0, 0.2],
    )
    def test_smoke(self, shape, extra_scale_factor) -> None:
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        polargrad_opt = polargrad.PolarGrad(
            [test_param],
            extra_scale_factor=extra_scale_factor,
        )
        polargrad_opt.step()

    @parameterized.product(
        shape=[(4, 8), (16, 16), (32, 64), (13, 17)],
        extra_scale_factor=[0.25, 0.125],
    )
    def test_orthogonalize_fn_matches_ref(self, shape, extra_scale_factor) -> None:
        dummy_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device=self.device))
        dummy_grad = torch.full(shape, 0.5, dtype=torch.float32, device=self.device)

        # Set num_ns_steps to 0 to skip Newton-Schulz iterations and only normalize the input gradient.
        polargrad_opt = polargrad.PolarGrad([dummy_param], num_ns_steps=0, extra_scale_factor=extra_scale_factor)
        norm_grad = torch.nn.functional.normalize(dummy_grad, p=2, dim=(-2, -1), eps=1e-7)

        # Assert normalization took effect
        self.assertFalse((norm_grad == 1).all())

        ref_scale = (norm_grad * dummy_grad).sum()
        ref_out = norm_grad * ref_scale * extra_scale_factor

        test_out = polargrad_opt.scaled_orthogonalize_fn(dummy_grad)

        torch.testing.assert_close(
            ref_out,
            test_out,
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    absltest.main()
