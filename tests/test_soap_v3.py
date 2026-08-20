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
from typing import override

import torch
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.legacy_soap import rekls, soap
from emerging_optimizers.shampoo.soap_v3 import KlMSoap, KlSoapPreconditioner, KlSoapV3, ReklsV3


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class KlSoapPreconditionerTest(parameterized.TestCase):
    @parameterized.parameters((8, 16), (16, 8), (12, 12))
    def test_init_state_layout(self, m: int, n: int) -> None:
        state = KlSoapPreconditioner.init_state((m, n), torch.device(FLAGS.device))

        expected_shapes = {
            "exp_avg": (m, n),
            "exp_avg_sq": (m, n),
            "L": (m, m),
            "R": (n, n),
            "Q_L": (m, m),
            "Q_R": (n, n),
            "eigvals_L": (m,),
            "eigvals_R": (n,),
        }
        self.assertCountEqual(state, expected_shapes)
        for key, shape in expected_shapes.items():
            self.assertEqual(state[key].shape, shape, msg=key)
            self.assertEqual(state[key].dtype, torch.float32, msg=key)

        assert_equal(state["Q_L"], torch.eye(m, device=FLAGS.device))
        assert_equal(state["Q_R"], torch.eye(n, device=FLAGS.device))

    def test_init_state_rejects_non_2d(self) -> None:
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            KlSoapPreconditioner.init_state((2, 3, 4), torch.device(FLAGS.device))

    @parameterized.parameters((8, 16), (16, 8), (12, 12))
    def test_rebind_state_binds_current_tensors_back(self, m: int, n: int) -> None:
        state = KlSoapPreconditioner.init_state((m, n), torch.device(FLAGS.device))
        preconditioner = KlSoapPreconditioner(state, 1e-8)
        preconditioner.step(torch.randn(m, n, device=FLAGS.device), 0.95)
        preconditioner.rebind_state(state)

        # step() replaces the eigenbasis and eigenvalue tensors rather than writing into them, so
        # rebind_state is what keeps the optimizer state in sync.
        self.assertIs(state["Q_L"], preconditioner.eigenbasis_pair.L)
        self.assertIs(state["Q_R"], preconditioner.eigenbasis_pair.R)
        self.assertIs(state["eigvals_L"], preconditioner.eigvals_pair.L)
        self.assertIs(state["exp_avg"], preconditioner.exp_avg)

    def test_rebind_state_raise_on_missing_key(self) -> None:
        state = KlSoapPreconditioner.init_state((8, 16), torch.device(FLAGS.device))
        preconditioner = KlSoapPreconditioner(state, 1e-8)

        del state["exp_avg"]
        with self.assertRaisesRegex(KeyError, "missing keys.*exp_avg"):
            preconditioner.rebind_state(state)

    @parameterized.parameters((8, 16), (16, 8), (12, 12))
    def test_update_kronecker_factors_matches_legacy(self, m: int, n: int) -> None:
        state = KlSoapPreconditioner.init_state((m, n), torch.device(FLAGS.device))
        preconditioner = KlSoapPreconditioner(state, 1e-8)
        preconditioner.init_step(torch.randn(m, n, device=FLAGS.device), 0.0)

        reference_factors = [
            preconditioner.kronecker_factor_pair.L.clone(),
            preconditioner.kronecker_factor_pair.R.clone(),
        ]
        grad = torch.randn(m, n, device=FLAGS.device)
        soap.update_kronecker_factors_kl_shampoo(
            reference_factors,
            grad,
            0.95,
            eigenbasis_list=[preconditioner.eigenbasis_pair.L, preconditioner.eigenbasis_pair.R],
            eigvals_list=[preconditioner.eigvals_pair.L, preconditioner.eigvals_pair.R],
            eps=1e-8,
        )
        preconditioner.update_kronecker_factors(grad, 0.95)

        assert_equal(preconditioner.kronecker_factor_pair.L, reference_factors[0])
        assert_equal(preconditioner.kronecker_factor_pair.R, reference_factors[1])


class SoapV3AgainstLegacyTest(parameterized.TestCase):
    @parameterized.parameters(
        {"m": 4, "n": 4, "atol": 1e-5, "rtol": 1e-5},
        {"m": 8, "n": 4, "atol": 1e-4, "rtol": 1e-4},
        {"m": 33, "n": 17, "atol": 1e-3, "rtol": 1e-3},
    )
    def test_3steps_close_to_legacy(self, m: int, n: int, atol: float, rtol: float) -> None:
        raw = torch.randint(-3, 4, (m, n), device=FLAGS.device, dtype=torch.float)

        # Testing aruments are chosen to have best chance of exactly matching reference
        test_kwargs = {
            "lr": 2,
            "betas": (1 / 2, 1 / 4),
            "shampoo_beta": 1 / 4,
            "eps": 1 / 8,
            "weight_decay": 1 / 16,
        }

        ref_param = raw.clone()
        ref_opt = soap.SOAP([ref_param], use_kl_shampoo=True, **test_kwargs)

        test_param = raw.clone()
        test_opt = KlSoapV3([test_param], **test_kwargs)

        for _ in range(3):
            grad = torch.randint_like(raw, -3, 4)
            test_param.grad = grad.clone()
            ref_param.grad = grad.clone()
            ref_opt.step()
            test_opt.step()
            test_param.grad = None
            ref_param.grad = None

            torch.testing.assert_close(test_param, ref_param, atol=atol, rtol=rtol)

            ref_state = ref_opt.state_dict()["state"][0]
            test_state = test_opt.state_dict()["state"][0]
            for key in ref_state.keys():
                torch.testing.assert_close(test_state[key], ref_state[key], atol=atol, rtol=rtol)

    @parameterized.parameters((5, 5), (16, 32), (63, 31), (127, 129))
    def test_tensordot_patched_5steps_matches_legacy(self, m, n):
        """Test aims exactly match legacy with use of tensordot

        Despite different abstraction, the only functional difference between V3 and legacy is use of matmul
        vs. tensordot in projections. Creating a subclass that uses legacy project_in/out to exactly match legacy
        """

        class PatchedConditioner(KlSoapPreconditioner):
            @override
            def project_in(self, x):
                return soap.project_in(x, self.eigenbasis_pair)

            @override
            def project_out(self, x):
                return soap.project_out(x, self.eigenbasis_pair)

        class PatchedKlSoap(KlSoapV3):
            PreconditionerCls = PatchedConditioner

        raw = torch.randn((m, n), device=FLAGS.device, dtype=torch.float)

        test_kwargs = {
            "lr": 2,
            "eps": 1 / 8,
        }

        ref_param = raw.clone()
        ref_opt = soap.SOAP([ref_param], use_kl_shampoo=True, **test_kwargs)

        test_param = raw.clone()
        test_opt = PatchedKlSoap([test_param], **test_kwargs)

        for _ in range(5):
            grad = torch.randn_like(raw)
            test_param.grad = grad.clone()
            ref_param.grad = grad.clone()
            ref_opt.step()
            test_opt.step()
            test_param.grad = None
            ref_param.grad = None

            assert_equal(test_param, ref_param)

            ref_state = ref_opt.state_dict()["state"][0]
            test_state = test_opt.state_dict()["state"][0]
            for key in ref_state.keys():
                assert_equal(test_state[key], ref_state[key])

    @parameterized.parameters((5, 5), (16, 32), (63, 31), (127, 129))
    def test_project_in_out_matches_legacy(self, m: int, n: int) -> None:
        device = torch.device(FLAGS.device)
        state = KlSoapPreconditioner.init_state((m, n), device)
        state["Q_L"] = torch.randint(-3, 4, (m, m), device=device, dtype=torch.float)
        state["Q_R"] = torch.randint(-3, 4, (n, n), device=device, dtype=torch.float)
        preconditioner = KlSoapPreconditioner(state, 1e-8)

        x = torch.randint(-3, 4, (m, n), device=device, dtype=torch.float)

        assert_equal(preconditioner.project_in(x), soap.project_in(x, preconditioner.eigenbasis_pair))
        assert_equal(preconditioner.project_out(x), soap.project_out(x, preconditioner.eigenbasis_pair))


class ReklsV3AgainstLegacyTest(parameterized.TestCase):
    @parameterized.parameters(
        {"m": 8, "n": 4, "atol": 1e-4, "rtol": 1e-4},
        {"m": 17, "n": 33, "atol": 1e-2, "rtol": 1e-2},
    )
    def test_3steps_closes_to_legacy(self, m: int, n: int, atol: float, rtol: float) -> None:
        raw = torch.randint(-3, 4, (m, n), device=FLAGS.device, dtype=torch.float)

        test_kwargs = {
            "lr": 2,
            "betas": (1 / 2, 1 / 4),
            "shampoo_beta": 1 / 4,
            "eps": 1 / 8,
            "weight_decay": 1 / 16,
        }

        ref_param = raw.clone()
        ref_opt = rekls.REKLS([ref_param], **test_kwargs)

        test_param = raw.clone()
        test_opt = ReklsV3([test_param], **test_kwargs)

        for _ in range(3):
            grad = torch.randint_like(raw, -3, 4)
            test_param.grad = grad.clone()
            ref_param.grad = grad.clone()
            ref_opt.step()
            test_opt.step()
            test_param.grad = None
            ref_param.grad = None

            torch.testing.assert_close(test_param, ref_param, atol=atol, rtol=rtol)

            ref_state = ref_opt.state_dict()["state"][0]
            test_state = test_opt.state_dict()["state"][0]
            for key in ref_state.keys():
                torch.testing.assert_close(test_state[key], ref_state[key], atol=atol, rtol=rtol)


class KlMSoapTest(parameterized.TestCase):
    @parameterized.product(shape=[(8, 5), (5, 8), (16, 16)])
    def test_smoke(self, shape) -> None:
        p = torch.nn.Parameter(torch.randn(shape, device=FLAGS.device))
        initial = p.detach().clone()

        opt = KlMSoap([p], lr=1e-2, weight_decay=0.01)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt.step()

        self.assertTrue(torch.isfinite(p).all())
        self.assertFalse(torch.equal(p.detach(), initial))
        self.assertEqual(opt.state[p]["step"], 3)

    def test_rejects_non_2d(self) -> None:
        p = torch.nn.Parameter(torch.randn(2, 3, 4, device=FLAGS.device))
        p.grad = torch.randn_like(p)
        opt = KlMSoap([p], lr=1e-2)
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            opt.step()


class SoapV3MultiStreamTest(parameterized.TestCase):
    """Tests that the v3 optimizers with stream_list produce identical results to without."""

    @classmethod
    def setUpClass(cls):
        if FLAGS.device == "cpu":
            cls.skipTest(cls, "SoapV3MultiStreamTest requires GPU")
        cls.device = FLAGS.device

    @parameterized.parameters(KlSoapV3, ReklsV3, KlMSoap)  # type: ignore[misc]
    def test_8streams_matches_no_streams(self, opt_cls):
        torch.manual_seed(42)
        num_steps = 10
        shapes = [(5, 3), (8, 4), (3, 7), (6, 6), (4, 5), (10, 3), (3, 9), (7, 4), (5, 5), (8, 6)]

        common_kwargs = dict(
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
            shampoo_beta=0.95,
        )

        # Create two sets of identical parameters
        params_no_stream = [
            torch.randn(s, requires_grad=True, device=self.device, dtype=torch.bfloat16) for s in shapes
        ]
        params_with_stream = [p.clone().detach().requires_grad_(True) for p in params_no_stream]

        opt_no_stream = opt_cls(params_no_stream, **common_kwargs)
        stream_list = [torch.cuda.Stream() for _ in range(8)]
        opt_with_stream = opt_cls(params_with_stream, **common_kwargs, stream_list=stream_list)

        grads_per_step = [
            [torch.randn(s, device=self.device, dtype=torch.bfloat16) for s in shapes] for _ in range(num_steps)
        ]

        for step in range(num_steps):
            for p, g in zip(params_no_stream, grads_per_step[step]):
                p.grad = g.clone()
            for p, g in zip(params_with_stream, grads_per_step[step]):
                p.grad = g.clone()

            opt_no_stream.step()
            opt_with_stream.step()
            torch.cuda.synchronize()

            for i, (p_no, p_with) in enumerate(zip(params_no_stream, params_with_stream)):
                assert_equal(
                    p_with,
                    p_no,
                    msg=lambda msg: f"Parameter {i} mismatch at step {step}:\n{msg}",
                )

            for p in params_no_stream + params_with_stream:
                p.grad = None


if __name__ == "__main__":
    absltest.main()
