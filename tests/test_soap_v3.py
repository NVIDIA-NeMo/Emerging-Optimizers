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
import torch
from _comparison import assert_equal
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.legacy_soap import SOAP, soap
from emerging_optimizers.shampoo.soap_v3 import KlSoapPreconditioner, KlSoapV3


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
        with self.assertRaisesRegex(ValueError, "only supported for 2D"):
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
    @parameterized.parameters((4, 4), (8, 4))
    def test_small_input_5steps_matches_legacy(self, m: int, n: int) -> None:
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
        ref_opt = SOAP([ref_param], use_kl_shampoo=True, **test_kwargs)

        test_param = raw.clone()
        test_opt = KlSoapV3([test_param], **test_kwargs)

        for _ in range(5):
            grad = torch.randint_like(raw, -3, 4)
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

    @parameterized.parameters((32, 16), (17, 33))
    def test_medium_input_2steps_closes_to_legacy(self, m: int, n: int) -> None:
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
        ref_opt = SOAP([ref_param], use_kl_shampoo=True, **test_kwargs)

        test_param = raw.clone()
        test_opt = KlSoapV3([test_param], **test_kwargs)

        for _ in range(2):
            grad = torch.randint_like(raw, -3, 4)
            test_param.grad = grad.clone()
            ref_param.grad = grad.clone()
            ref_opt.step()
            test_opt.step()
            test_param.grad = None
            ref_param.grad = None

            # Legacy uses tensordot for projection which can't match matmul exactly
            torch.testing.assert_close(test_param, ref_param, atol=1e-3, rtol=1e-3)

            # States should still match exactly
            ref_state = ref_opt.state_dict()["state"][0]
            test_state = test_opt.state_dict()["state"][0]
            for key in ref_state.keys():
                torch.testing.assert_close(test_state[key], ref_state[key], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    absltest.main()
