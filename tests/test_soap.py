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
import math
from functools import partial
from typing import Any

import soap_reference
import torch
from absl.testing import absltest, parameterized

from emerging_optimizers import soap
from emerging_optimizers.soap import rekls, soap
from emerging_optimizers.soap.soap import (
    _clip_update_rms_in_place,
    _is_eigenbasis_update_step,
)
from emerging_optimizers.utils.precondition_schedules import LinearSchedule


def kl_shampoo_update_ref(
    kronecker_factor_list: list[torch.Tensor],
    grad: torch.Tensor,
    eigenbasis_list: list[torch.Tensor],
    shampoo_beta: float,
    eps: float,
    eigval_exp: float = -1.0,
) -> None:
    """Reference implementation of KL-Shampoo update.

    Using same functionality implemented by different people as testing reference. The chance of two
    independent implementations having the same bug is very low.

    """
    if grad.dim() != 2:
        raise ValueError("KL-Shampoo mathematical correction is only supported for 2D tensors")
    # scale the gradient matrix by the approximate eigenvalues and the eigenbasis
    # G@Q_R@λ_R^(−1)@Q_R.T@G.T/dim(GG.T) and G.T@Q_L@λ_L^(−1)@Q_L.T@G/dim(G.TG)
    scale_factors = [
        1
        / grad.shape[idx]
        * (torch.diag(eigenbasis_list[idx].T @ kronecker_factor_list[idx] @ eigenbasis_list[idx]) + eps) ** eigval_exp
        for idx in range(len(kronecker_factor_list))
    ]
    kronecker_product_corrections = [
        (eigenbasis_list[idx] * scale_factors[idx][None, :]) @ eigenbasis_list[idx].T
        for idx in range(len(kronecker_factor_list))
    ]
    kronecker_product_updates = [
        grad @ kronecker_product_corrections[1] @ grad.T,
        grad.T @ kronecker_product_corrections[0] @ grad,
    ]
    for idx in range(len(kronecker_factor_list)):
        kronecker_factor_list[idx].lerp_(kronecker_product_updates[idx], 1 - shampoo_beta)


class SoapFunctionsTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(13)

    def test_init_preconditioner_multidim_tensor_shapes(self) -> None:
        """Tests init_preconditioner with a multi-dimensional tensor."""
        grad = torch.randn(3, 4, 5)
        state: dict[str, Any] = {}
        state["GG"] = soap.init_kronecker_factors(grad, precondition_1d=False)
        self.assertEqual(len(state["GG"]), grad.dim())
        self.assertEqual(state["GG"][0].shape, (3, 3))
        self.assertEqual(state["GG"][1].shape, (4, 4))
        self.assertEqual(state["GG"][2].shape, (5, 5))

    @parameterized.parameters(
        (1,),
        (2,),
        (3,),
    )
    def test_adam_warmup_steps(self, adam_warmup_steps: int) -> None:
        """Tests that adam_warmup_steps causes state["Q"] to be None until the specified steps are completed."""

        param = torch.randn(5, 3, requires_grad=True, device="cuda")

        optimizer = soap.SOAP(
            [param],
            lr=0.001,
            weight_decay=0.01,
            adam_warmup_steps=adam_warmup_steps,
            precondition_frequency=1,
        )

        for step in range(adam_warmup_steps):
            param.grad = torch.randn_like(param)
            optimizer.step()
            state = optimizer.state[param]

            self.assertNotIn("Q", state)

        for step in range(adam_warmup_steps, adam_warmup_steps + 3):
            param.grad = torch.randn_like(param)
            optimizer.step()
            state = optimizer.state[param]

            # Verify Q has the right shape (a list with tensors for each dim)
            self.assertIsInstance(state["Q"], list)
            self.assertEqual(len(state["Q"]), param.dim())
            # Verify Q has the right shape (a list with square eigenvector matrices for each dim)
            self.assertEqual(state["Q"][0].shape, (5, 5))
            self.assertEqual(state["Q"][1].shape, (3, 3))

    def test_update_kronecker_factors(self) -> None:
        max_dim = 8
        shampoo_beta = 0.9
        dim0, dim1, dim2 = 3, max_dim + 2, 5
        grad = torch.randn(dim0, dim1, dim2)

        # Initialize factors
        initial_factors = soap.init_kronecker_factors(grad, precondition_1d=False)

        kronecker_factors = [f.clone() for f in initial_factors]

        soap.update_kronecker_factors(
            kronecker_factor_list=kronecker_factors,
            grad=grad,
            shampoo_beta=shampoo_beta,
            precondition_1d=False,
        )

        self.assertEqual(len(kronecker_factors), grad.dim())

        contract_dims_0 = [1, 2]
        outer_product_0 = torch.tensordot(grad, grad, dims=[contract_dims_0] * 2)
        expected_factor_0 = initial_factors[0] * shampoo_beta + outer_product_0 * (1 - shampoo_beta)
        torch.testing.assert_close(kronecker_factors[0], expected_factor_0, atol=1e-6, rtol=1e-6)

        contract_dims_2 = [0, 1]
        outer_product_2 = torch.tensordot(grad, grad, dims=[contract_dims_2] * 2)
        expected_factor_2 = initial_factors[2] * shampoo_beta + outer_product_2 * (1 - shampoo_beta)
        torch.testing.assert_close(kronecker_factors[2], expected_factor_2, atol=1e-6, rtol=1e-6)

    @parameterized.parameters(
        (4, 5),
        (3, 3),
        (5, 4),
    )
    def test_tensordot_vs_matmul(self, m, n):
        # Create tensors with random eigenvectors for rotation matrices QL and QR
        grad = torch.randn(m, n)
        left_matrix = torch.randn(m, m)
        Q_L = torch.linalg.qr(left_matrix + left_matrix.T).Q
        right_matrix = torch.randn(n, n)
        Q_R = torch.linalg.qr(right_matrix + right_matrix.T).Q

        # Test that project operation to eigenbasis is correct
        # Calculate using sequential tensordot as used by the code
        grad_intermediate = torch.tensordot(grad, Q_L, dims=([0], [0]))
        # Check that grad_intermediate is transposed
        self.assertTrue(grad_intermediate.dim() == grad.transpose(0, 1).dim())
        grad_td = torch.tensordot(grad_intermediate, Q_R, dims=([0], [0]))
        # Calculate using pure sequential matmul
        grad_pt = Q_L.transpose(0, 1).matmul(grad).matmul(Q_R)
        self.assertTrue(torch.allclose(grad_td, grad_pt, atol=1e-6))

        # Test that project_back operation out of eigenbasis is correct
        # Calculate using sequential tensordot as used by the code
        grad_intermediate = torch.tensordot(grad, Q_L, dims=([0], [1]))
        # Check that grad_intermediate is transposed
        self.assertTrue(grad_intermediate.dim() == grad.transpose(0, 1).dim())
        grad_td = torch.tensordot(grad_intermediate, Q_R, dims=([0], [1]))
        # Calculate using pure sequential matmul
        grad_pt = Q_L.matmul(grad).matmul(Q_R.transpose(0, 1))
        self.assertTrue(torch.allclose(grad_td, grad_pt, atol=1e-6))

    @parameterized.parameters(  # type: ignore[misc]
        {"N": 4, "M": 8},
        {"N": 16, "M": 8},
        {"N": 32, "M": 8},
    )
    def test_project_and_project_back(self, N: int, M: int) -> None:
        """Tests that projecting a tensor to eigenbasis of QL and QR and back

        The projected tensor should approximately recover the original tensor.
        """
        torch.manual_seed(0)
        # Create a random tensor to project in and out of eigenbasis
        grad = torch.randn(M, N)
        # Create a state with 2 orthonormal matrix.
        Q_L = torch.linalg.qr(torch.randn(M, M))[0]
        Q_R = torch.linalg.qr(torch.randn(N, N))[0]
        orthonormal_matrix_list = [Q_L, Q_R]

        projected = soap.precondition(
            grad=grad,
            eigenbasis_list=orthonormal_matrix_list,
            dims=[[0], [0]],
        )
        recov = soap.precondition(
            grad=projected,
            eigenbasis_list=orthonormal_matrix_list,
            dims=[[0], [1]],
        )
        # Check that the recovered tensor is close to the original.
        torch.testing.assert_close(
            grad,
            recov,
            atol=1e-6,
            rtol=1e-6,
            msg="Project and project_back did not recover the original tensor.",
        )

    @parameterized.parameters(
        (5, 10, 10, False),
        (15, 10, 5, True),
        (20, 10, 10, True),
        (21, 10, 10, False),
        (30, 10, 10, True),
        (31, 10, 10, False),
    )
    def test_is_eigenbasis_update_step_fixed_frequency(
        self, step: int, adam_warmup_steps: int, precondition_frequency: int, expected: bool
    ) -> None:
        """Test _is_eigenbasis_update_step with fixed frequency."""
        result = _is_eigenbasis_update_step(step, adam_warmup_steps, precondition_frequency)
        self.assertEqual(result, expected)

    def test_soap_optimizer_fixed_frequency(self) -> None:
        """Test that SOAP optimizer can be created with fixed precondition frequency (default case)."""
        param = torch.randn(10, 5, requires_grad=True)
        optimizer = soap.SOAP([param], lr=1e-3, precondition_frequency=10)
        self.assertEqual(optimizer.precondition_frequency, 10)

    def test_soap_optimizer_class_based_schedule(self) -> None:
        """Test that SOAP optimizer can be created with class-based precondition frequency schedule."""
        param = torch.randn(10, 5, requires_grad=True)
        schedule = LinearSchedule(min_freq=2, max_freq=10, transition_steps=100)
        optimizer = soap.SOAP([param], lr=1e-3, precondition_frequency=schedule)
        self.assertTrue(optimizer.precondition_frequency == schedule)

        self.assertEqual(schedule(0), 2)
        self.assertEqual(schedule(50), 6)
        self.assertEqual(schedule(100), 10)

        adam_warmup = 1

        self.assertTrue(_is_eigenbasis_update_step(10, adam_warmup, schedule))
        self.assertFalse(_is_eigenbasis_update_step(11, adam_warmup, schedule))
        self.assertTrue(_is_eigenbasis_update_step(60, adam_warmup, schedule))
        self.assertFalse(_is_eigenbasis_update_step(61, adam_warmup, schedule))
        self.assertTrue(_is_eigenbasis_update_step(120, adam_warmup, schedule))
        self.assertFalse(_is_eigenbasis_update_step(121, adam_warmup, schedule))

    @parameterized.parameters(
        (1.0,),
        (0.0,),
        (0.5,),
    )
    def test_clip_update_rms(self, max_rms: float) -> None:
        """Test that _clip_update_rms works by clipping the update RMS to max_rms in place."""
        # test for 5 different u values
        u_s = [
            torch.tensor([4.0, -1.0, 1.0, -1.0, 1.0], device="cuda"),
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.0], device="cuda"),
            torch.tensor([0.8, 0.0, 0.0, 0.0, 0.0], device="cuda"),
        ]
        for u in u_s:
            u_clipped = u.clone()
            _clip_update_rms_in_place(u_clipped, max_rms=max_rms)
            if max_rms == 0:
                self.assertTrue(torch.linalg.norm(u_clipped) == torch.linalg.norm(u))
            else:
                self.assertTrue(torch.linalg.norm(u_clipped) / math.sqrt(u.numel()) <= max_rms)

    @parameterized.parameters(
        (4, 5),
        (3, 3),
        (5, 4),
    )
    def test_kl_shampoo_update(self, m, n):
        rand_exp_fn = partial(torch.randint, low=-4, high=-1, dtype=torch.float32, device="cuda")
        kronecker_factor_list = [
            2 ** rand_exp_fn(size=(m, m)),
            2 ** rand_exp_fn(size=(n, n)),
        ]
        kronecker_factor_list_ref = [f.clone() for f in kronecker_factor_list]

        test_grad = 2 ** rand_exp_fn(size=(m, n))
        eigenbasis_list = [2 ** rand_exp_fn(size=(m, m)), 2 ** rand_exp_fn(size=(n, n))]
        kwargs = dict(
            grad=test_grad,
            shampoo_beta=0.5,
            eps=1e-8,
            eigval_exp=-1.0,
            eigenbasis_list=eigenbasis_list,
        )
        kl_shampoo_update_ref(kronecker_factor_list_ref, **kwargs)
        soap.update_kronecker_factors_kl_shampoo(kronecker_factor_list, **kwargs)

        torch.testing.assert_close(kronecker_factor_list[0], kronecker_factor_list_ref[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(kronecker_factor_list[1], kronecker_factor_list_ref[1], atol=1e-6, rtol=1e-6)


class SoapTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(15)

    def setUp(self):
        self.default_config = {
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "precondition_frequency": 1,
            "shampoo_beta": 0.95,
            "precondition_1d": False,
            "adam_warmup_steps": 1,
            "fp32_matmul_prec": "highest",
            "use_adaptive_criteria": False,
            "power_iter_steps": 1,
        }

    def test_10steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device="cuda")
        optimizer = soap.SOAP(
            [param],
            **self.default_config,
        )

        for _ in range(10):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

    def test_with_kl_shampoo_10steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device="cuda")
        optimizer = soap.SOAP(
            [param],
            **self.default_config,
            use_kl_shampoo=True,
        )

        for _ in range(10):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

    def test_rekls_5steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device="cuda")
        optimizer = rekls.REKLS(
            [param],
            lr=self.default_config["lr"],
        )

        for _ in range(5):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None


class SoapVsReferenceTest(parameterized.TestCase):
    """Tests that compare SOAP implementation against reference implementation."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(17)

    @parameterized.product(
        shape=[(3, 3), (5, 3), (10, 10), (15, 31)],
        num_steps=[2, 5, 7],
        precondition_frequency=[1, 2, 5],
        correct_bias=[False, True],
    )
    def test_update_matches_reference(
        self, shape: tuple, num_steps: int, precondition_frequency: int, correct_bias: bool
    ):
        """Test that SOAP optimizer matches reference implementation for basic config."""
        # Create two identical parameters
        param_test = torch.randint(-2, 3, shape, dtype=torch.float32, device="cuda")
        param_ref = param_test.clone()

        # NOTE: eps is smaller than usual because reference implementation of Soap applies eps differently than
        # torch.optim.AdamW when correct_bias is True.
        if correct_bias and shape == (15, 31):
            self.skipTest("Skipping large tensor test with correct_bias.")
        common_kwargs = dict(
            lr=2,
            betas=(0.75, 0.75),
            shampoo_beta=0.5,
            eps=1e-15,
            weight_decay=0.125,
            precondition_frequency=precondition_frequency,
            correct_bias=correct_bias,
        )

        test_optimizer = soap.SOAP(
            [param_test],
            **common_kwargs,
            adam_warmup_steps=0,
            fp32_matmul_prec="highest",
            qr_fp32_matmul_prec="highest",
            correct_shampoo_beta_bias=False,
        )
        ref_optimizer = soap_reference.ReferenceSoap(
            [param_ref],
            **common_kwargs,
        )
        # Run optimization steps with identical gradients
        for step in range(num_steps):
            grad = torch.randint_like(param_test, -2, 3)

            # Apply same gradient to both
            param_test.grad = grad.clone()
            param_ref.grad = grad.clone()

            # Step both optimizers
            test_optimizer.step()
            ref_optimizer.step()

            torch.testing.assert_close(
                param_test,
                param_ref,
                atol=1e-4,
                rtol=1e-4,
                msg=lambda msg: f"Parameter mismatch at step {step}:\n{msg}",
            )

            param_test.grad = None
            param_ref.grad = None

    @parameterized.product(
        shape=[(3, 3), (5, 3), (10, 10), (15, 31)],
        num_steps=[2, 5, 7],
        precondition_frequency=[1, 2, 5],
    )
    def test_eigenbasis_matches_reference(self, shape: tuple, num_steps: int, precondition_frequency: int):
        param_soap = torch.randint(-2, 3, shape, dtype=torch.float32, device="cuda")
        param_ref = param_soap.clone()

        # Disable parameter updates, only test kronecker factors and eigenbases
        common_kwargs = dict(
            lr=0,
            betas=(0, 0),
            shampoo_beta=0.75,
            eps=1e-8,
            weight_decay=0,
            precondition_frequency=precondition_frequency,
            correct_bias=False,
        )

        test_optimizer = soap.SOAP(
            [param_soap],
            **common_kwargs,
            weight_decay_method="l2",
            adam_warmup_steps=0,
            fp32_matmul_prec="highest",
            qr_fp32_matmul_prec="highest",
        )
        ref_optimizer = soap_reference.ReferenceSoap(
            [param_ref],
            **common_kwargs,
        )

        for step in range(num_steps):
            grad = torch.randint_like(param_soap, -2, 3)
            param_soap.grad = grad.clone()
            param_ref.grad = grad.clone()

            test_optimizer.step()
            ref_optimizer.step()

            param_soap.grad = None
            param_ref.grad = None

            test_state = test_optimizer.state[param_soap]
            ref_state = ref_optimizer.state[param_ref]

            torch.testing.assert_close(
                test_state["GG"],
                ref_state["GG"],
                atol=1e-5,
                rtol=1e-5,
            )

            for eigenbasis_test, eigenbasis_ref in zip(test_state["Q"], ref_state["Q"]):
                torch.testing.assert_close(
                    eigenbasis_test,
                    eigenbasis_ref,
                    atol=1e-4,
                    rtol=1e-4,
                    msg=lambda msg: f"Eigenbasis mismatch at step {step}:\n{msg}",
                )

            # Compare step counters
            self.assertEqual(test_state["step"], ref_state["step"])


if __name__ == "__main__":
    absltest.main()
