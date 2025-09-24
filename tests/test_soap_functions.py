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
from typing import Any
import math
from absl.testing import absltest, parameterized
import torch

from emerging_optimizers.soap.soap import (
    init_kronecker_factors,
    precondition,
    update_kronecker_factors,
    SOAP,
    _get_precondition_frequency,
    _is_eigenbasis_update_step,
    _clip_update_rms_in_place,
)

from emerging_optimizers.utils.precondition_schedules import LinearSchedule


class SoapFunctionsTest(parameterized.TestCase):

    def test_init_preconditioner_multidim_tensor_shapes(self) -> None:
        """Tests init_preconditioner with a multi-dimensional tensor."""
        grad = torch.randn(3, 4, 5)
        state: dict[str, Any] = {}
        # No merge_dims: each dimension gets its own preconditioner unless dimension > max_precond_dim.
        state["GG"] = init_kronecker_factors(grad, precondition_1d=False, max_precond_dim=8192)
        self.assertEqual(len(state["GG"]), grad.dim())
        self.assertEqual(state["GG"][0].shape, (3, 3))
        self.assertEqual(state["GG"][1].shape, (4, 4))
        self.assertEqual(state["GG"][2].shape, (5, 5))

    def test_init_kronecker_factors_max_precond_dim(self) -> None:
        """Tests init_kronecker_factors respects max_precond_dim."""
        max_dim = 8
        grad = torch.randn(3, max_dim + 2, 5)  # Second dimension exceeds max_dim
        kronecker_factors = init_kronecker_factors(grad, precondition_1d=False, max_precond_dim=max_dim)

        self.assertEqual(len(kronecker_factors), grad.dim())
        # Dimension 0 (size 3) <= max_dim
        self.assertEqual(kronecker_factors[0].shape, (3, 3))
        # Dimension 1 (size max_dim + 2) > max_dim -> Should be empty
        self.assertEqual(kronecker_factors[1].shape, (0,))
        self.assertEqual(kronecker_factors[1].numel(), 0)
        # Dimension 2 (size 5) <= max_dim
        self.assertEqual(kronecker_factors[2].shape, (5, 5))

    @parameterized.parameters(
        (1,),
        (2,),
        (3,),
    )
    def test_adam_warmup_steps(self, adam_warmup_steps: int) -> None:
        """Tests that adam_warmup_steps causes state["Q"] to be None until the specified steps are completed."""

        param = torch.randn(5, 3, requires_grad=True)

        optimizer = SOAP(
            [param],
            lr=0.001,
            weight_decay=0.01,
            adam_warmup_steps=adam_warmup_steps,
            precondition_frequency=1,
        )

        for step in range(adam_warmup_steps - 1):
            param.grad = torch.randn_like(param)
            optimizer.step()
            state = optimizer.state[param]

            self.assertNotIn("Q", state, f"Q should not exist at step {step}")

        for step in range(adam_warmup_steps - 1, adam_warmup_steps + 3):
            param.grad = torch.randn_like(param)
            optimizer.step()
            state = optimizer.state[param]

            self.assertIn("Q", state, f"Q should exist at step {step}")
            self.assertIsNotNone(state["Q"], f"Q should not be None at step {step}")
            # Verify Q has the right shape (a list with tensors for each dim)
            self.assertIsInstance(state["Q"], list)
            self.assertEqual(len(state["Q"]), param.dim())
            # Verify Q has the right shape (a list with square eigenvector matrices for each dim)
            self.assertEqual(state["Q"][0].shape, (5, 5))
            self.assertEqual(state["Q"][1].shape, (3, 3))

    def test_update_kronecker_factors(self) -> None:
        """Tests update_kronecker_factors, including max_precond_dim handling."""
        max_dim = 8
        shampoo_beta = 0.9
        dim0, dim1, dim2 = 3, max_dim + 2, 5
        grad = torch.randn(dim0, dim1, dim2)

        # Initialize factors
        initial_factors = init_kronecker_factors(grad, precondition_1d=False, max_precond_dim=max_dim)

        kronecker_factors = [f.clone() for f in initial_factors]

        update_kronecker_factors(
            kronecker_factor_list=kronecker_factors,
            grad=grad,
            shampoo_beta=shampoo_beta,
            precondition_1d=False,
            max_precond_dim=max_dim,
        )

        self.assertEqual(len(kronecker_factors), grad.dim())

        # Dimension 0 (size 3) <= max_dim: Should be updated
        contract_dims_0 = [1, 2]
        outer_product_0 = torch.tensordot(grad, grad, dims=[contract_dims_0] * 2)
        expected_factor_0 = initial_factors[0] * shampoo_beta + outer_product_0 * (1 - shampoo_beta)
        torch.testing.assert_close(kronecker_factors[0], expected_factor_0, atol=1e-6, rtol=1e-6)

        # Dimension 1 (size 10) > max_dim: Should NOT be updated (still empty)
        self.assertEqual(kronecker_factors[1].shape, (0,))
        self.assertEqual(kronecker_factors[1].numel(), 0)

        # Check it's the same object or has same properties as initial empty tensor
        self.assertTrue(torch.equal(kronecker_factors[1], initial_factors[1]))

        # Dimension 2 (size 5) <= max_dim: Should be updated
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
        l = torch.randn(m, m)
        Q_L = torch.linalg.qr(l + l.T).Q
        r = torch.randn(n, n)
        Q_R = torch.linalg.qr(r + r.T).Q

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
        """Tests that projecting a tensor to eigenbasis of QL and QR and then projecting it back results in the original tensor.

        The projected tensor should approximately recover the original tensor.
        """
        torch.manual_seed(0)
        # Create a random tensor to project in and out of eigenbasis
        grad = torch.randn(M, N)
        # Create a state with 2 orthonormal matrix.
        Q_L = torch.linalg.qr(torch.randn(M, M))[0]
        Q_R = torch.linalg.qr(torch.randn(N, N))[0]
        orthonormal_matrix_list = [Q_L, Q_R]

        projected = precondition(
            grad=grad,
            eigenbasis_list=orthonormal_matrix_list,
            dims=[[0], [0]],
        )
        recov = precondition(
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

    def test_get_precondition_frequency_fixed(self) -> None:
        """Test that _get_precondition_frequency works with fixed frequency (default case)."""
        freq = _get_precondition_frequency(10, 100)
        self.assertEqual(freq, 10)

    @parameterized.parameters(
        (5, 10, 20, 10, False),
        (15, 10, 20, 10, True),
        (20, 10, 15, 10, True),
        (21, 10, 15, 10, False),
        (30, 10, 15, 10, True),
        (31, 10, 15, 10, False),
    )
    def test_is_eigenbasis_update_step_fixed_frequency(
        self, step: int, adam_warmup_steps: int, precondition_warmup: int, precondition_frequency: int, expected: bool
    ) -> None:
        """Test _is_eigenbasis_update_step with fixed frequency."""
        result = _is_eigenbasis_update_step(step, adam_warmup_steps, precondition_warmup, precondition_frequency)
        self.assertEqual(result, expected)

    def test_soap_optimizer_fixed_frequency(self) -> None:
        """Test that SOAP optimizer can be created with fixed precondition frequency (default case)."""
        param = torch.randn(10, 5, requires_grad=True)
        optimizer = SOAP([param], lr=1e-3, precondition_frequency=10)
        self.assertEqual(optimizer.param_groups[0]["precondition_frequency"], 10)

    def test_soap_optimizer_class_based_schedule(self) -> None:
        """Test that SOAP optimizer can be created with class-based precondition frequency schedule."""
        param = torch.randn(10, 5, requires_grad=True)
        schedule = LinearSchedule(min_freq=2, max_freq=10, transition_steps=100)
        optimizer = SOAP([param], lr=1e-3, precondition_frequency=schedule)
        self.assertTrue((optimizer.param_groups[0]["precondition_frequency"]) == schedule)

        self.assertEqual(schedule(0), 2)
        self.assertEqual(schedule(50), 6)
        self.assertEqual(schedule(100), 10)

        adam_warmup = 1
        precondition_warmup = 0

        self.assertTrue(_is_eigenbasis_update_step(10, adam_warmup, precondition_warmup, schedule))
        self.assertFalse(_is_eigenbasis_update_step(11, adam_warmup, precondition_warmup, schedule))
        self.assertTrue(_is_eigenbasis_update_step(60, adam_warmup, precondition_warmup, schedule))
        self.assertFalse(_is_eigenbasis_update_step(61, adam_warmup, precondition_warmup, schedule))
        self.assertTrue(_is_eigenbasis_update_step(120, adam_warmup, precondition_warmup, schedule))
        self.assertFalse(_is_eigenbasis_update_step(121, adam_warmup, precondition_warmup, schedule))

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


if __name__ == "__main__":
    absltest.main()
