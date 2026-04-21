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
from itertools import chain

import soap_reference
import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.soap import REKLS, SOAP, soap
from emerging_optimizers.soap.soap import _clip_update_rms_in_place


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


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
        cls.device = FLAGS.device

    def test_init_kronecker_factors_2d_tensor_shapes(self) -> None:
        """Tests init_kronecker_factors with a 2D tensor."""
        grad = torch.randn(3, 4)
        L, R = soap.init_kronecker_factors(grad.shape)
        self.assertEqual(L.shape, (3, 3))
        self.assertEqual(R.shape, (4, 4))

    def test_update_kronecker_factors(self) -> None:
        shampoo_beta = 0.9
        dim0, dim1 = 3, 10
        grad = torch.randn(dim0, dim1)

        # Initialize factors
        initial_L, initial_R = soap.init_kronecker_factors(grad.shape)
        kronecker_factors = [initial_L.clone(), initial_R.clone()]

        soap.update_kronecker_factors(
            kronecker_factor_list=kronecker_factors,
            grad=grad,
            shampoo_beta=shampoo_beta,
        )

        self.assertEqual(len(kronecker_factors), 2)

        # L = GG^T
        outer_product_L = torch.tensordot(grad, grad, dims=[[1], [1]])
        expected_L = initial_L * shampoo_beta + outer_product_L * (1 - shampoo_beta)
        torch.testing.assert_close(kronecker_factors[0], expected_L, atol=1e-6, rtol=1e-6)

        # R = G^TG
        outer_product_R = torch.tensordot(grad, grad, dims=[[0], [0]])
        expected_R = initial_R * shampoo_beta + outer_product_R * (1 - shampoo_beta)
        torch.testing.assert_close(kronecker_factors[1], expected_R, atol=1e-6, rtol=1e-6)

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
            grad,
            eigenbasis_list=orthonormal_matrix_list,
            dims=[[0], [0]],
        )
        recov = soap.precondition(
            projected,
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
        (1.0,),
        (0.0,),
        (0.5,),
    )
    def test_clip_update_rms(self, max_rms: float) -> None:
        """Test that _clip_update_rms works by clipping the update RMS to max_rms in place."""
        # test for 5 different u values
        u_s = [
            torch.tensor([4.0, -1.0, 1.0, -1.0, 1.0], device=self.device),
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.0], device=self.device),
            torch.tensor([0.8, 0.0, 0.0, 0.0, 0.0], device=self.device),
        ]
        for u in u_s:
            u_clipped = u.clone()
            _clip_update_rms_in_place(u_clipped, max_rms=max_rms)
            if max_rms == 0:
                self.assertTrue(torch.linalg.norm(u_clipped) == torch.linalg.norm(u))
            else:
                self.assertTrue(torch.linalg.norm(u_clipped) / math.sqrt(u.numel()) <= max_rms)

    @parameterized.product(  # type: ignore[misc]
        M=[8, 16, 33],
        N=[4, 8, 33],
        use_eigh=[True, False],
    )
    def test_update_eigenbasis_and_exp_avgs(self, M: int, N: int, use_eigh: bool) -> None:
        """Tests that update_eigenbasis_and_exp_avgs returns valid outputs.

        Verifies output shapes, eigenbasis orthogonality, and that the round-trip
        projection (original → eigenbasis → original → new eigenbasis) preserves the
        norm of exp_avg.
        """
        # Create symmetric positive definite kronecker factors
        g = torch.randn(M, N, device=self.device)
        L = g @ g.T
        R = g.T @ g
        kronecker_factor_list = [L, R]

        # Create orthonormal eigenbasis matrices
        Q_L = torch.linalg.qr(torch.randn(M, M, device=self.device)).Q
        Q_R = torch.linalg.qr(torch.randn(N, N, device=self.device)).Q
        eigenbasis_list = [Q_L, Q_R]

        exp_avg_sq = torch.abs(torch.randn(M, N, device=self.device))
        exp_avg = torch.randn(M, N, device=self.device)
        exp_avg_norm_before = torch.linalg.norm(exp_avg)

        updated_eigenbasis_list, updated_exp_avg, updated_exp_avg_sq = soap.update_eigenbasis_and_exp_avgs(
            kronecker_factor_list=kronecker_factor_list,
            eigenbasis_list=eigenbasis_list,
            exp_avg_sq=exp_avg_sq,
            exp_avg=exp_avg,
            use_eigh=use_eigh,
        )

        # Check output shapes
        self.assertEqual(len(updated_eigenbasis_list), 2)
        self.assertEqual(updated_eigenbasis_list[0].shape, (M, M))
        self.assertEqual(updated_eigenbasis_list[1].shape, (N, N))
        self.assertEqual(updated_exp_avg.shape, (M, N))
        self.assertEqual(updated_exp_avg_sq.shape, (M, N))

        # Check eigenbasis orthogonality
        for Q in updated_eigenbasis_list:
            identity = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
            torch.testing.assert_close(
                Q.T @ Q,
                identity,
                atol=1e-5,
                rtol=1e-5,
                msg="Updated eigenbasis is not orthogonal.",
            )

        # exp_avg is projected via orthogonal transforms, so norm should be preserved
        torch.testing.assert_close(
            torch.linalg.norm(updated_exp_avg),
            exp_avg_norm_before,
            atol=1e-5,
            rtol=1e-5,
            msg="exp_avg norm not preserved after eigenbasis update.",
        )

    @parameterized.parameters(
        (4, 5),
        (3, 3),
        (5, 4),
    )
    def test_kl_shampoo_update(self, m, n):
        rand_exp_fn = partial(torch.randint, low=-4, high=-1, dtype=torch.float32, device=self.device)
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

    def test_init_kronecker_factors_non_2d_raises_type_error(self) -> None:
        """Test that init_kronecker_factors raises TypeError for non-2D shape."""
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            soap.init_kronecker_factors((3,))

    def test_kl_shampoo_correction_non_2d_raises_type_error(self) -> None:
        """Test that update_kronecker_factors_kl_shampoo raises TypeError for non-2D grad."""
        grad = torch.randn(3, device=self.device)
        kronecker_factors = [torch.eye(3, device=self.device)]
        eigenbasis = [torch.eye(3, device=self.device)]
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            soap.update_kronecker_factors_kl_shampoo(
                kronecker_factors, grad=grad, shampoo_beta=0.9, eps=1e-8, eigenbasis_list=eigenbasis
            )

    def test_soap_non_2d_param_raises_type_error(self) -> None:
        """Test that SOAP raises TypeError for non-2D parameter during step."""
        param = torch.randn(10, requires_grad=True, device=self.device)
        optimizer = SOAP([param], lr=0.001)
        param.grad = torch.randn_like(param)
        with self.assertRaisesRegex(TypeError, "only supported for 2D"):
            optimizer.step()

    @parameterized.parameters(  # type: ignore[misc]
        {"shape": (15, 31)},
        {"shape": (31, 15)},
        {"shape": (255, 257)},
    )
    def test_move_states_to_cpu_and_back_preserves_stats(self, shape: tuple) -> None:
        """State must round-trip bit-exactly through ``move_states_to_cpu`` + ``move_states_to_gpu``."""
        if self.device == "cpu":
            self.skipTest("cpu_states_buffer requires pinned memory and a CUDA device")

        m, n = shape
        required_numel = 2 * m * n + 2 * m * m + 2 * n * n
        cpu_buffer = torch.empty(required_numel, dtype=torch.float32, pin_memory=True)

        param = torch.randn(shape, requires_grad=True, device=self.device)
        optimizer = SOAP([param], lr=1e-3, cpu_states_buffer=cpu_buffer)

        # Run a few steps to populate non-trivial state (past the step-0 bootstrap branch).
        for _ in range(3):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

        # Snapshot the current state.
        snapshot = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in optimizer.state[param].items()
        }

        # Round-trip through pinned CPU memory.
        optimizer.move_states_to_cpu()
        optimizer.move_states_to_gpu()

        # Every state entry must match the snapshot exactly.
        for key, expected in snapshot.items():
            actual = optimizer.state[param][key]
            if isinstance(expected, torch.Tensor):
                self.assertEqual(actual.device, expected.device, msg=f"device mismatch for '{key}'")
                self.assertEqual(actual.shape, expected.shape, msg=f"shape mismatch for '{key}'")
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=0,
                    rtol=0,
                    msg=lambda msg, key=key: f"State '{key}' mismatch after CPU round-trip:\n{msg}",
                )
            else:
                self.assertEqual(actual, expected, msg=f"Non-tensor '{key}' mismatch")


class SoapTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = FLAGS.device

    def setUp(self):
        self.default_config = {
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "shampoo_beta": 0.95,
            "fp32_matmul_prec": "highest",
            "power_iter_steps": 1,
        }

    def test_10steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device=self.device)
        optimizer = SOAP(
            [param],
            **self.default_config,
        )

        for _ in range(10):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

    def test_with_kl_shampoo_10steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device=self.device)
        optimizer = SOAP(
            [param],
            **self.default_config,
            use_kl_shampoo=True,
        )

        for _ in range(10):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

    def test_rekls_5steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device=self.device)
        optimizer = REKLS(
            [param],
            lr=self.default_config["lr"],
        )

        for _ in range(5):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

    def test_bfloat16_5steps_smoke(self):
        param = torch.randn(5, 3, requires_grad=True, device=self.device, dtype=torch.bfloat16)
        optimizer = SOAP(
            [param],
            **self.default_config,
        )

        for _ in range(5):
            param.grad = torch.randn_like(param)
            optimizer.step()
            param.grad = None

    def test_serialized_offload_matches_no_offload(self) -> None:
        """SOAP with per-step CPU offload round-trip must match SOAP without offload exactly."""
        if self.device == "cpu":
            self.skipTest("cpu_states_buffer requires pinned memory and a CUDA device")

        shapes = [(5, 3), (10, 10), (15, 31), (31, 15)]
        num_steps = 5

        params_no_offload = [torch.randn(s, requires_grad=True, device=self.device) for s in shapes]
        params_offload = [p.clone().detach().requires_grad_(True) for p in params_no_offload]

        required_numel = sum(2 * m * n + 2 * m * m + 2 * n * n for m, n in shapes)
        cpu_buffer = torch.empty(required_numel, dtype=torch.float32, pin_memory=True)

        opt_no_offload = SOAP(params_no_offload, **self.default_config)
        opt_offload = SOAP(params_offload, **self.default_config, cpu_states_buffer=cpu_buffer)

        for step in range(num_steps):
            grads = [torch.randn_like(p) for p in params_no_offload]
            for p, g in zip(params_no_offload, grads):
                p.grad = g.clone()
            for p, g in zip(params_offload, grads):
                p.grad = g.clone()

            opt_no_offload.step()

            opt_offload.move_states_to_gpu()
            opt_offload.step()
            opt_offload.move_states_to_cpu()

            for i, (p_off, p_no_off) in enumerate(zip(params_offload, params_no_offload)):
                torch.testing.assert_close(
                    p_off,
                    p_no_off,
                    atol=0,
                    rtol=0,
                    msg=lambda msg, step=step, i=i: (f"Param {i} (shape={shapes[i]}) mismatch at step {step}:\n{msg}"),
                )

            for p in chain(params_no_offload, params_offload):
                p.grad = None

    def test_async_offload_matches_no_offload(self) -> None:
        """SOAP with per-step CPU offload round-trip on a dedicated stream must match SOAP without offload exactly."""
        if self.device == "cpu":
            self.skipTest("cpu_states_buffer requires pinned memory and a CUDA device")

        shapes = [(5, 3), (10, 10), (15, 31), (31, 15)]
        num_steps = 5

        params_no_offload = [torch.randn(s, requires_grad=True, device=self.device) for s in shapes]
        params_offload = [p.clone().detach().requires_grad_(True) for p in params_no_offload]

        required_numel = sum(2 * m * n + 2 * m * m + 2 * n * n for m, n in shapes)
        cpu_buffer = torch.empty(required_numel, dtype=torch.float32, pin_memory=True)

        opt_no_offload = SOAP(params_no_offload, **self.default_config)
        opt_offload = SOAP(params_offload, **self.default_config, cpu_states_buffer=cpu_buffer)

        offload_stream = torch.cuda.Stream()

        for step in range(num_steps):
            grads = [torch.randn_like(p) for p in params_no_offload]
            for p, g in zip(params_no_offload, grads):
                p.grad = g.clone()
            for p, g in zip(params_offload, grads):
                p.grad = g.clone()

            opt_no_offload.step()

            # Reload on offload_stream; main stream waits for H2D before running step.
            reload_done = opt_offload.move_states_to_gpu(stream=offload_stream)
            torch.cuda.current_stream().wait_event(reload_done)
            opt_offload.step()
            # Offload stream waits for step() to finish before issuing D2H.
            offload_stream.wait_stream(torch.cuda.current_stream())
            opt_offload.move_states_to_cpu(stream=offload_stream)

            for i, (p_off, p_no_off) in enumerate(zip(params_offload, params_no_offload)):
                torch.testing.assert_close(
                    p_off,
                    p_no_off,
                    atol=0,
                    rtol=0,
                    msg=lambda msg, step=step, i=i: (f"Param {i} (shape={shapes[i]}) mismatch at step {step}:\n{msg}"),
                )

            for p in chain(params_no_offload, params_offload):
                p.grad = None

    @parameterized.parameters(  # type: ignore[misc]
        # Not 1D.
        {
            "shape": (100, 10),
            "dtype": torch.float32,
            "offload_device": "cpu",
            "pinned": True,
            "exc": TypeError,
            "regex": "must be 1D",
        },
        # Wrong dtype.
        {
            "shape": (1000,),
            "dtype": torch.float64,
            "offload_device": "cpu",
            "pinned": True,
            "exc": TypeError,
            "regex": "must be float32",
        },
        # Wrong device.
        {
            "shape": (1000,),
            "dtype": torch.float32,
            "offload_device": "cuda",
            "pinned": False,
            "exc": TypeError,
            "regex": "must be on CPU",
        },
        # Too few elements for the required state.
        {
            "shape": (1,),
            "dtype": torch.float32,
            "offload_device": "cpu",
            "pinned": True,
            "exc": ValueError,
            "regex": "need at least",
        },
        # Valid layout but not pinned — logs an error instead of raising.
        {
            "shape": (1000,),
            "dtype": torch.float32,
            "offload_device": "cpu",
            "pinned": False,
            "exc": None,
            "regex": "not pinned",
        },
    )
    def test_cpu_states_buffer_validation(self, shape, dtype, offload_device, pinned, exc, regex) -> None:
        """``cpu_states_buffer`` must be 1D float32 pinned CPU memory large enough for all state."""
        if offload_device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        param = torch.randn(5, 3, requires_grad=True, device=self.device)
        buffer = torch.empty(shape, dtype=dtype, device=offload_device, pin_memory=pinned)
        if exc is not None:
            with self.assertRaisesRegex(exc, regex):
                SOAP([param], lr=1e-3, cpu_states_buffer=buffer)
        else:
            with self.assertLogs(level="ERROR") as cm:
                SOAP([param], lr=1e-3, cpu_states_buffer=buffer)
            self.assertTrue(
                any(regex in record for record in cm.output),
                msg=f"Expected log containing {regex!r}, got {cm.output}",
            )


class SoapMultiStreamTest(parameterized.TestCase):
    """Tests that SOAP with stream_list produces identical results to without."""

    @classmethod
    def setUpClass(cls):
        if FLAGS.device == "cpu":
            cls.skipTest(cls, "SoapStreamTest requires GPU")
        cls.device = FLAGS.device

    @parameterized.parameters(  # type: ignore[misc]
        {"use_kl_shampoo": False, "use_eigh": False},
        {"use_kl_shampoo": False, "use_eigh": True},
        {"use_kl_shampoo": True, "use_eigh": False},
    )
    def test_8streams_matches_no_streams(self, use_kl_shampoo: bool, use_eigh: bool):
        """Test that SOAP with 8 CUDA streams produces the same results as without."""
        torch.manual_seed(42)
        num_steps = 10
        shapes = [(5, 3), (8, 4), (3, 7), (6, 6), (4, 5), (10, 3), (3, 9), (7, 4), (5, 5), (8, 6)]

        common_kwargs = dict(
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
            shampoo_beta=0.95,
            fp32_matmul_prec="highest",
            use_kl_shampoo=use_kl_shampoo,
            use_eigh=use_eigh,
        )

        # Create two sets of identical parameters
        params_no_stream = [
            torch.randn(s, requires_grad=True, device=self.device, dtype=torch.bfloat16) for s in shapes
        ]
        params_with_stream = [p.clone().detach().requires_grad_(True) for p in params_no_stream]

        opt_no_stream = SOAP(params_no_stream, **common_kwargs)
        stream_list = [torch.cuda.Stream() for _ in range(8)]
        opt_with_stream = SOAP(params_with_stream, **common_kwargs, stream_list=stream_list)

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
                torch.testing.assert_close(
                    p_with,
                    p_no,
                    atol=0,
                    rtol=0,
                    msg=lambda msg: f"Parameter {i} mismatch at step {step}:\n{msg}",
                )

            for p in params_no_stream + params_with_stream:
                p.grad = None


class SoapVsReferenceTest(parameterized.TestCase):
    """Tests that compare SOAP implementation against reference implementation."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(17)
        if FLAGS.device == "cpu":
            cls.skipTest(cls, "SoapVsReferenceTest requires GPU")
        cls.device = FLAGS.device

    @parameterized.product(
        shape=[(3, 3), (5, 3), (10, 10), (15, 31)],
        num_steps=[2, 5, 7],
        correct_bias=[False, True],
    )
    def test_update_matches_reference(self, shape: tuple, num_steps: int, correct_bias: bool):
        """Test that SOAP optimizer matches reference implementation for basic config."""
        # Create two identical parameters
        param_test = torch.randint(-2, 3, shape, dtype=torch.float32, device=self.device)
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
            correct_bias=correct_bias,
        )

        test_optimizer = SOAP(
            [param_test],
            **common_kwargs,
            fp32_matmul_prec="highest",
            qr_fp32_matmul_prec="highest",
            correct_shampoo_beta_bias=False,
        )
        ref_optimizer = soap_reference.ReferenceSoap(
            [param_ref],
            **common_kwargs,
            precondition_frequency=1,
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
        correct_bias=[False, True],
    )
    def test_update_with_cpu_offload_matches_reference(self, shape: tuple, num_steps: int, correct_bias: bool):
        """SOAP with CPU offload round-trip each step should still match the reference."""
        param_test = torch.randint(-2, 3, shape, dtype=torch.float32, device=self.device)
        param_ref = param_test.clone()

        if correct_bias and shape == (15, 31):
            self.skipTest("Skipping large tensor test with correct_bias.")

        common_kwargs = dict(
            lr=2,
            betas=(0.75, 0.75),
            shampoo_beta=0.5,
            eps=1e-15,
            weight_decay=0.125,
            correct_bias=correct_bias,
        )

        m, n = shape
        required_numel = 2 * m * n + 2 * m * m + 2 * n * n
        cpu_states_buffer = torch.empty(required_numel, dtype=torch.float32, pin_memory=True)

        test_optimizer = SOAP(
            [param_test],
            **common_kwargs,
            fp32_matmul_prec="highest",
            qr_fp32_matmul_prec="highest",
            correct_shampoo_beta_bias=False,
            cpu_states_buffer=cpu_states_buffer,
        )
        ref_optimizer = soap_reference.ReferenceSoap(
            [param_ref],
            **common_kwargs,
            precondition_frequency=1,
        )

        for step in range(num_steps):
            grad = torch.randint_like(param_test, -2, 3)
            param_test.grad = grad.clone()
            param_ref.grad = grad.clone()

            ref_optimizer.step()
            test_optimizer.step()
            # Round-trip state through pinned CPU memory each step.
            test_optimizer.move_states_to_cpu()
            test_optimizer.move_states_to_gpu()

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
    )
    def test_eigenbasis_matches_reference(self, shape: tuple, num_steps: int):
        param_soap = torch.randint(-2, 3, shape, dtype=torch.float32, device=self.device)
        param_ref = param_soap.clone()

        # Disable parameter updates, only test kronecker factors and eigenbases
        common_kwargs = dict(
            lr=0,
            betas=(0, 0),
            shampoo_beta=0.75,
            eps=1e-8,
            weight_decay=0,
            correct_bias=False,
        )

        test_optimizer = SOAP(
            [param_soap],
            **common_kwargs,
            weight_decay_method="l2",
            fp32_matmul_prec="highest",
            qr_fp32_matmul_prec="highest",
        )
        ref_optimizer = soap_reference.ReferenceSoap(
            [param_ref],
            **common_kwargs,
            precondition_frequency=1,
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
                [test_state["L"], test_state["R"]],
                ref_state["GG"],
                atol=1e-5,
                rtol=1e-5,
            )

            for eigenbasis_test, eigenbasis_ref in zip([test_state["Q_L"], test_state["Q_R"]], ref_state["Q"]):
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
