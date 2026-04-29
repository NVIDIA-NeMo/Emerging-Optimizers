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
from copy import deepcopy

import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers import utils
from emerging_optimizers.orthogonalized_optimizers import muon, muon_utils


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


_SM_VERSION = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)


def newton_schulz_ref(x: torch.Tensor, coefficient_sets: list[tuple[float, float, float]]) -> torch.Tensor:
    """Reference Newton-Schulz iteration to compute the zeroth power / orthogonalization of x."""
    # Muon is not for 1d parameters
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions since Muon is not for 1d parameters.")

    steps = len(coefficient_sets)

    # transpose tensor to perform whitening on the smaller dimension
    needs_transpose = x.size(-2) > x.size(-1)
    if needs_transpose:
        x = x.mT

    # Ensure spectral norm is at most 1
    X = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min_(1e-7)

    # Perform the NS iterations
    for i in range(steps):
        with utils.fp32_matmul_precision("highest"):
            a, b, c = coefficient_sets[i]
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

    # undo transpose if necessary
    if needs_transpose:
        X = X.mT
    return X


class TestNewtonSchulz(parameterized.TestCase):
    def setUp(self):
        self.prev_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.device = FLAGS.device

    def tearDown(self):
        torch.set_float32_matmul_precision(self.prev_precision)

    @parameterized.parameters(
        (512, 512),
        (512, 256),
        (256, 512),
    )
    def test_newtonschulz5_svd_close(self, dim1, dim2):
        shape = (dim1, dim2)
        x = torch.randn(*shape, device=self.device, dtype=torch.float32)
        out_zeropowerns = muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic")
        U, _, V = torch.linalg.svd(x, full_matrices=False)
        out_zeropower_svd = (U @ V).float()
        # Check that the outputs are close.
        # Note: Due to the nature of the approximation and different computation paths and the bfloat16 conversion
        # will lead to differences and need a higher tolerance.
        # This is expected behavior, see https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        torch.testing.assert_close(
            out_zeropowerns.float(),
            out_zeropower_svd.float(),
            atol=1e-1,
            rtol=1e-7,
        )

    @parameterized.parameters(
        (512, 512),
        (512, 256),
        (256, 512),
    )
    def test_newtonschulz5_close_to_reference(self, dim1, dim2):
        x = torch.randn(dim1, dim2, device=self.device, dtype=torch.float32)
        out_zeropower_test = muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic")
        out_zeropowerns_ref = newton_schulz_ref(
            x,
            coefficient_sets=muon_utils._COEFFICIENT_SETS["quintic"],
        )

        torch.testing.assert_close(
            out_zeropower_test,
            out_zeropowerns_ref,
            atol=1e-6,
            rtol=1e-7,
        )

    @parameterized.parameters(
        (2, 256, 256),
        (4, 128, 256),
        (3, 256, 128),
    )
    def test_newtonschulz_3d_input_closes_to_per_slice(self, batch, dim1, dim2):
        x = torch.randint(-3, 4, (batch, dim1, dim2), device=self.device, dtype=torch.float32)
        out_3d = muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic")
        out_per_slice = torch.stack(
            [muon_utils.newton_schulz(x[i], steps=5, coefficient_type="quintic") for i in range(batch)]
        )
        torch.testing.assert_close(out_3d, out_per_slice, atol=1e-6, rtol=0)

    @parameterized.parameters(
        (511, 513),
        (511, 257),
        (257, 513),
    )
    def test_newtonschulz_custom_coeff_close_to_reference(self, dim1, dim2):
        x = 2 ** torch.randint(-10, -5, (dim1, dim2), device=self.device, dtype=torch.float32)

        test_coefficient_sets = [
            (3, 5, 7),
            (11, 13, 17),
        ]
        out_zeropower_test = muon_utils.newton_schulz(
            x,
            steps=2,
            coefficient_type="custom",
            custom_coefficient_sets=test_coefficient_sets,
        )
        out_zeropowerns_ref = newton_schulz_ref(
            x,
            coefficient_sets=test_coefficient_sets,
        )

        torch.testing.assert_close(
            out_zeropower_test,
            out_zeropowerns_ref,
            atol=0,
            rtol=1e-6,
        )

    @parameterized.product(
        size=[(512, 512), (512, 256), (256, 512)],
        coefficient_type=["polar_express", "deepseekv4"],
    )
    def test_polar_express_and_deepseekv4_10steps_better_than_quintic(self, size, coefficient_type):
        dim1, dim2 = size
        # Create a matrix with terrible condition number
        min_dim = min(dim1, dim2)

        # Generate proper random orthogonal matrices for SVD structure
        random_left = torch.randn(dim1, min_dim, device=self.device, dtype=torch.float32)
        random_right = torch.randn(dim2, min_dim, device=self.device, dtype=torch.float32)
        # orthogonalize the random matrices using QR decomposition
        u, _ = torch.linalg.qr(random_left)
        v, _ = torch.linalg.qr(random_right)

        # Create singular values with terrible condition number (range from 1e6 to 1e-6)
        singular_values = torch.logspace(6, -6, min_dim, device=self.device, dtype=torch.float32)

        # Construct the matrix with terrible condition number using proper SVD: U @ diag(S) @ V^T
        # condition number = 1e12
        x = u @ torch.diag(singular_values) @ v.T

        # Compare polar express vs quintic Newton-Schulz methods
        out_svd = (u @ v.T).float()
        out_polar_express = muon_utils.newton_schulz(x, steps=10, coefficient_type=coefficient_type)
        out_quintic = muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic")

        l2_norm_diff_polar = torch.norm(out_polar_express.float() - out_svd.float(), p=2)
        l2_norm_diff_quintic = torch.norm(out_quintic.float() - out_svd.float(), p=2)

        logging.info(f"{coefficient_type} norm difference: {l2_norm_diff_polar:.6f}")
        logging.info(f"Quintic norm difference: {l2_norm_diff_quintic:.6f}")

        self.assertLess(
            l2_norm_diff_polar,
            l2_norm_diff_quintic,
            f"{coefficient_type} norm is larger than Quintic norm: {l2_norm_diff_polar:.6f} > {l2_norm_diff_quintic:.6f}",
        )

    @parameterized.parameters(
        (511, 513),
        (511, 257),
        (257, 513),
    )
    def test_polar_express_9steps_close_to_reference(self, dim1, dim2):
        x = torch.randn(dim1, dim2, device=self.device, dtype=torch.float32)
        out_pe9 = muon_utils.newton_schulz(x, steps=9, coefficient_type="polar_express")

        coeff = deepcopy(muon_utils._COEFFICIENT_SETS["polar_express"])
        coeff.append(coeff[-1])
        out_ref = newton_schulz_ref(x, coefficient_sets=coeff)
        torch.testing.assert_close(out_pe9, out_ref, atol=2e-6, rtol=1e-7)

    @parameterized.parameters(
        (511, 513),
        (511, 257),
        (257, 513),
    )
    def test_deepseekv4_close_to_reference(self, dim1, dim2):
        x = torch.randn(dim1, dim2, device=self.device, dtype=torch.float32)
        out_dsv4 = muon_utils.newton_schulz(x, steps=10, coefficient_type="deepseekv4")

        coeff = deepcopy(muon_utils._COEFFICIENT_SETS["deepseekv4"])
        out_ref = newton_schulz_ref(x, coefficient_sets=coeff)
        torch.testing.assert_close(out_dsv4, out_ref, atol=2e-6, rtol=1e-7)

    @parameterized.parameters(
        (512, 512),
        (512, 256),
        (256, 512),
    )
    def test_cans_close_to_reference(self, dim1, dim2):
        x = torch.randn(dim1, dim2, device=self.device, dtype=torch.float32)
        out_cans_test = muon_utils.newton_schulz(x, steps=5, coefficient_type="cans")
        out_cans_ref = newton_schulz_ref(x, coefficient_sets=muon_utils._COEFFICIENT_SETS["cans"])

        torch.testing.assert_close(
            out_cans_test,
            out_cans_ref,
            atol=1e-5,
            rtol=1e-7,
        )

    @parameterized.parameters(
        (511, 513),
        (511, 257),
        (257, 513),
    )
    def test_cans_9steps_close_to_reference(self, dim1, dim2):
        x = torch.randn(dim1, dim2, device=self.device, dtype=torch.float32)
        out_cans9 = muon_utils.newton_schulz(x, steps=9, coefficient_type="cans")
        coeff = deepcopy(muon_utils._COEFFICIENT_SETS["cans"])
        # CANS uses repeat_last, so repeat the last tuple for remaining steps.
        coeff.extend([coeff[-1]] * 4)
        out_ref = newton_schulz_ref(x, coefficient_sets=coeff)
        torch.testing.assert_close(out_cans9, out_ref, atol=2e-6, rtol=1e-7)

    @parameterized.parameters(
        ((10,),),
        ((2, 3, 4, 5),),
    )
    def test_newton_schulz_wrong_input_shape_raises_type_error(self, shape) -> None:
        """Test that newton_schulz raises TypeError for non-2D/3D input."""
        x = torch.randn(*shape, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "must be 2d or 3d"):
            muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic")

    def test_newton_schulz_non_fp32_raises_type_error(self) -> None:
        """Test that newton_schulz raises TypeError for non-float32 input."""
        x = torch.randn(5, 7, device=self.device, dtype=torch.float64)
        with self.assertRaisesRegex(TypeError, "float32.*float64"):
            muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic")

    def test_newton_schulz_custom_without_coefficients_raises_value_error(self) -> None:
        """Test that newton_schulz raises ValueError for custom type without coefficient_sets."""
        x = torch.randn(5, 7, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "custom_coefficient_sets must be provided"):
            muon_utils.newton_schulz(x, steps=5, coefficient_type="custom")

    def test_newton_schulz_invalid_coefficient_type_raises_value_error(self) -> None:
        """Test that newton_schulz raises ValueError for invalid coefficient_type."""
        x = torch.randn(5, 7, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "Invalid coefficient type.*nonexistent"):
            muon_utils.newton_schulz(x, steps=5, coefficient_type="nonexistent")

    def test_newton_schulz_use_syrk_with_3d_raises_type_error(self) -> None:
        """Test that newton_schulz raises TypeError for 3D input with use_syrk=True."""
        x = torch.randn(2, 4, 8, device=self.device, dtype=torch.float32)
        with utils.fp32_matmul_precision("medium"), self.assertRaisesRegex(TypeError, "use_syrk does not support"):
            muon_utils.newton_schulz(x, steps=5, coefficient_type="quintic", use_syrk=True)


class TestMuonUtils(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.product(
        size_pairs=[(512, 512), (512, 256), (256, 512), (97, 37), (37, 97)],
        mode=["shape_scaling", "spectral", "unit_rms_norm"],
    )
    def test_get_scale_factor(self, size_pairs, mode):
        size_out, size_in = size_pairs
        scale = muon.get_muon_scale_factor(size_out, size_in, mode)
        if mode == "shape_scaling":
            self.assertEqual(scale, math.sqrt(max(1, size_out / size_in)))
        elif mode == "spectral":
            self.assertEqual(scale, math.sqrt(max(size_out, size_in)))
        elif mode == "unit_rms_norm":
            self.assertEqual(scale, math.sqrt(size_out / size_in))
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def test_get_coefficient_iterator_empty_raises_value_error(self) -> None:
        """Test that get_coefficient_iterator raises ValueError for empty coefficient_sets."""
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            list(muon_utils.get_coefficient_iterator(5, []))

    def test_get_coefficient_iterator_invalid_mode_raises_value_error(self) -> None:
        """Test that get_coefficient_iterator raises ValueError for invalid mode."""
        with self.assertRaisesRegex(ValueError, "Invalid mode.*invalid"):
            list(muon_utils.get_coefficient_iterator(5, [(1.0, 2.0, 3.0)], mode="invalid"))

    def test_newton_schulz_tp_invalid_partition_dim_raises_value_error(self) -> None:
        """Test that newton_schulz_tp raises ValueError for invalid partition_dim."""
        x = torch.randn(5, 7, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "Invalid partition_dim.*2"):
            muon_utils.newton_schulz_tp(
                x, steps=5, coefficient_type="quintic", tp_group=None, partition_dim=2, tp_mode="distributed"
            )

    def test_newton_schulz_tp_invalid_tp_mode_raises_value_error(self) -> None:
        """Test that newton_schulz_tp raises ValueError for invalid tp_mode."""
        x = torch.randn(5, 7, device=self.device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "Invalid tp_mode.*invalid"):
            muon_utils.newton_schulz_tp(
                x, steps=5, coefficient_type="quintic", tp_group=None, partition_dim=0, tp_mode="invalid"
            )


class TestBatchedNewtonSchulzStep(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device
        self.prev_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision(self.prev_precision)

    @parameterized.parameters(
        (2, 16, 16),
        (4, 16, 32),
        (16, 128, 128),
        (32, 128, 256),
    )
    def test_batched_newton_schulz_step_close_to_unbatched(self, batch, dim1, dim2):
        x = torch.randint(-3, 4, (batch, dim1, dim2), device=self.device, dtype=torch.float32)
        x = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=1e-7)

        a, b, c = 0.5, 1, 0.25
        batched = muon_utils.batched_newton_schulz_step(x, a, b, c)
        per_item = torch.stack([muon_utils.newton_schulz_step(x[i], a, b, c) for i in range(batch)])

        torch.testing.assert_close(batched, per_item, atol=1e-8, rtol=0)


@absltest.skipIf(
    _SM_VERSION not in ((8, 0), (9, 0), (10, 0), (10, 3)),
    f"Correctness of Triton kernel on SM {_SM_VERSION} cannot be guaranteed.",
)
class TestNewtonSchulzStepWithTsyrk(parameterized.TestCase):
    def setUp(self):
        self.device = FLAGS.device

    @parameterized.parameters(
        (32, 32),
        (32, 64),
    )
    def test_match_newton_schulz_step_by_gemm(self, dim1, dim2):
        x = torch.randint(-2, 3, (dim1, dim2), device=self.device, dtype=torch.bfloat16)
        test_out = muon_utils.newton_schulz_step_tsyrk(x, 2**-1, 2**-2, 2**-3)
        test_ref = muon_utils.newton_schulz_step(x, 2**-1, 2**-2, 2**-3)

        torch.testing.assert_close(test_out, test_ref, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
