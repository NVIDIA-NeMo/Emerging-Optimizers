import torch
from absl import testing
from absl.testing import parameterized

from emerging_optimizers.psgd.psgd_kron_contractions import (
    _mode_n_mul_and_permute,
    apply_kronecker_factors,
    apply_preconditioner,
    partial_contraction,
)
from emerging_optimizers.utils import fp32_matmul_precision


class TestPSGDKronContractions(parameterized.TestCase):
    """Test cases for PSGD Kronecker contractions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    @parameterized.parameters(
        (2, 3, 3),
        (2, 3, 4),
        (2, 3, 5),
    )
    def test_partial_contraction_matches_reconstructed(self, size1: int, size2: int, size3: int) -> None:
        """Test partial_contraction matches reconstructed."""
        G1 = torch.randn(size1, size2, size3, device=self.device)
        G2 = torch.randn(size1, size2, size3, device=self.device)
        with fp32_matmul_precision("highest"):
            result = partial_contraction(G1, G2, axis=1)
            reconstructed = torch.tensordot(G1, G2, dims=([0, 2], [0, 2]))
        torch.testing.assert_close(result, reconstructed)

    def test_apply_kronecker_factors_matches_reconstructed(self) -> None:
        """Test apply_kronecker_factors matches reconstructed."""
        Q_list = [
            torch.triu(torch.randn(2, 2, device=self.device)),
            torch.triu(torch.randn(3, 3, device=self.device)),
            torch.triu(torch.randn(3, 3, device=self.device)),
        ]
        X = torch.randn(2, 3, 3, device=self.device)
        with fp32_matmul_precision("highest"):
            result = apply_kronecker_factors(Q_list, X)
            Y = X

            temp = torch.tensordot(Q_list[0], Y, dims=([1], [0]))
            nd = Y.dim()
            perm = list(range(1, 0 + 1)) + [0] + list(range(0 + 1, nd))  # [1, 0, 2]
            Y = temp.permute(perm)

            temp = torch.tensordot(Q_list[1], Y, dims=([1], [1]))
            nd = Y.dim()
            perm = list(range(1, 1 + 1)) + [0] + list(range(1 + 1, nd))  # [1, 0, 2]
            Y = temp.permute(perm)

            temp = torch.tensordot(Q_list[2], Y, dims=([1], [2]))
            nd = Y.dim()
            perm = list(range(1, 2 + 1)) + [0] + list(range(2 + 1, nd))  # [1, 2, 0]
            reconstructed = temp.permute(perm)

        torch.testing.assert_close(result, reconstructed)

    def test_apply_preconditioner_matches_reconstructed(self) -> None:
        """Test apply_preconditioner matches manual reconstruction for 2D tensor."""
        Q_list = [torch.triu(torch.randn(3, 3, device=self.device)), torch.triu(torch.randn(4, 4, device=self.device))]
        X = torch.randn(3, 4, device=self.device)

        with fp32_matmul_precision("highest"):
            result = apply_preconditioner(Q_list, X)

            # Manual reconstruction: precompute Q^T @ Q matrices then apply them
            # Create full preconditioner matrices Q^T @ Q
            QTQ_list = [q.T @ q for q in Q_list]

            # Apply the preconditioner matrices
            Y = X

            # Apply QTQ_list[0] to dimension 0
            temp = torch.tensordot(QTQ_list[0], Y, dims=([1], [0]))
            nd = Y.dim()
            perm = list(range(1, 0 + 1)) + [0] + list(range(0 + 1, nd))
            Y = temp.permute(perm)

            # Apply QTQ_list[1] to dimension 1
            temp = torch.tensordot(QTQ_list[1], Y, dims=([1], [1]))
            nd = Y.dim()
            perm = list(range(1, 1 + 1)) + [0] + list(range(1 + 1, nd))
            reconstructed = temp.permute(perm)

        torch.testing.assert_close(result, reconstructed)

    @parameterized.parameters(
        (2, 3, 5, 0),
        (2, 3, 5, 1),
        (2, 3, 5, 2),
        (4, 6, 2, 1),
    )
    def test_mode_n_mul_and_permute_shapes(self, dim0: int, dim1: int, dim2: int, mode: int) -> None:
        """Test `_mode_n_mul_and_permute` with non-uniform shapes and different modes."""
        X = torch.randn(dim0, dim1, dim2, device=self.device)
        input_shape = X.shape

        input_dim = input_shape[mode]
        output_dim = 7  # arbitrary output dimension
        M = torch.randn(output_dim, input_dim, device=self.device)

        result = _mode_n_mul_and_permute(X, M, mode)

        # Verify output shape: same as input but dimension `mode` replaced by output_dim
        expected_shape = list(input_shape)
        expected_shape[mode] = output_dim
        self.assertEqual(result.shape, torch.Size(expected_shape))


if __name__ == "__main__":
    testing.absltest.main()
