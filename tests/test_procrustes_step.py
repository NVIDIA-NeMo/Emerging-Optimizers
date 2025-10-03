import math

import torch
from absl import testing
from absl.testing import parameterized

from emerging_optimizers.psgd.procrustes_step import procrustes_step
from emerging_optimizers.utils import fp32_matmul_precision


class ProcrustesStepTest(parameterized.TestCase):
    """Test cases for procrustes_step function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _procrustes_objective(self, Q: torch.Tensor) -> torch.Tensor:
        """Helper function to compute Procrustes objective ||Q^H Q - I||_F^2."""
        return torch.linalg.matrix_norm(Q.H @ Q - torch.eye(Q.size(0), dtype=Q.dtype, device=Q.device), ord="fro") ** 2

    def test_improves_orthogonality_simple_case(self) -> None:
        """Test that procrustes_step doesn't worsen orthogonality for a simple case."""

        # Make a SPD non-orthogonal matrix
        Q = torch.randn(2, 2, device=self.device, dtype=torch.float32)
        Q = Q @ Q.T

        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=1 / 16)

        final_obj = self._procrustes_objective(Q)

        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-6)

    def test_modifies_matrix_in_place(self) -> None:
        """Test that procrustes_step modifies the matrix in place."""
        Q = torch.randn(3, 3, device=self.device)
        Q_original_id = id(Q)

        procrustes_step(Q, max_step_size=1 / 16)

        self.assertEqual(id(Q), Q_original_id)

    @parameterized.parameters(
        (8,),
        (128,),
        (1024,),
    )
    def test_minimal_change_when_already_orthogonal(self, size: int) -> None:
        """Test that procrustes_step makes minimal changes to an already orthogonal matrix."""
        # Create an orthogonal matrix using QR decomposition
        A = torch.randn(size, size, device=self.device, dtype=torch.float32)
        with fp32_matmul_precision("highest"):
            Q, _ = torch.linalg.qr(A)

        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=1 / 16)

        final_obj = self._procrustes_objective(Q)

        # For already orthogonal matrices, the objective should remain small
        self.assertLess(final_obj.item(), 1e-5)
        self.assertLess(final_obj.item(), initial_obj.item() + 1e-5)

    def test_handles_small_norm_gracefully(self) -> None:
        """Test that procrustes_step handles matrices with small R norm improvement."""
        # Create a matrix very close to orthogonal
        A = torch.randn(3, 3, device=self.device, dtype=torch.float32)
        with fp32_matmul_precision("highest"):
            Q, _ = torch.linalg.qr(A)
        # Add tiny perturbation
        Q += 1e-10 * torch.randn_like(Q, dtype=torch.float32)

        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=1 / 16)

        final_obj = self._procrustes_objective(Q)

        self.assertLess(final_obj.item(), 1e-6)
        self.assertLess(final_obj.item(), initial_obj.item() + 1e-6)

    @parameterized.parameters(
        (1 / 64,),
        (1 / 32,),
        (1 / 16,),
        (1 / 8,),
    )
    def test_different_step_sizes(self, max_step_size: float) -> None:
        """Test procrustes_step improvement with different step sizes."""
        perturbation = 1e-1 * torch.randn(10, 10, device=self.device, dtype=torch.float32) / math.sqrt(10)
        Q = torch.linalg.qr(torch.randn(10, 10, device=self.device, dtype=torch.float32)).Q + perturbation
        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=max_step_size)

        final_obj = self._procrustes_objective(Q)

        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-4)

    @parameterized.parameters(
        (8,),
        (64,),
        (512,),
        (8192,),
    )
    def test_different_matrix_sizes(self, size: int) -> None:
        """Test procrustes_step improvement with different matrix sizes."""
        # Create a non-orthogonal matrix by scaling an orthogonal one
        A = torch.randn(size, size, device=self.device, dtype=torch.float32)
        with fp32_matmul_precision("highest"):
            Q_orth, _ = torch.linalg.qr(A)
        Q = Q_orth + 1e-2 * torch.randn(size, size, device=self.device, dtype=torch.float32) / math.sqrt(size)
        max_step_size = 0.5 * size ** (-1 / 3)
        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=max_step_size)

        final_obj = self._procrustes_objective(Q)

        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-3)

    def test_preserves_determinant_sign_for_real_matrices(self) -> None:
        """Test that procrustes_step preserves the sign of determinant for real matrices."""
        # Create real matrices with positive and negative determinants
        Q_pos = torch.tensor([[2.0, 0.1], [0.1, 1.5]], device=self.device, dtype=torch.float32)  # det > 0
        Q_neg = torch.tensor([[-2.0, 0.1], [0.1, 1.5]], device=self.device, dtype=torch.float32)  # det < 0

        initial_det_pos = torch.det(Q_pos)
        initial_det_neg = torch.det(Q_neg)

        procrustes_step(Q_pos, max_step_size=1 / 16)
        procrustes_step(Q_neg, max_step_size=1 / 16)

        final_det_pos = torch.det(Q_pos)
        final_det_neg = torch.det(Q_neg)

        # Signs should be preserved
        self.assertGreater(initial_det_pos.item() * final_det_pos.item(), 0)
        self.assertGreater(initial_det_neg.item() * final_det_neg.item(), 0)


if __name__ == "__main__":
    testing.absltest.main()
