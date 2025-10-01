import torch
from absl import testing
from absl.testing import parameterized

from emerging_optimizers.psgd.procrustes_step import procrustes_step
from emerging_optimizers.utils import fp32_matmul_precision


class ProcrustesStepTest(parameterized.TestCase):
    """Test cases for procrustes_step function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)  # For reproducible tests
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _procrustes_objective(self, Q):
        """Helper function to compute Procrustes objective ||Q^H Q - I||_F^2."""
        return torch.linalg.matrix_norm(Q.H @ Q - torch.eye(Q.size(0), dtype=Q.dtype, device=Q.device), ord="fro") ** 2

    def test_improves_orthogonality_simple_case(self):
        """Test that procrustes_step doesn't worsen orthogonality for a simple case."""
        # Create a non-orthogonal matrix
        Q = torch.tensor([[2.0, 0.5], [0.0, 1.5]], device=self.device, dtype=torch.float32)

        # Measure initial objective
        initial_obj = self._procrustes_objective(Q)

        # Apply procrustes step
        procrustes_step(Q, max_step_size=1 / 16)

        # Measure final objective
        final_obj = self._procrustes_objective(Q)

        # Should improve (reduce) the objective or remain the same (if tr_RQ <= 0)
        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-6)

    def test_modifies_matrix_in_place(self):
        """Test that procrustes_step modifies the matrix in place."""
        Q = torch.randn(3, 3, device=self.device)
        Q_original_id = id(Q)

        procrustes_step(Q, max_step_size=1 / 16)

        # Should be the same tensor object
        self.assertEqual(id(Q), Q_original_id)
        # The tensor object should remain the same, regardless of whether data changed

    def test_minimal_change_when_already_orthogonal(self):
        """Test that procrustes_step makes minimal changes to an already orthogonal matrix."""
        # Create an orthogonal matrix using QR decomposition
        A = torch.randn(4, 4, device=self.device)
        with fp32_matmul_precision("highest"):
            Q, _ = torch.linalg.qr(A)

        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=1 / 16)

        final_obj = self._procrustes_objective(Q)

        # For already orthogonal matrices, the objective should remain small
        self.assertLess(final_obj.item(), 1e-4)
        # And shouldn't get significantly worse
        self.assertLess(final_obj.item(), initial_obj.item() + 1e-4)

    def test_handles_small_norm_gracefully(self):
        """Test that procrustes_step handles matrices with small R norm."""
        # Create a matrix very close to orthogonal
        A = torch.randn(3, 3, device=self.device)
        with fp32_matmul_precision("highest"):
            Q, _ = torch.linalg.qr(A)
        # Add tiny perturbation
        Q += 1e-10 * torch.randn_like(Q)

        initial_obj = self._procrustes_objective(Q)

        # Should not crash
        procrustes_step(Q, max_step_size=1 / 16)

        final_obj = self._procrustes_objective(Q)

        # Should maintain good orthogonality (objective should remain small)
        self.assertLess(final_obj.item(), 1e-6)
        # Shouldn't get significantly worse
        self.assertLess(final_obj.item(), initial_obj.item() + 1e-6)

    @parameterized.parameters(
        (1 / 32,),
        (1 / 16,),
        (1 / 8,),
        (1 / 6,),
    )
    def test_different_step_sizes(self, max_step_size):
        """Test procrustes_step with different step sizes."""
        Q = torch.linalg.qr(torch.randn(10, 10, device=self.device)).Q + 1e-1 * torch.randn(10, 10, device=self.device)
        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=max_step_size)

        final_obj = self._procrustes_objective(Q)

        # Should not make orthogonality worse (algorithm only updates when beneficial)
        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-5)

    @parameterized.parameters(
        (8,),
        (64,),
        (512,),
        (8192,),
    )
    def test_different_matrix_sizes(self, size):
        """Test procrustes_step with different matrix sizes."""
        # Create a non-orthogonal matrix by scaling an orthogonal one
        A = torch.randn(size, size, device=self.device)
        with fp32_matmul_precision("highest"):
            Q_orth, _ = torch.linalg.qr(A)
        Q = Q_orth + 1e-2 * torch.randn(size, size, device=self.device)
        max_step_size = 0.5 * size ** (-1 / 3)
        initial_obj = self._procrustes_objective(Q)

        procrustes_step(Q, max_step_size=max_step_size)

        final_obj = self._procrustes_objective(Q)

        # Algorithm should not make orthogonality significantly worse
        self.assertLessEqual(final_obj.item(), initial_obj.item() + 1e-3)

    @parameterized.parameters(
        (1,),
        (2,),
        (4,),
        (8,),
    )
    def test_multiple_iterations_converge(self, num_iter):
        """Test that multiple procrustes steps eventually improve orthogonality."""
        Q = torch.tensor([[3.0, 0.5], [0.2, 2.0]], device=self.device)

        initial_obj = self._procrustes_objective(Q).item()

        procrustes_step(Q, max_step_size=1 / 16, num_iter=num_iter)

        final_obj = self._procrustes_objective(Q).item()

        # After many iterations, should significantly improve orthogonality
        self.assertLess(final_obj, initial_obj)
        # Should achieve reasonable orthogonality
        self.assertLess(final_obj, initial_obj * 0.5)

    @parameterized.parameters(
        (1,),
        (2,),
        (4,),
        (8,),
    )
    def test_functional_behavior_with_good_case(self, num_iter):
        """Test procrustes_step with a case that's likely to show improvement."""
        # Create a matrix that's close to orthogonal but not quite
        eps = 1e-2
        Q_orth = torch.eye(2, device=self.device)
        Q = Q_orth + eps * torch.randn(2, 2, device=self.device)

        initial_obj = self._procrustes_objective(Q)

        # Apply several steps to ensure we see improvement
        procrustes_step(Q, max_step_size=1 / 16, num_iter=num_iter)

        final_obj = self._procrustes_objective(Q)

        # Should improve orthogonality
        self.assertLess(final_obj, initial_obj)
        # Should get reasonably close to orthogonal
        self.assertLess(final_obj, 0.1)

    def test_preserves_determinant_sign_for_real_matrices(self):
        """Test that procrustes_step preserves the sign of determinant for real matrices."""
        # Create real matrices with positive and negative determinants
        Q_pos = torch.tensor([[2.0, 0.1], [0.1, 1.5]], device=self.device)  # det > 0
        Q_neg = torch.tensor([[-2.0, 0.1], [0.1, 1.5]], device=self.device)  # det < 0

        initial_det_pos = torch.det(Q_pos)
        initial_det_neg = torch.det(Q_neg)

        procrustes_step(Q_pos, max_step_size=1 / 16, num_iter=1)
        procrustes_step(Q_neg, max_step_size=1 / 16, num_iter=1)

        final_det_pos = torch.det(Q_pos)
        final_det_neg = torch.det(Q_neg)

        # Signs should be preserved
        self.assertGreater(initial_det_pos.item() * final_det_pos.item(), 0)
        self.assertGreater(initial_det_neg.item() * final_det_neg.item(), 0)


if __name__ == "__main__":
    torch.manual_seed(42)
    testing.absltest.main()
