import torch
import torch.nn as nn
from absl.testing import absltest, parameterized

from emerging_optimizers.orthogonalized_optimizers.adaptive_orthogonalized_optimizer import (
    AdaptiveOrthogonalizedOptimizer,
)


class AdaptiveOrthogonalizedOptimizerTest(parameterized.TestCase):
    @parameterized.product(
        shape=[(5, 7), (33, 65), (127, 257)],
        second_moment_method=["adamuon", "normuon"],
        use_nesterov=[True, False],
    )
    def test_smoke(self, shape, second_moment_method, use_nesterov) -> None:
        """Smoke test AdaptiveOrthogonalizedOptimizer with both second moment methods."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        adaptive_opt = AdaptiveOrthogonalizedOptimizer(
            [test_param],
            lr=0.01,
            momentum_beta=0.9,
            weight_decay=0.01,
            use_nesterov=use_nesterov,
            moment2_method=second_moment_method,
            beta2=0.999,
            eps=1e-8,
            weight_decay_method="decoupled",
            fp32_matmul_prec="highest",
        )
        adaptive_opt.step()

    @parameterized.parameters(
        {"shape": (8, 16), "second_moment_method": "adamuon"},
        {"shape": (16, 8), "second_moment_method": "normuon"},
    )
    def test_second_moment_matches_shapes(self, shape, second_moment_method) -> None:
        """Test that second moment buffers are properly initialized."""
        test_param = nn.Parameter(torch.randint(-5, 5, shape, dtype=torch.float32, device="cuda"))
        test_param.grad = torch.randint_like(test_param, -5, 5)

        adaptive_opt = AdaptiveOrthogonalizedOptimizer(
            [test_param],
            lr=0.01,
            momentum_beta=0.9,
            weight_decay=0.0,
            use_nesterov=False,
            moment2_method=second_moment_method,
            beta2=0.999,
            eps=1e-8,
            weight_decay_method="decoupled",
            fp32_matmul_prec="highest",
        )

        # Run one step to initialize buffers
        adaptive_opt.step()

        # Check that second moment buffer was created
        state = adaptive_opt.state[test_param]
        self.assertIn("moment2_buffer", state)
        self.assertIn("momentum_buffer", state)

        # Check second moment buffer shape
        second_moment = state["moment2_buffer"]
        if second_moment_method == "adamuon":
            # Full elementwise buffer
            self.assertEqual(second_moment.shape, test_param.shape)
        elif second_moment_method == "normuon":
            # Reduced shape buffer
            avg_dim = -1 if shape[-2] >= shape[-1] else -2
            expected_shape = list(shape)
            expected_shape[avg_dim] = 1
            self.assertEqual(list(second_moment.shape), expected_shape)

    def test_requires_second_moment_method(self) -> None:
        """Test that AdaptiveOrthogonalizedOptimizer requires second_moment_method."""
        test_param = nn.Parameter(torch.randint(-5, 5, (8, 16), dtype=torch.float32, device="cuda"))

        with self.assertRaises(TypeError):
            AdaptiveOrthogonalizedOptimizer(
                [test_param],
                lr=0.01,
                momentum_beta=0.9,
                weight_decay=0.0,
                use_nesterov=False,
                moment2_method=None,  # Should raise error
                beta2=0.999,
                eps=1e-8,
                weight_decay_method="decoupled",
                fp32_matmul_prec="highest",
            )


if __name__ == "__main__":
    absltest.main()
