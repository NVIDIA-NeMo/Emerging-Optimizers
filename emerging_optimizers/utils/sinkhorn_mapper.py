import torch
import torch.nn.functional as F


__all__ = [
    "SinkhornMapper",
]


class SinkhornMapper:
    """
    Applies the Sinkhorn-Knopp mapping in place on the input tensor:
    Input -> [Exp] -> [Iterative Row/Col Normalization]

    Args:
        t_max: The number of iterations to run the Sinkhorn-Knopp mapping.
        epsilon: The epsilon value to use for the Sinkhorn-Knopp mapping for numerical stability.
    """

    def __init__(self, t_max: int = 20, epsilon: float = 1e-8):
        self.t_max = t_max
        self.epsilon = epsilon

    @torch.no_grad()
    def _sinkhorn_inplace(self, x: torch.Tensor) -> None:
        # Enforce positivity via exp in place
        x.exp_()

        # Iterative normalization of rows and columns
        for _ in range(self.t_max):
            # Normalize columns
            F.normalize(x, p=1, dim=-2, eps=self.epsilon, out=x)
            # Normalize rows
            F.normalize(x, p=1, dim=-1, eps=self.epsilon, out=x)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> None:
        self._sinkhorn_inplace(x)
