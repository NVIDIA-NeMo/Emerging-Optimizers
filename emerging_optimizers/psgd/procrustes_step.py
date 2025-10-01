import torch

from emerging_optimizers.psgd.psgd_utils import norm_lower_bound_skew
from emerging_optimizers.utils import fp32_matmul_precision


def procrustes_step(Q, max_step_size=1 / 8, num_iters=1):
    r"""In-place online solver for the orthogonal Procrustes problem.

        min_U || U Q - I ||_F,   s.t. U^H U = I
    by rotating Q as exp(a R) Q, where R = Q^H - Q is the generator and ||a R|| < 1.

    `max_step_size` should be less than 1/4 as we only expand exp(a R) to its 2nd order term.

    Args:
        Q: Tensor of shape (n, n), general square matrix to orthogonalize.
        max_step_size: Maximum step size for the line search. Default is 1/8.
        num_iters: Number of iterations to run. Default is 1.

    """
    for _ in range(num_iters):
        with fp32_matmul_precision("highest"):
            R = Q.H - Q
            R /= norm_lower_bound_skew(R) + torch.finfo(R.dtype).smallest_normal
            RQ = R @ Q
            tr_RQ = torch.trace(RQ)
            RRQ = R @ RQ
            tr_RRQ = torch.trace(RRQ)
            step_size = torch.clamp(-tr_RQ / tr_RRQ, min=0, max=max_step_size)
            a = torch.where(tr_RRQ < 0, step_size, max_step_size)
            # rotate Q as exp(a R) Q ~ (I + a R + a^2 R^2/2) Q with an optimal step size by line search
            # for 2nd order Taylor expansion, we only expand exp(a R) to its 2nd term.
            Q.add_(a * (RQ + 0.5 * a * RRQ))
