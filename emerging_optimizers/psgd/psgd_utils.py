from typing import List

import torch


__all__ = [
    "balance_q_in_place",
    "solve_triangular_right",
    "norm_lower_bound_spd",
    "norm_lower_bound_skew",
]


@torch.compile  # type: ignore[misc]
def balance_q_in_place(Q_list: List[torch.Tensor]) -> None:
    """Balance the dynamic ranges of kronecker factors in place to prevent numerical underflow or overflow.

    Each tensor in `Q_list` is rescaled so that its maximum absolute entry
    becomes the geometric mean of all factors original maxima. This preserves
    the overall product of norms (and thus the scale of the Kronecker product)
    while avoiding numerical underflow or overflow when factors have widely
    differing magnitudes.

    Given tensors :math:`Q_1, Q_2, \\ldots, Q_n`:

    1. Compute max-absolute norms: :math:`\\|Q_i\\|_\\infty = \\max(|Q_i|)` for :math:`i = 1, \\ldots, n`
    2. Compute geometric mean: :math:`g = \\left(\\prod_{i=1}^{n} \\|Q_i\\|_\\infty \\right)^{1/n}`
    3. Rescale each tensor: :math:`Q_i \\leftarrow Q_i \\cdot \\frac{g}{\\|Q_i\\|_\\infty}`

    This ensures :math:`\\|Q_i\\|_\\infty = g` for all :math:`i`, while preserving the norm of
    the Kronecker product :math:`Q_1 \\otimes Q_2 \\otimes \\cdots \\otimes Q_n`.

    Args:
        Q_list: List of Q (e.g. the Kronecker factors), each tensor will be modified in place.

    Returns:
        None

    """
    if (order := len(Q_list)) <= 1:
        return

    # 1) Compute max‐abs norm of each factor
    norms = [torch.max(torch.abs(Q)) for Q in Q_list]

    # 2) Compute geometric mean of those norms
    stacked = torch.stack(norms)
    gmean = torch.prod(stacked) ** (1.0 / order)

    # 3) Rescale each factor so its max‐abs entry == geometric mean
    for Q, norm in zip(Q_list, norms):
        Q.mul_(gmean / norm)


@torch.compile  # type: ignore[misc]
def solve_triangular_right(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Solve system of linear equations y A = X with upper-triangular A for y.

    This wraps `torch.linalg.solve_triangular` with `left=False`
    so that you compute :math:`X A^{-1}`. If :math:`X` is 1D, we temporarily
    add a leading batch dimension to satisfy the solver's requirements
    and then remove it from the result.

    Args:
        X: Tensor of shape :math:`(..., n)` or :math:`(n,)` containing one or more
           right-hand sides.
        A: Upper-triangular square matrix of shape :math:`(n, n)`.

    Returns:
        A Tensor of the same shape as X, equal to :math:`X A^{-1}`.

    """
    if X.dim() > 1:
        return torch.linalg.solve_triangular(A, X, upper=True, left=False)
    # `torch.linalg.solve_triangular` requires at least 2D RHS, so we reshape
    batch_result = torch.linalg.solve_triangular(A, X[None, :], upper=True, left=False)
    solution = batch_result.squeeze(0)  # Remove the added batch dimension to get back to 1D
    return solution


@torch.compile  # type: ignore[misc]
def norm_lower_bound_spd(A: torch.Tensor, k: int = 4, half_iters: int = 2) -> torch.Tensor:
    r"""Returns a cheap lower bound for the spectral norm of a symmetric positive definite matrix.

    Uses numerically stable subspace iteration with a random initialization that aligns with the
    largest row of A to approximate the dominant eigenspace of A. From Xi-Lin Li.

    Args:
        A: Tensor of shape :math:`(n, n)`, symmetric positive definite.
        k: Dimension of the subspace.
        half_iters: Half of the number of subspace iterations.
    Returns:
        A scalar giving a lower bound on :math:`\\|A\\|_2`.
    """
    # Smallest representable normal number for numerical stability
    smallest_normal = torch.finfo(A.dtype).smallest_normal

    # Compute normalizing factor from the largest diagonal entry to prevent overflow/underflow
    normalization = A.diagonal().amax() + smallest_normal
    A = A / normalization

    # Find the row index with the largest 2-norm to initialize our subspace
    # This helps the algorithm converge faster to the dominant eigenspace
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))

    # Initialize random subspace matrix V of shape (k, n)
    # k vectors of dimension n will span our subspace approximation
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)

    # Rotate the random vectors to align with the dominant row A[j]
    # This initialization trick makes the subspace iteration more robust for low-rank matrices
    # The sign function ensures proper alignment of each random vector with A[j]
    V = A[j] + torch.sign(torch.sum(A[j] * V, dim=1, keepdim=True)) * V

    # Perform subspace iteration
    for _ in range(half_iters):
        V = V @ A
        # Normalize each row of V to prevent exponential growth/decay in the subspace iteration
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + smallest_normal
        # V approximates the dominant eigenspace of A^2
        V = V @ A

    # Compute the final estimate: find the largest 2-norm among the k vectors
    # and scale back by the normalization factor to get the actual bound
    return normalization * torch.amax(torch.linalg.vector_norm(V, dim=1))


@torch.compile  # type: ignore[misc]
def norm_lower_bound_skew(A: torch.Tensor, iters: int = 3, eps: float = 1e-9) -> torch.Tensor:
    """Compute a cheap lower bound on the spectral norm (largest eigenvalue) of skew-symmetric matrix.

    This uses a power iteration method:

    1. Normalize :math:`A` by its largest absolute entry to avoid overflow.
    2. Find the row :math:`j` of :math:`A_{\\text{scaled}}` with the largest 2-norm.
    3. Initialize vector :math:`v` from :math:`A_{\\text{scaled}}[j]`.
    4. Perform power iteration for `iters` steps: :math:`v \\leftarrow v \\cdot A_{\\text{scaled}}`.
    5. Estimate the norm by :math:`\\|v \\cdot A_{\\text{scaled}}\\|_2`, then rescale.

    Note: For skew-symmetric matrices, all diagonal entries are zero and :math:`A^T = -A`.
    From Xi-Lin Li.

    Args:
        A: Tensor of shape :math:`(n, n)`, skew-symmetric.
        iters: Number of power iteration steps to perform.
        eps: Small value to add to the diagonal of A to avoid division by zero.

    Returns:
        A scalar Tensor giving a lower bound on :math:`\\|A\\|_2`.

    """
    # Normalize to avoid extreme values, by extracting the max absolute value
    max_abs = torch.max(torch.abs(A))

    if max_abs <= torch.finfo(A.dtype).smallest_normal:
        return max_abs

    A_scaled = A / max_abs

    # Pick row with largest 2-norm (skew-symmetric matrices have zero diagonal)
    row_norms = torch.linalg.norm(A_scaled, dim=1)
    j = torch.argmax(row_norms)

    # Initialize with the dominant row
    v = A_scaled[j].clone()

    # Power iteration steps
    for _ in range(iters):
        # Normalize to prevent overflow
        v_norm = torch.linalg.norm(v) + eps
        v = v / v_norm

        # Apply matrix
        v = v @ A_scaled

    # Final normalization and bound computation
    v_norm = torch.linalg.norm(v) + eps
    v = v / v_norm

    # Compute bound and rescale
    bound = torch.linalg.norm(v @ A_scaled)
    return max_abs * bound


def norm_lower_bound_skh(A, k=32, half_iters=2):
    """
    Returns a cheap lower bound for the spectral norm of a skew-Hermitian matrix A,
        k: the dim of subspace, suggesting 128 for bfloat16, 32 for float32 and 4 for float64 (tested on my laptop 4070 GPU);
        half_iters: half of the number of subspace iterations, suggesting 2.
    A rough norm estimation is good, and we don't orthonormaliz the subspace vectors.

    The initial noise space V is rotated such that its centroid aligns with the largest row of A.
    Hence, each row of V and the largest row of A has an angle about acos(1/sqrt(k)) when k << dim(A).
    This feature makes the subspace iteration more robust for large matrices with very low rank.
    A simplified branchless approximate implementation is provided here.
    """
    smallest_normal = torch.finfo(A.dtype).smallest_normal
    normalizing_factor = A.abs().amax() + smallest_normal
    A = A / normalizing_factor  # (complex tensor) / (subnormal number) could produce inf or nan unexpectedly
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    V = A[j] + torch.sgn(torch.sum(A[j] * V.conj(), dim=1, keepdim=True)) * V  # torch.sign for real
    for _ in range(half_iters):
        V = V @ A
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + smallest_normal
        V = V @ A
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))
