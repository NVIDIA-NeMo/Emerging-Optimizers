# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Derive a <=10-step CANS-style coefficient schedule for the matrix inverse fourth root.

Coupled iteration on ``M_0 = A / (1.01 * inf_norm)`` with ``q_k(m) = beta_k + alpha_k * m``:
``X_{k+1} = q_k(M_k) X_k``, ``M_{k+1} = q_k(M_k)^4 M_k``, so ``X_N = M_0^{-1/4} M_N^{1/4}``.
Each eigenvalue follows ``m -> f(m) = m * q(m)^4``. The per-step design maximizes the image
interval ratio ``l'/u'``; the ratio is invariant to scaling ``q``, so only the root location
``r = -alpha / beta`` is optimized (1-D), and the scale is fixed by renormalizing the image
interval top to 1. The greedy objective is exact for the schedule family: the final relative
error after the free geometric recentering is ``rho_N**-0.125 - 1``, a monotone function of
the final ratio alone. The printed table folds the 1.01 spectral safety margin and the
recentering scalar following the convention of ``mat_root_inv_via_scaled_cans``.
"""

import math

import numpy as np
import torch
from absl import app, flags


FLAGS = flags.FLAGS

CoeffTable = tuple[tuple[float, float], ...]


def f_eval(m: np.ndarray | float, r: float) -> np.ndarray | float:
    """Evaluate the scalar step map ``f(m) = m * (1 - r * m)**4``."""
    return m * (1.0 - r * m) ** 4


def f_extrema(r: float, lo: float, up: float) -> tuple[float, float]:
    """Return the extrema of ``f`` over ``[lo, up]`` from the closed-form critical point."""
    f_lo = f_eval(lo, r)
    f_up = f_eval(up, r)
    f_min = min(f_lo, f_up)
    f_max = max(f_lo, f_up)
    m_star = 1.0 / (5.0 * r)
    if lo <= m_star <= up:
        f_max = max(f_max, f_eval(m_star, r))
    return f_min, f_max


def step_ratio(r: float, lo: float, up: float) -> float:
    """Return the image interval ratio ``f_min / f_max`` for root location ``r``."""
    f_min, f_max = f_extrema(r, lo, up)
    return f_min / f_max


def optimize_step(lo: float, up: float, root_margin: float) -> float:
    """Find the root location maximizing the image interval ratio over ``[lo, up]``."""
    r_hi = 1.0 / (root_margin * up)
    scan = np.geomspace(r_hi * 1e-6, r_hi, 4001)
    ratios = np.array([step_ratio(float(r), lo, up) for r in scan])
    i = int(np.argmax(ratios))
    a = float(scan[max(i - 1, 0)])
    b = float(scan[min(i + 1, len(scan) - 1)])
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    for _ in range(100):
        if step_ratio(c, lo, up) > step_ratio(d, lo, up):
            b = d
        else:
            a = c
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
    return 0.5 * (a + b)


def derive_schedule(
    l0: float, steps: int, root_margin: float
) -> tuple[CoeffTable, list[tuple[float, float, float]], float]:
    """Greedily derive the per-step coefficient pairs starting from the interval ``[l0, 1]``."""
    pairs = []
    rows = []
    lo, up = l0, 1.0
    grid = np.geomspace(l0, 1.0, 100001)
    for _ in range(steps):
        r = optimize_step(lo, up, root_margin)
        f_min, f_max = f_extrema(r, lo, up)
        scale = f_max**-0.25
        beta, alpha = scale, -r * scale
        pairs.append((beta, alpha))
        new_lo = f_min / f_max
        rows.append((lo, new_lo / lo, 1.0 / r))
        grid = grid * (beta + alpha * grid) ** 4
        if grid.max() > 1.0 + 1e-9 or grid.min() < new_lo * (1.0 - 1e-9):
            raise AssertionError(
                f"interval tracking violated: grid in [{grid.min():.3e}, {grid.max():.3e}], expected [{new_lo:.3e}, 1]"
            )
        lo, up = new_lo, 1.0
    return tuple(pairs), rows, lo


def fold_coefficients(pairs: CoeffTable, rho_final: float, safety: float) -> CoeffTable:
    """Fold the safety margin and the final geometric recentering into the coefficient table."""
    folded = [(beta, alpha / safety) for beta, alpha in pairs[:-1]]
    beta, alpha = pairs[-1]
    recenter = rho_final**-0.125
    folded.append((beta * recenter / safety**0.25, alpha * recenter / safety**1.25))
    return tuple(folded)


def run_schedule_scalar(folded: CoeffTable, lam: np.ndarray, dtype: type) -> tuple[np.ndarray, np.ndarray]:
    """Run the folded schedule on scalar eigenvalues in the given dtype, mirroring the GEMM order."""
    lam = lam.astype(dtype)
    x = np.ones_like(lam)
    mu = lam.copy()
    coeffs = [(dtype(b), dtype(a)) for b, a in folded]
    for beta, alpha in coeffs[:-1]:
        x = beta * x + alpha * (mu * x)
        q = beta + alpha * mu
        s = q * q
        mu = (s * s) * mu
    beta, alpha = coeffs[-1]
    x = beta * x + alpha * (mu * x)
    return x, mu


def scalar_max_rel_err(folded: CoeffTable, lam: np.ndarray, dtype: type) -> float:
    """Return the max relative error of the schedule output against ``lam**-0.25``."""
    x, _ = run_schedule_scalar(folded, lam, dtype)
    return float(np.abs(x.astype(np.float64) * lam**0.25 - 1.0).max())


def newton_baseline_rel_err(l0: float, steps: int) -> float:
    """Return the recentered max relative error of plain Newton at the same step count."""
    lam = np.geomspace(l0, 1.0, 20001)
    x = np.ones_like(lam)
    mu = lam.copy()
    for _ in range(steps):
        q = (5.0 - mu) / 4.0
        x = x * q
        mu = mu * q**4
    factor = x * lam**0.25
    center = math.sqrt(factor.min() * factor.max())
    return float(np.abs(factor / center - 1.0).max())


def fourth_root_inv_matrix(x: torch.Tensor, folded: CoeffTable) -> torch.Tensor:
    """Apply the folded schedule to a symmetric positive-definite matrix."""
    inf_norm = torch.linalg.matrix_norm(x, ord=float("inf"))
    m = x / inf_norm
    identity = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    z = identity.clone()
    for beta, alpha in folded[:-1]:
        z = torch.addmm(z, m, z, beta=beta, alpha=alpha)
        t = alpha * m
        t.diagonal().add_(beta)
        s = t @ t
        m = (s @ s) @ m
    beta, alpha = folded[-1]
    z = torch.addmm(z, m, z, beta=beta, alpha=alpha)
    z = z * inf_norm ** (-0.25)
    return (z + z.T) * 0.5


def spd_with_cond(n: int, cond: float, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Build a symmetric positive-definite matrix with a log-spaced spectrum of the given condition number."""
    gen = torch.Generator().manual_seed(42)
    q_mat, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    eigenvalues = torch.logspace(0.0, -math.log10(cond), n, dtype=torch.float64)
    mat = (q_mat * eigenvalues) @ q_mat.T
    mat = (mat + mat.T) * 0.5
    return mat.to(device=device, dtype=dtype)


def matrix_check(folded: CoeffTable, n: int, cond: float, dtype: torch.dtype, device: str) -> tuple[float, float]:
    """Return the relative spectral error against an eigh reference and the whitening deviation."""
    mat = spd_with_cond(n, cond, dtype, device)
    w, v = torch.linalg.eigh(mat.double())
    ref = ((v * w.pow(-0.25)) @ v.mT).to(dtype)
    z = fourth_root_inv_matrix(mat, folded)
    rel_err = (torch.linalg.matrix_norm(z - ref, 2) / torch.linalg.matrix_norm(ref, 2)).item()
    w2 = z @ z
    whiten_dev = (w2 @ mat @ w2 - torch.eye(n, device=device, dtype=dtype)).abs().max().item()
    return rel_err, whiten_dev


def main(argv: list[str]) -> None:
    """Derive, verify, and print the schedules for all requested design intervals."""
    del argv
    torch.set_float32_matmul_precision("highest")
    np.seterr(over="raise", invalid="raise")

    target_bound = 1e-3
    target_fp32 = 2e-3
    selected = None

    for l0_str in FLAGS.l0_values:
        l0 = float(l0_str)
        pairs, rows, rho = derive_schedule(l0, FLAGS.steps, FLAGS.root_margin)
        folded = fold_coefficients(pairs, rho, FLAGS.safety)
        bound = rho**-0.125 - 1.0

        lam_design = np.geomspace(FLAGS.safety * l0, 1.0, 200001)
        err64 = scalar_max_rel_err(folded, lam_design, np.float64)
        err32 = scalar_max_rel_err(folded, lam_design, np.float32)

        lam_below = np.geomspace(l0 * 1e-3, FLAGS.safety * l0, 2001)
        x_below, mu_below = run_schedule_scalar(folded, lam_below, np.float64)
        below_ok = bool((x_below > 0).all() and (mu_below > 0).all() and mu_below.max() < FLAGS.safety + 1e-6)
        err_half = scalar_max_rel_err(folded, np.array([l0 / 2.0]), np.float64)
        err_tenth = scalar_max_rel_err(folded, np.array([l0 / 10.0]), np.float64)

        newton = newton_baseline_rel_err(l0, FLAGS.steps)

        print(f"\n{'=' * 100}")
        print(f"l0 = {l0:.0e}   (kappa coverage ~ {1.0 / (FLAGS.safety * l0):.2e})")
        print(f"{'=' * 100}")
        print(f"{'k':>2}  {'beta (folded)':>22}  {'alpha (folded)':>22}  {'l_k':>10}  {'ratio gain':>10}  {'root':>7}")
        for k, ((beta, alpha), (lo, gain, root)) in enumerate(zip(folded, rows)):
            print(f"{k:>2}  {beta:>22.15f}  {alpha:>22.15f}  {lo:>10.3e}  {gain:>10.3f}  {root:>7.4f}")
        print(f"final ratio rho_N = {rho:.10f}")
        print(f"guaranteed scalar rel err (recentered): {bound:.3e}")
        print(f"float64 scalar sim max rel err:         {err64:.3e}")
        print(f"float32 scalar sim max rel err:         {err32:.3e}")
        print(f"newton (5/4, -1/4) baseline, same N:    {newton:.3e}")
        print(f"below-design eigenvalues stay positive/bounded: {below_ok}")
        print(f"rel err at l0/2: {err_half:.3e}   at l0/10: {err_tenth:.3e}")

        if bound <= target_bound and err32 <= target_fp32:
            selected = (l0, folded)

    if selected is None:
        print("\nNo l0 in the sweep met the accuracy targets.")
        return

    l0, folded = selected
    print(f"\n{'=' * 100}")
    print(f"SELECTED: l0 = {l0:.0e} (smallest meeting bound <= {target_bound:.0e}, fp32 sim <= {target_fp32:.0e})")
    print(f"{'=' * 100}")
    print("_CANS_4TH_ROOT_COEFFS = (")
    for beta, alpha in folded:
        print(f"    ({beta!r}, {alpha!r}),")
    print(")")

    print(f"\nmatrix check on device={FLAGS.device} (rel spectral err vs float64 eigh ref; max |Z^2 A Z^2 - I|):")
    for n in (64, 256):
        for cond in (1e2, 1e4, 1e6):
            rel_err, whiten = matrix_check(folded, n, cond, torch.float32, FLAGS.device)
            print(f"  fp32  n={n:>4} cond={cond:.0e}:  rel_err={rel_err:.3e}  whiten_dev={whiten:.3e}")
    design_cond = 0.9 / (FLAGS.safety * l0)
    for n in (64, 256):
        rel_err, whiten = matrix_check(folded, n, design_cond, torch.float64, FLAGS.device)
        print(f"  fp64  n={n:>4} cond={design_cond:.0e}:  rel_err={rel_err:.3e}  whiten_dev={whiten:.3e}")


if __name__ == "__main__":
    flags.DEFINE_list("l0_values", ["1e-4", "1e-5", "1e-6", "1e-7"], "Design interval lower bounds to sweep.")
    flags.DEFINE_integer("steps", 10, "Number of schedule steps.")
    flags.DEFINE_float("safety", 1.01, "Spectral safety margin folded into the coefficients.")
    flags.DEFINE_float("root_margin", 1.02, "Minimum gap between the polynomial root and the interval top.")
    flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device for the matrix-level check.")
    app.run(main)
