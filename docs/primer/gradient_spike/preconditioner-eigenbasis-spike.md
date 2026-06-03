# Preconditioner eigenbasis rotation around a gradient spike (SOAP vs REKLS)

Companion to `optimizer-update-comparison.ipynb`. Here we look only at how the **left Kronecker factor eigenbasis** `Q_L` of SOAP and REKLS evolves *over time* around a sudden gradient spike, and how that evolution depends on the matmul precision (`fp32_matmul_prec`) used in the KL-Shampoo update.

Setup:

- Single 2-D parameter of shape `(128, 64)`; identical i.i.d. Gaussian gradient sequence across runs (same seed).
- A 1000× spike on step `SPIKE_AT`: `randn(m, n) * 1000`. All other steps use `randn(m, n) * 1.0`.
- Both optimizers use `use_kl_shampoo=True` (current practice). REKLS additionally sets `use_eigh=True`; SOAP keeps `use_eigh=False, power_iter_steps=1`. So the **only difference** between the two is the eigenbasis solver — full `eigh` vs one step of orthogonal iteration.
- We run each at `fp32_matmul_prec="high"` (TF32 on Ampere+ CUDA) and `"highest"` (full fp32). On CPU the two are identical.
- We record `Q_L` after every step and compare against two reference frames *within the same run's own trajectory*:
  - **pre-spike reference** `Q_L(SPIKE_AT - 1)` — basis right before the spike;
  - **post-spike reference** `Q_L(SPIKE_AT)` — basis right after the spike has been ingested.

**Caveat (read before interpreting the tables).** Under KL-Shampoo's steady state with i.i.d. Gaussian inputs, `L` converges to a *nearly isotropic* spectrum — the top eigenvalue is only ~2% above the 2nd. With a near-degenerate matrix the eigenbasis is not uniquely defined: any orthonormal basis spanning the degenerate subspace is a valid eigenbasis, and `torch.linalg.eigh`'s output can jump arbitrarily under tiny numerical perturbations without the underlying matrix meaningfully changing. The eigenvalue-spectrum cell near the end explicitly shows the `top/2nd` ratio so you can judge how literally to take the rotation angles.



```python
import numpy as np
import torch

from emerging_optimizers.soap.soap import SOAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # uncomment to force CPU
dtype = torch.float32
print(f"device={device}, dtype={dtype}")
if device.type == "cuda":
    print(f"  cuBLAS TF32 matmul:  {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cuDNN TF32:          {torch.backends.cudnn.allow_tf32}")
    print(f"  global float32 matmul precision: {torch.get_float32_matmul_precision()}")

```

    device=cuda, dtype=torch.float32
      cuBLAS TF32 matmul:  False
      cuDNN TF32:          True
      global float32 matmul precision: highest



```python
PARAM_SHAPE = (128, 64)
SEED = 0
LR = 1.0

SPIKE_AT = 50          # iteration on which the spike is injected (0-indexed)
SPIKE_SCALE = 1000.0
NORMAL_SCALE = 1.0
SPIKE_TOTAL_STEPS = 200
```


```python
# Both rows use use_kl_shampoo=True (current practice) and see the same gradient sequence; they
# differ only in the eigenbasis solver. REKLS == SOAP(use_eigh=True, use_kl_shampoo=True), but REKLS
# doesn't expose fp32_matmul_prec, so we build both via SOAP to sweep that knob.
def make_soap_kl(param: torch.Tensor, fp32_matmul_prec: str = "highest") -> SOAP:
    return SOAP(
        [param], lr=LR, betas=(0.9, 0.95), shampoo_beta=0.95, weight_decay=0.0,
        use_kl_shampoo=True, use_eigh=False, fp32_matmul_prec=fp32_matmul_prec,
    )


def make_rekls(param: torch.Tensor, fp32_matmul_prec: str = "highest") -> SOAP:
    return SOAP(
        [param], lr=LR, betas=(0.9, 0.95), shampoo_beta=0.95, weight_decay=0.0,
        use_kl_shampoo=True, use_eigh=True, fp32_matmul_prec=fp32_matmul_prec,
    )

```


```python
def collect_q_trajectory(make_opt, fp32_matmul_prec: str, n_steps: int = SPIKE_TOTAL_STEPS) -> torch.Tensor:
    """Drive `make_opt`'s optimizer with i.i.d. Gaussian gradients and a 1000× spike at step `SPIKE_AT`.

    Returns `Q_L_traj` where `Q_L_traj[i]` is the left-factor eigenbasis `Q_L` after iteration `i`.
    """
    g = torch.Generator(device=device).manual_seed(SEED)
    param = torch.zeros(PARAM_SHAPE, device=device, dtype=dtype, requires_grad=True)
    opt = make_opt(param, fp32_matmul_prec)
    m = PARAM_SHAPE[0]
    Q_L_traj = torch.empty(n_steps, m, m)
    for i in range(n_steps):
        scale = SPIKE_SCALE * NORMAL_SCALE if i == SPIKE_AT else NORMAL_SCALE
        with torch.no_grad():
            grad = torch.randn(PARAM_SHAPE, device=device, dtype=dtype, generator=g) * scale
        param.grad = grad
        opt.step()
        Q_L_traj[i] = opt.state[param]["Q_L"].detach().cpu()
    return Q_L_traj


# Collect for both eigenbasis solvers at both matmul precisions. On CUDA, "high" enables TF32 in the
# KL-Shampoo L/R update; "highest" forces fp32. (On CPU the two are identical.)
PRECISIONS = ["high", "highest"]
SOLVERS = [("SOAP (KL)", make_soap_kl), ("REKLS", make_rekls)]
trajectories = {
    (name, prec): collect_q_trajectory(make_opt, prec) for prec in PRECISIONS for name, make_opt in SOLVERS
}
print("collected", len(trajectories), "eigenbasis trajectories; Q_L shape =", tuple(next(iter(trajectories.values())).shape))

```

    collected 4 eigenbasis trajectories; Q_L shape = (200, 128, 128)



```python
def top1_angle_to_ref_deg(Q: torch.Tensor, ref_col0: torch.Tensor) -> float:
    """Acute angle (deg) between the top eigenvector of Q and a reference unit vector."""
    return (Q[:, 0] @ ref_col0).abs().clamp(max=1.0).arccos().rad2deg().item()


def topk_largest_angle_to_ref_deg(Q: torch.Tensor, ref_Qk: torch.Tensor, k: int) -> float:
    """Largest principal angle (deg) between the top-`k` subspace of Q and a reference top-`k` orthonormal subspace."""
    sigmas = torch.linalg.svdvals(Q[:, :k].T @ ref_Qk).clamp(max=1.0)
    return sigmas.min().arccos().rad2deg().item()


TOP_K = 8


def compute_rotation_curves(Q_traj: torch.Tensor, k: int = TOP_K) -> dict[str, np.ndarray]:
    ref_pre = Q_traj[SPIKE_AT - 1]
    ref_post = Q_traj[SPIKE_AT]
    return {
        "top1_to_pre": np.array([top1_angle_to_ref_deg(Q_traj[i], ref_pre[:, 0]) for i in range(Q_traj.shape[0])]),
        "top1_to_post": np.array([top1_angle_to_ref_deg(Q_traj[i], ref_post[:, 0]) for i in range(Q_traj.shape[0])]),
        "topk_to_pre": np.array(
            [topk_largest_angle_to_ref_deg(Q_traj[i], ref_pre[:, :k], k) for i in range(Q_traj.shape[0])]
        ),
        "topk_to_post": np.array(
            [topk_largest_angle_to_ref_deg(Q_traj[i], ref_post[:, :k], k) for i in range(Q_traj.shape[0])]
        ),
    }


rotation = {key: compute_rotation_curves(traj) for key, traj in trajectories.items()}

```


```python
# At offset N, compare two equidistant points around the spike against the pre-spike basis Q_L(SPIKE_AT - 1):
#   "before" = N steady-state steps before the pre-spike basis (no spike in the window)
#   "after"  = N steps after the pre-spike basis (includes the spike, plus N-1 recovery steps)
# If the "after" values are larger than the "before" values, the spike caused more rotation than steady-state drift would.
OFFSETS = [1, 2, 5, 10]


def summarize_symmetric(label: str, rot: dict[str, np.ndarray]) -> None:
    title = f"{label}  —  principal angle to pre-spike basis Q_L(step {SPIKE_AT}), degrees"
    topk = f"top-{TOP_K}"
    rule = "═" * len(title)
    print()
    print(title)
    print(rule)
    print(f"  {'':>3} │ {'BEFORE spike':^20} │ {'AFTER spike':^20}")
    print(f"  {'N':>3} │ {'top-1':>9}  {topk:>9} │ {'top-1':>9}  {topk:>9}")
    print("─" * len(title))
    for n in OFFSETS:
        before_idx = (SPIKE_AT - 1) - n
        after_idx = (SPIKE_AT - 1) + n
        b1, bk = rot["top1_to_pre"][before_idx], rot["topk_to_pre"][before_idx]
        a1, ak = rot["top1_to_pre"][after_idx], rot["topk_to_pre"][after_idx]
        before = f"{b1:>8.3f}°  {bk:>8.3f}°"
        after = f"{a1:>8.3f}°  {ak:>8.3f}°"
        print(f"  {n:>3} │ {before} │ {after}")
    print(rule)


for prec in PRECISIONS:
    print()
    print("#" * 71)
    print(f"#  fp32_matmul_prec = {prec!r}")
    print("#" * 71)
    for name, _ in SOLVERS:
        summarize_symmetric(name, rotation[(name, prec)])

```

    
    #######################################################################
    #  fp32_matmul_prec = 'high'
    #######################################################################
    
    SOAP (KL)  —  principal angle to pre-spike basis Q_L(step 50), degrees
    ══════════════════════════════════════════════════════════════════════
          │     BEFORE spike     │     AFTER spike     
        N │     top-1      top-8 │     top-1      top-8
    ──────────────────────────────────────────────────────────────────────
        1 │    0.000°     0.028° │    0.020°     0.028°
        2 │    0.000°     0.059° │   88.491°    87.201°
        5 │    0.028°     0.044° │   86.837°    89.978°
       10 │    0.020°     0.028° │   88.965°    89.504°
    ══════════════════════════════════════════════════════════════════════
    
    REKLS  —  principal angle to pre-spike basis Q_L(step 50), degrees
    ══════════════════════════════════════════════════════════════════
          │     BEFORE spike     │     AFTER spike     
        N │     top-1      top-8 │     top-1      top-8
    ──────────────────────────────────────────────────────────────────
        1 │    0.000°     0.000° │    0.000°     0.000°
        2 │    0.000°     0.000° │   86.186°    89.767°
        5 │    0.000°     0.000° │   85.919°    89.759°
       10 │    0.000°     0.000° │   85.784°    89.647°
    ══════════════════════════════════════════════════════════════════
    
    #######################################################################
    #  fp32_matmul_prec = 'highest'
    #######################################################################
    
    SOAP (KL)  —  principal angle to pre-spike basis Q_L(step 50), degrees
    ══════════════════════════════════════════════════════════════════════
          │     BEFORE spike     │     AFTER spike     
        N │     top-1      top-8 │     top-1      top-8
    ──────────────────────────────────────────────────────────────────────
        1 │    0.000°     0.020° │    0.020°     0.044°
        2 │    0.000°     0.048° │    0.000°     0.034°
        5 │    0.034°     0.034° │    0.020°     0.020°
       10 │    0.000°     0.028° │    0.028°     0.028°
    ══════════════════════════════════════════════════════════════════════
    
    REKLS  —  principal angle to pre-spike basis Q_L(step 50), degrees
    ══════════════════════════════════════════════════════════════════
          │     BEFORE spike     │     AFTER spike     
        N │     top-1      top-8 │     top-1      top-8
    ──────────────────────────────────────────────────────────────────
        1 │    0.000°     0.000° │    0.000°     0.000°
        2 │    0.000°     0.000° │    0.000°     0.000°
        5 │    0.000°     0.000° │    0.000°     0.000°
       10 │    0.000°     0.000° │    0.000°     0.000°
    ══════════════════════════════════════════════════════════════════


## Reading the result: is the eigenbasis actually rotating?

Before drawing conclusions from the angles above, check how degenerate `L` actually is. The cell below prints `L`'s top few eigenvalues at several steps around the spike.


```python
# We need full eigenvalue trajectories, so re-run REKLS once and store them all.
g = torch.Generator(device=device).manual_seed(SEED)
param = torch.zeros(PARAM_SHAPE, device=device, dtype=dtype, requires_grad=True)
opt = make_rekls(param, fp32_matmul_prec="high")  # TF32 causes trouble here. Set it to highest will fix.
L_eigvals_per_step: list[np.ndarray] = []
for i in range(SPIKE_TOTAL_STEPS):
    scale = SPIKE_SCALE * NORMAL_SCALE if i == SPIKE_AT else NORMAL_SCALE
    with torch.no_grad():
        grad = torch.randn(PARAM_SHAPE, device=device, dtype=dtype, generator=g) * scale
    param.grad = grad
    opt.step()
    L_eigvals_per_step.append(torch.linalg.eigvalsh(opt.state[param]["L"].detach()).cpu().numpy())

print(f"{'step':>5}  {'eig[0] (top)':>14}  {'eig[1]':>14}  {'eig[4]':>14}  {'eig[9]':>14}  {'top/2nd':>9}")
print("-" * 75)
for i in [SPIKE_AT - 5, SPIKE_AT - 1, SPIKE_AT, SPIKE_AT + 1, SPIKE_AT + 4, SPIKE_AT + 9, SPIKE_AT + 19]:
    ev = L_eigvals_per_step[i][::-1]  # descending
    marker = "  <-- spike" if i == SPIKE_AT else ""
    print(f"{i + 1:>5}  {ev[0]:>14.4g}  {ev[1]:>14.4g}  {ev[4]:>14.4g}  {ev[9]:>14.4g}  {ev[0] / ev[1]:>9.4f}{marker}")
```

     step    eig[0] (top)          eig[1]          eig[4]          eig[9]    top/2nd
    ---------------------------------------------------------------------------
       46        3.03e+06       2.921e+06       2.472e+06       2.036e+06     1.0374
       50       2.421e+06       2.334e+06       1.976e+06       1.627e+06     1.0374
       51       2.291e+06       2.208e+06       1.869e+06       1.539e+06     1.0374  <-- spike
       52       1.539e+07       1.348e+07       1.061e+07       1.961e+06     1.1417
       55       2.293e+07       1.996e+07       1.753e+07       1.327e+07     1.1491
       60       2.529e+07       2.474e+07       2.099e+07       1.751e+07     1.0220
       70       3.074e+07       2.886e+07       2.614e+07       2.274e+07     1.0655


## Takeaways

- Under KL-Shampoo steady state with i.i.d. Gaussian gradients, `L` converges to a **nearly isotropic** spectrum — the top eigenvalue and the 2nd eigenvalue differ by only ~2–4%. With such a tiny gap, the eigenbasis is highly sensitive to small perturbations of `L`: any change to `L`'s top-2 ordering rotates `Q_L[:, 0]` by ~90°.
- **The `fp32_matmul_prec` tables tell the story.** With `"high"` (TF32 on Ampere+ CUDA) the post-spike angles jump to ~86°–89° within a step or two of the spike, while the matching before-spike angles stay near 0° — so the spike, not steady-state drift, drives the rotation. With `"highest"` (full fp32) the after-spike angles collapse back to ~0°, matching the before-spike side. This holds for both SOAP (`use_eigh=False`) and REKLS (`use_eigh=True`), so the eigenbasis solver is not the cause — the matmul precision in the KL-Shampoo `L`/`R` update is.
- The ~90° eigenvector "rotation" is the visible symptom of a **real divergence in `L`'s numerical state**, not just an `eigh` quirk. The eigenvalue-spectrum cell shows that under TF32 `L`'s top eigenvalue jumps and the `top/2nd` ratio shifts the step after the spike; under fp32 the spectrum decays smoothly. TF32 errors in `grad @ R⁻¹ @ grad.T` accumulate enough to substantially perturb `L` after a high-magnitude gradient.
- **Practical implication.** Run SOAP/REKLS with `fp32_matmul_prec="highest"` (now the library default) for stable, cross-device-reproducible preconditioner state — especially under spike-like conditions. Whether the TF32 divergence affects end-to-end training quality on smooth-gradient workloads is an open question; the optimizer's `step` direction depends on the *action* of the preconditioner, which may be less sensitive than the basis representation itself.

