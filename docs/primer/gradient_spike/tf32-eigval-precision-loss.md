# TF32 `Q^T L Q` precision loss — standalone demo

Build a symmetric matrix `L` with a realistic wide-dynamic-range spectrum (top eigenvalue ~10⁶, smallest ~10⁻¹ — span of 7 orders of magnitude). Take its exact eigendecomposition `L = Q diag(λ) Q^T`, then try to *recover* `λ` by computing `diag(Q^T L Q)` under different matmul precisions.



```python
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # uncomment to force CPU
dtype = torch.float32
print(f"device={device}, dtype={dtype}")
if device.type == "cuda":
    print(f"  cuBLAS TF32 matmul:  {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  global float32 matmul precision: {torch.get_float32_matmul_precision()}")
else:
    print("  (running on CPU — fp32 'high' and 'highest' are identical, so this notebook won't show a precision difference unless run on CUDA.)")

```

    device=cuda, dtype=torch.float32
      cuBLAS TF32 matmul:  False
      global float32 matmul precision: highest


## Construct `L` with a realistic SOAP-like spectrum

Pick `m = 128` and assign true eigenvalues:

- Top 5 nearly degenerate around ~2.5e6 (mimicking KL-Shampoo's near-isotropic dominant subspace).
- The remaining 123 decay log-uniformly down to ~0.1 (the tail).

Then generate a random orthogonal `Q` via QR of a Gaussian, and form `L = Q diag(λ) Q^T`. We build in fp64 for clean ground truth, then cast to fp32 for the test. By construction, computing `diag(Q^T L Q)` should return `λ` exactly (any deviation is matmul precision error).


```python
m = 128

# True eigenvalues: top 5 near-degenerate around 2.5e6, tail decays log-uniformly to 0.1.
true_eigvals_f64 = torch.empty(m, dtype=torch.float64)
top = 2.5e6
true_eigvals_f64[:5] = torch.tensor([top, top * 0.98, top * 0.96, top * 0.94, top * 0.92], dtype=torch.float64)
true_eigvals_f64[5:] = torch.logspace(np.log10(top * 0.90), -1.0, m - 5, dtype=torch.float64)

# Random orthogonal Q via QR of a Gaussian matrix (built in fp64 for cleanliness).
g = torch.Generator().manual_seed(0)
A = torch.randn(m, m, generator=g, dtype=torch.float64)
Q_f64, _ = torch.linalg.qr(A)

# Assemble L = Q diag(λ) Q^T in fp64, symmetrize, then cast to fp32 on device.
L_f64 = Q_f64 @ torch.diag(true_eigvals_f64) @ Q_f64.T
L_f64 = (L_f64 + L_f64.T) / 2

L = L_f64.to(dtype).to(device)
Q = Q_f64.to(dtype).to(device)
true_eigvals = true_eigvals_f64.to(dtype)

print(f"L: shape={tuple(L.shape)}, dtype={L.dtype}, device={L.device}")
print(f"Spectrum:")
print(f"  top 5     : {true_eigvals[:5].tolist()}")
print(f"  bottom 5  : {true_eigvals[-5:].tolist()}")
print(f"  dynamic range: {(true_eigvals[0] / true_eigvals[-1]).item():.2e}")
```

    L: shape=(128, 128), dtype=torch.float32, device=cuda:0
    Spectrum:
      top 5     : [2500000.0, 2450000.0, 2400000.0, 2350000.0, 2300000.0]
      bottom 5  : [0.1742028146982193, 0.1516321748495102, 0.13198591768741608, 0.1148851215839386, 0.10000000149011612]
      dynamic range: 2.50e+07


## Compute `diag(Q^T L Q)` under different matmul precisions

Two ways to control matmul precision in PyTorch:

- `torch.set_float32_matmul_precision("highest")` — full fp32 (24-bit mantissa).
- `torch.set_float32_matmul_precision("high")` — TF32 on Ampere+ CUDA (10-bit mantissa); identical to `"highest"` on CPU.

Run on CUDA to see the TF32 precision loss. On CPU both runs produce the same output.



```python
def diag_QtLQ(L_in: torch.Tensor, Q_in: torch.Tensor) -> torch.Tensor:
    """Compute diag(Q.T @ L @ Q) without materializing the full product (matches SOAP's `eig_utils.conjugate(..., diag=True)`)."""
    QtL = Q_in.T @ L_in
    return (QtL * Q_in.T).sum(dim=-1)


def run_at_precision(prec: str) -> torch.Tensor:
    """Run the diag(Q.T L Q) computation under a particular matmul precision setting."""
    prev_global = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if prec == "highest":
        # Belt-and-suspenders: also flip the cuBLAS knob to ensure no TF32 on CUDA.
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision(prec)
    try:
        return diag_QtLQ(L, Q).cpu()
    finally:
        torch.set_float32_matmul_precision(prev_global)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


eigvals_highest = run_at_precision("highest")  # full fp32
eigvals_high = run_at_precision("high")        # TF32 on CUDA, fp32 on CPU

print(f"{'idx':>4}  {'true':>14}  {'fp32 (highest)':>16}  {'fp32 (high/TF32)':>18}")
print("-" * 60)
for i in [0, 1, 4, 20, 50, 80, 100, 115, 120, 124, 125, 126, 127]:
    print(
        f"  {i:>3}  {true_eigvals[i].item():>14.5g}  {eigvals_highest[i].item():>16.5g}  "
        f"{eigvals_high[i].item():>18.5g}"
    )

```

     idx            true    fp32 (highest)    fp32 (high/TF32)
    ------------------------------------------------------------
        0         2.5e+06           2.5e+06          2.4999e+06
        1        2.45e+06          2.45e+06          2.4501e+06
        4         2.3e+06           2.3e+06          2.2999e+06
       20      2.8069e+05        2.8069e+05          2.8067e+05
       50          4368.3            4368.3              4366.8
       80          67.983            67.979              75.758
      100          4.2376            4.2453             -5.1326
      115         0.52865           0.52618               16.15
      120         0.26415           0.26051              4.6133
      124         0.15163           0.15539             -12.385
      125         0.13199           0.13418             -1.5987
      126         0.11489           0.11431             -27.317
      127             0.1           0.10366              -6.226

