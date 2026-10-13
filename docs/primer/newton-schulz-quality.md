# Newton–Schulz orthogonalization quality

This doc summarizes quality of orthogonalization by Newton-Schulz compared polar decomposition (QDWH).

## Run settings

| Setting | Value |
| --- | --- |
| Matrix shape | `(3072, 3584)`. From Kimi-K3 MoE layers |
| Inputs | FP32 Gaussian `N(0, 1)`, uniform `[0, 1)`, Hilbert, and dense logspace |
| Logspace input | `U @ diag(s) @ V.T`; independent Gaussian QR factors `U`, `V`; `s` logarithmically spaced from `1e-30` to `1e10`; FP64 reconstruction cast to FP32 |
| Steps | 5, 8, 10, 16; `cubic5` at 5 only |
| Polar Express schedule | Upstream `repeat_penultimate` coefficient iterator (main `4ebce60`) |
| NS precision | FP32 (`highest`) and BF16 (`medium`); FP32 normalization and returned output |
| BF16 reduced-precision reduction | `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` |
| Reference | `torch.linalg.polar` in FP64 on the transposed input; factor transposed back |

```bash
python -m benchmarks.newton_schulz_quality \
  --steps=5,8,10,16 --spectra=gaussian,uniform,hilbert,logspace \
  --output=/tmp/newton-schulz-quality.json \
  --plot=/tmp/newton-schulz-quality.png
```

## Results

Dense logspace input after FP32 conversion: measured singular values `1.54` to `1e+10`, condition number `6.47e+09`.

For original input `A`, NS output `X`, and FP64 reference polar factors `A = H Q`:

$$
\begin{aligned}
\text{Relative polar error} &= \frac{\lVert X-Q\rVert_F}{\lVert Q\rVert_F} \\
\text{Relative reconstruction error} &= \frac{\lVert HX-A\rVert_F}{\lVert A\rVert_F}
\end{aligned}
$$

![Relative polar and reconstruction errors for Gaussian, uniform, Hilbert, and dense logspace inputs. Solid lines: FP32. Dashed lines: BF16.](../assets/newton-schulz-quality.png)
