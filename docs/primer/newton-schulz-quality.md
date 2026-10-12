# Newton–Schulz quality

## Run settings

| Setting | Value |
| --- | --- |
| Matrix shape | `(3072, 3584)`. From Kimi-K3 MoE layers |
| Inputs | FP32 Gaussian `N(0, 1)`, uniform `[0, 1)`, and Hilbert |
| Steps | 5, 8, 10, 16; `cubic5` at 5 only |
| NS precision | FP32 (`highest`) and BF16 (`medium`); FP32 normalization and returned output |
| BF16 reduced-precision reduction | `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` |
| Reference | `torch.linalg.polar` in FP64 on the transposed input; factor transposed back |

```bash
python -m benchmarks.newton_schulz_quality \
  --steps=5,8,10,16 --spectra=gaussian,uniform,hilbert \
  --output=/tmp/newton-schulz-quality.json \
  --plot=/tmp/newton-schulz-quality.png
```

## Results

For NS output $X$ and reference $Q$, relative polar error is $\|X-Q\|_F / \|Q\|_F$; orthogonality error is $\|XX^T-I_{3072}\|_F / \sqrt{3072}$.

![Relative polar and orthogonality errors for Gaussian, uniform, and Hilbert inputs. Solid lines: FP32. Dashed lines: BF16.](../assets/newton-schulz-quality.png)

