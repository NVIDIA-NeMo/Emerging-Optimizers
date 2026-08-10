# Gradient spikes and the SOAP preconditioner

A set of executed-notebook walkthroughs of how different optimizers respond to a sudden gradient spike, and how that exposed a TF32 precision bug in the SOAP/REKLS KL-Shampoo preconditioner. Read in order:

1. **Optimizer update comparison** — per-step update magnitude and spike-recovery behavior of AdamW, LaProp, Muon, SOAP, and REKLS.
2. **Preconditioner eigenbasis rotation** — how the SOAP/REKLS eigenbasis `Q_L` rotates around a spike, and how it depends on `fp32_matmul_prec`.
3. **TF32 eigenvalue precision loss** — a standalone demo of the underlying `diag(Qᵀ L Q)` precision failure that drives the rotation.

```{toctree}
:caption: Gradient spike
:hidden:

optimizer-update-comparison.md
preconditioner-eigenbasis-spike.md
tf32-eigval-precision-loss.md
```
