# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`emerging_optimizers` is an NVIDIA research package implementing matrix-preconditioned and orthogonalization-based optimizers (Muon, SOAP, PSGD, Lion, Scion, etc.) for LLM training, plus supporting Triton kernels and EVD/Sinkhorn utilities. Python 3.12+, PyTorch 2.0+. Dependency management is **uv** (`uv.lock` is committed).

## Environment setup

The canonical environment is the NGC PyTorch container (`docker/Dockerfile.ci` builds from `nvcr.io/nvidia/pytorch:26.02-py3`); torch, triton, and the CUDA libs come from the base image, **not** uv. The Dockerfile is the source of truth for how to bring up an env that matches CI:

```
# Inside the NGC pytorch container:
export PIP_CONSTRAINT=""                  # NGC sets a pip constraint that breaks uv resolution
export UV_PROJECT_ENVIRONMENT=/opt/venv
uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT"
uv sync --link-mode symlink --locked --all-groups \
    --no-install-package absl-py --no-install-package torch \
    --no-install-package triton \
    --no-install-package nvidia-cublas-cu12 --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu12 --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cudnn-cu12 --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-cufile-cu12 --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-cusparselt-cu12 --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-nvjitlink-cu12 --no-install-package nvidia-nvtx-cu12
```

Key points:

- `--system-site-packages` is required so the venv inherits the container's torch/CUDA — without it `uv sync` will try to install incompatible CUDA wheels.
- The `--no-install-package` allowlist (torch, triton, all `nvidia-*-cu12` wheels, plus absl-py which the container also pins) prevents uv from clobbering the container's stack. If you add a dep that pulls in a new CUDA wheel, extend this list.
- `--all-groups` installs the `dev`, `test`, and `docs` groups together; use this rather than `uv sync` alone.
- `--link-mode symlink` is used because the bind-mount layout makes hardlinks fail.
- Outside the NGC container (e.g. local laptop), drop `--system-site-packages` and the `--no-install-package` flags; uv will install vanilla torch from PyPI.

## Common commands

Use `uv run <cmd>` so commands resolve against the locked environment.

- **Lint / format**: `uv run ruff check --fix .` and `uv run ruff format .` — CI does **not** auto-fix; pre-commit runs both.
- **Type check**: mypy is configured strictly for the `emerging_optimizers/` tree (`disallow_untyped_defs = True`); tests, benchmarks, docker, and docs are excluded. Pre-commit runs mypy with the same scope.
- **Pre-commit**: `uv run pre-commit run --all-files`. Note the `no-underscore-md` local hook — Markdown filenames must use hyphens, not underscores (except under `.github/`).
- **Add a dependency**: `uv add <pkg>` for required, `uv add --optional --extra <extra> <pkg>` for optional, `uv add --group <group> <pkg>` for dev/docs/test groups. Commit `uv.lock` and `pyproject.toml` together.

## Tests

Tests use `absl.testing` (not pytest) and accept absl flags. Standard form:

```
python tests/test_<name>.py --device={cpu|cuda} [--seed=42] -v -2 --xml_output_file=<path>.xml
```

Distributed CPU tests are launched via `torchrun`:

```
torchrun --nproc_per_node=<n> --no-python python tests/test_distributed_muon_utils_cpu.py --device=cpu -v -2
```

CI test suites are encoded as shell scripts in `tests/ci/`:

- `L0_Tests_CPU.sh` — all `tests/test_distributed_*_cpu.py` at 8 and 4 ranks (rekls also at 1 rank) + a couple of CPU-only tests.
- `L0_Tests_GPU.sh` — every `tests/test_*.py` (excluding `*_cpu.py`) on `--device=cuda`, run twice (random seed, then `--seed=42`), plus the `tests/convergence/*_test.py` set.
- `L1_Tests_GPU.sh` — convergence tests at `--seed=77` plus the `moe_c4_convergence.py` runs against muon/soap with a loss target.

Coverage is collected with `coverage run -p --source=emerging_optimizers ...` and aggregated; configuration in `pyproject.toml` (`[tool.coverage.*]`) excludes Triton-decorated code, `if TYPE_CHECKING`, and abstract methods.

Run a single test class/method using absl conventions, e.g. `python tests/test_orthogonalized_optimizer.py OrthogonalizedOptimizerTest.test_smoke --device=cpu -v -2`.

### Authoring tests

- **Every test file must define `--device` and `--seed` flags.** CI invokes every test under both `--device=cpu` and `--device=cuda`, and runs each GPU test twice (random seed, then `--seed=42` / `--seed=77`). Tests without these flags will be invoked with `--device=...` and fail at flag parsing. Match the existing pattern: `flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], ...)`, `flags.DEFINE_integer("seed", None, ...)`, and a `setUpModule` that seeds when `FLAGS.seed is not None`.
- **Don't override `torch.testing.assert_close`'s `msg`; append to it.** The default message contains the diff/atol/rtol summary, which is invaluable when debugging CI failures. Pass `msg=` as a callable that takes the default and returns a wrapped string — never a bare string that replaces the default. Canonical pattern (from the [PyTorch 2.11 docs](https://docs.pytorch.org/docs/2.11/testing.html#module-torch.testing)):

  ```python
  torch.testing.assert_close(
      actual, expected, msg=lambda msg: f"Header\n\n{msg}\n\nFooter"
  )
  ```

- **`atol` and `rtol` must be set together or not at all** in `torch.testing.assert_close`. Passing only one raises `ValueError`. Use both or neither (the default tolerances are dtype-aware).

## Architecture

### Package layout (`emerging_optimizers/`)

- `registry.py` — global `_OPTIMIZERS` dict with a `@register_optimizer("name")` decorator; `get_optimizer_cls` / `get_configured_optimizer_cls` provide lookup. **Important: a class is only in the registry once its module has been imported.** External callers must import the optimizer module (e.g. `from emerging_optimizers.orthogonalized_optimizers import muon`) before `get_optimizer_cls("muon")` will find it.
- `mixin.py` — `WeightDecayMixin` with four `weight_decay_method` modes: `decoupled`, `independent`, `l2`, `palm`. Optimizers that include this mixin set `self.weight_decay_method` and call `_apply_weight_decay_inplace(p, grad, lr, wd)` inside `step`.
- `orthogonalized_optimizers/` — Muon and friends (`muon`, `adaptive_muon`, `muon_hyperball`, `mop`, `polargrad`, `scion`, `sinkhorn_muon`, `spel`). All extend `OrthogonalizedOptimizer` (`orthogonalized_optimizer.py`), which implements the EMA-momentum + (optional Nesterov) + `scaled_orthogonalize_fn` pipeline. Subclasses customize behavior by overriding `orthogonalize(p, grad, **group_kwargs)` (e.g. for split-QKV preconditioning) or the `pre_weight_update_fn_inplace` / `post_weight_update_fn_inplace` hooks (used by hyperball-style updates that preserve weight norms). `muon_utils` holds the Newton–Schulz iteration coefficients and helpers, including the tensor-parallel `newton_schulz_tp`; `spectral_clipping_utils` implements spectral norm clipping.
- `soap/` — the SOAP family, sharing `soap_utils.py`: `soap.py` (`SOAP`, plus `StackedSoap` which optimizes batched/3D params through transient 2D stacking), `rekls.py` (REKLS), `tp_rekls.py` (experimental tensor-parallel REKLS), and `moso.py` (MOSO, momentum one-sided SOAP). Uses Kronecker-factored preconditioning and tracks eigenbases per-parameter.
- `psgd/` — `PSGDPro` (`psgd.py`, registered as `psgd_pro`) plus `psgd_kron_contractions.py` (Einsum contractions for triangular factors), `procrustes_step.py`, and `psgd_utils.py`.
- `scalar_optimizers/` — element-wise optimizers (`lion`, `signum`, `laprop`, `sim_ademamix`), defined in `__init__.py` on top of the `base.py` hierarchy (`SingleMomentumOptimizer` / `TwoMomentsOptimizer`); `update_functions/` holds the reusable per-algorithm update primitives (adam, lion, laprop, ademamix, madam, signum).
- `riemannian_optimizers/` — `normalized_optimizer.py` (manifold-normalized updates; `oblique_sgd`, `oblique_adam`).
- `contrib/` — contributed/experimental optimizers kept out of the registry (`muown.py`: `Muown`, a Muon variant).
- `triton_kernels/syrk.py` — Triton SYRK kernel used to accelerate Gram-matrix computations in preconditioner updates. Triton is an optional/test-only dep; gate imports.
- `utils/` — `eig.py` (eigendecomposition helpers), `sinkhorn_mapper.py` (`SinkhornMapper`), `modules.py`, and the `fp32_matmul_precision` context manager (sets `torch.set_float32_matmul_precision` for the duration of orthogonalization). `FP32MatmulPrecT = Literal["highest", "high", "medium"]` is the canonical type alias.

### Subclassing pattern (orthogonalized optimizers)

`OrthogonalizedOptimizer.step()` is the canonical loop:

1. `_init_group` lazily creates `momentum_buffer` per param.
2. Apply weight decay via the mixin.
3. EMA-update momentum (`lerp_(grad, 1 - momentum)`).
4. Optional Nesterov blend.
5. Wrap orthogonalization in `fp32_matmul_precision(self.fp32_matmul_prec)` and call `self.orthogonalize(p, grad, **group_kwargs)`.
6. Call `pre_weight_update_fn_inplace` → `p.add_(orth_grad, alpha=-lr)` → `post_weight_update_fn_inplace`.

Don't override `step` — override `orthogonalize` (and the pre/post hooks if you need norm preservation). The default `orthogonalize` only supports 2D parameters; conv weights must be reshaped (im2col) by the subclass.

### Tests structure

- Unit tests in `tests/test_*.py` (absl flags: `--device`, `--seed`).
- Convergence tests in `tests/convergence/` — file pattern `*_test.py` (note: not `test_*`); they're picked up only by L0/L1 GPU CI scripts.
- `tests/soap_reference.py` is a reference implementation, not a test.

## Conventions

- Use **abseil-py** instead of stdlib equivalents: `from absl import logging` (not `logging`), `absl.testing` (not `unittest`), `absl.flags` (not `argparse`).
- Line length 120 for Python (ruff configured to 119), 100 for C++.
- Follow Google Python style. "Mixed case" (camelCase, e.g. `getFoo`, `myVariable`) is disallowed — Google style already covers the rest. PyTorch idioms (`x`, `dX`, importing functions/classes directly) are allowed despite Google style.
- Markdown filenames must use hyphens (`my-doc.md`), not underscores — pre-commit hook will reject.
- Commits must be DCO-signed: `git commit -s ...`.

## CI specifics

CI runs in containers built via `docker/Dockerfile.ci`. Key env flags set by the test scripts: `TORCH_COMPILE_DISABLE=1` (CPU and L0 GPU scripts) and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` (GPU scripts) — preserve these when reproducing CI failures locally.

## Rules for agent

* Never write or modify comment including docstring unless explicitly asked.
* Never run tests unless explicitly asked.
