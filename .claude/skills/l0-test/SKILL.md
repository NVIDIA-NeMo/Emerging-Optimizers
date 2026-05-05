---
name: l0-test
description: Reproduce all L0 PR CI checks locally for the Emerging-Optimizers repo — pre-commit lint, L0 CPU tests, and L0 GPU tests inside the NGC pytorch container. Use before opening or updating a PR to catch CI failures locally.
allowed-tools: Bash
---

# L0 Test (PR CI reproduction)

Runs the same checks that gate a pull request in `.github/workflows/cicd-main.yml` and `code-linting.yml`:

1. **Lint** — `pre-commit run --all-files` (ruff check, ruff format, mypy, copyright/EOF/whitespace, no-underscore-md).
2. **L0 CPU** — `tests/ci/L0_Tests_CPU.sh` (distributed muon utils at `nproc=4,8`, plus `test_scalar_optimizers.py` and `test_procrustes_step.py`).
3. **L0 GPU** — `tests/ci/L0_Tests_GPU.sh` inside the NGC pytorch image declared in `docker/Dockerfile.ci` (read its `FROM` line — do **not** hard-code a tag, so the skill picks up version bumps automatically), with `--gpus all`. Runs every `tests/test_*.py` (excluding `*_cpu.py`) twice — once with a random seed, once with `--seed=42` — then the convergence `tests/convergence/*_test.py` set.

Not covered: the `copyright-check` external workflow (uses `NVIDIA-NeMo/FW-CI-templates`) and L1 long-running convergence runs.

## Args

Default (no args): run all three stages in order, stopping on first failure.

- `lint` — only stage 1.
- `cpu` — only stage 2.
- `gpu` — only stage 3.
- `nogpu` — stages 1 + 2 (skip the GPU container).
- `keepgoing` — don't stop on the first failed stage; run all and report at the end.
- `affected` — instead of running every test in stages 2 and 3, only run tests associated with files changed vs `main` (see below). Stage 1 always runs the full lint regardless. If no tests are affected, stages 2 and 3 print `no affected tests` and exit 0.

Args can be combined, e.g. `affected gpu`, `cpu keepgoing`.

### `affected` — change-aware test scoping

Compute the set of changed Python files vs `main` (committed-since-branch + working-tree + untracked) and reduce that to a set of test files:

```bash
# All Python files changed on this branch vs main, plus working-tree and untracked.
mapfile -t changed < <(
    {
        git diff --name-only main...HEAD --
        git diff --name-only HEAD --
        git ls-files --others --exclude-standard
    } | sort -u | grep '\.py$' || true
)

tests_to_run=()

# 1. Test files that were changed directly: run them as-is.
for f in "${changed[@]}"; do
    case "$f" in
        tests/test_*.py|tests/convergence/*_test.py) tests_to_run+=("$f") ;;
    esac
done

# 2. Source files under emerging_optimizers/: find tests that import them.
for f in "${changed[@]}"; do
    case "$f" in
        emerging_optimizers/*.py)
            mod=${f%.py}; mod=${mod//\//.}     # path → dotted module
            # Match `from <mod>` (submodule import), `import <mod>`, and prefix imports of submodules.
            mapfile -t hits < <(grep -lE "(^|[^.\w])(from[[:space:]]+${mod//./\\.}|import[[:space:]]+${mod//./\\.})([[:space:]]|\.|$)" tests/test_*.py tests/convergence/*_test.py 2>/dev/null || true)
            tests_to_run+=("${hits[@]}")
            ;;
    esac
done

# Dedupe.
mapfile -t tests_to_run < <(printf '%s\n' "${tests_to_run[@]}" | sort -u)
```

Then in stage 2 / stage 3, iterate over `${tests_to_run[@]}` instead of the full `find` results. Filtering rules:

- Stage 2 (CPU) runs only the affected files in `{tests/test_distributed_muon_utils_cpu.py, tests/test_scalar_optimizers.py, tests/test_procrustes_step.py}`. The distributed test runs at both `nproc=4,8` only if it's in the affected set.
- Stage 3 (GPU) runs the intersection of `tests_to_run` with `tests/test_*.py` (excluding `*_cpu.py`) twice (random + `--seed=42`), and the intersection with `tests/convergence/*_test.py` once at `--seed=42`.
- If `tests_to_run` is empty after filtering, that stage prints `no affected tests` and exits 0.

**Caveats:**

- The import-grep is purely syntactic. It won't catch tests that pull in a module via `__init__.py` re-exports (e.g. `from emerging_optimizers.orthogonalized_optimizers import muon` triggered by a change to `mop.py` — these `__init__.py` files do `from .mop import *`). To compensate: when an `__init__.py` is changed, treat every test that imports the package as affected. The grep already does this since the import lines reference the package path.
- Changes to `emerging_optimizers/registry.py` or `emerging_optimizers/mixin.py` realistically affect every optimizer test. Don't try to be clever — if the grep finds many tests, run them all.
- Changes outside `emerging_optimizers/` and `tests/` (e.g. `pyproject.toml`, `docker/`, `.github/`, `docs/`) do not map to any test; `affected` will report `no affected tests`. Use the unscoped form before merging if those changed.

## Prerequisites

- `uv` installed on host (https://docs.astral.sh/uv/).
- `docker` with the `nvidia` runtime configured, plus `--gpus all` access. Verify with `docker info | grep nvidia` and `nvidia-smi -L`.
- The NGC pytorch image referenced by `docker/Dockerfile.ci`'s `FROM` line (~22 GB). Pulled on demand if missing; an `nvcr.io` login may be required. Resolve the tag at runtime, do not hard-code it:

  ```bash
  IMAGE=$(awk 'tolower($1)=="from" {print $2; exit}' docker/Dockerfile.ci)
  ```

## Execution

### Stage 1 — Lint

```bash
uv run pre-commit run --all-files --show-diff-on-failure --color=always
```

This matches `.github/workflows/code-linting.yml` exactly. Pre-commit runs the hooks defined in `.pre-commit-config.yaml`: ruff (`--fix` then isort-only `--fix` then ruff-format), mypy (scoped to `emerging_optimizers/`), end-of-file-fixer, trailing-whitespace, and the local `no-underscore-md` hook.

### Stage 2 — L0 CPU tests

Run on the host (no container needed). Don't `bash tests/ci/L0_Tests_CPU.sh` directly — that script generates JUnit XML and coverage files we don't need locally. Inline the equivalent invocations and track failures:

```bash
export TORCH_COMPILE_DISABLE=1
failed=()
for n in 8 4; do
    # Use `python -m torch.distributed.run` rather than the bare `torchrun` entry point.
    # In a `--system-site-packages` uv venv with torch in --no-install-package, `torchrun`
    # often resolves to /usr/local/bin/torchrun, which spawns /usr/bin/python3 (system),
    # which doesn't see the editable emerging_optimizers install.
    python -m torch.distributed.run --nproc_per_node=$n tests/test_distributed_muon_utils_cpu.py -v -2
    [[ ${PIPESTATUS[0]} -eq 0 ]] || failed+=("test_distributed_muon_utils_cpu (n=$n)")
done
for t in tests/test_scalar_optimizers.py tests/test_procrustes_step.py; do
    python "$t" --device=cpu -v -2 || failed+=("$t")
done
```

Note: stage 2 must run inside the NGC container, not on the host — `tests/test_distributed_muon_utils_cpu.py` imports `numpy`, which is pre-installed in the NGC image but not pulled in by `uv sync` against a host venv.

### Stage 3 — L0 GPU tests (NGC container)

Same idea inside the NGC container — skip the shell script's XML/coverage scaffolding and run python directly. Resolve the image tag from `docker/Dockerfile.ci` so a `FROM` bump there flows through automatically:

```bash
IMAGE=$(awk 'tolower($1)=="from" {print $2; exit}' docker/Dockerfile.ci)
docker run --rm --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  -e PIP_CONSTRAINT="" \
  -e UV_PROJECT_ENVIRONMENT=/opt/venv \
  -e PATH="/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  -e TORCH_COMPILE_DISABLE=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  "$IMAGE" \
  bash -euc '
    curl -LsSf https://astral.sh/uv/0.9.3/install.sh | sh >/dev/null 2>&1
    uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT" >/dev/null 2>&1
    uv sync --link-mode copy --locked --all-groups \
        --no-install-package absl-py --no-install-package torch --no-install-package triton \
        --no-install-package nvidia-cublas-cu12 --no-install-package nvidia-cuda-cupti-cu12 \
        --no-install-package nvidia-cuda-nvrtc-cu12 --no-install-package nvidia-cuda-runtime-cu12 \
        --no-install-package nvidia-cudnn-cu12 --no-install-package nvidia-cufft-cu12 \
        --no-install-package nvidia-cufile-cu12 --no-install-package nvidia-curand-cu12 \
        --no-install-package nvidia-cusolver-cu12 --no-install-package nvidia-cusparse-cu12 \
        --no-install-package nvidia-cusparselt-cu12 --no-install-package nvidia-nccl-cu12 \
        --no-install-package nvidia-nvjitlink-cu12 --no-install-package nvidia-nvtx-cu12 >/dev/null 2>&1

    failed=()
    # all tests/test_*.py except *_cpu.py, with random seed
    for t in $(find tests -maxdepth 1 -type f -name "test_*.py" ! -name "*_cpu.py" | sort); do
        python "$t" --device=cuda -v -2 || failed+=("$t (random seed)")
    done
    # then again with --seed=42
    for t in $(find tests -maxdepth 1 -type f -name "test_*.py" ! -name "*_cpu.py" | sort); do
        python "$t" --device=cuda --seed=42 -v -2 || failed+=("$t (seed=42)")
    done
    # convergence suite
    for t in $(find tests/convergence -type f -name "*_test.py" | sort); do
        python "$t" --device=cuda --seed=42 -v -2 || failed+=("$t (seed=42)")
    done

    if (( ${#failed[@]} )); then
        printf "FAIL: %s\n" "${failed[@]}" >&2
        exit 1
    fi
  '
```

The container setup mirrors `docker/Dockerfile.ci`:

- `PIP_CONSTRAINT=""` clears the NGC image's pip constraint that breaks uv resolution.
- `--system-site-packages` makes the uv venv inherit the container's torch + CUDA stack.
- The `--no-install-package` allowlist prevents uv from clobbering torch, triton, absl-py, or any `nvidia-*-cu12` wheel. **If a new dep pulls in a CUDA wheel, extend this list and update `docker/Dockerfile.ci`.**
- `--link-mode copy` because bind-mount layout makes hardlinks fail.
- `TORCH_COMPILE_DISABLE`, `CUDA_VISIBLE_DEVICES`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` are set on the docker invocation since we're not sourcing `L0_Tests_GPU.sh`.

## Reporting

For each stage, surface only what failed. Don't dump full passing logs; absl already prints `[OK]/[FAIL]` per test.

- Stage 1: pre-commit's own diff on failure.
- Stages 2 and 3: aggregate the `failed` array; print one line per failed test file (and which seed it was running under) plus the absl tail with the actual error. If everything passed, a one-line `L0 CPU OK` / `L0 GPU OK` is enough.

No JUnit XML, no `coverage run` wrapping — those are for CI's reporting pipeline, not local iteration. If the user asks for coverage afterward, they can rerun under `coverage run -p` separately.

## Common gotchas

- **GPU not visible inside container** → check `docker info | grep -i runtime` shows `nvidia`, and that `--gpus all` is passed. The NGC image won't fall back to CPU.
- **`uv sync` reinstalling torch** → you forgot `--system-site-packages` on the venv create, or dropped a `--no-install-package` flag.
- **Tests fail only with `--seed=42`** → likely a tolerance issue. The L0 GPU script runs the suite twice (random + fixed) for exactly this reason. See history of `tests/test_sinkhorn.py` for a recent example (commit `b64b9b2` relaxed a tolerance).
- **Parameterized test selection failing** → absl synthesizes `_0/_1/...` suffixes for `parameterized.product` / `parameterized.parameters`; select by class name (`ClassName`), not bare method.
- **Pre-commit modifies files** → ruff and end-of-file-fixer auto-fix; rerun until clean.

## Do not

- Do not invoke tests via `pytest`; this repo uses `absl.testing` and depends on `--device` / `--seed` flags.
- Do not skip stage 1 just because tests pass — CI gates on lint independently.
- Do not run L1 GPU tests as part of this skill; they're long-running convergence runs and gated separately.
