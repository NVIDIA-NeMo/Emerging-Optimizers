---
name: l0-test
description: Reproduce all L0 PR CI checks locally for the Emerging-Optimizers repo — pre-commit lint on host, then all CPU and GPU tests inside the NGC pytorch container. Use before opening or updating a PR to catch CI failures locally.
allowed-tools: Bash
---

# L0 Test (PR CI reproduction)

Runs the same checks that gate a pull request in `.github/workflows/cicd-main.yml` and `code-linting.yml`:

1. **Lint** (host) — `pre-commit run --all-files` (ruff check, ruff format, mypy, copyright/EOF/whitespace, no-underscore-md).
2. **L0 CPU tests** (container) — distributed muon utils at `nproc=4,8`, plus `test_scalar_optimizers.py` and `test_procrustes_step.py`.
3. **L0 GPU tests** (container) — every `tests/test_*.py` (excluding `*_cpu.py`) twice — once with a random seed, once with `--seed=42` — then the convergence `tests/convergence/*_test.py` set.

Stages 2 and 3 share a single NGC container session: lint runs on the host, then a single `docker run` against the image declared in `docker/Dockerfile.ci` (the `FROM` line — do **not** hard-code a tag, so the skill picks up version bumps automatically) with `--gpus all` performs both CPU and GPU passes. **All test execution happens inside the container; never run tests on the host.** Reasons:

- `tests/test_distributed_muon_utils_cpu.py` imports `numpy`, which is preinstalled in the NGC image but not pulled in by `uv sync` against a host venv.
- The NGC torch build (`2.11.0a0+...nv26.02`) and CUDA stack are what CI uses — host torch may differ in version, build flags, or CUDA major.
- `torchrun` resolution differs between host and container; we use `python -m torch.distributed.run` everywhere so it's deterministic.

Not covered: the `copyright-check` external workflow (uses `NVIDIA-NeMo/FW-CI-templates`) and L1 long-running convergence runs.

## Args

Default (no args): run all three stages in order, stopping on first failure.

- `lint` — only stage 1 (host).
- `cpu` — only stage 2 (container, CPU tests).
- `gpu` — only stage 3 (container, GPU tests).
- `nogpu` — stages 1 + 2 (skip GPU tests; container still launches for stage 2).
- `keepgoing` — don't stop on the first failed stage; run all and report at the end.
- `affected` — instead of running every test in stages 2 and 3, only run tests associated with files changed vs `main` (see below). Stage 1 always runs the full lint regardless. If no tests are affected, stages 2 and 3 print `no affected tests` and exit 0.

Args can be combined, e.g. `affected gpu`, `cpu keepgoing`.

### `affected` — change-aware test scoping

Compute the set of changed Python files vs `main` (committed-since-branch + working-tree + untracked) and reduce that to a set of test files. Run this on the host before launching the container, then pass the resolved list in via env or argv:

```bash
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
            mapfile -t hits < <(grep -lE "(^|[^.\w])(from[[:space:]]+${mod//./\\.}|import[[:space:]]+${mod//./\\.})([[:space:]]|\.|$)" tests/test_*.py tests/convergence/*_test.py 2>/dev/null || true)
            tests_to_run+=("${hits[@]}")
            ;;
    esac
done

mapfile -t tests_to_run < <(printf '%s\n' "${tests_to_run[@]}" | sort -u | grep -v '^$' || true)
```

Filtering rules inside the container:

- Stage 2 (CPU) runs only the affected files in `{tests/test_distributed_muon_utils_cpu.py, tests/test_scalar_optimizers.py, tests/test_procrustes_step.py}`. The distributed test runs at both `nproc=4,8` only if it's in the affected set.
- Stage 3 (GPU) runs the intersection of `tests_to_run` with `tests/test_*.py` (excluding `*_cpu.py`) twice (random + `--seed=42`), and the intersection with `tests/convergence/*_test.py` once at `--seed=42`.
- If `tests_to_run` is empty after filtering, that stage prints `no affected tests` and exits 0.

**Caveats:**

- The import-grep is purely syntactic. It won't catch tests that pull in a module via `__init__.py` re-exports (e.g. `from emerging_optimizers.orthogonalized_optimizers import muon` triggered by a change to `mop.py` — these `__init__.py` files do `from .mop import *`). To compensate: when an `__init__.py` is changed, treat every test that imports the package as affected. The grep already does this since the import lines reference the package path.
- Changes to `emerging_optimizers/registry.py` or `emerging_optimizers/mixin.py` realistically affect every optimizer test. Don't try to be clever — if the grep finds many tests, run them all.
- Changes outside `emerging_optimizers/` and `tests/` (e.g. `pyproject.toml`, `docker/`, `.github/`, `docs/`) do not map to any test; `affected` will report `no affected tests`. Use the unscoped form before merging if those changed.

## Prerequisites

- `uv` installed on host (https://docs.astral.sh/uv/) — only used by stage 1 (`uv run pre-commit ...`).
- `docker` with the `nvidia` runtime configured, plus `--gpus all` access. Verify with `docker info | grep nvidia` and `nvidia-smi -L`.
- The NGC pytorch image referenced by `docker/Dockerfile.ci`'s `FROM` line (~22 GB). Pulled on demand if missing; an `nvcr.io` login may be required.

**Don't hard-code versions in this skill.** The image tag, the uv version, and the `--no-install-package` allowlist are all defined in `docker/Dockerfile.ci`. Resolve them at runtime so a Dockerfile bump flows through automatically:

```bash
IMAGE=$(awk 'tolower($1)=="from" {print $2; exit}' docker/Dockerfile.ci)
UV_VERSION=$(awk '$1=="ARG" && $2 ~ /^UV_VERSION=/ {sub(/^UV_VERSION=/, "", $2); print $2; exit}' docker/Dockerfile.ci)
NO_INSTALL_FLAGS=$(grep -oE -- '--no-install-package [a-zA-Z0-9._-]+' docker/Dockerfile.ci | sort -u | tr '\n' ' ')
```

If a value isn't in `docker/Dockerfile.ci`, prefer reading it from `pyproject.toml`/`uv.lock` or relying on whatever's installed in the container — never paste a literal version string into this skill.

## Execution

### Stage 1 — Lint (host)

```bash
uv run pre-commit run --all-files --show-diff-on-failure --color=always
```

This matches `.github/workflows/code-linting.yml` exactly. Pre-commit runs the hooks defined in `.pre-commit-config.yaml`: ruff (`--fix` then isort-only `--fix` then ruff-format), mypy (scoped to `emerging_optimizers/`), end-of-file-fixer, trailing-whitespace, and the local `no-underscore-md` hook. This is the only stage that runs on the host.

### Stages 2 & 3 — All tests inside the NGC container

A single `docker run` performs the complete uv setup (per `docker/Dockerfile.ci`) and then executes the CPU and GPU test passes inline. Skip the `tests/ci/L0_Tests_*.sh` wrappers — they generate JUnit XML and coverage files that aren't useful locally. The script below tracks failures in a `failed=()` array and exits non-zero if any test failed.

Resolve `IMAGE`, `UV_VERSION`, and `NO_INSTALL_FLAGS` from `docker/Dockerfile.ci` (see Prerequisites) and pass them in:

```bash
docker run --rm --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  -e PIP_CONSTRAINT="" \
  -e UV_PROJECT_ENVIRONMENT=/opt/venv \
  -e PATH="/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  -e TORCH_COMPILE_DISABLE=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e UV_VERSION="$UV_VERSION" \
  -e NO_INSTALL_FLAGS="$NO_INSTALL_FLAGS" \
  "$IMAGE" \
  bash -euc '
    # ----- venv setup (mirror of docker/Dockerfile.ci) -----
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh >/dev/null 2>&1
    uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT" >/dev/null 2>&1
    # NO_INSTALL_FLAGS is intentionally word-split here.
    uv sync --link-mode copy --locked --all-groups $NO_INSTALL_FLAGS >/dev/null 2>&1

    failed=()

    # ----- Stage 2: L0 CPU tests -----
    # Use `python -m torch.distributed.run` instead of bare `torchrun`. With --system-site-packages
    # and torch in --no-install-package, `torchrun` resolves to /usr/local/bin/torchrun and spawns
    # /usr/bin/python3 (system), which lacks the editable emerging_optimizers install.
    for n in 8 4; do
        python -m torch.distributed.run --nproc_per_node=$n tests/test_distributed_muon_utils_cpu.py -v -2
        [[ ${PIPESTATUS[0]} -eq 0 ]] || failed+=("CPU dist n=$n: tests/test_distributed_muon_utils_cpu.py")
    done
    for t in tests/test_scalar_optimizers.py tests/test_procrustes_step.py; do
        python "$t" --device=cpu -v -2 || failed+=("CPU: $t")
    done

    # ----- Stage 3: L0 GPU tests -----
    for t in $(find tests -maxdepth 1 -type f -name "test_*.py" ! -name "*_cpu.py" | sort); do
        python "$t" --device=cuda -v -2 || failed+=("GPU random: $t")
    done
    for t in $(find tests -maxdepth 1 -type f -name "test_*.py" ! -name "*_cpu.py" | sort); do
        python "$t" --device=cuda --seed=42 -v -2 || failed+=("GPU seed=42: $t")
    done
    for t in $(find tests/convergence -type f -name "*_test.py" | sort); do
        python "$t" --device=cuda --seed=42 -v -2 || failed+=("GPU conv seed=42: $t")
    done

    if (( ${#failed[@]} )); then
        printf "FAIL: %s\n" "${failed[@]}" >&2
        exit 1
    fi
  '
```

When `cpu` or `gpu` is selected, drop the corresponding block from the inline script. When `affected` is selected, replace the bare `for t in $(find ...)` lists with iteration over the resolved `tests_to_run` array (passed in via env or by templating it into the script).

The container setup mirrors `docker/Dockerfile.ci`:

- `PIP_CONSTRAINT=""` clears the NGC image's pip constraint that breaks uv resolution.
- `--system-site-packages` makes the uv venv inherit the container's torch + CUDA stack.
- `$NO_INSTALL_FLAGS` (resolved from `docker/Dockerfile.ci`) prevents uv from clobbering torch, triton, absl-py, or any `nvidia-*-cu12` wheel. **If a new dep pulls in a CUDA wheel, extend the `--no-install-package` lines in `docker/Dockerfile.ci` — this skill picks up the change automatically.**
- `--link-mode copy` because bind-mount layout makes hardlinks fail.
- `TORCH_COMPILE_DISABLE`, `CUDA_VISIBLE_DEVICES`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` are set on the docker invocation since we're not sourcing the `tests/ci/L0_Tests_*.sh` scripts.

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

- **Do not run any test on the host** — always inside the NGC container. The host venv may not have numpy, may have a different torch build, and `torchrun` resolves differently. Stage 1 lint is the only host-side step.
- Do not invoke tests via `pytest`; this repo uses `absl.testing` and depends on `--device` / `--seed` flags.
- Do not skip stage 1 just because tests pass — CI gates on lint independently.
- Do not run L1 GPU tests as part of this skill; they're long-running convergence runs and gated separately.
