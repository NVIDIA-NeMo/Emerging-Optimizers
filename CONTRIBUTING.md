# Contributing To Emerging-Optimizers

## Dependency

Use [abseil-py](https://github.com/abseil/abseil-py/tree/main)'s **logging**, **testing** and **flags** instead of Python's own **logging**, **unittest** and **argparse**.

We use [uv](https://docs.astral.sh/uv/) for managing dependencies. For reproducible builds, our project tracks the generated `uv.lock` file in the repository.
On a weekly basis, the CI attemps an update of the lock file to test against upstream dependencies.

New required dependencies can be added by `uv add $DEPENDENCY`.

New optional dependencies can be added by `uv add --optional --extra $EXTRA $DEPENDENCY`.

`EXTRA` refers to the subgroup of extra-dependencies to which you're adding the new dependency.
Example: For adding a TRT-LLM specific dependency, run `uv add --optional --extra trtllm $DEPENDENCY`.

New dependencies to a dependency group can be added with `uv add --group $GROUP $DEPENDENCY`. Dependency groups are specific to uv and are also optional dependencies but not intended to be used with the package such as docs dependencies.

Alternatively, the `pyproject.toml` file can also be modified directly.

Adding a new dependency will update UV's lock-file. Please check this into your branch:

```bash
git add uv.lock pyproject.toml
git commit -m "build: Adding dependencies"
git push
```

### 🧹 Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. CI does not auto-fix linting and formatting issues, but most issues can be fixed by running the following command:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Coding Style

We generally follow [Google's style guides](https://google.github.io/styleguide/) , with some exceptions:

* Line length extended to 120 for Python and 100 for C++ code.
* Common use in PyTorch, which are prohibited by Google style, are allowed, including but not limited to:
  * Import function, class, not just module
  * Some special variable name, `x`, `dX`, etc.
* Allow common capitalized naming in Triton code.

Although common, **mixed case is not allowed** in any code.

Run pre-commit at local before submitting merge request. You can also read [.pre-commit-config.yaml]( .pre-commit-config.yaml) to understand what are being forced. The **flake8** and **mypy** settings are inherited from PyTorch.

## Test

All tests should be placed under [tests](tests). We aim for 100% test coverage for this tiny project.

We use [abseil-py](https://github.com/abseil/abseil-py/tree/main) **testing** because it is easier to launch multi process than alternatives.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```
