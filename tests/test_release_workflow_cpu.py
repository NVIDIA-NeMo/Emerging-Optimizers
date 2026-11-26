# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["pyyaml>=6.0,<7"]
# ///
"""Exercise release guards without credentials, uploads, or package builds."""

import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load(next((ROOT / ".github/workflows").glob("release.y*ml")).read_text())
JOBS = WORKFLOW["jobs"]
FREEZE_JOBS = yaml.safe_load((ROOT / ".github/workflows/release-freeze.yml").read_text())["jobs"]


def evaluate(expression: str, context: dict[str, object]) -> bool:
    """Evaluate the boolean and equality subset used by release job guards."""
    source = expression.strip().removeprefix("${{").removesuffix("}}")
    source = source.replace("!cancelled()", "True").replace("!failure()", "True")
    source = re.sub(r"\b(?:github|inputs|needs)\.[\w.-]+", lambda match: repr(context[match[0]]), source)
    source = source.replace("&&", " and ").replace("||", " or ")
    source = re.sub(r"!(?!=)", "not ", source)
    return bool(eval(" ".join(source.split()), {"__builtins__": {}}, {}))


class ReleaseGuards(unittest.TestCase):
    """Cover publication eligibility and unsuccessful or skipped dependencies."""

    def setUp(self) -> None:
        """Create a successful manual release context for the canonical repo."""
        self.context = {
            "github.repository": "NVIDIA-NeMo/Emerging-Optimizers",
            "github.event_name": "workflow_dispatch",
            "inputs.dry-run": False,
            "needs.validate-inputs.result": "success",
            "needs.release-workflow-tests.result": "success",
            "needs.bump.result": "success",
            "needs.build-wheel.result": "success",
            "needs.publish-wheel.result": "success",
        }

    def test_publish_requires_manual_non_dry_run(self) -> None:
        """Pushes and dry runs cannot request the production upload job."""
        for event in ("push", "workflow_dispatch"):
            for dry_run in (False, True):
                with self.subTest(event=event, dry_run=dry_run):
                    context = dict(self.context, **{"github.event_name": event, "inputs.dry-run": dry_run})
                    self.assertEqual(
                        evaluate(JOBS["publish-wheel"]["if"], context), event == "workflow_dispatch" and not dry_run
                    )
        self.context["github.repository"] = "example/fork"
        self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_missing_build_blocks_upload(self) -> None:
        """Failed, cancelled and skipped builds cannot reach publication."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.build-wheel.result"] = state
                self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_regression_failures_block_version_preparation(self) -> None:
        """Require the regression job to succeed before version preparation."""
        self.assertIn("release-workflow-tests", JOBS["bump"]["needs"])
        self.assertTrue(evaluate(JOBS["bump"]["if"], self.context))
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.release-workflow-tests.result"] = state
                self.assertFalse(evaluate(JOBS["bump"]["if"], self.context))

    def test_real_finalize_requires_upload(self) -> None:
        """A missing upload cannot finalize a real release; dry runs may continue."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.publish-wheel.result"] = state
                self.context["inputs.dry-run"] = False
                self.assertFalse(evaluate(JOBS["finalize"]["if"], self.context))
                self.context["inputs.dry-run"] = True
                self.assertTrue(evaluate(JOBS["finalize"]["if"], self.context))

    def test_summary_rejects_missing_required_jobs(self) -> None:
        """Run the actual shell summary against failures and permitted skips."""
        script = JOBS["release-summary"]["steps"][0]["run"]
        for publish in (False, True):
            baseline = {name: {"result": "success"} for name in JOBS["release-summary"]["needs"]}
            baseline["publish-wheel"]["result"] = "success" if publish else "skipped"
            for job in (
                "release-workflow-tests",
                "validate-inputs",
                "bump",
                "build-wheel",
                "publish-wheel",
                "finalize",
            ):
                for state in ("success", "failure", "cancelled", "skipped"):
                    with self.subTest(publish=publish, job=job, state=state):
                        needs = {name: dict(value) for name, value in baseline.items()}
                        needs[job]["result"] = state
                        env = dict(
                            os.environ,
                            NEEDS=json.dumps(needs),
                            PUBLISH_EXPECTED=str(publish).lower(),
                            RELEASE_SKIPPED="false",
                        )
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True
                        )
                        expected = state == ("skipped" if job == "publish-wheel" and not publish else "success")
                        self.assertEqual(result.returncode == 0, expected, result.stderr.decode())

    def test_validate_ref_treats_input_as_data(self) -> None:
        """Only full immutable SHAs pass; branch names and shell payloads fail."""
        script = JOBS["validate-inputs"]["steps"][0]["run"]
        for ref in ("a" * 40, "main", "abc1234", "$(exit 0)", "a" * 40 + "\n"):
            with self.subTest(ref=ref):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env=dict(os.environ, RELEASE_REF=ref),
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, ref == "a" * 40)

    def test_upload_has_no_checkout_or_password(self) -> None:
        """Keep source execution and build credentials outside the OIDC job."""
        self.assertEqual(JOBS["build-wheel"]["permissions"], {"contents": "read"})
        self.assertNotIn("secrets", JOBS["build-wheel"])
        upload = JOBS["publish-wheel"]
        self.assertEqual(upload["permissions"], {"id-token": "write"})
        self.assertEqual(upload["environment"]["name"], "pypi")
        self.assertEqual(len(upload["steps"]), 2)
        for step in upload["steps"]:
            self.assertNotIn("run", step)
            self.assertNotIn("password", step.get("with", {}))
            self.assertNotIn("checkout", step["uses"])

    def test_dispatch_ref_guard_preserves_push_rehearsals(self) -> None:
        """Reject manual runs on non-main refs before the version bump."""
        guard = next(step for step in JOBS["validate-inputs"]["steps"] if step["name"] == "Require dispatch from main")
        for event in ("push", "workflow_dispatch"):
            for ref in ("refs/heads/main", "refs/heads/r0.5.0", "refs/heads/pull-request/123", "refs/tags/main"):
                with self.subTest(event=event, ref=ref):
                    rejected = evaluate(guard["if"], {"github.event_name": event, "github.ref": ref})
                    self.assertEqual(rejected, event == "workflow_dispatch" and ref != "refs/heads/main")
        self.assertNotEqual(subprocess.run(["bash", "-c", guard["run"]], capture_output=True).returncode, 0)


class ReleaseCommitGuards(unittest.TestCase):
    """Run the real source validation steps against local GitHub API fixtures."""

    target_sha = "2" * 40
    candidate = "3" * 40
    api = """
gh() {
  [[ $1 == api && $3 == --jq && $# == 4 ]] || return 1
  case "$2" in
    "repos/$GITHUB_REPOSITORY/branches/$TARGET_BRANCH") reply=$BRANCH_REPLY ;;
    "repos/$GITHUB_REPOSITORY/compare/$EXPECTED_COMPARE") reply=$COMPARE_REPLY ;;
    *) return 1 ;;
  esac
  [[ -n "$reply" ]] || return 1
  jq -r "$4" <<< "$reply"
}
"""

    def run_step(self, step, **inputs):
        """Execute a workflow step, returning its exit status and Actions outputs."""
        env = dict(
            os.environ,
            GITHUB_REPOSITORY="NVIDIA-NeMo/Emerging-Optimizers",
            RELEASE_REF=self.candidate,
            TARGET_BRANCH="r0.7.0",
            BRANCH_REPLY=json.dumps({"protected": True, "commit": {"sha": self.target_sha}}),
            COMPARE_REPLY=json.dumps({"status": "ahead"}),
            EXPECTED_COMPARE=f"{self.candidate}...{self.target_sha}",
        )
        env.update(inputs)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.touch()
            env["GITHUB_OUTPUT"] = str(output)
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", self.api + step["run"]], env=env, capture_output=True
            )
            return result.returncode, output.read_text()

    def test_release_target_ancestry(self):
        """Only ancestors of the protected target pass."""
        step = next(s for s in JOBS["validate-inputs"]["steps"] if s["name"].startswith("Verify release commit"))
        self.assertEqual(step["if"], "github.event_name == 'workflow_dispatch'")
        for status in ("ahead", "identical", "behind", "diverged"):
            with self.subTest(status=status):
                code, _ = self.run_step(step, COMPARE_REPLY=json.dumps({"status": status}))
                self.assertEqual(code == 0, status in {"ahead", "identical"})
        for bad_reply in ("", json.dumps({"protected": False, "commit": {"sha": self.target_sha}})):
            self.assertNotEqual(self.run_step(step, BRANCH_REPLY=bad_reply)[0], 0)
        self.assertNotEqual(self.run_step(step, COMPARE_REPLY="")[0], 0)

    def test_release_target_uses_the_maintainer_branch_rule(self):
        """Only main and branches covered by the release rule qualify as targets."""
        step = next(s for s in JOBS["validate-inputs"]["steps"] if s["name"].startswith("Verify release commit"))
        for branch in ("main", "r0.7.0", "v0.2.0"):
            with self.subTest(branch=branch):
                self.assertEqual(self.run_step(step, TARGET_BRANCH=branch)[0], 0)
        for branch in ("v1-train-client", "r0.7.0-preview", "r1x2x3", "topic/prepare-release"):
            with self.subTest(branch=branch):
                self.assertNotEqual(self.run_step(step, TARGET_BRANCH=branch)[0], 0)

    def test_freeze_uses_main_as_the_source_branch(self):
        """Emerging-Optimizers opts into the shared workflow's cutoff policy for main."""
        inputs = FREEZE_JOBS["code-freeze"]["with"]
        self.assertEqual(inputs["freeze-source-branch"], "main")
        self.assertEqual(inputs["freeze-commit"], "${{ inputs.freeze-commit }}")

    def test_release_jobs_use_the_protected_app(self):
        """Scope GitHub writes to the release environment and declare its key explicitly."""
        for job in (FREEZE_JOBS["code-freeze"], JOBS["bump"], JOBS["finalize"]):
            with self.subTest(workflow=job["uses"]):
                self.assertEqual(job["with"]["release-environment"], "emerging-optimizers-release")
                self.assertNotIn("app-id", job["with"])
                self.assertEqual(job["secrets"]["RELEASE_APP_PRIVATE_KEY"], "${{ secrets.RELEASE_APP_PRIVATE_KEY }}")
                self.assertNotIn("BOT_KEY", job["secrets"])
                self.assertNotIn("PAT", job["secrets"])
        self.assertEqual(JOBS["bump"]["permissions"], {"contents": "read"})
        self.assertEqual(JOBS["finalize"]["permissions"], {"contents": "read", "pull-requests": "read"})


class ContainerBuildRevision(unittest.TestCase):
    """Verify that the CI container and its mounted checkout test the same vetted revision."""

    jobs = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())["jobs"]
    head = "1" * 40
    merge = "2" * 40
    base = "3" * 40

    def resolve(self, event="push", ref="pull-request/42", **responses):
        """Run the actual resolver with metadata fixtures and no GitHub connection."""
        step = next(step for step in self.jobs["cicd-container-build"]["steps"] if step.get("id") == "revision")
        api = """
gh() {
  [[ $1 == api && $3 == --jq && $# == 4 ]] || return 1
  case "$2" in
    "repos/$GITHUB_REPOSITORY/pulls/42") response=$PR ;;
    "repos/$GITHUB_REPOSITORY/git/commits/$MERGE_SHA") response=$COMMIT ;;
    *) return 1 ;;
  esac
  [[ -n "$response" ]] || return 1
  jq -r "$4" <<< "$response"
}
"""
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.touch()
            env = dict(
                os.environ,
                GITHUB_OUTPUT=str(output),
                GITHUB_REPOSITORY="example/repo",
                GITHUB_EVENT_NAME=event,
                GITHUB_REF_NAME=ref,
                GITHUB_SHA=self.head,
                MERGE_SHA=self.merge,
                PR=json.dumps({"head": {"sha": self.head}, "merge_commit_sha": self.merge}),
                COMMIT=json.dumps({"parents": [{"sha": self.base}, {"sha": self.head}]}),
            )
            env.update(responses)
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", api + step["run"]], env=env, capture_output=True
            )
            return result.returncode, output.read_text()

    def test_resolve_mirror_and_default_branch(self):
        """Use the synthetic merge for mirrors and the event SHA for pushes to main and schedules."""
        self.assertEqual(self.resolve(), (0, f"sha={self.merge}\n"))
        for event in ("push", "schedule"):
            with self.subTest(event=event):
                self.assertEqual(self.resolve(event=event, ref="main", PR="", COMMIT=""), (0, f"sha={self.head}\n"))

    def test_mirror_metadata_must_verify(self):
        """Return no revision if either API fails, the mirror is stale or the merge cannot be verified."""
        for responses in (
            {"PR": ""},
            {"COMMIT": ""},
            {"PR": json.dumps({"head": {"sha": self.base}, "merge_commit_sha": self.merge})},
            {"PR": json.dumps({"head": {"sha": self.head}, "merge_commit_sha": None})},
            {"PR": json.dumps({"head": {"sha": self.head}, "merge_commit_sha": "123"})},
            {"COMMIT": json.dumps({"parents": [{"sha": self.head}]})},
            {"COMMIT": json.dumps({"parents": [{"sha": self.head}, {"sha": self.base}]})},
        ):
            with self.subTest(responses=responses):
                status, output = self.resolve(**responses)
                self.assertNotEqual(status, 0)
                self.assertEqual(output, "")

    def test_build_and_test_use_verified_outputs(self):
        """Bind checkout and mounted sources to the selected SHA and tests to the built image digest."""
        build = self.jobs["cicd-container-build"]
        checkout = next(step for step in build["steps"] if step.get("name") == "Checkout")
        self.assertEqual(checkout["with"]["ref"], build["outputs"]["tested_sha"])
        self.assertEqual(build["outputs"]["image_digest"], "${{ steps.image.outputs.digest }}")
        tested_sha = "${{ needs.cicd-container-build.outputs.tested_sha }}"
        for name in ("cicd-unit-tests", "cicd-nightly-tests"):
            with self.subTest(job=name):
                job = self.jobs[name]
                self.assertIn("needs.cicd-container-build.result == 'success'", job["if"])
                checkout, tests = job["steps"][:2]
                self.assertEqual(checkout["with"]["ref"], tested_sha)
                self.assertEqual(tests["with"]["checkout-ref"], tested_sha)
                self.assertTrue(
                    tests["with"]["container-image"].endswith(
                        "@${{ needs.cicd-container-build.outputs.image_digest }}"
                    )
                )
        template = yaml.safe_load((ROOT / ".github/actions/test-template/action.yml").read_text())
        mounted = next(step for step in template["runs"]["steps"] if step["name"] == "Checkout repository")
        self.assertEqual(mounted["with"]["ref"], "${{ inputs.checkout-ref }}")


class WorkflowCredentials(unittest.TestCase):
    """Keep the catalog reader, release identity and reviewer credential in their own jobs."""

    def test_mirror_guardwords_uses_only_its_deploy_key(self):
        """Check the head SHA before handing the vetted mirror to the pinned scanner."""
        workflows = ROOT / ".github/workflows"
        guard = yaml.load((workflows / "guardwords-check.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(guard["on"], {"push": {"branches": ["pull-request/[0-9]+"]}})
        self.assertEqual(guard["permissions"], {"contents": "read", "pull-requests": "read"})
        resolve = guard["jobs"]["pr_info"]
        verify = resolve["steps"][-1]
        self.assertIn("steps.pr.outputs.pr-info).head.sha", verify["env"]["PR_HEAD_SHA"])
        self.assertEqual(verify["run"], 'test "$PR_HEAD_SHA" = "$GITHUB_SHA"')
        scan = guard["jobs"]["guardwords-check"]
        self.assertEqual(scan["needs"], "pr_info")
        self.assertEqual(scan["secrets"], {"GUARDWORDS_DEPLOY_KEY": "${{ secrets.GUARDWORDS_DEPLOY_KEY }}"})
        self.assertRegex(scan["uses"], r"/_guardwords_check\.yml@[a-f0-9]{40}$")

    def test_coverage_status_and_queue_credentials(self):
        """Use the native status permission and keep classic deployment review on the main dispatcher."""
        workflows = ROOT / ".github/workflows"
        ci = yaml.safe_load((workflows / "cicd-main.yml").read_text())
        coverage = ci["jobs"]["Coverage_Fake"]
        self.assertEqual(coverage["permissions"], {"statuses": "write"})
        self.assertNotIn("BOT_KEY", str(coverage))
        self.assertNotIn("write-all", str(ci))
        queue = yaml.safe_load((workflows / "cicd-approve-test-queue.yml").read_text())["jobs"]["approve-queue"]
        self.assertEqual(queue["if"], "github.ref == 'refs/heads/main'")
        self.assertEqual(queue["environment"], "main")
        review = next(step for step in queue["steps"] if step.get("name") == "Approve waiting deployments")
        self.assertEqual(review["env"]["READ_TOKEN"], "${{ github.token }}")
        self.assertEqual(review["env"]["APPROVAL_TOKEN"], "${{ secrets.PAT }}")
        self.assertIn('item["environment"]["name"] == "test"', review["run"])


if __name__ == "__main__":
    unittest.main()
