"""Regression tests for contributor-check baseline validation."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "contributor-check.yml"


def _baseline_script() -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("run: |\n") + len("run: |\n")
    end = workflow.index(
        "# Find any new author emails after the event's real baseline.",
        start,
    )
    return textwrap.dedent(workflow[start:end])


def _git_repo(tmp_path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
    )
    (tmp_path / "file.txt").write_text("root\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "file.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "root"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _run_baseline_check(
    tmp_path: Path,
    *,
    event_name: str,
    baseline: str,
) -> subprocess.CompletedProcess[str]:
    output = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": event_name,
            "PR_BASE_SHA": baseline if event_name == "pull_request" else "",
            "PUSH_BEFORE_SHA": baseline if event_name == "push" else "",
            "GITHUB_OUTPUT": str(output),
        }
    )
    return subprocess.run(
        ["bash", "-c", _baseline_script()],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("event_name", ["pull_request", "push"])
@pytest.mark.parametrize("baseline", ["HEAD", "0", "1" * 39, "0" * 41])
def test_malformed_baseline_fails_closed(
    tmp_path: Path,
    event_name: str,
    baseline: str,
) -> None:
    _git_repo(tmp_path)

    result = _run_baseline_check(
        tmp_path,
        event_name=event_name,
        baseline=baseline,
    )

    assert result.returncode == 2, result.stderr


def test_exact_all_zero_push_baseline_is_initial_push(tmp_path: Path) -> None:
    _git_repo(tmp_path)

    result = _run_baseline_check(
        tmp_path,
        event_name="push",
        baseline="0" * 40,
    )

    assert result.returncode == 0, result.stderr


def test_valid_commit_sha_is_accepted_for_pull_request(tmp_path: Path) -> None:
    root_sha = _git_repo(tmp_path)

    result = _run_baseline_check(
        tmp_path,
        event_name="pull_request",
        baseline=root_sha,
    )

    assert result.returncode == 0, result.stderr
