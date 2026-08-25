"""Regression tests for pull-request changed-file collection."""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION = REPO_ROOT / ".github" / "actions" / "detect-changes" / "action.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yaml"
HISTORY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "history-check.yml"


def _collection_script() -> str:
    action = ACTION.read_text(encoding="utf-8")
    start = action.index("run: |\n") + len("run: |\n")
    end = action.index('echo "Changed files:"', start)
    return textwrap.dedent(action[start:end])


def _history_check_script() -> str:
    workflow = HISTORY_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("        run: |\n") + len("        run: |\n")
    end = workflow.index("\n\n      - name: Upload review status artifact", start)
    return textwrap.dedent(workflow[start:end])


def _run_collection(
    tmp_path: Path,
    *,
    metadata: str,
    metadata_after_files: str | None = None,
    files: str,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh.log"
    metadata_file = tmp_path / "metadata"
    metadata_after_file = tmp_path / "metadata-after"
    metadata_calls_file = tmp_path / "metadata-calls"
    files_file = tmp_path / "files"
    metadata_file.write_text(metadata, encoding="utf-8")
    metadata_after_file.write_text(metadata_after_files or "", encoding="utf-8")
    files_file.write_text(files, encoding="utf-8")
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        f"""#!/bin/sh
printf '%s\\n' \"$*\" >> {shlex.quote(str(log))}
for arg in \"$@\"; do
  case \"$arg\" in
    repos/acme/hermes/pulls/7/files*) cat {shlex.quote(str(files_file))}; exit 0 ;;
    repos/acme/hermes/pulls/7)
      calls=0
      [ -f {shlex.quote(str(metadata_calls_file))} ] && calls=$(cat {shlex.quote(str(metadata_calls_file))})
      calls=$((calls + 1))
      printf '%s' "$calls" > {shlex.quote(str(metadata_calls_file))}
      if [ "$calls" -gt 1 ] && [ -s {shlex.quote(str(metadata_after_file))} ]; then
        cat {shlex.quote(str(metadata_after_file))}
      else
        cat {shlex.quote(str(metadata_file))}
      fi
      printf '%s\\n' ''
      exit 0 ;;
    repos/acme/hermes/compare/*) printf '%s\\n' 'old-path.py'; exit 0 ;;
  esac
done
exit 1
""",
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "REPO": "acme/hermes",
            "EVENT_NAME": "pull_request",
            "BASE_SHA": "a" * 40,
            "HEAD_SHA": "b" * 40,
            "PR_NUMBER": "7",
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            _collection_script() + '\nprintf \'CHANGED_BEGIN\\n%sCHANGED_END\\n\' "$CHANGED"\n',
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )


def _log(tmp_path: Path) -> str:
    return (tmp_path / "gh.log").read_text(encoding="utf-8")


def test_collection_uses_all_pull_request_file_pages(tmp_path: Path) -> None:
    result = _run_collection(
        tmp_path,
        metadata=f'{"a" * 40}\t{"b" * 40}\t2',
        files="one.py\ntwo.md\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split("CHANGED_BEGIN\n", 1)[1] == "one.py\ntwo.mdCHANGED_END\n"
    assert "::warning::" not in result.stdout
    assert "/pulls/7/files?per_page=100" in _log(tmp_path)
    assert "/compare/" not in _log(tmp_path)


@pytest.mark.parametrize(
    "metadata",
    [
        f'{"a" * 40}\t{"b" * 40}\t3001',
        f'{"a" * 40}\t{"c" * 40}\t2',
    ],
)
def test_collection_fails_open_on_unusable_pr_file_metadata(
    tmp_path: Path,
    metadata: str,
) -> None:
    result = _run_collection(
        tmp_path,
        metadata=metadata,
        files="should-not-be-used.py\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("CHANGED_BEGIN\nCHANGED_END\n")
    assert "::warning::" in result.stdout
    assert "/pulls/7/files?per_page=100" not in _log(tmp_path)
    assert "/compare/" not in _log(tmp_path)


def test_collection_fails_open_if_pr_changes_after_file_pagination(tmp_path: Path) -> None:
    result = _run_collection(
        tmp_path,
        metadata=f'{"a" * 40}\t{"b" * 40}\t2',
        metadata_after_files=f'{"a" * 40}\t{"c" * 40}\t2',
        files="stale.py\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("CHANGED_BEGIN\nCHANGED_END\n")
    assert "changed during file collection" in result.stdout


def test_detector_timeout_covers_paginated_file_collection() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    detect_block = workflow.split("  detect:\n", 1)[1].split("\n  # ", 1)[0]

    assert "timeout-minutes: 6" in detect_block


def test_contributor_attribution_runs_for_every_pull_request() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    contributor_block = workflow.split("  contributor-check:\n", 1)[1].split("\n  uv-lockfile:", 1)[0]

    assert "if: needs.detect.outputs.event_name == 'pull_request'" in contributor_block
    assert "if: needs.detect.outputs.python == 'true'" not in contributor_block


def test_history_check_uses_the_pull_request_base_sha() -> None:
    workflow = HISTORY_WORKFLOW.read_text(encoding="utf-8")
    ci_workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "base_sha:" in workflow
    assert "head_sha:" in workflow
    assert "ref: ${{ inputs.head_sha }}" in workflow
    assert 'PR_BASE_SHA: ${{ inputs.base_sha }}' in workflow
    assert 'git merge-base "$PR_BASE_SHA" HEAD' in workflow
    assert 'head_sha: ${{ github.event.pull_request.head.sha }}' in ci_workflow
    assert "origin/main" not in workflow


def test_history_check_rejects_unrelated_pull_request_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git("add", "base.txt")
    git("commit", "-qm", "base")
    base_sha = git("rev-parse", "HEAD")

    git("checkout", "--orphan", "unrelated")
    git("rm", "-q", "-rf", ".")
    (repo / "head.txt").write_text("unrelated\n", encoding="utf-8")
    git("add", "head.txt")
    git("commit", "-qm", "unrelated head")

    github_output = tmp_path / "github-output"
    env = os.environ.copy()
    env.update({"GITHUB_OUTPUT": str(github_output), "PR_BASE_SHA": base_sha})
    result = subprocess.run(
        ["bash", "-c", _history_check_script()],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "no common ancestor" in result.stdout
    assert '"unrelated histories"' in github_output.read_text(encoding="utf-8")
