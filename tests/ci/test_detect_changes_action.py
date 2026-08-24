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


def _collection_script() -> str:
    action = ACTION.read_text(encoding="utf-8")
    start = action.index("run: |\n") + len("run: |\n")
    end = action.index('echo "Changed files:"', start)
    return textwrap.dedent(action[start:end])


def _run_collection(
    tmp_path: Path,
    *,
    metadata: str,
    files: str,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh.log"
    metadata_file = tmp_path / "metadata"
    files_file = tmp_path / "files"
    metadata_file.write_text(metadata, encoding="utf-8")
    files_file.write_text(files, encoding="utf-8")
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        f"""#!/bin/sh
printf '%s\\n' \"$*\" >> {shlex.quote(str(log))}
for arg in \"$@\"; do
  case \"$arg\" in
    repos/acme/hermes/pulls/7/files*) cat {shlex.quote(str(files_file))}; exit 0 ;;
    repos/acme/hermes/pulls/7) cat {shlex.quote(str(metadata_file))}; printf '%s\\n' ''; exit 0 ;;
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
