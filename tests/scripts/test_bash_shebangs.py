"""Tests for the portable bash shebang checker."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-bash-shebangs.py"


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


def test_missing_explicit_path_fails_closed_and_reports_path(tmp_path):
    missing = tmp_path / "missing.sh"
    result = run_checker(str(missing))
    assert result.returncode == 2
    assert str(missing) in result.stderr
    assert "does not exist" in result.stderr


def test_markdown_only_fixture_is_checked(tmp_path):
    markdown = tmp_path / "guide.md"
    bad_shebang = "#!" + "/bin/bash\n"
    markdown.write_text(f"```bash\n{bad_shebang}```\n", encoding="utf-8")
    result = run_checker(str(markdown))
    assert result.returncode == 1
    assert "guide.md:2" in result.stdout


def test_markdown_extension_is_checked(tmp_path):
    markdown = tmp_path / "guide.markdown"
    bad_shebang = "#!" + "/bin/bash\n"
    markdown.write_text(f"```bash\n{bad_shebang}```\n", encoding="utf-8")
    result = run_checker(str(markdown))
    assert result.returncode == 1
    assert "guide.markdown:2" in result.stdout


def test_bare_inline_allowance_is_rejected(tmp_path):
    script = tmp_path / "bare-ok.sh"
    script.write_text(
        "# shebang: ok\n" + "#!" + "/bin/bash\n", encoding="utf-8"
    )
    result = run_checker(str(script))
    assert result.returncode == 1
    assert "bare-ok.sh:2" in result.stdout


def test_allowlist_entries_require_a_reason(tmp_path):
    script = tmp_path / "shim.sh"
    script.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("shim.sh\n", encoding="utf-8")
    result = run_checker("--allowlist", str(allowlist), str(script))
    assert result.returncode == 2
    assert "trailing # reason" in result.stderr


def test_allowlist_entries_require_a_repo_relative_path(tmp_path):
    script = tmp_path / "shim.sh"
    script.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("shim.sh # local shim\n", encoding="utf-8")
    result = run_checker("--allowlist", str(allowlist), str(script))
    assert result.returncode == 2
    assert "repo-relative POSIX glob" in result.stderr


def test_reasoned_repo_relative_allowlist_can_skip_a_specific_file():
    fixture = REPO_ROOT / "tests/scripts/_temporary-shebang-allowlist-fixture.sh"
    allowlist = REPO_ROOT / "tests/scripts/_temporary-shebang-allowlist.txt"
    fixture.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    allowlist.write_text(
        "tests/scripts/_temporary-shebang-allowlist-fixture.sh # platform shim\n",
        encoding="utf-8",
    )
    try:
        result = run_checker("--allowlist", str(allowlist), str(fixture))
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    finally:
        fixture.unlink(missing_ok=True)
        allowlist.unlink(missing_ok=True)


def test_extensionless_executable_is_checked(tmp_path):
    script = tmp_path / "run-helper"
    script.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    script.chmod(0o755)
    result = run_checker(str(script))
    assert result.returncode == 1
    assert "run-helper:1" in result.stdout


def test_git_scope_lookup_failure_exits_two():
    result = run_checker("--diff", "not-a-real-ref-for-shebang-policy")
    assert result.returncode == 2
    assert "git diff" in result.stderr


def test_git_scope_lookup_permission_failure_fails_closed(monkeypatch):
    spec = importlib.util.spec_from_file_location("check_bash_shebangs", SCRIPT)
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    def raise_permission(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(checker.subprocess, "check_output", raise_permission)

    try:
        checker.git_paths(["git", "diff"])
    except checker.GitLookupError as exc:
        assert "git diff failed" in str(exc)
    else:
        raise AssertionError("permission failure did not become GitLookupError")


def test_scan_catches_markdown_and_honors_above_line_suppression(tmp_path):
    old_shebang = "#!" + "/bin/bash\n"
    markdown = tmp_path / "optional-skills" / "example.md"
    markdown.parent.mkdir()
    markdown.write_text(f"```bash\n{old_shebang}```\n", encoding="utf-8")

    result = run_checker(str(markdown))
    assert result.returncode == 1
    assert "example.md:2" in result.stdout

    suppressed = tmp_path / "suppressed.sh"
    suppressed.write_text(
        "# shebang: ok fixture-only example\n#!" + "/bin/bash\n",
        encoding="utf-8",
    )
    result = run_checker(str(suppressed))
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_scan_flags_embedded_shebang_strings_and_honors_suppression(tmp_path):
    generated = tmp_path / "generated.py"
    generated.write_text('value = "#!' + '/bin/bash\\n"\n', encoding="utf-8")

    result = run_checker(str(generated))
    assert result.returncode == 1, f"{result.stdout}\n{result.stderr}"
    assert "embedded" in result.stdout

    suppressed = tmp_path / "suppressed.py"
    suppressed.write_text(
        "# shebang: ok documents the pattern under test\n"
        'value = "#!' + '/bin/bash\\n"\n',
        encoding="utf-8",
    )
    result = run_checker(str(suppressed))
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_bare_ok_variants_do_not_suppress(tmp_path):
    for bad_marker in (
        "# shebang: ok",
        "# shebang: OK:",
        "# shebang: ok   ",
    ):
        fixture = tmp_path / "bare-ok.sh"
        fixture.write_text(bad_marker + "\n#!" + "/bin/bash\n", encoding="utf-8")
        result = run_checker(str(fixture))
        assert result.returncode == 1, f"{bad_marker!r}: {result.stdout}"
        assert "shebang" in result.stdout


def test_allowlist_skips_matched_paths_but_not_others():
    allowed_file = REPO_ROOT / "tests/scripts/_temporary-termux-shim.sh"
    allowed_file.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    other_file = REPO_ROOT / "tests/scripts/_temporary-normal.sh"
    other_file.write_text("#!" + "/bin/bash\n", encoding="utf-8")

    allowlist = REPO_ROOT / "tests/scripts/_temporary-shebang-allowlist.txt"
    allowlist.write_text(
        "# termux launcher shim: no usable env at kernel exec time\n"
        "tests/scripts/_temporary-termux-shim.sh # platform shim\n",
        encoding="utf-8",
    )
    try:
        result = run_checker("--allowlist", str(allowlist), str(allowed_file))
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

        result = run_checker("--allowlist", str(allowlist), str(other_file))
        assert result.returncode == 1
        assert "tests/scripts/_temporary-normal.sh:1" in result.stdout
    finally:
        allowed_file.unlink(missing_ok=True)
        other_file.unlink(missing_ok=True)
        allowlist.unlink(missing_ok=True)


def test_allowlist_glob_does_not_cross_path_separators():
    allowed_file = REPO_ROOT / "tests/scripts/_temporary-allowlist-top.sh"
    nested_file = REPO_ROOT / "tests/scripts/_temporary-allowlist-nested/evil.sh"
    allowed_file.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    nested_file.parent.mkdir(exist_ok=True)
    nested_file.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    allowlist = REPO_ROOT / "tests/scripts/_temporary-shebang-allowlist.txt"
    allowlist.write_text(
        "tests/scripts/*.sh # only direct scripts under tests/scripts\n",
        encoding="utf-8",
    )
    try:
        result = run_checker("--allowlist", str(allowlist), str(allowed_file))
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

        result = run_checker("--allowlist", str(allowlist), str(nested_file))
        assert result.returncode == 1
        assert "tests/scripts/_temporary-allowlist-nested/evil.sh:1" in result.stdout
    finally:
        allowed_file.unlink(missing_ok=True)
        nested_file.unlink(missing_ok=True)
        nested_file.parent.rmdir()
        allowlist.unlink(missing_ok=True)


def test_allowlist_glob_is_root_anchored():
    allowed_file = REPO_ROOT / "tests/scripts/_temporary-root-anchor.sh"
    prefixed_file = REPO_ROOT / "_temporary-prefix/tests/scripts/_temporary-root-anchor.sh"
    allowlist = REPO_ROOT / "tests/scripts/_temporary-shebang-allowlist.txt"
    allowed_file.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    prefixed_file.parent.mkdir(parents=True, exist_ok=True)
    prefixed_file.write_text("#!" + "/bin/bash\n", encoding="utf-8")
    allowlist.write_text(
        "tests/scripts/*.sh # direct scripts under the repository root\n",
        encoding="utf-8",
    )
    try:
        result = run_checker("--allowlist", str(allowlist), str(allowed_file))
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

        result = run_checker("--allowlist", str(allowlist), str(prefixed_file))
        assert result.returncode == 1
        assert "_temporary-prefix/tests/scripts/_temporary-root-anchor.sh:1" in result.stdout
    finally:
        allowed_file.unlink(missing_ok=True)
        prefixed_file.unlink(missing_ok=True)
        for parent in (
            prefixed_file.parent,
            prefixed_file.parent.parent,
            prefixed_file.parent.parent.parent,
        ):
            parent.rmdir()
        allowlist.unlink(missing_ok=True)


def test_missing_allowlist_file_fails_closed(tmp_path):
    fixture = tmp_path / "clean.sh"
    fixture.write_text("#!" + "/usr/bin/env bash\n", encoding="utf-8")
    result = run_checker("--allowlist", str(tmp_path / "nope.txt"), str(fixture))
    assert result.returncode == 2
    assert "cannot read allowlist" in result.stderr
