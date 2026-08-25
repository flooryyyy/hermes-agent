#!/usr/bin/env python3
"""Find hardcoded ``#!/bin/bash`` shebangs in repository text files.

The checker scans staged files by default, the whole repository with ``--all``,
changed files with ``--diff REF``, or explicit files/directories. Markdown and
embedded generated-script strings are checked as well as shell scripts.

Suppressions require ``# shebang: ok <reason>`` on the line immediately above
the match; a bare marker is rejected. Whole-file exceptions use a root-anchored
repo-relative POSIX glob with a required trailing ``# reason`` in the file passed to
``--allowlist``. Missing paths, filesystem errors, and failed git scope lookups
fail closed with exit status 2.

Exit codes: 0 clean, 1 violations, 2 scope/read/tooling error.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
GIT_LOOKUP_TIMEOUT_SECONDS = 30

TEXT_SUFFIXES = {
    ".bash",
    ".bat",
    ".cmd",
    ".css",
    ".html",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".markdown",
    ".mdx",
    ".nix",
    ".ps1",
    ".py",
    ".pyi",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}
EXCLUDED_FILES = {"scripts/check-bash-shebangs.py"}

SUPPRESS_MARKER = re.compile(
    r"#\s*shebang\s*:\s*ok[\s:]+(?P<reason>\S.*?)\s*$", re.IGNORECASE
)
SHEBANG_LINE = re.compile(r"^#![ \t]*/bin/bash(?:[ \t]|$)")
EMBEDDED_SHEBANG = re.compile(r"#![ \t]*/bin/bash(?![A-Za-z0-9_.-])")


class GitLookupError(RuntimeError):
    """A git query needed to determine the scan scope failed."""


class ScopeError(RuntimeError):
    """A requested path could not be inspected."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag hardcoded /bin/bash shebangs.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="Scan the full repository.")
    parser.add_argument("--diff", metavar="REF", help="Scan paths changed from REF.")
    parser.add_argument(
        "--allowlist",
        metavar="FILE",
        help="Repo-relative glob exceptions; every entry needs a trailing # reason.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow an empty default staged scope.",
    )
    return parser.parse_args(argv)


def git_paths(command: list[str]) -> list[Path]:
    try:
        output = subprocess.check_output(
            command,
            cwd=REPO_ROOT,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_LOOKUP_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        raise GitLookupError(f"{' '.join(command)} failed: {exc}") from exc
    return [REPO_ROOT / name for name in output.splitlines() if name.strip()]


def get_staged_files() -> list[Path]:
    return git_paths(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def get_diff_files(ref: str) -> list[Path]:
    return git_paths(["git", "diff", ref, "--name-only", "--diff-filter=ACMR"])


def _validate_allowlist_pattern(pattern: str, line_number: int) -> None:
    if (
        not pattern
        or pattern.startswith("/")
        or pattern.startswith("./")
        or "\\" in pattern
        or "/" not in pattern
        or any(part == ".." for part in pattern.split("/"))
        or any(char.isspace() for char in pattern)
    ):
        raise ValueError(
            f"allowlist line {line_number}: expected a repo-relative POSIX glob"
        )


def load_allowlist(path: str) -> list[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read allowlist {path}: {exc}") from exc

    patterns: list[str] = []
    for line_number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" not in stripped:
            raise ValueError(
                f"allowlist line {line_number}: entry needs a trailing # reason"
            )
        pattern, reason = (part.strip() for part in stripped.split("#", 1))
        _validate_allowlist_pattern(pattern, line_number)
        if not reason:
            raise ValueError(f"allowlist line {line_number}: reason is required")
        patterns.append(pattern)
    return patterns


def _is_excluded(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return True
    try:
        return path.relative_to(REPO_ROOT).as_posix() in EXCLUDED_FILES
    except ValueError:
        return False


def _is_scan_candidate(path: Path) -> bool:
    if _is_excluded(path):
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.suffix:
        return False
    mode = path.stat().st_mode
    return bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def collect_files(paths: Iterable[Path]) -> tuple[list[Path], list[str]]:
    """Collect candidates and return all filesystem errors instead of skipping."""
    files: list[Path] = []
    errors: list[str] = []
    seen: set[Path] = set()

    def add_file(path: Path) -> None:
        try:
            if path.is_file() and _is_scan_candidate(path) and path not in seen:
                files.append(path)
                seen.add(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    def walk_error(exc: OSError) -> None:
        errors.append(f"{getattr(exc, 'filename', None) or '<unknown>'}: {exc}")

    for path in paths:
        try:
            if not path.exists():
                errors.append(f"{path}: does not exist")
                continue
            if path.is_file():
                add_file(path)
                continue
            if not path.is_dir():
                errors.append(f"{path}: is not a regular file or directory")
                continue
            for root, dirs, names in os.walk(path, onerror=walk_error):
                dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
                for name in names:
                    add_file(Path(root) / name)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return files, errors


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _allowlist_matches(path: str, pattern: str) -> bool:
    """Match a repo-relative glob from the repository root."""
    path_parts = PurePosixPath(path).parts
    pattern_parts = PurePosixPath(pattern).parts

    @cache
    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        if pattern_parts[pattern_index] == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], pattern_parts[pattern_index])
            and match(path_index + 1, pattern_index + 1)
        )

    return match(0, 0)


def report(path: str, lineno: int, line: str, *, embedded: bool) -> None:
    kind = "embedded /bin/bash shebang string" if embedded else "hardcoded /bin/bash shebang"
    print(f"{path}:{lineno}: [{kind}]")
    print(f"    {line.strip()}")
    print("    Fix: #!/usr/bin/env bash")
    print("    Or add # shebang: ok <reason> immediately above the match.")
    print()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        allowlist = load_allowlist(args.allowlist) if args.allowlist else []
        if args.all:
            roots = [REPO_ROOT]
        elif args.diff:
            roots = get_diff_files(args.diff)
        elif args.paths:
            roots = [path.expanduser().resolve(strict=False) for path in args.paths]
        else:
            roots = get_staged_files()
            if not roots:
                if args.allow_empty:
                    print("OK no staged files to scan (--allow-empty)")
                    return 0
                raise ScopeError(
                    "no staged files to scan; pass --all, --diff REF, or explicit paths"
                )
    except (GitLookupError, ScopeError, ValueError) as exc:
        print(f"X {exc}", file=sys.stderr)
        return 2

    files, errors = collect_files(roots)
    matches = 0
    files_scanned = 0
    for path in files:
        rel = display_path(path)
        if any(_allowlist_matches(rel, pattern) for pattern in allowlist):
            continue
        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(keepends=True)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        files_scanned += 1
        for lineno, line in enumerate(lines, start=1):
            is_shebang = lineno == 1 and SHEBANG_LINE.match(line) is not None
            embedded = not is_shebang and EMBEDDED_SHEBANG.search(line) is not None
            if not (is_shebang or embedded):
                continue
            if lineno > 1 and SUPPRESS_MARKER.search(lines[lineno - 2]):
                continue
            report(rel, lineno, line, embedded=embedded)
            matches += 1

    if errors:
        print("X filesystem scope/read errors (scan is not trustworthy):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2
    if matches:
        print(
            f"X {matches} hardcoded /bin/bash shebang(s) found across "
            f"{files_scanned} file(s) scanned.",
            file=sys.stderr,
        )
        return 1
    print(f"OK no hardcoded /bin/bash shebangs ({files_scanned} file(s) scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
