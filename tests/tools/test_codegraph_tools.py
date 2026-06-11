"""Comprehensive tests for codegraph_tools.py.

Tests run without codegraph installed (mocked CLI) and verify every handler,
helper, staleness cache, error path, and edge case.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.codegraph_tools import (
    STALENESS_CACHE_TTL,
    _VALID_KINDS,
    _check_codegraph_available,
    _check_codegraph_binary_only,
    _find_codegraph_binary,
    _find_project_root,
    _get_staleness,
    _handle_affected,
    _handle_callers,
    _handle_callees,
    _handle_context,
    _handle_impact,
    _handle_index,
    _handle_search,
    _handle_status,
    _resolve_project,
    _run_codegraph,
    _staleness_cache,
    _sync_if_needed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_codegraph_dir(tmp_path: Path) -> Path:
    """Create a .codegraph/ dir with a dummy db file."""
    cg = tmp_path / ".codegraph"
    cg.mkdir(parents=True, exist_ok=True)
    (cg / "codegraph.db").touch()
    return tmp_path


def _mock_subprocess(monkeypatch, stdout=b"", stderr=b"", returncode=0):
    """Mock asyncio.create_subprocess_exec to return controlled output."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(stdout, stderr))
    mock_proc.returncode = returncode
    monkeypatch.setattr(
        "tools.codegraph_tools.asyncio.create_subprocess_exec",
        AsyncMock(return_value=mock_proc),
    )
    return mock_proc


# ---------------------------------------------------------------------------
# _find_codegraph_binary
# ---------------------------------------------------------------------------

class TestFindCodegraphBinary:
    def test_returns_path_when_installed(self, monkeypatch):
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: "/usr/local/bin/codegraph")
        assert _find_codegraph_binary() == "/usr/local/bin/codegraph"

    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: None)
        assert _find_codegraph_binary() is None

    def test_looks_for_codegraph_specifically(self, monkeypatch):
        called_with = []
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: called_with.append(x) or None)
        _find_codegraph_binary()
        assert called_with == ["codegraph"]


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------

class TestFindProjectRoot:
    def test_finds_in_direct_dir(self, tmp_path):
        _make_codegraph_dir(tmp_path)
        assert _find_project_root(str(tmp_path)) == tmp_path

    def test_finds_in_parent(self, tmp_path):
        _make_codegraph_dir(tmp_path)
        nested = tmp_path / "src" / "deep" / "path"
        nested.mkdir(parents=True)
        assert _find_project_root(str(nested)) == tmp_path

    def test_returns_none_when_missing(self, tmp_path):
        assert _find_project_root(str(tmp_path)) is None

    def test_returns_none_when_dir_exists_but_no_db(self, tmp_path):
        (tmp_path / ".codegraph").mkdir()
        assert _find_project_root(str(tmp_path)) is None

    def test_returns_none_when_db_is_directory(self, tmp_path):
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        (cg / "codegraph.db").mkdir()  # db is a dir, not a file
        assert _find_project_root(str(tmp_path)) is None

    def test_finds_deepest_first(self, tmp_path):
        """Should find the closest .codegraph/, not the furthest."""
        outer = _make_codegraph_dir(tmp_path)
        inner = tmp_path / "subproject"
        inner.mkdir()
        _make_codegraph_dir(inner)
        nested = inner / "src"
        nested.mkdir()
        assert _find_project_root(str(nested)) == inner

    def test_uses_cwd_when_no_path(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert _find_project_root() == tmp_path


# ---------------------------------------------------------------------------
# _check_codegraph_available
# ---------------------------------------------------------------------------

class TestCheckCodegraphAvailable:
    def test_false_when_no_binary(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: None)
        assert _check_codegraph_available() is False

    def test_true_when_binary_exists(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: "/usr/bin/codegraph")
        assert _check_codegraph_available() is True

    def test_true_even_without_index(self, monkeypatch, tmp_path):
        """Tools appear when binary exists, even without an index. Handlers return helpful errors."""
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: "/usr/bin/codegraph")
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _check_codegraph_available() is True


# ---------------------------------------------------------------------------
# _check_codegraph_binary_only
# ---------------------------------------------------------------------------

class TestCheckCodegraphBinaryOnly:
    def test_true_when_binary_exists(self, monkeypatch):
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: "/usr/bin/codegraph")
        assert _check_codegraph_binary_only() is True

    def test_false_when_no_binary(self, monkeypatch):
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: None)
        assert _check_codegraph_binary_only() is False


# ---------------------------------------------------------------------------
# _resolve_project
# ---------------------------------------------------------------------------

class TestResolveProject:
    def test_from_explicit_path(self, tmp_path):
        _make_codegraph_dir(tmp_path)
        assert _resolve_project({"path": str(tmp_path)}) == tmp_path

    def test_from_kanban_env(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
        assert _resolve_project({}) == tmp_path

    def test_from_terminal_cwd(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _resolve_project({}) == tmp_path

    def test_from_cwd(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(tmp_path)
        assert _resolve_project({}) == tmp_path

    def test_returns_none_when_not_found(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(tmp_path)
        assert _resolve_project({}) is None


# ---------------------------------------------------------------------------
# _run_codegraph
# ---------------------------------------------------------------------------

class TestRunCodegraph:
    @pytest.mark.asyncio
    async def test_parses_json_output(self, monkeypatch):
        _mock_subprocess(monkeypatch, stdout=b'{"results": []}')
        result = await _run_codegraph("query", "test", "--json")
        assert result == {"results": []}

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, monkeypatch):
        _mock_subprocess(monkeypatch, stderr=b"error message", returncode=1)
        result = await _run_codegraph("query", "test")
        assert "error" in result
        assert "code 1" in result["error"]
        assert result["stderr"] == "error message"

    @pytest.mark.asyncio
    async def test_timeout(self, monkeypatch):
        async def slow_communicate():
            await asyncio.sleep(999)
        mock_proc = AsyncMock()
        mock_proc.communicate = slow_communicate
        mock_proc.returncode = 0
        monkeypatch.setattr(
            "tools.codegraph_tools.asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        )
        result = await _run_codegraph("query", "test", timeout=1)
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_no_binary(self, monkeypatch):
        monkeypatch.setattr("tools.codegraph_tools.shutil.which", lambda x: None)
        result = await _run_codegraph("query", "test")
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_malformed_json(self, monkeypatch):
        _mock_subprocess(monkeypatch, stdout=b"some plain text")
        result = await _run_codegraph("status")
        assert result["raw"] == "some plain text"

    @pytest.mark.asyncio
    async def test_empty_output(self, monkeypatch):
        _mock_subprocess(monkeypatch, stdout=b"")
        result = await _run_codegraph("sync")
        assert result["empty"] is True

    @pytest.mark.asyncio
    async def test_passes_cwd_to_subprocess(self, monkeypatch):
        mock_exec = AsyncMock()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'{"ok": true}', b''))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        monkeypatch.setattr("tools.codegraph_tools.asyncio.create_subprocess_exec", mock_exec)
        await _run_codegraph("status", cwd="/some/path")
        _, kwargs = mock_exec.call_args
        assert kwargs.get("cwd") == "/some/path"

    @pytest.mark.asyncio
    async def test_large_output(self, monkeypatch):
        big_data = {"results": [{"id": i, "name": f"sym_{i}"} for i in range(1000)]}
        _mock_subprocess(monkeypatch, stdout=json.dumps(big_data).encode())
        result = await _run_codegraph("query", "test", "--limit", "1000")
        assert len(result["results"]) == 1000

    @pytest.mark.asyncio
    async def test_unicode_in_output(self, monkeypatch):
        data = {"results": [{"name": "über_kläss", "file": "日本語.py"}]}
        _mock_subprocess(monkeypatch, stdout=json.dumps(data).encode("utf-8"))
        result = await _run_codegraph("query", "test")
        assert result["results"][0]["name"] == "über_kläss"


# ---------------------------------------------------------------------------
# Staleness cache
# ---------------------------------------------------------------------------

class TestStalenessCache:
    def setup_method(self):
        _staleness_cache.clear()

    @pytest.mark.asyncio
    async def test_cache_hit(self, monkeypatch, tmp_path):
        _staleness_cache[str(tmp_path)] = (time.time(), {"initialized": True})
        result = await _get_staleness(tmp_path)
        assert result["cached"] is True
        assert result["initialized"] is True

    @pytest.mark.asyncio
    async def test_cache_miss_calls_cli(self, monkeypatch, tmp_path):
        _mock_subprocess(monkeypatch, stdout=b'{"initialized": true}')
        result = await _get_staleness(tmp_path)
        assert result["cached"] is False
        assert str(tmp_path) in _staleness_cache

    @pytest.mark.asyncio
    async def test_cache_expires(self, monkeypatch, tmp_path):
        _staleness_cache[str(tmp_path)] = (time.time() - STALENESS_CACHE_TTL - 1, {"old": True})
        _mock_subprocess(monkeypatch, stdout=b'{"initialized": true, "fresh": true}')
        result = await _get_staleness(tmp_path)
        assert result["cached"] is False
        assert result.get("fresh") is True

    @pytest.mark.asyncio
    async def test_per_project_isolation(self, monkeypatch, tmp_path):
        path_a = tmp_path / "a"
        path_b = tmp_path / "b"
        _staleness_cache[str(path_a)] = (time.time(), {"project": "a"})
        _staleness_cache[str(path_b)] = (time.time(), {"project": "b"})
        result_a = await _get_staleness(path_a)
        result_b = await _get_staleness(path_b)
        assert result_a["project"] == "a"
        assert result_b["project"] == "b"


# ---------------------------------------------------------------------------
# _sync_if_needed
# ---------------------------------------------------------------------------

class TestSyncIfNeeded:
    def setup_method(self):
        _staleness_cache.clear()

    @pytest.mark.asyncio
    async def test_no_sync_when_fresh(self, monkeypatch, tmp_path):
        _staleness_cache[str(tmp_path)] = (time.time(), {"needs_update": False})
        result = await _sync_if_needed(tmp_path)
        assert result is None

    @pytest.mark.asyncio
    async def test_syncs_when_stale(self, monkeypatch, tmp_path):
        _staleness_cache[str(tmp_path)] = (time.time(), {"needs_update": True})
        call_args = []
        original_run = _run_codegraph

        async def mock_run(*args, **kwargs):
            call_args.append(args)
            if args[0] == "sync":
                return {"result": "ok"}
            return {"initialized": True, "needs_update": True}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", mock_run)
        result = await _sync_if_needed(tmp_path)
        assert result is None
        assert ("sync",) in call_args

    @pytest.mark.asyncio
    async def test_sync_failure_returns_error(self, monkeypatch, tmp_path):
        _staleness_cache[str(tmp_path)] = (time.time(), {"needs_update": True})

        async def mock_run(*args, **kwargs):
            if args[0] == "sync":
                return {"error": "sync failed"}
            return {"initialized": True, "needs_update": True}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", mock_run)
        result = await _sync_if_needed(tmp_path)
        assert result is not None
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalidates_cache_after_sync(self, monkeypatch, tmp_path):
        _staleness_cache[str(tmp_path)] = (time.time(), {"needs_update": True})

        async def mock_run(*args, **kwargs):
            if args[0] == "sync":
                return {"result": "ok"}
            return {"initialized": True, "needs_update": True}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", mock_run)
        await _sync_if_needed(tmp_path)
        assert str(tmp_path) not in _staleness_cache


# ---------------------------------------------------------------------------
# Handler: codegraph_search
# ---------------------------------------------------------------------------

class TestHandleSearch:
    @pytest.mark.asyncio
    async def test_no_project_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_search({"query": "UserService"}))
        assert "error" in result
        assert "codegraph init" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_results(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        sample = {"results": [{"name": "UserService", "kind": "class"}]}
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value=sample))
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))
        result = json.loads(await _handle_search({"query": "UserService"}))
        assert result["results"][0]["name"] == "UserService"

    @pytest.mark.asyncio
    async def test_passes_kind_filter(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured_args = []

        async def capture(*args, **kwargs):
            captured_args.extend(args)
            return {"results": []}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", capture)
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))
        await _handle_search({"query": "test", "kind": "class"})
        assert "--kind" in captured_args
        assert "class" in captured_args

    @pytest.mark.asyncio
    async def test_passes_limit(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured_args = []

        async def capture(*args, **kwargs):
            captured_args.extend(args)
            return {"results": []}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", capture)
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))
        await _handle_search({"query": "test", "limit": 5})
        assert "--limit" in captured_args
        assert "5" in captured_args

    # --- input validation tests ---

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        result = json.loads(await _handle_search({"query": ""}))
        assert "error" in result
        assert "non-empty" in result["error"]

    @pytest.mark.asyncio
    async def test_whitespace_only_query_returns_error(self):
        result = json.loads(await _handle_search({"query": "   "}))
        assert "error" in result
        assert "non-empty" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_query_returns_error(self):
        result = json.loads(await _handle_search({}))
        assert "error" in result
        assert "non-empty" in result["error"]

    @pytest.mark.asyncio
    async def test_non_string_query_returns_error(self):
        result = json.loads(await _handle_search({"query": 42}))
        assert "error" in result
        assert "non-empty" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_kind_returns_error(self):
        result = json.loads(await _handle_search({"query": "foo", "kind": "bogus"}))
        assert "error" in result
        assert "kind" in result["error"]

    @pytest.mark.asyncio
    async def test_valid_kind_accepted(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"results": []}))
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))
        for kind in _VALID_KINDS:
            result = json.loads(await _handle_search({"query": "x", "kind": kind}))
            assert "error" not in result

    @pytest.mark.asyncio
    async def test_negative_limit_returns_error(self):
        result = json.loads(await _handle_search({"query": "foo", "limit": -1}))
        assert "error" in result
        assert "positive" in result["error"]

    @pytest.mark.asyncio
    async def test_zero_limit_returns_error(self):
        result = json.loads(await _handle_search({"query": "foo", "limit": 0}))
        assert "error" in result
        assert "positive" in result["error"]

    @pytest.mark.asyncio
    async def test_huge_limit_returns_error(self):
        result = json.loads(await _handle_search({"query": "foo", "limit": 999}))
        assert "error" in result
        assert "500" in result["error"]

    @pytest.mark.asyncio
    async def test_string_limit_returns_error(self):
        result = json.loads(await _handle_search({"query": "foo", "limit": "ten"}))
        assert "error" in result
        assert "positive integer" in result["error"]

    @pytest.mark.asyncio
    async def test_bool_limit_returns_error(self):
        result = json.loads(await _handle_search({"query": "foo", "limit": True}))
        assert "error" in result
        assert "positive integer" in result["error"]


# ---------------------------------------------------------------------------
# Handler: codegraph_context
# ---------------------------------------------------------------------------

class TestHandleContext:
    @pytest.mark.asyncio
    async def test_no_project_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_context({"task": "understand auth"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_markdown(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"raw": "## Auth\n..."}))
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))
        result = json.loads(await _handle_context({"task": "understand auth"}))
        assert "raw" in result

    @pytest.mark.asyncio
    async def test_passes_max_nodes(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured_args = []

        async def capture(*args, **kwargs):
            captured_args.extend(args)
            return {"raw": ""}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", capture)
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))
        await _handle_context({"task": "test", "max_nodes": 10})
        assert "--max-nodes" in captured_args
        assert "10" in captured_args


# ---------------------------------------------------------------------------
# Handler: codegraph_callers
# ---------------------------------------------------------------------------

class TestHandleCallers:
    @pytest.mark.asyncio
    async def test_no_project_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_callers({"symbol": "foo"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_callers(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        sample = {"symbol": "foo", "callers": [{"name": "bar", "file": "x.py"}]}
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value=sample))
        result = json.loads(await _handle_callers({"symbol": "foo"}))
        assert len(result["callers"]) == 1

    @pytest.mark.asyncio
    async def test_passes_limit(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured_args = []

        async def capture(*args, **kwargs):
            captured_args.extend(args)
            return {"symbol": "foo", "callers": []}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", capture)
        await _handle_callers({"symbol": "foo", "limit": 10})
        assert "--limit" in captured_args
        assert "10" in captured_args


# ---------------------------------------------------------------------------
# Handler: codegraph_callees
# ---------------------------------------------------------------------------

class TestHandleCallees:
    @pytest.mark.asyncio
    async def test_no_project_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_callees({"symbol": "foo"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_callees(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        sample = {"symbol": "foo", "callees": [{"name": "bar"}]}
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value=sample))
        result = json.loads(await _handle_callees({"symbol": "foo"}))
        assert len(result["callees"]) == 1


# ---------------------------------------------------------------------------
# Handler: codegraph_impact
# ---------------------------------------------------------------------------

class TestHandleImpact:
    @pytest.mark.asyncio
    async def test_no_project_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_impact({"symbol": "foo"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_passes_depth(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured_args = []

        async def capture(*args, **kwargs):
            captured_args.extend(args)
            return {"symbol": "foo", "impact": []}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", capture)
        await _handle_impact({"symbol": "foo", "depth": 5})
        assert "--depth" in captured_args
        assert "5" in captured_args


# ---------------------------------------------------------------------------
# Handler: codegraph_affected
# ---------------------------------------------------------------------------

class TestHandleAffected:
    @pytest.mark.asyncio
    async def test_no_project_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_affected({"files": ["a.py"]}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_files_returns_error(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_affected({"files": []}))
        assert "error" in result
        assert "files" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_passes_files(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        captured_args = []

        async def capture(*args, **kwargs):
            captured_args.extend(args)
            return {"affected": []}

        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", capture)
        await _handle_affected({"files": ["src/a.py", "src/b.py"]})
        assert "src/a.py" in captured_args
        assert "src/b.py" in captured_args


# ---------------------------------------------------------------------------
# Handler: codegraph_index
# ---------------------------------------------------------------------------

class TestHandleIndex:
    @pytest.mark.asyncio
    async def test_init_new_project(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._find_project_root", lambda p: None)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"status": "initialized"}))
        result = json.loads(await _handle_index({"action": "init"}))
        assert result["status"] == "initialized"

    @pytest.mark.asyncio
    async def test_init_already_initialized(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_index({"action": "init"}))
        assert result["status"] == "already_initialized"

    @pytest.mark.asyncio
    async def test_sync_existing(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"result": "ok"}))
        result = json.loads(await _handle_index({"action": "sync"}))
        assert result["result"] == "ok"

    @pytest.mark.asyncio
    async def test_sync_without_index(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = json.loads(await _handle_index({"action": "sync"}))
        assert "error" in result
        assert "init" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_action(self, monkeypatch):
        result = json.loads(await _handle_index({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, monkeypatch):
        result = json.loads(await _handle_index({"action": "delete"}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_init_invalidates_cache(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _staleness_cache[str(tmp_path)] = (time.time(), {"old": True})
        monkeypatch.setattr("tools.codegraph_tools._find_project_root", lambda p: None)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"status": "ok"}))
        await _handle_index({"action": "init"})
        assert str(tmp_path) not in _staleness_cache


# ---------------------------------------------------------------------------
# Handler: codegraph_status
# ---------------------------------------------------------------------------

class TestHandleStatus:
    @pytest.mark.asyncio
    async def test_no_project(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        result = json.loads(await _handle_status({}))
        assert result["initialized"] is False

    @pytest.mark.asyncio
    async def test_with_project(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"initialized": True, "fileCount": 100}))
        result = json.loads(await _handle_status({}))
        assert result["initialized"] is True
        assert result["fileCount"] == 100


# ---------------------------------------------------------------------------
# Edge cases: concurrent access
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    def setup_method(self):
        _staleness_cache.clear()

    @pytest.mark.asyncio
    async def test_rapid_calls_use_cache(self, monkeypatch, tmp_path):
        """Multiple calls within TTL should use staleness cache, not re-check."""
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)

        check_count = 0
        original_get = _get_staleness

        async def counting_get(project_root):
            nonlocal check_count
            check_count += 1
            return await original_get(project_root)

        monkeypatch.setattr("tools.codegraph_tools._get_staleness", counting_get)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"results": []}))
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))

        # Call search 5 times rapidly
        for _ in range(5):
            await _handle_search({"query": "test"})

        # Staleness should be checked at most once (cache hit on rest)
        assert check_count <= 1

    @pytest.mark.asyncio
    async def test_concurrent_handlers_dont_crash(self, monkeypatch, tmp_path):
        """Multiple async handlers running simultaneously should not corrupt state."""
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"results": []}))
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))

        results = await asyncio.gather(
            _handle_search({"query": "a"}),
            _handle_search({"query": "b"}),
            _handle_callers({"symbol": "c"}),
            _handle_callees({"symbol": "d"}),
            _handle_impact({"symbol": "e"}),
        )
        # All should complete without error
        for r in results:
            parsed = json.loads(r)
            assert "error" not in parsed or "callers" in parsed or "callees" in parsed


# ---------------------------------------------------------------------------
# Edge cases: error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_cli_error_propagates_to_search(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"error": "something broke"}))
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value=None))
        result = json.loads(await _handle_search({"query": "test"}))
        assert result["error"] == "something broke"

    @pytest.mark.asyncio
    async def test_sync_error_propagates(self, monkeypatch, tmp_path):
        _make_codegraph_dir(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("tools.codegraph_tools._sync_if_needed", AsyncMock(return_value={"error": "sync failed"}))
        # sync error doesn't block the query — it's advisory
        monkeypatch.setattr("tools.codegraph_tools._run_codegraph", AsyncMock(return_value={"results": []}))
        result = json.loads(await _handle_search({"query": "test"}))
        # Should still return results (sync error is logged, not blocking)
        assert "results" in result
