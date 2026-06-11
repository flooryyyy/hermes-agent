"""CodeGraph native tools — tree-sitter code intelligence for Hermes agents.

Wraps the CodeGraph CLI to provide symbol search, call graph traversal,
impact analysis, and context generation. No MCP, no external dependencies
beyond the ``codegraph`` CLI binary.

Auto-gated: tools only appear when .codegraph/codegraph.db exists in the
current workspace.
"""

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Staleness cache — avoid sync-spam on rapid tool calls
# ---------------------------------------------------------------------------

_staleness_cache: dict[str, tuple[float, dict]] = {}  # path -> (last_check_ts, status_result)
STALENESS_CACHE_TTL = 60  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_codegraph_binary() -> Optional[str]:
    """Find the codegraph binary, return path or None."""
    return shutil.which("codegraph")


def _find_project_root(path: Optional[str] = None) -> Optional[Path]:
    """Find the nearest .codegraph/ directory from cwd or given path."""
    search = Path(path) if path else Path.cwd()
    for d in [search, *search.parents]:
        db = d / ".codegraph" / "codegraph.db"
        if db.is_file():
            return d
    return None


async def _run_codegraph(*args: str, cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a codegraph CLI command. Returns parsed JSON or error dict."""
    binary = _find_codegraph_binary()
    if not binary:
        return {
            "error": "codegraph binary not found. Install: "
            "curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh"
        }

    cmd = [binary, *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        if proc.returncode != 0:
            return {
                "error": f"codegraph exited with code {proc.returncode}",
                "stderr": stderr.decode().strip(),
            }

        output = stdout.decode().strip()
        if not output:
            return {"result": "ok", "empty": True}

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"result": "ok", "raw": output}

    except asyncio.TimeoutError:
        return {"error": f"codegraph timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


async def _get_staleness(project_root: Path) -> dict:
    """Check index staleness. Cached for 60s to prevent sync-spam."""
    cache_key = str(project_root)
    now = time.time()

    if cache_key in _staleness_cache:
        last_check, cached_result = _staleness_cache[cache_key]
        if now - last_check < STALENESS_CACHE_TTL:
            cached_result["cached"] = True
            return cached_result

    result = await _run_codegraph("status", "--json", cwd=str(project_root))
    _staleness_cache[cache_key] = (now, result)
    result["cached"] = False
    return result


async def _sync_if_needed(project_root: Path) -> Optional[dict]:
    """Run codegraph sync if index might be stale. Returns error dict if sync fails."""
    status = await _get_staleness(project_root)

    # If status reports the index needs updating, sync it
    if isinstance(status, dict) and status.get("needs_update"):
        # Invalidate cache after sync
        _staleness_cache.pop(str(project_root), None)
        result = await _run_codegraph("sync", cwd=str(project_root), timeout=60)
        if "error" in result:
            return result
    return None


# ---------------------------------------------------------------------------
# Check function — gates tool availability
# ---------------------------------------------------------------------------

def _check_codegraph_available() -> bool:
    """True if codegraph binary exists.

    We don't check for .codegraph/ here because the workspace isn't known at
    schema-build time (TERMINAL_CWD is set later, during tool execution).
    Handlers return a helpful error if no index is found.
    """
    return bool(_find_codegraph_binary())


def _check_codegraph_binary_only() -> bool:
    """True if codegraph binary exists (for codegraph_index which works without an index)."""
    return bool(_find_codegraph_binary())


# ---------------------------------------------------------------------------
# Resolve project root from args or env
# ---------------------------------------------------------------------------

def _resolve_project(args: dict) -> Optional[Path]:
    """Resolve project root from args['path'] or workspace env.

    Tries each candidate in order and returns the first that has a .codegraph/ index.
    This handles the case where TERMINAL_CWD points to a parent dir that doesn't
    contain .codegraph/ but the actual working directory does.
    """
    candidates = []
    if args.get("path"):
        candidates.append(args["path"])
    if os.environ.get("HERMES_KANBAN_WORKSPACE"):
        candidates.append(os.environ["HERMES_KANBAN_WORKSPACE"])
    if os.environ.get("TERMINAL_CWD"):
        candidates.append(os.environ["TERMINAL_CWD"])
    candidates.append(os.getcwd())

    for candidate in candidates:
        result = _find_project_root(candidate)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Tool: codegraph_search
# ---------------------------------------------------------------------------

_CODEGRAPH_SEARCH_SCHEMA = {
    "name": "codegraph_search",
    "description": (
        "Search for code symbols (functions, classes, methods, variables) by name or pattern. "
        "Uses a pre-indexed code knowledge graph — faster and more accurate than grep for "
        "finding definitions, declarations, and structural code elements. "
        "Returns matching symbols with file paths, line numbers, and kind. "
        "Prefer this over grep/read_file when looking for WHERE something is defined."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Symbol name or pattern (e.g. 'UserService', 'handleLogin', 'auth_middleware')",
            },
            "kind": {
                "type": "string",
                "description": "Filter by symbol kind",
                "enum": ["function", "class", "method", "variable", "interface", "type", "enum", "struct"],
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20)",
            },
            "path": {
                "type": "string",
                "description": "Project root path. Auto-detected from workspace if omitted.",
            },
        },
        "required": ["query"],
    },
}


_VALID_KINDS = frozenset({"function", "class", "method", "variable", "interface", "type", "enum", "struct"})


async def _handle_search(args: dict, **kw) -> str:
    """Search for code symbols by name or pattern.

    Validates all inputs before invoking the codegraph CLI:
      - ``query`` must be a non-empty string (whitespace-only rejected).
      - ``kind`` (optional) must be one of the recognised symbol kinds.
      - ``limit`` (optional) must be a positive integer ≤ 500.

    Args:
        args: Tool arguments.  ``query`` is required; ``kind``, ``limit``,
              and ``path`` are optional.

    Returns:
        JSON-encoded string with either a ``results`` list or an ``error``.
    """
    # --- input validation ---
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return tool_error(
            "'query' is required and must be a non-empty string"
        )
    query = query.strip()

    kind = args.get("kind")
    if kind is not None and kind not in _VALID_KINDS:
        return tool_error(
            f"'kind' must be one of: {', '.join(sorted(_VALID_KINDS))}"
        )

    limit = args.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return tool_error("'limit' must be a positive integer")
        if limit > 500:
            return tool_error("'limit' must be ≤ 500")

    # --- resolve project & sync ---
    project = _resolve_project(args)
    if not project:
        return tool_error(
            "No .codegraph/ found in workspace. Run `codegraph init -i` in your project root first."
        )

    await _sync_if_needed(project)

    # --- build CLI args ---
    cmd_args = ["query", query, "--json"]
    if kind:
        cmd_args.extend(["--kind", kind])
    if limit:
        cmd_args.extend(["--limit", str(limit)])

    result = await _run_codegraph(*cmd_args, cwd=str(project))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: codegraph_context
# ---------------------------------------------------------------------------

_CODEGRAPH_CONTEXT_SCHEMA = {
    "name": "codegraph_context",
    "description": (
        "Build a focused context bundle for understanding an area of a codebase. "
        "Given a natural language task description, returns relevant symbols, their source code, "
        "call relationships, and file locations — all in one call. "
        "This is the FASTEST way to understand unfamiliar code. Prefer this over multiple "
        "grep/read_file calls when you need to understand how something works or where to make changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural language description of what you're trying to understand",
            },
            "max_nodes": {
                "type": "integer",
                "description": "Max symbols to include (default 30). Lower = faster.",
            },
            "path": {
                "type": "string",
                "description": "Project root path. Auto-detected from workspace if omitted.",
            },
        },
        "required": ["task"],
    },
}


async def _handle_context(args: dict, **kw) -> str:
    project = _resolve_project(args)
    if not project:
        return tool_error(
            "No .codegraph/ found in workspace. Run `codegraph init -i` in your project root first."
        )

    await _sync_if_needed(project)

    cmd_args = ["context", args["task"], "--format", "markdown"]
    if args.get("max_nodes"):
        cmd_args.extend(["--max-nodes", str(args["max_nodes"])])

    result = await _run_codegraph(*cmd_args, cwd=str(project), timeout=60)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: codegraph_callers
# ---------------------------------------------------------------------------

_CODEGRAPH_CALLERS_SCHEMA = {
    "name": "codegraph_callers",
    "description": (
        "Find all functions/methods that CALL a given symbol. "
        "Essential for understanding the impact of changing a function — "
        "shows every callsite in the codebase. "
        "Faster than grep because it understands method resolution, imports, and inheritance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Function/method name to find callers of",
            },
            "limit": {
                "type": "integer",
                "description": "Max callers to return (default 20)",
            },
            "path": {
                "type": "string",
                "description": "Project root path. Auto-detected from workspace if omitted.",
            },
        },
        "required": ["symbol"],
    },
}


async def _handle_callers(args: dict, **kw) -> str:
    project = _resolve_project(args)
    if not project:
        return tool_error(
            "No .codegraph/ found in workspace. Run `codegraph init -i` in your project root first."
        )

    cmd_args = ["callers", args["symbol"], "--json"]
    if args.get("limit"):
        cmd_args.extend(["--limit", str(args["limit"])])

    result = await _run_codegraph(*cmd_args, cwd=str(project))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: codegraph_callees
# ---------------------------------------------------------------------------

_CODEGRAPH_CALLEES_SCHEMA = {
    "name": "codegraph_callees",
    "description": (
        "Find all functions/methods that a given symbol CALLS. "
        "Shows the dependencies of a function — what it relies on. "
        "Use before refactoring to understand what your function depends on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Function/method name to find callees of",
            },
            "limit": {
                "type": "integer",
                "description": "Max callees to return (default 20)",
            },
            "path": {
                "type": "string",
                "description": "Project root path. Auto-detected from workspace if omitted.",
            },
        },
        "required": ["symbol"],
    },
}


async def _handle_callees(args: dict, **kw) -> str:
    project = _resolve_project(args)
    if not project:
        return tool_error(
            "No .codegraph/ found in workspace. Run `codegraph init -i` in your project root first."
        )

    cmd_args = ["callees", args["symbol"], "--json"]
    if args.get("limit"):
        cmd_args.extend(["--limit", str(args["limit"])])

    result = await _run_codegraph(*cmd_args, cwd=str(project))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: codegraph_impact
# ---------------------------------------------------------------------------

_CODEGRAPH_IMPACT_SCHEMA = {
    "name": "codegraph_impact",
    "description": (
        "Analyze the full transitive impact of changing a symbol. "
        "Returns all code that would be affected: direct callers, their callers, "
        "test files, and the full dependency chain. Use before refactoring to "
        "understand the blast radius of your changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to analyze impact of",
            },
            "depth": {
                "type": "integer",
                "description": "How many levels of callers to trace (default 3)",
            },
            "path": {
                "type": "string",
                "description": "Project root path. Auto-detected from workspace if omitted.",
            },
        },
        "required": ["symbol"],
    },
}


async def _handle_impact(args: dict, **kw) -> str:
    project = _resolve_project(args)
    if not project:
        return tool_error(
            "No .codegraph/ found in workspace. Run `codegraph init -i` in your project root first."
        )

    cmd_args = ["impact", args["symbol"], "--json"]
    if args.get("depth"):
        cmd_args.extend(["--depth", str(args["depth"])])

    result = await _run_codegraph(*cmd_args, cwd=str(project))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: codegraph_affected
# ---------------------------------------------------------------------------

_CODEGRAPH_AFFECTED_SCHEMA = {
    "name": "codegraph_affected",
    "description": (
        "Find test files affected by changes to source files. "
        "Traces import dependencies transitively to determine which tests "
        "would be impacted. Use after making changes to know which tests to run."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Changed source file paths (relative to project root)",
            },
            "path": {
                "type": "string",
                "description": "Project root path. Auto-detected from workspace if omitted.",
            },
        },
        "required": ["files"],
    },
}


async def _handle_affected(args: dict, **kw) -> str:
    project = _resolve_project(args)
    if not project:
        return tool_error(
            "No .codegraph/ found in workspace. Run `codegraph init -i` in your project root first."
        )

    files = args.get("files")
    if not files:
        return tool_error("'files' is required — provide a list of changed source file paths")

    cmd_args = ["affected", *files, "--json"]
    result = await _run_codegraph(*cmd_args, cwd=str(project))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: codegraph_index
# ---------------------------------------------------------------------------

_CODEGRAPH_INDEX_SCHEMA = {
    "name": "codegraph_index",
    "description": (
        "Initialize or sync a CodeGraph index for a project. "
        "Use action='init' to create a new index (first time — can take 10-60s on large repos). "
        "Use action='sync' to update an existing index after file changes (fast, incremental)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["init", "sync"],
                "description": "init = create index, sync = update existing",
            },
            "path": {
                "type": "string",
                "description": "Project root path. Defaults to current directory.",
            },
        },
        "required": ["action"],
    },
}


async def _handle_index(args: dict, **kw) -> str:
    action = args.get("action")
    if not action:
        return tool_error("'action' is required — use 'init' or 'sync'")

    project_path = args.get("path") or os.getcwd()

    if action == "init":
        # Check if already initialized
        existing = _find_project_root(project_path)
        if existing:
            return json.dumps({
                "status": "already_initialized",
                "path": str(existing),
                "hint": "Use action='sync' to update an existing index.",
            })

        result = await _run_codegraph("init", "-i", cwd=project_path, timeout=120)
        # Invalidate staleness cache for this path
        _staleness_cache.pop(project_path, None)
        return json.dumps(result, ensure_ascii=False)

    elif action == "sync":
        project = _find_project_root(project_path)
        if not project:
            return tool_error(
                "No .codegraph/ found. Use action='init' first to create an index."
            )

        _staleness_cache.pop(str(project), None)
        result = await _run_codegraph("sync", cwd=str(project), timeout=60)
        return json.dumps(result, ensure_ascii=False)

    else:
        return tool_error(f"Unknown action '{action}'. Use 'init' or 'sync'.")


# ---------------------------------------------------------------------------
# Tool: codegraph_status
# ---------------------------------------------------------------------------

_CODEGRAPH_STATUS_SCHEMA = {
    "name": "codegraph_status",
    "description": (
        "Check CodeGraph index status for a project: whether it exists, "
        "how many files/symbols are indexed, when it was last synced, "
        "and whether it may be stale. Use before other codegraph tools "
        "to verify the index is healthy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project root path. Auto-detected from workspace if omitted.",
            },
        },
    },
}


async def _handle_status(args: dict, **kw) -> str:
    project = _resolve_project(args)
    if not project:
        return json.dumps({
            "initialized": False,
            "hint": "No .codegraph/ found in workspace. Run `codegraph init -i` to create an index.",
        })

    result = await _run_codegraph("status", "--json", cwd=str(project))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="codegraph_search",
    toolset="codegraph",
    schema=_CODEGRAPH_SEARCH_SCHEMA,
    handler=_handle_search,
    check_fn=_check_codegraph_available,
    is_async=True,
    emoji="🔍",
)

registry.register(
    name="codegraph_context",
    toolset="codegraph",
    schema=_CODEGRAPH_CONTEXT_SCHEMA,
    handler=_handle_context,
    check_fn=_check_codegraph_available,
    is_async=True,
    emoji="🧭",
)

registry.register(
    name="codegraph_callers",
    toolset="codegraph",
    schema=_CODEGRAPH_CALLERS_SCHEMA,
    handler=_handle_callers,
    check_fn=_check_codegraph_available,
    is_async=True,
    emoji="📞",
)

registry.register(
    name="codegraph_callees",
    toolset="codegraph",
    schema=_CODEGRAPH_CALLEES_SCHEMA,
    handler=_handle_callees,
    check_fn=_check_codegraph_available,
    is_async=True,
    emoji="📤",
)

registry.register(
    name="codegraph_impact",
    toolset="codegraph",
    schema=_CODEGRAPH_IMPACT_SCHEMA,
    handler=_handle_impact,
    check_fn=_check_codegraph_available,
    is_async=True,
    emoji="💥",
)

registry.register(
    name="codegraph_affected",
    toolset="codegraph",
    schema=_CODEGRAPH_AFFECTED_SCHEMA,
    handler=_handle_affected,
    check_fn=_check_codegraph_available,
    is_async=True,
    emoji="🧪",
)

registry.register(
    name="codegraph_index",
    toolset="codegraph",
    schema=_CODEGRAPH_INDEX_SCHEMA,
    handler=_handle_index,
    check_fn=_check_codegraph_binary_only,
    is_async=True,
    emoji="📦",
)

registry.register(
    name="codegraph_status",
    toolset="codegraph",
    schema=_CODEGRAPH_STATUS_SCHEMA,
    handler=_handle_status,
    check_fn=_check_codegraph_available,
    is_async=True,
    emoji="📊",
)
