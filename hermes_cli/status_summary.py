# hermes_cli/status_summary.py — session overview for /status
#
# Gathers: active kanban tasks, done-today, pipeline state, pending topics,
# background processes. Single entry: gather_status_data() -> dict

import datetime
import os
import sqlite3
import subprocess

KANBAN_DB_PATH = "/home/floory/.hermes/kanban.db"
BENCHMARK_PATH = "/home/floory/FYP-AI-Workspace/model-training/benchmarks/v3-results-summary.md"
PENDING_TOPICS_PATH = "/home/floory/Documents/Obsidian Vault/pending-topics.md"


def _read_kanban_active() -> list[dict]:
    """Running + blocked + ready + todo tasks."""
    tasks = []
    try:
        if not os.path.isfile(KANBAN_DB_PATH):
            return tasks
        conn = sqlite3.connect(KANBAN_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, assignee, status, priority "
            "FROM tasks WHERE status IN ('running','blocked','ready','todo') "
            "ORDER BY CASE status "
            "  WHEN 'running' THEN 1 WHEN 'blocked' THEN 2 "
            "  WHEN 'ready' THEN 3 WHEN 'todo' THEN 4 "
            "END, created_at DESC"
        )
        for row in cur.fetchall():
            tasks.append({
                "id": row[0],
                "title": row[1],
                "assignee": row[2],
                "status": row[3],
                "priority": row[4],
            })
        conn.close()
    except Exception:
        pass
    return tasks


def _read_kanban_done_today() -> list[dict]:
    """Tasks completed since midnight local."""
    tasks = []
    try:
        if not os.path.isfile(KANBAN_DB_PATH):
            return tasks
        today_start = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        conn = sqlite3.connect(KANBAN_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, assignee, completed_at "
            "FROM tasks WHERE status='done' AND completed_at > ? "
            "ORDER BY completed_at DESC LIMIT 20",
            (today_start,),
        )
        for row in cur.fetchall():
            ts = datetime.datetime.fromtimestamp(row[3]).strftime("%H:%M") if row[3] else "?"
            tasks.append({
                "id": row[0],
                "title": row[1],
                "assignee": row[2],
                "time": ts,
            })
        conn.close()
    except Exception:
        pass
    return tasks


def _read_pending_topics() -> list[str]:
    """Parse active pending topics from Obsidian vault."""
    topics = []
    try:
        if not os.path.isfile(PENDING_TOPICS_PATH):
            return topics
        with open(PENDING_TOPICS_PATH) as f:
            text = f.read()
        # Grab lines under ## Active
        in_active = False
        for line in text.splitlines():
            if line.strip() == "## Active":
                in_active = True
                continue
            if in_active:
                if line.startswith("## "):
                    break
                stripped = line.strip().strip("- ")
                if stripped and not stripped.startswith("<!--"):
                    topics.append(stripped)
    except Exception:
        pass
    return topics


def _read_benchmark_blob() -> dict | None:
    """Quick parse of v3 benchmark for summary."""
    data = None
    try:
        if not os.path.isfile(BENCHMARK_PATH):
            return None
        with open(BENCHMARK_PATH) as f:
            text = f.read()
        data = {}
        for line in text.splitlines():
            if "IFEval prompt_strict" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    data["ifeval"] = parts[2]
            elif "Style composite" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    data["style"] = parts[2]
            if line.startswith("**Model:**"):
                data["model"] = line.split("**Model:**", 1)[-1].strip().strip("**")
    except Exception:
        pass
    return data


def _running_processes() -> list[str]:
    """Check for known background processes."""
    procs = []
    try:
        # Check for ssh / training / etc.
        result = subprocess.run(
            ["pgrep", "-a", "-f", "python.*train"], capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) >= 2:
                    procs.append(f"training ({parts[1][:60]})")
                else:
                    procs.append(f"training (pid {parts[0]})")
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["pgrep", "-a", "-f", "llama-server"], capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                procs.append("llama-server (inference)")
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["pgrep", "-a", "-f", "ssh.*100.93.40.51"], capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                procs.append("ssh desktop")
    except Exception:
        pass
    return procs


def gather_status_data() -> dict:
    """Gather everything for /status display."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    active_tasks = _read_kanban_active()
    done_today = _read_kanban_done_today()
    topics = _read_pending_topics()
    benchmark = _read_benchmark_blob()
    procs = _running_processes()

    # Distinguish running vs blocked vs ready/todo
    running = [t for t in active_tasks if t["status"] == "running"]
    blocked = [t for t in active_tasks if t["status"] == "blocked"]
    pending = [t for t in active_tasks if t["status"] in ("ready", "todo")]

    return {
        "timestamp": now,
        "running_tasks": running,
        "blocked_tasks": blocked,
        "pending_tasks": pending,
        "done_today": done_today,
        "pending_topics": topics,
        "benchmark": benchmark,
        "background_procs": procs,
    }
