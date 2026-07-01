# hermes_cli/pipeline_status.py – shared data gatherer for /pipeline
#
# Reads model deployment, benchmark results, kanban tasks, desktop disk.
# Single entry point: gather_pipeline_data() -> dict
# CLI and gateway format the output themselves.

import datetime
import os
import sqlite3
import subprocess

BENCHMARK_PATH = "/home/floory/FYP-AI-Workspace/model-training/benchmarks/v3-results-summary.md"
KANBAN_DB_PATH = "/home/floory/.hermes/kanban.db"

# Source of truth: NixOS llama-swap config at /etc/nixos/modules/services/llama-cpp.nix
# Primary models extracted from that config (the ones actually used by Hermes)
MODELS = [
    {
        "id": "qwen3.6-35b-a3b-thinking",
        "name": "Qwen3.6 35B A3B Thinking (IQ3_S) (100k)",
        "path": "/mnt/ssd/qwen/qwen3.6-35b-a3b/qwen3.6-35b-a3b.Q4_K_H.gguf",
        "size_gb": 17,
        "reasoning": True,
    },
    {
        "id": "qwen3.6-35b-a3b-instruct",
        "name": "Qwen3.6 35B A3B Instruct (IQ3_S) (64k)",
        "path": "/mnt/ssd/qwen/qwen3.6-35b-a3b/qwen3.6-35b-a3b.Q4_K_H.gguf",
        "size_gb": 17,
        "reasoning": False,
    },
    {
        "id": "carnice-v2-27b",
        "name": "Carnice V2 27B (Q5_K_M) (128k)",
        "path": "/var/lib/llm-models/kai-os/carnice-v2-27b/carnice-v2-27b-Q5_K_M.gguf",
        "size_gb": 17,
    },
    {
        "id": "qwen3.6-27b-thinking",
        "name": "Qwen3.6 27B Thinking (Q6_K_H) (128k)",
        "path": "/mnt/ssd/qwen/qwen3.6-27b/qwen3.6-27b.Q6_K_H.gguf",
        "size_gb": 17,
    },
]


def _read_benchmarks() -> dict:
    """Read v3 benchmark summary."""
    data = {
        "file_found": False,
        "model": None,
        "ifeval_prompt_strict": None,
        "ifeval_vs_v2": None,
        "ifeval_vs_baseline": None,
        "style_composite": None,
        "style_vs_v2": None,
        "style_vs_baseline": None,
        "verdict": None,
    }
    try:
        with open(BENCHMARK_PATH) as f:
            text = f.read()
        data["file_found"] = True
        for line in text.splitlines():
            if line.startswith("**Model:**"):
                data["model"] = line.split("**Model:**", 1)[-1].strip().strip("**")
            elif "IFEval prompt_strict" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    data["ifeval_prompt_strict"] = parts[2]
                if len(parts) >= 4:
                    data["ifeval_vs_v2"] = parts[3]
                if len(parts) >= 5:
                    data["ifeval_vs_baseline"] = parts[4]
            elif "Style composite" in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    data["style_composite"] = parts[2]
                if len(parts) >= 4:
                    data["style_vs_v2"] = parts[3]
                if len(parts) >= 5:
                    data["style_vs_baseline"] = parts[4]
            elif line.startswith("**Verdict:**"):
                data["verdict"] = line.split("**Verdict:**", 1)[-1].strip()
    except FileNotFoundError:
        pass
    return data


def _read_kanban_tasks() -> dict:
    """Query kanban DB for active tasks."""
    result = {
        "available": False,
        "tasks": [],
        "error": None,
    }
    try:
        if not os.path.isfile(KANBAN_DB_PATH) or os.path.getsize(KANBAN_DB_PATH) == 0:
            result["error"] = "DB empty or missing"
            return result
        conn = sqlite3.connect(KANBAN_DB_PATH)
        cur = conn.cursor()
        # Test if DB is valid
        cur.execute("SELECT COUNT(*) FROM tasks")
        count = cur.fetchone()[0]
        if count == 0:
            result["available"] = True
            result["tasks"] = []
            conn.close()
            return result
        cur.execute(
            "SELECT id, title, assignee, status, priority "
            "FROM tasks WHERE status NOT IN ('done','cancelled','archived') "
            "ORDER BY created_at DESC LIMIT 15"
        )
        for row in cur.fetchall():
            result["tasks"].append({
                "id": row[0],
                "title": row[1],
                "assignee": row[2],
                "status": row[3],
                "priority": row[4],
            })
        conn.close()
        result["available"] = True
    except sqlite3.DatabaseError as e:
        result["error"] = f"DB corrupted: {e}"
    except Exception as e:
        result["error"] = str(e)
    return result


def _check_desktop_disk() -> dict:
    """Attempt SSH to desktop for /var/lib/llm-models/ usage."""
    result = {
        "reachable": False,
        "usage_percent": None,
        "used_gb": None,
        "total_gb": None,
        "error": None,
    }
    try:
        r = subprocess.run(
            [
                    "ssh", "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no",
                    "100.93.40.51",
                "df -B1 /var/lib/llm-models/ 2>/dev/null | tail -1",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split()
            if len(parts) >= 4:
                total_b = int(parts[1]) if parts[1].isdigit() else 0
                used_b = int(parts[2]) if parts[2].isdigit() else 0
                if total_b > 0:
                    result["usage_percent"] = round(used_b / total_b * 100, 1)
                    result["used_gb"] = round(used_b / (1024**3), 1)
                    result["total_gb"] = round(total_b / (1024**3), 1)
                    result["reachable"] = True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        result["error"] = str(e)
    return result


def _get_pending_steps(benchmarks: dict) -> list[str]:
    """Infer next pipeline steps from benchmark state."""
    steps = []
    if benchmarks.get("ifeval_prompt_strict"):
        score = benchmarks["ifeval_prompt_strict"]
        try:
            if float(score) < 0.5:
                steps.append("Train v4 — improve IFEval (try alpha=8, r=1 or r=2)")
        except ValueError:
            pass
    if benchmarks.get("file_found"):
        steps.append("Re-evaluate v2 on 200 samples for fair comparison")
    if benchmarks.get("style_composite") and benchmarks.get("style_vs_baseline"):
        comp = benchmarks["style_composite"]
        vs_base = benchmarks["style_vs_baseline"]
        steps.append("Plan multi-task training: style + IFEval mix")
    steps.append("Run cleanup: archive old GGUF files from /var/lib/llm-models/")
    return steps


def gather_pipeline_data() -> dict:
    """Gather all pipeline status data. Returns dict with sections."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    benchmarks = _read_benchmarks()
    kanban = _read_kanban_tasks()
    disk = _check_desktop_disk()
    steps = _get_pending_steps(benchmarks)

    # Determine which model is "current" (latest trained)
    current_model = benchmarks.get("model") or "qwen3.6-35b-a3b-thinking (default)"

    return {
        "timestamp": now,
        "current_model": current_model,
        "deployed_models": MODELS,
        "benchmarks": benchmarks,
        "kanban": kanban,
        "desktop_disk": disk,
        "pending_steps": steps,
        "llama_swap_port": 8000,
        "llama_swap_host": "100.93.40.51",
    }
