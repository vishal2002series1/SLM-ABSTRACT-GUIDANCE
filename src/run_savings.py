"""Token-savings measurement: hybrid (SLM+Opus) vs pure-Opus baseline.

Segregable by workload so each can be run/measured independently:

    python src/run_savings.py --workload single     # single-file tasks 06-10
    python src/run_savings.py --workload codebase    # multi-file feature
    python src/run_savings.py --workload both        # both (default)

Each workload writes its OWN csv under data/, so results never mix:
    data/savings_single.csv
    data/savings_codebase.csv

For every task we record: hybrid cloud tokens (the loop's Opus spend), pure-Opus
baseline tokens (solo Opus), and the savings %. Hybrid success is also recorded so
a "0-token saving" isn't mistaken for a win when the hybrid actually failed.
"""
import os
import csv
import shutil
import argparse

from baseline import pure_opus_baseline_tokens, savings_percent

DATA_DIR = "data"
SINGLE_CSV = os.path.join(DATA_DIR, "savings_single.csv")
CODEBASE_CSV = os.path.join(DATA_DIR, "savings_codebase.csv")

HEADER = ["Task ID", "Hybrid Success", "Hybrid Cloud Tokens",
          "Pure Opus Baseline Tokens", "Token Savings %"]


def _stage_repo(files: dict):
    if os.path.exists("sandbox"):
        shutil.rmtree("sandbox")
    os.makedirs("sandbox", exist_ok=True)
    for filename, code in files.items():
        with open(os.path.join("sandbox", filename), "w") as f:
            f.write(code)


def _write_rows(path, rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def _fmt_pct(p):
    return "n/a" if p is None else f"{p}%"


# --------------------------------------------------------------------------- #
# Single-file workload (tasks 06-10 via the single-file orchestrator)
# --------------------------------------------------------------------------- #
def run_single():
    from extended_benchmarks import COMPLEX_BENCHMARK_TASKS
    from orchestrator import app

    rows = []
    print("\n##### WORKLOAD: single-file #####")
    for task in COMPLEX_BENCHMARK_TASKS:
        print(f"\n=== {task['id']} ===")
        primary = [f for f in task["files"].keys() if f != "test_suite.py"][0]
        test_code = task["files"].get("test_suite.py", "")

        # 1) Pure-Opus baseline (solo).
        baseline = pure_opus_baseline_tokens(
            task["description"], {primary: task["files"][primary]}, test_code)

        # 2) Hybrid pipeline.
        _stage_repo(task["files"])
        initial = {
            "task_description": task["description"],
            "target_file": primary,
            "test_code": test_code,
            "current_code": task["files"][primary],
            "last_error": "", "iterations": 0, "max_iterations": 6,
            "abstract_query": "", "abstract_guidance": "",
            "test_passed": False, "metrics_log": [], "local_attempt_failed": False,
            "attempt_history": [],
        }
        final = app.invoke(initial)
        hybrid = sum(e["total_tokens"] for e in final["metrics_log"])
        pct = savings_percent(baseline, hybrid)
        rows.append([task["id"], final["test_passed"], hybrid, baseline, _fmt_pct(pct)])
        print(f">> baseline={baseline}  hybrid={hybrid}  savings={_fmt_pct(pct)}  success={final['test_passed']}")

    _write_rows(SINGLE_CSV, rows)
    print(f"\nSaved single-file results to {SINGLE_CSV}")
    return rows


# --------------------------------------------------------------------------- #
# Codebase workload (multi-file feature via the multi-file orchestrator)
# --------------------------------------------------------------------------- #
def run_codebase():
    from codebase_benchmark import CODEBASE_FEATURE_TASK
    from codebase_orchestrator import app

    task = CODEBASE_FEATURE_TASK
    editable = task["editable_files"]
    test_code = task["files"][task["test_file"]]
    print("\n##### WORKLOAD: codebase (multi-file) #####")
    print(f"\n=== {task['id']} ===")

    # 1) Pure-Opus baseline over ALL editable files at once.
    baseline = pure_opus_baseline_tokens(
        task["description"], {f: task["files"][f] for f in editable}, test_code)

    # 2) Hybrid multi-file pipeline.
    _stage_repo(task["files"])
    initial = {
        "task_description": task["description"],
        "feature_name": task["feature_name"],
        "editable_files": editable,
        "test_code": test_code,
        "files": {f: task["files"][f] for f in editable},
        "last_error": "", "iterations": 0, "max_iterations": 8,
        "abstract_query": "", "abstract_guidance": "",
        "test_passed": False, "metrics_log": [], "local_attempt_failed": False,
        "attempt_history": [],
    }
    final = app.invoke(initial)
    hybrid = sum(e["total_tokens"] for e in final["metrics_log"])
    pct = savings_percent(baseline, hybrid)
    rows = [[task["id"], final["test_passed"], hybrid, baseline, _fmt_pct(pct)]]
    print(f">> baseline={baseline}  hybrid={hybrid}  savings={_fmt_pct(pct)}  success={final['test_passed']}")

    _write_rows(CODEBASE_CSV, rows)
    print(f"\nSaved codebase results to {CODEBASE_CSV}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Measure hybrid vs pure-Opus token savings.")
    parser.add_argument("--workload", choices=["single", "codebase", "both"],
                        default="both", help="Which workload to measure (default: both).")
    args = parser.parse_args()

    if args.workload in ("single", "both"):
        run_single()
    if args.workload in ("codebase", "both"):
        run_codebase()

    # Clean up the shared sandbox so it doesn't leak between runs.
    if os.path.exists("sandbox"):
        shutil.rmtree("sandbox")
    print("\nDone.")


if __name__ == "__main__":
    main()
