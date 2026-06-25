"""Run the codebase-level feature experiment.

Stages the multi-file mini-app into sandbox/, then drives the multi-file
orchestrator to add the discount-codes feature across models/storage/service.
Reports per-attempt file edits, cloud token spend, and final pass/fail.
"""
import os
import csv
import shutil

from codebase_benchmark import CODEBASE_FEATURE_TASK
from codebase_orchestrator import app

RESULTS_CSV = "data/codebase_feature_results.csv"


def stage_repo(task):
    if os.path.exists("sandbox"):
        shutil.rmtree("sandbox")
    os.makedirs("sandbox", exist_ok=True)
    for filename, code in task["files"].items():
        with open(os.path.join("sandbox", filename), "w") as f:
            f.write(code)


def run():
    task = CODEBASE_FEATURE_TASK
    print("=" * 66)
    print(f"CODEBASE FEATURE TEST: {task['id']} ({task['feature_name']})")
    print("=" * 66)

    stage_repo(task)
    editable = task["editable_files"]

    initial_state = {
        "task_description": task["description"],
        "feature_name": task["feature_name"],
        "editable_files": editable,
        "test_code": task["files"][task["test_file"]],
        "files": {f: task["files"][f] for f in editable},
        "last_error": "",
        "iterations": 0,
        "max_iterations": 8,  # multi-file is harder; allow a couple more rounds
        "abstract_query": "",
        "abstract_guidance": "",
        "test_passed": False,
        "metrics_log": [],
        "local_attempt_failed": False,
        "attempt_history": [],
    }

    final = app.invoke(initial_state)

    total_tokens = sum(e["total_tokens"] for e in final["metrics_log"])
    cloud_rounds = len(final["metrics_log"])

    os.makedirs("data", exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Task ID", "Feature", "Success", "Iterations", "Cloud Rounds", "Cloud Tokens"])
        w.writerow([task["id"], task["feature_name"], final["test_passed"],
                    final["iterations"], cloud_rounds, total_tokens])

    print("\n" + "=" * 66)
    print("CODEBASE FEATURE RESULT")
    print("=" * 66)
    print(f"Success           : {final['test_passed']}")
    print(f"Iterations        : {final['iterations']}")
    print(f"Cloud rounds       : {cloud_rounds}")
    print(f"Cloud tokens used  : {total_tokens}")
    print(f"Attempt sources    : {[a['source'] for a in final['attempt_history']]}")
    print(f"Attempt summaries  :")
    for i, a in enumerate(final["attempt_history"], 1):
        print(f"  A{i} [{a['source']}:{a['strategy']}] {a['change_summary']}")
    print(f"\nSaved to {RESULTS_CSV}")


if __name__ == "__main__":
    run()
