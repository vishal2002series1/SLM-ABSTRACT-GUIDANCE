import os
import csv
import shutil
import subprocess
from extended_benchmarks import COMPLEX_BENCHMARK_TASKS
from orchestrator import app

STRESS_CSV_PATH = "data/stress_test_results.csv"

def prepare_sandbox_repo(task):
    # Completely nuke and recreate sandbox to ensure no cross-contamination
    if os.path.exists("sandbox"):
        shutil.rmtree("sandbox")
    os.makedirs("sandbox", exist_ok=True)
    
    # Write the file framework dynamically
    for filename, code in task["files"].items():
        with open(f"sandbox/{filename}", "w") as f:
            f.write(code)

def run_stress_suite():
    print(f"Initializing Stress Test Suite across {len(COMPLEX_BENCHMARK_TASKS)} highly nested codebase tasks...")
    
    # Init ledger
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(STRESS_CSV_PATH):
        with open(STRESS_CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Task ID", "Success", "Total Loops Run", "Total Cloud Encounters", "Cloud Tokens Used"])

    for task in COMPLEX_BENCHMARK_TASKS:
        print(f"\n==================================================================")
        print(f"STRESS TESTING: {task['id']}")
        print(f"==================================================================")
        
        # Staging multi-file environment
        prepare_sandbox_repo(task)

        # The file the agent must actually edit (the one the test suite imports).
        primary_file = [f for f in task["files"].keys() if f != "test_suite.py"][0]

        # Read the baseline buggy code
        with open(f"sandbox/{primary_file}", "r") as f:
            buggy_content = f.read()

        initial_state = {
            "task_description": task["description"],
            "target_file": primary_file,
            "test_code": task["files"].get("test_suite.py", ""),
            "current_code": buggy_content,
            "last_error": "",
            "iterations": 0,
            "max_iterations": 6,
            "abstract_query": "",
            "abstract_guidance": "",
            "test_passed": False,
            "metrics_log": [],
            "local_attempt_failed": False,
            "attempt_history": []
        }

        # Invoke LangGraph loop. The orchestrator now edits `primary_file` directly,
        # so no cart.py copy shim is needed (tests import the real module).
        final_output = app.invoke(initial_state)

        # Compile metrics
        total_tokens = sum([entry["total_tokens"] for entry in final_output["metrics_log"]])
        
        with open(STRESS_CSV_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                task["id"],
                final_output["test_passed"],
                final_output["iterations"],
                len(final_output["metrics_log"]),
                total_tokens
            ])
            
        print(f"\nCompleted {task['id']}. Saved performance matrix to {STRESS_CSV_PATH}")

if __name__ == "__main__":
    run_stress_suite()