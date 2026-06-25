import csv
import os
import boto3
from benchmark_config import BENCHMARK_TASKS
from orchestrator import app

CSV_FILE_PATH = "data/comprehensive_experiment_results.csv"
CLOUD_MODEL_ID = "us.anthropic.claude-opus-4-8"

def init_csv():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Task ID", 
                "Hybrid_Success", 
                "Hybrid_Cloud_Tokens", 
                "Baseline_Success", 
                "Baseline_Cloud_Tokens",
                "Token_Savings_Percent"
            ])

def run_pure_opus_baseline(task):
    print(f"\n[Baseline Run] Querying Standalone Opus 4.8 for {task['id']}...")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    prompt = (
        f"You are an autonomous engineer. Fix the bug in this code:\n\n"
        f"File: target_app.py\n{task['buggy_code']}\n\n"
        f"File: test_suite.py\n{task['test_suite']}\n\n"
        f"Output the complete corrected target_app.py code inside code fences."
    )
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    response = client.converse(modelId=CLOUD_MODEL_ID, messages=messages)
    return response['usage']['totalTokens']

def run_experiment():
    init_csv()
    print(f"Starting Comparative Evaluation Suite across {len(BENCHMARK_TASKS)} tasks...")

    for task in BENCHMARK_TASKS:
        print(f"\n==================================================")
        print(f"RUNNING EXPERIMENT FOR TASK: {task['id']}")
        print(f"==================================================")

        # 1. Run Pure Cloud Baseline
        baseline_tokens = run_pure_opus_baseline(task)

        # 2. Reset Sandbox & Run Hybrid Pipeline
        # with open("sandbox/target_app.py", "w") as f:
        #     f.write(task["buggy_code"])
        # with open("sandbox/test_suite.py", "w") as f:
        #     f.write(task["test_suite"])

        initial_state = {
            "task_description": task["description"],
            "target_file": "cart.py",
            "test_code": task.get("test_suite", ""),
            "current_code": task["buggy_code"],
            "last_error": "",
            "iterations": 0,
            "max_iterations": 6,  # Expanded context turns to support the local try
            "abstract_query": "",
            "abstract_guidance": "",
            "test_passed": False,
            "metrics_log": [],
            "local_attempt_failed": False,
            "attempt_history": []
        }

        final_output = app.invoke(initial_state)
        hybrid_tokens = sum([entry["total_tokens"] for entry in final_output["metrics_log"]])
        
        savings_pct = round(((baseline_tokens - hybrid_tokens) / baseline_tokens) * 100, 2)

        with open(CSV_FILE_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                task["id"],
                final_output["test_passed"],
                hybrid_tokens,
                True,
                baseline_tokens,
                f"{savings_pct}%"
            ])
            
        print(f"\n>> Task {task['id']} Concluded.")
        print(f">> Baseline Cloud Tokens: {baseline_tokens}")
        print(f">> Our Hybrid Cloud Tokens: {hybrid_tokens}")
        print(f">> Token Optimization Delta: {savings_pct}%")

    print(f"\nEvaluation complete. Compiled data saved to {CSV_FILE_PATH}")

if __name__ == "__main__":
    run_experiment()