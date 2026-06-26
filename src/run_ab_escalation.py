"""A/B test: does the SLM abstraction step help or hurt cloud escalations?

Arm A "abstract" (default pipeline): SLM distills failure -> abstract query -> Opus.
Arm B "direct": skip abstraction; send Opus the raw symptom + actual code directly.

We run each task under BOTH arms (fresh sandbox each time) and compare, per task:
  * success
  * cloud rounds (escalations)
  * cloud tokens spent

Only tasks that ESCALATE exercise the difference (the SLM solves the rest alone with
0 cloud tokens in both arms). Those are highlighted in the summary.

Important: arm B sends source code to the cloud, trading away the privacy/IP property
that the abstract arm preserves. This harness measures the token/quality side only;
weigh that tradeoff separately.

Usage:
    python src/run_ab_escalation.py                 # both arms, all single-file tasks
    python src/run_ab_escalation.py --arm direct     # only the direct arm
    python src/run_ab_escalation.py --arm abstract    # only the abstract arm
"""
import os
import csv
import shutil
import argparse
import importlib

DATA_DIR = "data"
AB_CSV = os.path.join(DATA_DIR, "ab_escalation_results.csv")


def _stage(files):
    if os.path.exists("sandbox"):
        shutil.rmtree("sandbox")
    os.makedirs("sandbox", exist_ok=True)
    for name, code in files.items():
        with open(os.path.join("sandbox", name), "w") as f:
            f.write(code)


def run_arm(mode):
    """Run all single-file tasks under one escalation mode. Returns list of result dicts."""
    # Set BOTH switches BEFORE importing/reloading so the module-level constants pick
    # them up. FORCE_ESCALATE removes the stochastic autonomous-fix confound: both arms
    # escalate from the identical buggy code, so we measure ONLY guidance quality.
    os.environ["ESCALATION_MODE"] = mode
    os.environ["FORCE_ESCALATE"] = "1"
    import orchestrator
    importlib.reload(orchestrator)
    from extended_benchmarks import COMPLEX_BENCHMARK_TASKS

    print(f"\n{'#' * 66}\n# ARM: {mode}\n{'#' * 66}")
    results = []
    for task in COMPLEX_BENCHMARK_TASKS:
        print(f"\n=== [{mode}] {task['id']} ===")
        primary = [f for f in task["files"].keys() if f != "test_suite.py"][0]
        test_code = task["files"].get("test_suite.py", "")
        _stage(task["files"])
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
        final = orchestrator.app.invoke(initial)
        tokens = sum(e["total_tokens"] for e in final["metrics_log"])
        in_tok = sum(e["input_tokens"] for e in final["metrics_log"])
        out_tok = sum(e["output_tokens"] for e in final["metrics_log"])
        rounds = len(final["metrics_log"])
        results.append({
            "task": task["id"], "mode": mode, "success": final["test_passed"],
            "cloud_rounds": rounds, "cloud_tokens": tokens,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "escalated": rounds > 0,
        })
        print(f">> [{mode}] success={final['test_passed']} rounds={rounds} "
              f"tokens={tokens} (in={in_tok} out={out_tok})")
    return results


def summarize(all_results):
    by_task = {}
    for r in all_results:
        by_task.setdefault(r["task"], {})[r["mode"]] = r

    print(f"\n{'=' * 86}\nA/B ESCALATION COMPARISON (FORCE_ESCALATE: both arms start from identical buggy code)\n{'=' * 86}")
    print(f"{'Task':<34}{'Arm':<10}{'Success':<9}{'Rounds':<8}{'Total':<8}{'In':<8}{'Out':<7}")
    print("-" * 86)
    for task, arms in by_task.items():
        for mode, r in arms.items():
            print(f"{task:<34}{mode:<10}{str(r['success']):<9}{r['cloud_rounds']:<8}"
                  f"{r['cloud_tokens']:<8}{r['input_tokens']:<8}{r['output_tokens']:<7}")
        print("-" * 86)

    print("\nAGGREGATE (all tasks now escalate under FORCE_ESCALATE):")
    modes = sorted({r["mode"] for r in all_results})
    for mode in modes:
        rs = [r for r in all_results if r["mode"] == mode]
        if not rs:
            continue
        tot = sum(r["cloud_tokens"] for r in rs)
        rounds = sum(r["cloud_rounds"] for r in rs)
        ok = sum(1 for r in rs if r["success"])
        print(f"  {mode:<10} total_tokens={tot:<7} cloud_rounds={rounds:<4} success={ok}/{len(rs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["abstract", "direct", "both"], default="both")
    args = parser.parse_args()

    all_results = []
    if args.arm in ("abstract", "both"):
        all_results += run_arm("abstract")
    if args.arm in ("direct", "both"):
        all_results += run_arm("direct")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(AB_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task ID", "Arm", "Success", "Cloud Rounds", "Total Tokens",
                    "Input Tokens", "Output Tokens", "Escalated"])
        for r in all_results:
            w.writerow([r["task"], r["mode"], r["success"], r["cloud_rounds"],
                        r["cloud_tokens"], r["input_tokens"], r["output_tokens"], r["escalated"]])

    summarize(all_results)
    if os.path.exists("sandbox"):
        shutil.rmtree("sandbox")
    print(f"\nSaved A/B results to {AB_CSV}")


if __name__ == "__main__":
    main()
