"""Run the hybrid SLM+Opus agent over SWE-bench Lite smoke instances and write
predictions in the format the official harness expects.

    python src/run_swebench.py                       # default smoke set
    python src/run_swebench.py --instances a b c       # explicit instance ids
    python src/run_swebench.py --limit 3                # first N of smoke set

Then evaluate with the official harness (Docker):
    python -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Lite \
        --predictions_path data/swebench/preds/predictions.jsonl \
        --max_workers 1 --run_id hybrid_smoke --cache_level instance
"""
import os
import json
import argparse

from datasets import load_dataset
from swe_agent import solve_instance, MODEL_NAME

SMOKE = [
    "pallets__flask-4045",
    "psf__requests-2674",
    "pylint-dev__pylint-5859",
    "mwaskom__seaborn-3010",
    "psf__requests-3362",
]
PRED_PATH = "data/swebench/preds/predictions.jsonl"
SUMMARY_PATH = "data/swebench/preds/run_summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", help="explicit instance ids")
    ap.add_argument("--limit", type=int, default=None, help="first N of smoke set")
    ap.add_argument("--max-cloud", type=int, default=2)
    ap.add_argument("--n-files", type=int, default=6, help="how many BM25 candidate files to attempt")
    args = ap.parse_args()

    targets = args.instances or SMOKE
    if args.limit:
        targets = targets[:args.limit]

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_id = {x["instance_id"]: x for x in ds}

    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    results = []
    with open(PRED_PATH, "w") as pf:
        for sid in targets:
            if sid not in by_id:
                print(f"!! unknown instance {sid}, skipping")
                continue
            res = solve_instance(by_id[sid], max_cloud=args.max_cloud, n_files=args.n_files)
            pf.write(json.dumps(res["prediction"]) + "\n")
            results.append({k: v for k, v in res.items() if k != "prediction"})

    with open(SUMMARY_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}\nGENERATION SUMMARY\n{'=' * 60}")
    print(f"{'instance':<32}{'patch':<7}{'cloud':<7}{'tokens':<8}{'file'}")
    for r in results:
        print(f"{r['instance_id']:<32}{str(r['produced_patch']):<7}"
              f"{r['cloud_calls']:<7}{r['cloud_tokens']:<8}{r['target_file']}")
    total = sum(r["cloud_tokens"] for r in results)
    print(f"\nTotal cloud tokens: {total}")
    print(f"Predictions: {PRED_PATH}")
    print(f"Model name: {MODEL_NAME}")


if __name__ == "__main__":
    main()
