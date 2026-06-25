"""Pure-Opus baseline: how many cloud tokens to solve a task with NO local SLM.

This is the comparison point for the hybrid pipeline. Given a task's files and its
test suite, we ask Opus once to produce the full fix and record the tokens spent.
We do NOT execute the result -- the baseline measures the *cost* of the all-cloud
approach (the thing the hybrid is trying to reduce), consistent with how
run_evaluation.py already framed it.

Works for both workload shapes:
  * single-file: pass {filename: code} with one editable file
  * multi-file (codebase): pass {filename: code} for every editable file
"""
import boto3

CLOUD_MODEL_ID = "us.anthropic.claude-opus-4-8"
_client = None


def _bedrock():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name="us-east-1")
    return _client


def pure_opus_baseline_tokens(description: str, editable_files: dict, test_code: str) -> int:
    """Query standalone Opus to solve the task; return total tokens spent.

    editable_files: {filename: current_contents} for every file Opus may edit.
    """
    file_blocks = "\n\n".join(
        f"File: {name}\n```python\n{code}\n```" for name, code in editable_files.items()
    )
    prompt = (
        f"You are an autonomous engineer. Implement the task below by fixing/extending "
        f"the given files so the test suite passes.\n\n"
        f"Task: {description}\n\n"
        f"{file_blocks}\n\n"
        f"File: test_suite.py (fixed, do not change)\n```python\n{test_code}\n```\n\n"
        f"Output the complete corrected contents of every file you change, each inside "
        f"its own code fence labeled with the filename."
    )
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    response = _bedrock().converse(modelId=CLOUD_MODEL_ID, messages=messages)
    return response["usage"]["totalTokens"]


def savings_percent(baseline_tokens: int, hybrid_tokens: int):
    """Percent of cloud tokens saved by the hybrid vs pure Opus. None if baseline is 0."""
    if not baseline_tokens:
        return None
    return round((baseline_tokens - hybrid_tokens) / baseline_tokens * 100, 2)
