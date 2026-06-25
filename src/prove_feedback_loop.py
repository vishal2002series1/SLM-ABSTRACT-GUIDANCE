"""Deterministic proof of the multi-round cloud feedback loop.

We cannot reliably prove the escalation/feedback machinery by hoping a stochastic
SLM (and then Opus) both fail on a hard task -- that is fragile and token-costly.
Instead we stub the three external effects of the orchestrator:

  * local_slm.invoke   -> canned SLM responses (no Ollama)
  * bedrock_client.converse -> canned Opus responses, and CAPTURE the prompt sent
  * subprocess.run (pytest) -> always reports FAILURE

The real LangGraph `app` runs unmodified, so this exercises the genuine routing,
attempt-history accumulation, and prompt construction. We then ASSERT that:

  1. The loop performs MULTIPLE cloud rounds (not a single pass).
  2. Round 2's prompt to Opus contains the growing attempt digest (A1, A2, ...).
  3. Round 2's prompt flags the previous guidance as a dead end.
  4. attempt_history accumulates local + cloud entries in execution order.

Run: python src/prove_feedback_loop.py     (exits non-zero if any assertion fails)
"""
import os
import sys

import orchestrator as orch


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _SLMResponse:
    def __init__(self, content):
        self.content = content


class FakeSLM:
    """Stand-in for ChatOllama. Returns plausible content per node by sniffing the prompt."""
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        # abstract_problem asks for a single abstract query.
        if "purely abstract" in prompt:
            return _SLMResponse("How does the matching architecture reconcile variable-length wildcards?")
        # local fixes / apply_logic expect a python code block.
        return _SLMResponse("```python\nclass GlobMatcher:\n    def matches(self, p, t):\n        return False\n```")


class FakeBedrock:
    """Stand-in for the Bedrock client. Records every prompt and returns distinct guidance."""
    def __init__(self):
        self.prompts = []
        self.round = 0

    def converse(self, modelId, messages):
        prompt = messages[0]["content"][0]["text"]
        self.prompts.append(prompt)
        self.round += 1
        guidance = f"STRATEGY:round{self.round}-approach|STEPS:do-x~do-y|CONSTRAINT:only edit target"
        return {
            "output": {"message": {"content": [{"text": guidance}]}},
            "usage": {"inputTokens": 100 + self.round, "outputTokens": 50, "totalTokens": 150 + self.round},
        }


class _FakePytestResult:
    """Mimics subprocess.CompletedProcess for a failing pytest run."""
    returncode = 1
    stdout = ">       assert m.matches('a*a', 'aa') is True\nE       AssertionError: assert False is True"
    stderr = ""


def _fake_subprocess_run(*args, **kwargs):
    return _FakePytestResult()


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def run_proof():
    # The orchestrator writes edited files into a cwd-relative `sandbox/` dir.
    # subprocess (pytest) is stubbed, so the dir just needs to exist for the writes.
    os.makedirs("sandbox", exist_ok=True)
    with open(os.path.join("sandbox", "matcher.py"), "w") as f:
        f.write("class GlobMatcher: ...\n")

    fake_slm = FakeSLM()
    fake_bedrock = FakeBedrock()

    # Patch the orchestrator module globals in place; the compiled graph closes
    # over these names, so patching the module attribute is sufficient.
    orch.local_slm = fake_slm
    orch.bedrock_client = fake_bedrock
    orch.subprocess.run = _fake_subprocess_run

    initial_state = {
        "task_description": "Implement a glob matcher.",
        "target_file": "matcher.py",
        "test_code": "def test(): assert GlobMatcher().matches('a*a','aa')",
        "current_code": "class GlobMatcher: ...",
        "last_error": "",
        "iterations": 0,
        "max_iterations": 6,
        "abstract_query": "",
        "abstract_guidance": "",
        "test_passed": False,
        "metrics_log": [],
        "local_attempt_failed": False,
        "attempt_history": [],
    }

    final = orch.app.invoke(initial_state)

    # ----------------------------------------------------------------------- #
    # Assertions
    # ----------------------------------------------------------------------- #
    failures = []

    cloud_rounds = len(fake_bedrock.prompts)
    if cloud_rounds < 2:
        failures.append(f"[1] Expected >=2 cloud rounds, got {cloud_rounds}. Loop did not iterate.")

    if cloud_rounds >= 2:
        second = fake_bedrock.prompts[1]
        # The digest in round 2 must reference multiple prior attempts.
        if "A1[" not in second or "A2[" not in second:
            failures.append("[2] Round-2 prompt missing accumulated attempt digest (A1/A2).")
        # Round 2 must carry the dead-end flag referencing prior guidance.
        if "YOUR PREVIOUS GUIDANCE" not in second:
            failures.append("[3] Round-2 prompt did not flag previous guidance as a dead end.")
        # The two cloud prompts must differ (round-aware context, not a repeat).
        if fake_bedrock.prompts[0] == second:
            failures.append("[3b] Round-1 and round-2 prompts are identical (no new context).")

    history = final.get("attempt_history", [])
    sources = [a["source"] for a in history]
    if "local" not in sources or "cloud" not in sources:
        failures.append(f"[4] attempt_history missing local/cloud mix: {sources}")
    # Every non-final attempt should have its resulting_error paired in.
    unpaired = [a for a in history[:-1] if not a.get("resulting_error")]
    if unpaired:
        failures.append(f"[4b] {len(unpaired)} attempt(s) never got their resulting_error paired.")

    # ----------------------------------------------------------------------- #
    # Report
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 66)
    print("FEEDBACK-LOOP PROOF RESULTS")
    print("=" * 66)
    print(f"Cloud consult rounds         : {cloud_rounds}")
    print(f"Total iterations run         : {final['iterations']}")
    print(f"Attempt-history length       : {len(history)}")
    print(f"Attempt sources (in order)   : {sources}")
    print(f"Attempt strategies           : {[a['strategy'] for a in history]}")
    if cloud_rounds >= 2:
        print("\n--- Round-2 prompt sent to Opus (verbatim) ---")
        print(fake_bedrock.prompts[1])
        print("--- end ---")

    print("\n" + "-" * 66)
    if failures:
        print("RESULT: FAILED")
        for f in failures:
            print("  " + f)
        return 1
    print("RESULT: PASSED — multi-round feedback loop verified end-to-end.")
    print("  * Loop escalated to the cloud multiple times.")
    print("  * Round 2 carried the accumulated attempt history to Opus.")
    print("  * Round 2 flagged the prior guidance as a failed dead-end.")
    print("  * attempt_history recorded the local+cloud sequence with paired errors.")
    return 0


if __name__ == "__main__":
    sys.exit(run_proof())
