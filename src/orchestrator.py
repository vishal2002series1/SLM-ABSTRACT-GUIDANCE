import os
import subprocess
from typing import TypedDict, List, Dict, Any
import boto3
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


class AttemptRecord(TypedDict):
    source: str           # "local" or "cloud"
    strategy: str         # short label of what was attempted
    change_summary: str   # brief note of what the SLM changed
    resulting_error: str  # condensed error AFTER this attempt (filled by execute_and_test)


class AgentState(TypedDict):
    task_description: str
    target_file: str
    test_code: str
    current_code: str
    last_error: str
    iterations: int
    max_iterations: int
    abstract_query: str
    abstract_guidance: str
    test_passed: bool
    metrics_log: List[Dict[str, Any]]
    local_attempt_failed: bool
    attempt_history: List[AttemptRecord]


local_slm = ChatOllama(model="gemma4:e4b", temperature=0)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
CLOUD_MODEL_ID = "us.anthropic.claude-opus-4-8"

# Escalation mode A/B switch (env ESCALATION_MODE):
#   "abstract" (default): SLM distills the failure into a purely abstract query (no code
#                         leaves local); Opus reasons from that + the raw symptom.
#   "direct":             SKIP the SLM abstraction; send Opus the raw symptom + the actual
#                         code directly. Tests whether the abstraction step is a liability.
# Note: "direct" sends source to the cloud, trading away the privacy/IP property that
# "abstract" preserves -- this A/B measures what the abstraction costs vs. buys.
ESCALATION_MODE = os.environ.get("ESCALATION_MODE", "abstract")

# A/B fairness switch (env FORCE_ESCALATE=1): skip the stochastic autonomous local
# fix so BOTH arms escalate from the identical buggy starting code. Without this the
# SLM may solve a task alone in one arm and not the other, confounding the comparison
# of guidance quality. Off by default (normal hybrid behavior).
FORCE_ESCALATE = os.environ.get("FORCE_ESCALATE", "0") == "1"

# Shared scope constraints. The agent can ONLY edit the target source file; the test
# suite is ground truth and the environment is fixed. Stating this prevents the SLM/Opus
# from chasing unactionable fixes (e.g. "make the test async", "pip install X") and from
# changing a function's call contract (the async-coroutine rabbit hole seen in task_07).
def _action_space_rules(target: str) -> str:
    return (
        f"HARD CONSTRAINTS (non-negotiable):\n"
        f"1. You may ONLY modify `sandbox/{target}`. You CANNOT edit the test file, "
        f"install packages, or change the environment.\n"
        f"2. The test suite is FIXED and CORRECT. It defines the required behavior. "
        f"Never propose changing a test.\n"
        f"3. Match the call contract the tests assume. If a test calls `obj.method(...)` "
        f"synchronously and compares the result, the method MUST be synchronous "
        f"(do NOT make it `async`/return a coroutine).\n"
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _target_path(state: AgentState) -> str:
    """Resolve the sandbox file this run is actually responsible for editing."""
    return os.path.join("sandbox", state.get("target_file", "cart.py"))


def _extract_code(content: str) -> str:
    """Pull a python code block out of an LLM response, falling back to raw text."""
    if "```python" in content:
        return content.split("```python")[1].split("```")[0].strip()
    if "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content.strip()


def condense_error(raw: str, max_lines: int = 8, hard_cap: int = 600) -> str:
    """Reduce a full pytest dump to the few high-signal lines worth sending upstream."""
    if not raw:
        return ""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    signal_markers = ("Error", "assert", "Assertion", "FAILED", "Exception", "Traceback", "E ")
    signal = [l for l in lines if any(m in l for m in signal_markers)]
    chosen = (signal or lines)[-max_lines:]
    return " | ".join(chosen)[:hard_cap]


def _extract_strategy(toon: str) -> str:
    """Best-effort label of the strategy Opus proposed, for the attempt history."""
    if "STRATEGY:" in toon:
        return toon.split("STRATEGY:")[1].split("|")[0].split("~")[0].strip()[:40]
    return "unknown"


def summarize_history(history: List[AttemptRecord], limit: int = 3, per_err: int = 160) -> str:
    """Token-lean digest of recent attempts to hand to the cloud architect."""
    if not history:
        return "NONE"
    parts = []
    for i, a in enumerate(history[-limit:], 1):
        err = (a.get("resulting_error") or "still-failing")[:per_err]
        parts.append(f"A{i}[{a['source']}:{a['strategy']}]ERR:{err}")
    return "~".join(parts)


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def execute_and_test(state: AgentState) -> Dict[str, Any]:
    print(f"\n--- Node: Executing Repository Test Suite (Iteration {state['iterations'] + 1}) ---")
    path = _target_path(state)
    try:
        with open(path, "r") as f:
            code = f.read()
    except FileNotFoundError:
        code = ""

    result = subprocess.run(["pytest", "test_suite.py"], cwd="sandbox", capture_output=True, text=True)
    test_passed = result.returncode == 0
    full_error = "" if test_passed else result.stdout + "\n" + result.stderr

    if test_passed:
        print("[Repo Test Status]: SUCCESS! Feature implemented without regression.")
    else:
        print("[Repo Test Status]: FAILED. Compilation or assertion errors detected.")

    # Pair this outcome with the attempt that produced it, so the history records
    # "tried X -> got error Y" rather than a context-free single pass.
    history = state.get("attempt_history", [])
    if history and not history[-1].get("resulting_error"):
        history[-1]["resulting_error"] = "PASSED" if test_passed else condense_error(full_error)

    return {
        "current_code": code,
        "last_error": full_error,
        "test_passed": test_passed,
        "iterations": state["iterations"] + 1,
        "attempt_history": history,
    }


def local_autonomous_fix(state: AgentState) -> Dict[str, Any]:
    """SLM tries to fix on its own.

    Used both for the very first attempt AND for the self-debug pass that now
    follows every cloud guidance, before we are allowed to re-escalate.
    """
    have_guidance = bool(state.get("abstract_guidance"))
    if have_guidance:
        print("\n--- Node: Local SLM Self-Debugging Residual Error (post-guidance) ---")
        guidance_ctx = (
            f"You already applied this architect guidance, but the test STILL fails:\n"
            f"{state['abstract_guidance']}\n\n"
            f"Do not just re-apply it verbatim. Debug WHY the error below persists and correct it.\n\n"
        )
        strategy = "self-debug-post-guidance"
    else:
        print("\n--- Node: Local SLM Attempting Autonomous Fix ---")
        guidance_ctx = ""
        strategy = "autonomous-fix"

    target = state.get("target_file", "cart.py")
    test_ctx = (
        f"The fixed test suite that defines correct behavior:\n"
        f"```python\n{state.get('test_code', '')}\n```\n\n"
    )
    prompt = (
        f"Task: {state['task_description']}\n\n"
        f"{_action_space_rules(target)}\n"
        f"{test_ctx}"
        f"Current contents of `sandbox/{target}`:\n```python\n{state['current_code']}\n```\n\n"
        f"{guidance_ctx}"
        f"Test failure trace:\n{state['last_error']}\n\n"
        f"Update `sandbox/{target}` so all tests pass without breaking existing functionality. "
        f"Output ONLY the complete, functional code for the file inside markdown fences."
    )
    response = local_slm.invoke(prompt)
    code_block = _extract_code(response.content)

    with open(_target_path(state), "w") as f:
        f.write(code_block)
    print(f"[Local Edit Applied]: Modified sandbox/{target} contents.")

    history = state.get("attempt_history", [])
    history.append({
        "source": "local",
        "strategy": strategy,
        "change_summary": f"rewrote {target}",
        "resulting_error": "",
    })
    return {
        "current_code": code_block,
        "local_attempt_failed": True,
        "attempt_history": history,
    }


def abstract_problem(state: AgentState) -> Dict[str, Any]:
    # In "direct" mode we skip the SLM abstraction entirely -- consult_cloud sends
    # Opus the concrete symptom + code instead. This node becomes a no-op so the
    # graph topology is unchanged across both A/B arms.
    if ESCALATION_MODE == "direct":
        print("\n--- Node: Escalating to Cloud - [direct mode] skipping SLM abstraction ---")
        return {"abstract_query": ""}

    print("\n--- Node: Escalating to Cloud - Abstracting Code Errors ---")
    history_summary = summarize_history(state.get("attempt_history", []))
    prompt = (
        f"Task: {state['task_description']}\n"
        f"Error Logs: {condense_error(state['last_error'])}\n"
        f"Already-tried approaches that FAILED: {history_summary}\n\n"
        f"Identify the core algorithmic failure that the prior attempts did NOT resolve. "
        f"Output a single, purely abstract mathematical or architectural query that targets the "
        f"remaining defect. Do NOT include code, variable names, or syntax. Limit to 40 words."
    )
    response = local_slm.invoke(prompt)
    abstract_query = response.content.strip()
    print(f"[Abstract Query Formulated]: {abstract_query}...")
    return {"abstract_query": abstract_query}


def consult_cloud(state: AgentState) -> Dict[str, Any]:
    print("\n--- Node: Querying AWS Bedrock (Opus 4.8) with TOON Protocol ---")
    history_summary = summarize_history(state.get("attempt_history", []))

    # When we have already consulted the cloud, tell Opus exactly what IT advised
    # last time and what error persisted, so it can reject its own failed approach
    # rather than re-deriving it from a near-identical abstract query.
    prior_guidance = state.get("abstract_guidance", "")
    if prior_guidance:
        feedback_block = (
            f"YOUR PREVIOUS GUIDANCE was:\n{prior_guidance}\n"
            f"It was applied and the SLM then self-debugged, but the failure PERSISTS. "
            f"This means your previous strategy was wrong or misdiagnosed the root cause. "
            f"Treat it as a dead end and propose a DIFFERENT root-cause hypothesis.\n\n"
        )
    else:
        feedback_block = ""

    target = state.get("target_file", "cart.py")
    raw_symptom = condense_error(state.get("last_error", ""))

    if ESCALATION_MODE == "direct":
        # Direct mode: no SLM abstraction. Give Opus the concrete symptom + the actual
        # current code so it diagnoses from ground truth. (Sends source to the cloud.)
        print(f"[Context Sent to Cloud] Mode: DIRECT (symptom + code, no abstraction)")
        prompt = (
            f"A unit test is failing. Diagnose from the concrete evidence below.\n\n"
            f"Task: {state['task_description']}\n\n"
            f"ACTUAL test failure symptom:\n{raw_symptom}\n\n"
            f"Current `sandbox/{target}`:\n```python\n{state['current_code']}\n```\n\n"
            f"{_action_space_rules(target)}\n"
            f"{feedback_block}"
            f"Compact log of every approach ALREADY TRIED and the error it left behind:\n"
            f"{history_summary}\n\n"
            f"Do NOT repeat any strategy listed above. Provide a corrected, specific approach "
            f"that only changes `sandbox/{target}`. "
            f"You MUST use strict TOON (Token-Oriented Object Notation). "
            f"Do not use JSON, markdown, or conversational text. "
            f"Format strictly as: STRATEGY:<name>|STEPS:<step1>~<step2>|CONSTRAINT:<rule>\n"
            f"Maximum length: 75 words."
        )
    else:
        # Abstract mode (default): SLM's abstraction is primary, raw symptom is a
        # ground-truth check so Opus can overrule a misdiagnosis.
        print(f"[Context Sent to Cloud] Mode: ABSTRACT (SLM query + symptom)")
        prompt = (
            f"Abstract query (the SLM's interpretation, which MAY be wrong):\n{state['abstract_query']}\n\n"
            f"ACTUAL test failure symptom (ground truth — trust this over the interpretation):\n{raw_symptom}\n\n"
            f"{_action_space_rules(target)}\n"
            f"{feedback_block}"
            f"Compact log of every approach ALREADY TRIED and the error it left behind:\n"
            f"{history_summary}\n\n"
            f"Do NOT repeat any strategy listed above. If the abstract query contradicts the "
            f"actual symptom or the constraints, diagnose from the symptom instead. "
            f"Provide a corrected, more specific approach that only changes `sandbox/{target}`. "
            f"You MUST use strict TOON (Token-Oriented Object Notation). "
            f"Do not use JSON, markdown, or conversational text. "
            f"Format strictly as: STRATEGY:<name>|STEPS:<step1>~<step2>|CONSTRAINT:<rule>\n"
            f"Maximum length: 75 words."
        )
    print(f"[Context Sent to Cloud] Raw symptom: {raw_symptom[:160]}")
    print(f"[Context Sent to Cloud] Prior-attempt digest: {history_summary}")
    if prior_guidance:
        print("[Context Sent to Cloud] Flagged previous guidance as a failed dead-end.")
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    response = bedrock_client.converse(modelId=CLOUD_MODEL_ID, messages=messages)
    guidance = response['output']['message']['content'][0]['text']
    usage = response['usage']

    print(f"[TOON Guidance Received]: {guidance}")
    print(f"[Cloud Tokens Spent]: Input: {usage['inputTokens']} | Output: {usage['outputTokens']} | Total: {usage['totalTokens']}")

    current_metrics = state.get("metrics_log", [])
    current_metrics.append({
        "iteration": state["iterations"],
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "total_tokens": usage["totalTokens"],
    })
    return {"abstract_guidance": guidance, "metrics_log": current_metrics}


def apply_logic(state: AgentState) -> Dict[str, Any]:
    print("\n--- Node: Compiling Cloud TOON Architectural Strategy into Codebase ---")
    target = state.get("target_file", "cart.py")
    prompt = (
        f"Modify `sandbox/{target}` using this structural strategy.\n\n"
        f"{_action_space_rules(target)}\n"
        f"Fixed test suite (defines the required call contract):\n"
        f"```python\n{state.get('test_code', '')}\n```\n\n"
        f"Current Code:\n```python\n{state['current_code']}\n```\n\n"
        f"Error Trace:\n{condense_error(state['last_error'])}\n\n"
        f"TOON Guidance from Architect:\n{state['abstract_guidance']}\n\n"
        f"Task: Rewrite `sandbox/{target}` entirely to manifest this logic. "
        f"Output ONLY valid Python code inside markdown blocks."
    )
    response = local_slm.invoke(prompt)
    code_block = _extract_code(response.content)

    with open(_target_path(state), "w") as f:
        f.write(code_block)
    print("[Repository Mutated with Cloud Guidance]")

    history = state.get("attempt_history", [])
    history.append({
        "source": "cloud",
        "strategy": _extract_strategy(state["abstract_guidance"]),
        "change_summary": f"applied architect strategy to {target}",
        "resulting_error": "",
    })
    # Reset the local flag so the SLM gets a self-debug pass on the result of this
    # guidance BEFORE we are allowed to spend more cloud tokens re-escalating.
    return {
        "current_code": code_block,
        "attempt_history": history,
        "local_attempt_failed": False,
    }


def route_decision(state: AgentState):
    if state["test_passed"]:
        return "complete"
    if state["iterations"] >= state["max_iterations"]:
        return "abort"
    # A/B fairness: skip the stochastic autonomous local fix so both arms escalate
    # from the same buggy code. We only skip the FIRST (pre-guidance) local attempt;
    # post-guidance self-debug still runs (abstract_guidance is set by then).
    if FORCE_ESCALATE and not state.get("abstract_guidance"):
        return "escalate_to_cloud"
    if not state["local_attempt_failed"]:
        return "try_local_fix"
    return "escalate_to_cloud"


workflow = StateGraph(AgentState)
workflow.add_node("execute_and_test", execute_and_test)
workflow.add_node("local_autonomous_fix", local_autonomous_fix)
workflow.add_node("abstract_problem", abstract_problem)
workflow.add_node("consult_cloud", consult_cloud)
workflow.add_node("apply_logic", apply_logic)

workflow.set_entry_point("execute_and_test")
workflow.add_edge("local_autonomous_fix", "execute_and_test")
workflow.add_edge("abstract_problem", "consult_cloud")
workflow.add_edge("consult_cloud", "apply_logic")
workflow.add_edge("apply_logic", "execute_and_test")

workflow.add_conditional_edges(
    "execute_and_test",
    route_decision,
    {
        "complete": END,
        "abort": END,
        "try_local_fix": "local_autonomous_fix",
        "escalate_to_cloud": "abstract_problem",
    },
)
app = workflow.compile()

if __name__ == "__main__":
    pass
