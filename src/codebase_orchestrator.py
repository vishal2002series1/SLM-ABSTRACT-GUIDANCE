"""Multi-file orchestrator for codebase-level feature work.

The single-file orchestrator (orchestrator.py) reads one file and writes one file.
A cross-cutting feature spans several files whose interfaces must agree, so this
variant:

  * reads ALL editable files into one context (the "context window" the SLM sees),
  * asks the SLM to emit edits for as many files as needed using a delimited format,
  * parses that multi-file response back onto disk,
  * runs the repo test suite, and escalates to Opus with the same abstract/TOON
    protocol + attempt-history feedback as the single-file loop.

The file-bundling and response-parsing here ARE the context engineering: presenting
multiple files coherently and reliably recovering multiple files from one response
is the part that gets hard as the repo grows.
"""
import os
import re
import subprocess
from typing import TypedDict, List, Dict, Any
import boto3
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# Reuse the proven helpers from the single-file orchestrator.
from orchestrator import condense_error, summarize_history, _extract_strategy


class MFState(TypedDict):
    task_description: str
    feature_name: str
    editable_files: List[str]      # files the agent may edit
    test_code: str                 # ground-truth test suite
    files: Dict[str, str]          # current contents of every editable file
    last_error: str
    iterations: int
    max_iterations: int
    abstract_query: str
    abstract_guidance: str
    test_passed: bool
    metrics_log: List[Dict[str, Any]]
    local_attempt_failed: bool
    attempt_history: List[Dict[str, Any]]


local_slm = ChatOllama(model="gemma4:e4b", temperature=0)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
CLOUD_MODEL_ID = "us.anthropic.claude-opus-4-8"

SANDBOX = "sandbox"

# Sentinel format the SLM must use so we can recover multiple files from one reply.
FILE_BEGIN = "=== FILE: {name} ==="
FILE_RE = re.compile(r"=== FILE: (?P<name>[^\s=]+) ===\s*\n(?P<body>.*?)(?=\n=== FILE:|\Z)", re.DOTALL)


# --------------------------------------------------------------------------- #
# Context engineering helpers
# --------------------------------------------------------------------------- #
def bundle_files(files: Dict[str, str], names: List[str]) -> str:
    """Render the editable files into one delimited context block for the SLM."""
    chunks = []
    for n in names:
        body = files.get(n, "")
        chunks.append(f"{FILE_BEGIN.format(name=n)}\n```python\n{body}\n```")
    return "\n\n".join(chunks)


def parse_file_bundle(text: str, allowed: List[str]) -> Dict[str, str]:
    """Recover {filename: code} from a delimited multi-file response.

    Only files in `allowed` are accepted; code fences are stripped. Returns just
    the files the model actually emitted so untouched files are left as-is.
    """
    out = {}
    for m in FILE_RE.finditer(text):
        name = m.group("name").strip()
        if name not in allowed:
            continue
        body = m.group("body").strip()
        if "```" in body:
            # pull the first fenced block
            parts = body.split("```")
            # parts[1] may start with "python\n"
            code = parts[1]
            if code.startswith("python"):
                code = code[len("python"):]
            body = code.strip()
        out[name] = body
    return out


def _write_files(files: Dict[str, str]) -> None:
    for name, body in files.items():
        with open(os.path.join(SANDBOX, name), "w") as f:
            f.write(body if body.endswith("\n") else body + "\n")


def _read_files(names: List[str]) -> Dict[str, str]:
    out = {}
    for name in names:
        try:
            with open(os.path.join(SANDBOX, name), "r") as f:
                out[name] = f.read()
        except FileNotFoundError:
            out[name] = ""
    return out


def _scope_rules(editable: List[str]) -> str:
    return (
        f"HARD CONSTRAINTS (non-negotiable):\n"
        f"1. You may ONLY edit these files: {', '.join(editable)}. Do NOT edit the test file.\n"
        f"2. The test suite is FIXED and CORRECT; it defines the required behavior.\n"
        f"3. Keep interfaces consistent ACROSS files: a symbol you add in one file must be "
        f"imported/used with the same name and signature in the others.\n"
        f"4. Output format: for EVERY file you change, emit a block exactly like:\n"
        f"   === FILE: <filename> ===\n   ```python\n   <full file contents>\n   ```\n"
        f"   Emit the COMPLETE file, not a diff. Only include files you changed.\n"
    )


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def execute_and_test(state: MFState) -> Dict[str, Any]:
    print(f"\n--- Node: Executing Repository Test Suite (Iteration {state['iterations'] + 1}) ---")
    files = _read_files(state["editable_files"])

    result = subprocess.run(["pytest", "test_suite.py"], cwd=SANDBOX, capture_output=True, text=True)
    test_passed = result.returncode == 0
    full_error = "" if test_passed else result.stdout + "\n" + result.stderr

    if test_passed:
        print("[Repo Test Status]: SUCCESS! Feature implemented across files without regression.")
    else:
        print("[Repo Test Status]: FAILED. Compilation or assertion errors detected.")

    history = state.get("attempt_history", [])
    if history and not history[-1].get("resulting_error"):
        history[-1]["resulting_error"] = "PASSED" if test_passed else condense_error(full_error)

    return {
        "files": files,
        "last_error": full_error,
        "test_passed": test_passed,
        "iterations": state["iterations"] + 1,
        "attempt_history": history,
    }


def local_multifile_fix(state: MFState) -> Dict[str, Any]:
    have_guidance = bool(state.get("abstract_guidance"))
    if have_guidance:
        print("\n--- Node: Local SLM Self-Debugging Across Files (post-guidance) ---")
        guidance_ctx = (
            f"You already applied this architect guidance but tests STILL fail:\n"
            f"{state['abstract_guidance']}\n\n"
            f"Do not re-apply verbatim. Debug WHY the error persists across the files and fix it.\n\n"
        )
        strategy = "self-debug-post-guidance"
    else:
        print("\n--- Node: Local SLM Attempting Multi-File Feature Implementation ---")
        guidance_ctx = ""
        strategy = "autonomous-multifile"

    editable = state["editable_files"]
    prompt = (
        f"Task: implement the feature '{state['feature_name']}'.\n{state['task_description']}\n\n"
        f"{_scope_rules(editable)}\n"
        f"Fixed test suite (defines required behavior):\n```python\n{state['test_code']}\n```\n\n"
        f"Current repository files:\n{bundle_files(state['files'], editable)}\n\n"
        f"{guidance_ctx}"
        f"Latest test failure:\n{condense_error(state['last_error'])}\n\n"
        f"Implement the feature so all tests pass. Emit each changed file in the required "
        f"=== FILE: ... === format with complete contents."
    )
    response = local_slm.invoke(prompt)
    edits = parse_file_bundle(response.content, editable)
    if edits:
        _write_files(edits)
    print(f"[Local Edit Applied]: Modified {len(edits)} file(s): {sorted(edits.keys())}")

    history = state.get("attempt_history", [])
    history.append({
        "source": "local",
        "strategy": strategy,
        "change_summary": f"edited {sorted(edits.keys())}",
        "resulting_error": "",
    })
    return {
        "files": {**state["files"], **edits},
        "local_attempt_failed": True,
        "attempt_history": history,
    }


def abstract_problem(state: MFState) -> Dict[str, Any]:
    print("\n--- Node: Escalating to Cloud - Abstracting Multi-File Failure ---")
    history_summary = summarize_history(state.get("attempt_history", []))
    prompt = (
        f"Feature: {state['feature_name']}. {state['task_description']}\n"
        f"Error Logs: {condense_error(state['last_error'])}\n"
        f"Files involved: {', '.join(state['editable_files'])}\n"
        f"Already-tried approaches that FAILED: {history_summary}\n\n"
        f"Identify the core architectural failure spanning these files that prior attempts "
        f"did not resolve. Output a single, purely abstract architectural query (no code, no "
        f"variable names). Limit to 40 words."
    )
    response = local_slm.invoke(prompt)
    abstract_query = response.content.strip()
    print(f"[Abstract Query Formulated]: {abstract_query}...")
    return {"abstract_query": abstract_query}


def consult_cloud(state: MFState) -> Dict[str, Any]:
    print("\n--- Node: Querying AWS Bedrock (Opus 4.8) with TOON Protocol ---")
    history_summary = summarize_history(state.get("attempt_history", []))
    prior_guidance = state.get("abstract_guidance", "")
    feedback_block = ""
    if prior_guidance:
        feedback_block = (
            f"YOUR PREVIOUS GUIDANCE was:\n{prior_guidance}\n"
            f"It was applied and the SLM self-debugged, but the failure PERSISTS. Treat it as a "
            f"dead end and propose a DIFFERENT root-cause hypothesis.\n\n"
        )

    raw_symptom = condense_error(state.get("last_error", ""))
    prompt = (
        f"Abstract query (SLM interpretation, MAY be wrong):\n{state['abstract_query']}\n\n"
        f"ACTUAL test failure symptom (ground truth):\n{raw_symptom}\n\n"
        f"This is a MULTI-FILE feature across: {', '.join(state['editable_files'])}. "
        f"Guidance must keep interfaces consistent across files.\n\n"
        f"{feedback_block}"
        f"Compact log of approaches ALREADY TRIED and their errors:\n{history_summary}\n\n"
        f"Do NOT repeat a strategy listed above. Provide a corrected cross-file approach. "
        f"You MUST use strict TOON: STRATEGY:<name>|STEPS:<step1>~<step2>|CONSTRAINT:<rule>\n"
        f"Maximum length: 90 words."
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

    metrics = state.get("metrics_log", [])
    metrics.append({
        "iteration": state["iterations"],
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "total_tokens": usage["totalTokens"],
    })
    return {"abstract_guidance": guidance, "metrics_log": metrics}


def apply_logic(state: MFState) -> Dict[str, Any]:
    print("\n--- Node: Compiling Cloud TOON Strategy Across Files ---")
    editable = state["editable_files"]
    prompt = (
        f"Apply this architect strategy to implement '{state['feature_name']}' across files.\n\n"
        f"{_scope_rules(editable)}\n"
        f"Fixed test suite:\n```python\n{state['test_code']}\n```\n\n"
        f"Current files:\n{bundle_files(state['files'], editable)}\n\n"
        f"TOON Guidance from Architect:\n{state['abstract_guidance']}\n\n"
        f"Latest error:\n{condense_error(state['last_error'])}\n\n"
        f"Emit each changed file in the required === FILE: ... === format with complete contents."
    )
    response = local_slm.invoke(prompt)
    edits = parse_file_bundle(response.content, editable)
    if edits:
        _write_files(edits)
    print(f"[Repository Mutated with Cloud Guidance]: {sorted(edits.keys())}")

    history = state.get("attempt_history", [])
    history.append({
        "source": "cloud",
        "strategy": _extract_strategy(state["abstract_guidance"]),
        "change_summary": f"applied strategy to {sorted(edits.keys())}",
        "resulting_error": "",
    })
    return {
        "files": {**state["files"], **edits},
        "attempt_history": history,
        "local_attempt_failed": False,
    }


def route_decision(state: MFState):
    if state["test_passed"]:
        return "complete"
    if state["iterations"] >= state["max_iterations"]:
        return "abort"
    if not state["local_attempt_failed"]:
        return "try_local_fix"
    return "escalate_to_cloud"


workflow = StateGraph(MFState)
workflow.add_node("execute_and_test", execute_and_test)
workflow.add_node("local_multifile_fix", local_multifile_fix)
workflow.add_node("abstract_problem", abstract_problem)
workflow.add_node("consult_cloud", consult_cloud)
workflow.add_node("apply_logic", apply_logic)

workflow.set_entry_point("execute_and_test")
workflow.add_edge("local_multifile_fix", "execute_and_test")
workflow.add_edge("abstract_problem", "consult_cloud")
workflow.add_edge("consult_cloud", "apply_logic")
workflow.add_edge("apply_logic", "execute_and_test")
workflow.add_conditional_edges(
    "execute_and_test",
    route_decision,
    {
        "complete": END,
        "abort": END,
        "try_local_fix": "local_multifile_fix",
        "escalate_to_cloud": "abstract_problem",
    },
)
app = workflow.compile()

if __name__ == "__main__":
    pass
