import os
import subprocess
from typing import TypedDict, List, Dict, Any
import boto3
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    task_description: str
    current_code: str
    last_error: str
    iterations: int
    max_iterations: int
    abstract_query: str
    abstract_guidance: str
    test_passed: bool
    metrics_log: List[Dict[str, Any]]
    local_attempt_failed: bool

local_slm = ChatOllama(model="gemma4:e4b", temperature=0)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
CLOUD_MODEL_ID = "us.anthropic.claude-opus-4-8"

def execute_and_test(state: AgentState) -> Dict[str, Any]:
    print(f"\n--- Node: Executing Tests (Iteration {state['iterations'] + 1}) ---")
    try:
        with open("sandbox/target_app.py", "r") as f:
            code = f.read()
    except FileNotFoundError:
        code = ""

    result = subprocess.run(["pytest", "test_suite.py"], cwd="sandbox", capture_output=True, text=True)
    test_passed = result.returncode == 0
    last_error = "" if test_passed else result.stdout + "\n" + result.stderr
    
    if test_passed:
        print("[Local Test Status]: SUCCESS! Code is verified.")
    else:
        print("[Local Test Status]: FAILED.")

    return {
        "current_code": code,
        "last_error": last_error,
        "test_passed": test_passed,
        "iterations": state["iterations"] + 1
    }

def local_autonomous_fix(state: AgentState) -> Dict[str, Any]:
    print("\n--- Node: Local SLM Attempting Autonomous Fix (No Cloud) ---")
    prompt = (
        f"The Python code in `sandbox/target_app.py` is failing unit tests.\n"
        f"Current Code:\n```python\n{state['current_code']}\n```\n\n"
        f"Error Trace:\n{state['last_error']}\n\n"
        f"Task: Fix the bug yourself. Output ONLY the valid, complete Python code within markdown blocks."
    )
    response = local_slm.invoke(prompt)
    content = response.content
    if "```python" in content:
        code_block = content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        code_block = content.split("```")[1].split("```")[0].strip()
    else:
        code_block = content.strip()

    with open("sandbox/target_app.py", "w") as f:
        f.write(code_block)
    print("[Local Patch Applied]: Rewrote sandbox file autonomously.")
    return {"current_code": code_block, "local_attempt_failed": True}

def abstract_problem(state: AgentState) -> Dict[str, Any]:
    print("\n--- Node: Escalating to Cloud - Abstracting Code Errors ---")
    prompt = (
        f"Task: {state['task_description']}\n"
        f"Error Logs: {state['last_error']}\n\n"
        f"Identify the core algorithmic failure. Output a single, purely abstract mathematical or architectural query. "
        f"Do NOT include code, variable names, or syntax. Limit to 40 words."
    )
    response = local_slm.invoke(prompt)
    abstract_query = response.content.strip()
    print(f"[Abstract Query Formulated]: {abstract_query[:120]}...")
    return {"abstract_query": abstract_query}

def consult_cloud(state: AgentState) -> Dict[str, Any]:
    print("\n--- Node: Querying AWS Bedrock (Opus 4.8) with TOON Protocol ---")
    prompt = (
        f"Query: {state['abstract_query']}\n\n"
        f"Provide the algorithmic solution. You MUST use strict TOON (Token-Oriented Object Notation). "
        f"Do not use JSON, markdown, or conversational text. "
        f"Format strictly as: STRATEGY:<name>|STEPS:<step1>~<step2>|CONSTRAINT:<rule>\n"
        f"Maximum length: 75 words."
    )
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
        "total_tokens": usage["totalTokens"]
    })
    return {"abstract_guidance": guidance, "metrics_log": current_metrics}

def apply_logic(state: AgentState) -> Dict[str, Any]:
    print("\n--- Node: Injecting Cloud TOON Logic Back to Sandbox ---")
    prompt = (
        f"Modify `sandbox/target_app.py` based on this advice.\n"
        f"Current Code:\n```python\n{state['current_code']}\n```\n\n"
        f"Error Trace:\n{state['last_error']}\n\n"
        f"Architectural TOON Guidance:\n{state['abstract_guidance']}\n\n"
        f"Task: Rewrite the code completely. Output ONLY valid Python code inside markdown blocks."
    )
    response = local_slm.invoke(prompt)
    content = response.content
    if "```python" in content:
        code_block = content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        code_block = content.split("```")[1].split("```")[0].strip()
    else:
        code_block = content.strip()

    with open("sandbox/target_app.py", "w") as f:
        f.write(code_block)
    print("[Sandbox File Modded with Cloud Logic]")
    return {"current_code": code_block}

def route_decision(state: AgentState):
    if state["test_passed"]:
        return "complete"
    if state["iterations"] >= state["max_iterations"]:
        return "abort"
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
        "escalate_to_cloud": "abstract_problem"
    }
)
app = workflow.compile()

if __name__ == "__main__":
    pass