"""Hybrid SLM+Opus agent that produces SWE-bench patch predictions.

Pipeline per instance:
  1. Checkout repo at base_commit.
  2. Localize candidate files (swe_localize: BM25 + optional SLM re-rank).
  3. SLM attempts the fix by rewriting the localized file(s).
  4. Local feedback = does the edit import/compile cleanly? (Real tests live in the
     Docker eval, so we cannot run them here -- compile-check is our cheap signal.)
  5. If the SLM stalls, escalate to Opus for abstract TOON guidance (reusing the
     proven condense_error / attempt-history machinery), apply, recompile.
  6. Emit `git diff` as the model_patch in SWE-bench prediction format.

This is deliberately a SINGLE-pass-with-escalation adaptation of the sandbox loop,
scoped for the smoke test. It is NOT trying to be a SOTA SWE agent -- the research
question is how much the abstract-Opus guidance lifts the SLM, and at what token cost.
"""
import os
import re
import json
import subprocess

import boto3
from langchain_ollama import ChatOllama

from orchestrator import condense_error, summarize_history, _extract_strategy, _action_space_rules

REPO_ROOT = "data/swebench/repos"
MODEL_NAME = "slm-opus-hybrid"

local_slm = ChatOllama(model="gemma4:e4b", temperature=0)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
CLOUD_MODEL_ID = "us.anthropic.claude-opus-4-8"


# --------------------------------------------------------------------------- #
# Repo + patch helpers
# --------------------------------------------------------------------------- #
def checkout(instance):
    repo, sha, sid = instance["repo"], instance["base_commit"], instance["instance_id"]
    d = os.path.join(REPO_ROOT, sid)
    if not os.path.isdir(d):
        subprocess.run(["git", "clone", "--quiet", f"https://github.com/{repo}.git", d], check=True)
    # Hard reset to base_commit so re-runs start clean.
    subprocess.run(["git", "-C", d, "reset", "--hard", "--quiet"], check=True)
    subprocess.run(["git", "-C", d, "clean", "-fdq"], check=True)
    subprocess.run(["git", "-C", d, "checkout", "--quiet", sha], check=True)
    return d


def read_file(repo_dir, rel):
    try:
        with open(os.path.join(repo_dir, rel), "r", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def write_file(repo_dir, rel, content):
    with open(os.path.join(repo_dir, rel), "w") as f:
        f.write(content)


def compile_check(repo_dir, rel):
    """Cheap local feedback: does the file parse/byte-compile? Returns (ok, error)."""
    r = subprocess.run(["python", "-m", "py_compile", os.path.join(repo_dir, rel)],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stderr or r.stdout)


def git_diff(repo_dir):
    r = subprocess.run(["git", "-C", repo_dir, "diff"], capture_output=True, text=True)
    return r.stdout


def git_diff_for(repo_dir, rel):
    r = subprocess.run(["git", "-C", repo_dir, "diff", "--", rel], capture_output=True, text=True)
    return r.stdout


# Search/replace block format. Editing via exact-match SEARCH/REPLACE blocks (not
# full-file rewrites) is what keeps changes minimal and makes it IMPOSSIBLE to delete
# unseen code -- the root cause of the earlier file-destruction bug, where a truncated
# full-file rewrite silently dropped 1300+ lines yet still compiled.
_SR_RE = re.compile(
    r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
    re.DOTALL,
)


def parse_search_replace(text):
    """Return list of (search, replace) string pairs from the model's response."""
    return [(m.group(1), m.group(2)) for m in _SR_RE.finditer(text)]


def apply_search_replace(content, blocks):
    """Apply each (search, replace) to content. Returns (new_content, applied, reason).

    Requires each SEARCH to match EXACTLY ONCE -- ambiguous or missing matches are
    rejected so we never edit the wrong location or silently no-op.
    """
    new = content
    applied = 0
    for search, replace in blocks:
        if not search:
            return content, 0, "empty SEARCH block"
        count = new.count(search)
        if count == 0:
            return content, applied, f"SEARCH not found: {search[:60]!r}"
        if count > 1:
            return content, applied, f"SEARCH ambiguous ({count}x): {search[:60]!r}"
        new = new.replace(search, replace, 1)
        applied += 1
    return new, applied, "ok"


def _looks_destructive(old, new, max_shrink=0.5):
    """Guard: reject edits that delete a large fraction of the file (truncation/clobber)."""
    if len(old) == 0:
        return False
    return len(new) < len(old) * (1 - max_shrink)


# --------------------------------------------------------------------------- #
# Edit attempts
# --------------------------------------------------------------------------- #
def slm_edit_file(repo_dir, rel, issue_text, prior_error="", guidance=""):
    """Ask the SLM for SEARCH/REPLACE edits to a file. Returns (ok, error).

    Never rewrites the whole file: edits are applied as exact-match snippet swaps,
    then byte-compiled AND checked for destructive shrinkage.
    """
    current = read_file(repo_dir, rel)
    # For READING context we can show a large window; we no longer write this back,
    # so truncation here cannot destroy the file.
    view = current[:40000]
    note = "" if len(current) <= 40000 else "\n# NOTE: file shown is truncated; only reference code you can see.\n"
    guidance_block = f"\nArchitect guidance (TOON):\n{guidance}\n" if guidance else ""
    err_block = f"\nPrevious attempt problem:\n{prior_error[:600]}\n" if prior_error else ""
    prompt = (
        f"Resolve this software issue by editing `{rel}`.\n\n"
        f"Issue:\n{issue_text[:2000]}\n\n"
        f"Current `{rel}`:{note}\n```python\n{view}\n```\n"
        f"{guidance_block}{err_block}\n"
        f"Respond with ONE OR MORE edit blocks in EXACTLY this format:\n"
        f"<<<<<<< SEARCH\n<exact lines to find, copied verbatim>\n=======\n"
        f"<replacement lines>\n>>>>>>> REPLACE\n\n"
        f"The SEARCH text must match the file EXACTLY (including indentation) and be "
        f"unique. Make the MINIMAL change needed. Do not output the whole file."
    )
    resp = local_slm.invoke(prompt).content
    blocks = parse_search_replace(resp)
    if not blocks:
        return False, "no SEARCH/REPLACE blocks produced"
    new_code, applied, reason = apply_search_replace(current, blocks)
    if applied == 0:
        return False, f"no edit applied ({reason})"
    if _looks_destructive(current, new_code):
        return False, "edit rejected: would delete too much of the file"
    write_file(repo_dir, rel, new_code)
    return compile_check(repo_dir, rel)


def opus_guidance(issue_text, target_file, raw_error, history):
    """Abstract TOON guidance from Opus. Returns (guidance, usage_dict)."""
    history_summary = summarize_history(history)
    prompt = (
        f"A software issue must be fixed by editing `{target_file}`.\n\n"
        f"Issue (abstracted):\n{issue_text[:1200]}\n\n"
        f"Concrete obstacle so far:\n{condense_error(raw_error)}\n\n"
        f"{_action_space_rules(target_file)}\n"
        f"Approaches already tried and their result:\n{history_summary}\n\n"
        f"Provide a corrected, specific strategy to fix `{target_file}`. "
        f"You MUST use strict TOON: STRATEGY:<name>|STEPS:<s1>~<s2>|CONSTRAINT:<rule>\n"
        f"Maximum 90 words."
    )
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    resp = bedrock_client.converse(modelId=CLOUD_MODEL_ID, messages=messages)
    return resp["output"]["message"]["content"][0]["text"], resp["usage"]


def solve_instance(instance, max_cloud=2, n_files=5):
    """Run the hybrid pipeline on one instance. Returns a result dict + prediction.

    Gold patches in SWE-bench Lite are overwhelmingly SINGLE-FILE, and editing extra
    files only risks breaking PASS_TO_PASS tests. So we try BM25 candidates ONE AT A
    TIME and STOP at the first file that yields a compiling, non-empty edit. We trust
    BM25 ranking directly (the SLM re-rank was found to demote correct files).
    """
    from swe_localize import bm25_candidates

    sid = instance["instance_id"]
    issue = instance["problem_statement"]
    print(f"\n{'=' * 60}\nSOLVING {sid} ({instance['repo']})\n{'=' * 60}")

    repo_dir = checkout(instance)
    cands = bm25_candidates(repo_dir, issue, top_k=max(10, n_files))
    targets = [c[0] for c in cands[:n_files]]
    print(f"[Localized] BM25 top files: {targets}")
    if not targets:
        return _result(sid, False, 0, 0, "", "no candidate files")

    history = []
    cloud_calls = 0
    total_tokens = 0
    edited_ok = []

    for target in targets:
        # 1) SLM autonomous attempt on this file.
        ok, err = slm_edit_file(repo_dir, target, issue)
        history.append({"source": "local", "strategy": f"autonomous-edit:{target}",
                        "resulting_error": "compiles" if ok else condense_error(err)})
        print(f"[SLM edit] {target} -> {'compiles' if ok else err[:70]}")

        # 2) Escalate to Opus while the edit won't apply/compile (budget shared).
        while not ok and cloud_calls < max_cloud:
            cloud_calls += 1
            guidance, usage = opus_guidance(issue, target, err, history)
            total_tokens += usage["totalTokens"]
            print(f"[Opus #{cloud_calls}] {guidance[:110]}... (tokens={usage['totalTokens']})")
            history.append({"source": "cloud", "strategy": _extract_strategy(guidance),
                            "resulting_error": ""})
            ok, err = slm_edit_file(repo_dir, target, issue, prior_error=err, guidance=guidance)
            history[-1]["resulting_error"] = "compiles" if ok else condense_error(err)
            print(f"[SLM apply] {target} -> {'compiles' if ok else err[:70]}")

        if ok and git_diff_for(repo_dir, target).strip():
            edited_ok.append(target)
            print(f"[Accepted] minimal edit to {target}; stopping (single-file fix).")
            break  # single-file fix found -- don't risk damaging other files
        else:
            # Revert any partial change so it doesn't pollute the patch.
            subprocess.run(["git", "-C", repo_dir, "checkout", "--", target],
                           capture_output=True, text=True)

    patch = git_diff(repo_dir)
    produced = bool(patch.strip())
    print(f"[Patch] produced={produced} ({len(patch)} bytes), files={edited_ok}, "
          f"cloud_tokens={total_tokens}")
    return _result(sid, produced, cloud_calls, total_tokens, patch,
                   "" if edited_ok else "no-compiling-edit", ",".join(edited_ok))


def _result(sid, produced, cloud_calls, tokens, patch, note, target=None):
    return {
        "instance_id": sid,
        "produced_patch": produced,
        "cloud_calls": cloud_calls,
        "cloud_tokens": tokens,
        "target_file": target,
        "note": note,
        "prediction": {
            "instance_id": sid,
            "model_name_or_path": MODEL_NAME,
            "model_patch": patch,
        },
    }
