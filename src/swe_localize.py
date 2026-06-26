"""File localization for SWE-bench: pick candidate files to edit from a large repo.

The SLM cannot see a whole repo (django/sympy have thousands of files), so before
any editing we must narrow to a handful of likely-relevant files. This is the core
context-engineering step.

Strategy (two stages, both dependency-free):
  1. BM25 retrieval over all python files using the issue text as the query -> top-K.
  2. Optional SLM re-rank: show the SLM the issue + the top-K file paths with short
     snippets and ask which few to actually edit. (Cheap: paths + snippets, not full
     files.) Falls back to pure BM25 if the SLM is unavailable.

We index per-file (not per-chunk) for the smoke test; chunking is a later refinement.
"""
import os
import re
import math
from collections import Counter

CODE_EXT = (".py",)
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".tox", "build", "dist",
             ".eggs", "tests", "test", "docs", "doc", ".github"}
# Note: we skip test dirs for EDIT-candidate retrieval (SWE-bench fixes live in
# source, tests are provided by the harness). Keep this conservative.

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text):
    # Split snake_case/CamelCase identifiers so "separability_matrix" also matches
    # "separability" and "matrix" in the issue prose.
    toks = []
    for m in _TOKEN_RE.findall(text.lower()):
        toks.append(m)
        toks.extend(p for p in m.split("_") if p)
        toks.extend(p.lower() for p in re.findall(r"[a-z]+|[0-9]+", m) if p)
    return toks


def list_source_files(repo_dir):
    out = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(CODE_EXT):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, repo_dir)
                out.append(rel)
    return out


class BM25:
    """Minimal BM25 (no external deps)."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_tokens = [_tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for t in self.doc_tokens:
            for term in set(t):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }

    def score(self, query):
        q = _tokenize(query)
        scores = [0.0] * self.N
        for i in range(self.N):
            tf, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * num / den
            scores[i] = s
        return scores


def bm25_candidates(repo_dir, issue_text, top_k=10, max_bytes=200_000):
    """Return [(relpath, score)] of the top_k source files for the issue text."""
    files = list_source_files(repo_dir)
    docs = []
    for rel in files:
        try:
            with open(os.path.join(repo_dir, rel), "r", errors="ignore") as f:
                content = f.read(max_bytes)
        except OSError:
            content = ""
        # Weight the path heavily -- file/module names are strong signal.
        path_boost = (" " + rel.replace("/", " ").replace(".py", "")) * 3
        docs.append(content + path_boost)
    bm = BM25(docs)
    scores = bm.score(issue_text)
    ranked = sorted(zip(files, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def slm_rerank(slm, issue_text, candidates, repo_dir, pick=3, snippet_lines=25):
    """Ask the SLM to choose the few files to edit from BM25 candidates.

    candidates: [(relpath, score)]. Returns a list of relpaths (<= pick).
    Falls back to the top-`pick` BM25 files on any parsing failure.
    """
    blocks = []
    for rel, _ in candidates:
        try:
            with open(os.path.join(repo_dir, rel), "r", errors="ignore") as f:
                head = "".join(f.readlines()[:snippet_lines])
        except OSError:
            head = ""
        blocks.append(f"FILE: {rel}\n---\n{head}\n")
    listing = "\n".join(blocks)
    prompt = (
        f"A bug/feature issue must be fixed by editing source files.\n\n"
        f"Issue:\n{issue_text[:1500]}\n\n"
        f"Candidate files (path + first lines):\n{listing}\n\n"
        f"Which {pick} files are MOST likely to need editing to resolve this issue? "
        f"Reply with ONLY the file paths, one per line, no commentary."
    )
    fallback = [c[0] for c in candidates[:pick]]
    try:
        resp = slm.invoke(prompt).content
    except Exception:
        return fallback
    valid = {c[0] for c in candidates}
    picked = []
    for line in resp.splitlines():
        line = line.strip().strip("`-*0123456789. ")
        for cand in valid:
            if cand in line and cand not in picked:
                picked.append(cand)
    return picked[:pick] or fallback


def localize(repo_dir, issue_text, slm=None, top_k=10, pick=3):
    """Full localization: BM25 -> optional SLM re-rank. Returns list of relpaths."""
    cands = bm25_candidates(repo_dir, issue_text, top_k=top_k)
    if not cands:
        return []
    if slm is not None:
        return slm_rerank(slm, issue_text, cands, repo_dir, pick=pick)
    return [c[0] for c in cands[:pick]]
