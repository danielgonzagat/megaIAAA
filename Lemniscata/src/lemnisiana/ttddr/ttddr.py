from __future__ import annotations
import os, glob, math, random, json, pathlib
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

def active_retrieval(query: str, corpus_dir: str, topk: int = 4) -> List[Tuple[str, float]]:
    docs = []
    paths = []
    for p in glob.glob(os.path.join(corpus_dir, "**/*"), recursive=True):
        if os.path.isfile(p) and os.path.getsize(p) < 2_000_000:
            try:
                txt = open(p, "r", encoding="utf-8", errors="ignore").read()
                docs.append(txt); paths.append(p)
            except Exception: pass
    if not docs: return []
    vec = TfidfVectorizer(stop_words="english").fit(docs + [query])
    q = vec.transform([query])
    D = vec.transform(docs)
    import numpy as np
    sims = (D @ q.T).toarray().ravel()
    idx = np.argsort(-sims)[:topk]
    return [(paths[i], float(sims[i])) for i in idx]

def denoise_report(seed: str, steps: int, retrieval_cb, corpus_dir: str) -> str:
    report = seed
    for t in range(steps):
        gaps = identify_gaps(report)
        if gaps:
            q = gaps[0]
            hits = retrieval_cb(q, corpus_dir, topk=4)
            snippets = []
            for path, _ in hits:
                try:
                    snippets.append(open(path, "r", encoding="utf-8", errors="ignore").read()[:800])
                except Exception: pass
            report = refine(report, q, snippets)
        else:
            report = refine(report, None, [])
    return report

def identify_gaps(report: str) -> List[str]:
    cues = ["TBD","missing","unknown","???"]
    gaps = []
    for c in cues:
        if c in report: gaps.append("fill " + c)
    if not gaps and len(report) < 600:
        gaps.append("add background section")
    return gaps

def refine(report: str, query: str | None, snippets: List[str]) -> str:
    parts = []
    if query:
        parts.append(f"## Resolved: {query}\n")
    if snippets:
        parts.append("### Evidence\n" + "\n---\n".join(snippets[:2]) + "\n")
    parts.append(report + "\n")
    return "\n".join(parts)

def cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--emit", default="report.md")
    args = p.parse_args()
    seed = "# Research Report\n\nTBD\n"
    out = denoise_report(seed, args.steps, active_retrieval, args.corpus)
    open(args.emit, "w", encoding="utf-8").write(out)
    print(args.emit)
