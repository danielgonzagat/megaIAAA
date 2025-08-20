from __future__ import annotations
import os, json, statistics
from typing import Callable, Dict, List, Any

from ..providers.openai_provider import OpenAIClient
from ..providers.ollama_provider import OllamaClient

def llm_client_from_env():
    prov = os.getenv("LEM_PROVIDER","stub").lower()
    if prov == "openai":
        return OpenAIClient()
    elif prov == "ollama":
        return OllamaClient()
    else:
        return None  # stub

class DualVerifier:
    """Combines an LLM grader ('real' provider if available) with task-specific heuristics."""
    def __init__(self, mode: str = "real", rubric: str = "general", heuristic_fn: Callable[[str, Dict], float] | None = None):
        self.mode = mode
        self.rubric = rubric
        self.heuristic_fn = heuristic_fn

    def grade(self, prompt: str, answer: str, context: Dict | None = None) -> Dict[str, Any]:
        scores = {}
        if self.mode == "real":
            client = llm_client_from_env()
            if client is not None:
                msg = [
                    {"role":"system","content": f"You are a strict grader. Rubric: {self.rubric}. Return a JSON with fields 'score' (0..1) and 'rationale'."},
                    {"role":"user","content": f"PROMPT:\n{prompt}\n\nANSWER:\n{answer}"}
                ]
                try:
                    out = client.chat(msg, max_tokens=256, temperature=0.0)
                    j = _safejson(out)
                    if isinstance(j, dict) and "score" in j:
                        scores["llm"] = float(j["score"])
                        scores["llm_rationale"] = j.get("rationale","")
                    else:
                        scores["llm"] = 0.5
                        scores["llm_rationale"] = "fallback"
                except Exception as e:
                    scores["llm"] = 0.5
                    scores["llm_rationale"] = f"error:{e}"
            else:
                scores["llm"] = 0.5
                scores["llm_rationale"] = "stub-provider"
        # Heuristics
        if self.heuristic_fn:
            try:
                scores["heur"] = float(self.heuristic_fn(answer, context or {}))
            except Exception:
                scores["heur"] = 0.0
        # Aggregate
        vals = [v for k,v in scores.items() if k in ("llm","heur")]
        agg = statistics.fmean(vals) if vals else 0.0
        return {"score": agg, **scores}

def _safejson(txt: str):
    try:
        return json.loads(txt)
    except Exception:
        # attempt to extract a JSON object
        import re
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

def cli():
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--rubric", default="reasoning")
    p.add_argument("--mode", default="real")
    p.add_argument("--prompt", required=True)
    p.add_argument("--answer", required=True)
    args = p.parse_args()
    v = DualVerifier(mode=args.mode, rubric=args.rubric, heuristic_fn=lambda a,c: 1.0 if len(a.strip())>0 else 0.0)
    print(v.grade(args.prompt, args.answer))
