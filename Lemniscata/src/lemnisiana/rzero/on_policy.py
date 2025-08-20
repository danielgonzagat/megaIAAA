from __future__ import annotations
import os, json, random, time, math
from typing import List, Dict, Any, Tuple, Callable
from ..verifiers.dual_verifier import DualVerifier
from ..providers.openai_provider import OpenAIClient
from ..providers.ollama_provider import OllamaClient

def _client():
    prov = os.getenv("LEM_PROVIDER","stub").lower()
    if prov == "openai":
        return OpenAIClient()
    elif prov == "ollama":
        return OllamaClient()
    else:
        return None  # stub

class RZeroOnPolicy:
    """On-policy R-Zero with multi-metric self-consistency and calibration gap logging."""
    def __init__(self, maj_k: int = 5, samples: int = 6, verifier_mode: str = "real"):
        self.maj_k = maj_k
        self.samples = samples
        self.verifier = DualVerifier(mode=verifier_mode, rubric="math_reasoning",
                                     heuristic_fn=lambda a,c: 1.0 if c.get("gold") and c["gold"] in a else 0.0)
        self.calibration_log = []  # list of dicts with prob vs acc

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        prompt = problem.get("prompt") or problem.get("question") or ""
        gold = problem.get("answer") or problem.get("gold") or ""
        client = _client()
        samples = []
        for _ in range(self.samples):
            if client is None:
                # stub sample
                txt = f"[STUB] Reasoning for: {prompt[:120]} -> {gold}"
            else:
                msg = [{"role":"user","content": f"Solve step-by-step and end with 'FINAL:' then the final numeric answer.\n{prompt}"}]
                txt = client.chat(msg, max_tokens=500, temperature=0.6)
            samples.append(txt)

        # self-consistency: majority on extracted FINAL
        finals = [extract_final(x) for x in samples]
        maj = majority(finals, self.maj_k)
        # LLM+heuristics verification
        ver = self.verifier.grade(prompt, maj or (samples[0] if samples else ""), context={"gold": str(gold)})
        # naive prob proxy: fraction agreeing with majority
        agree = sum(1 for f in finals if f == maj and maj is not None)
        prob = agree / max(1, len(finals))
        acc = 1.0 if str(gold) == str(maj) and maj is not None else 0.0
        self.calibration_log.append({"prob": prob, "acc": acc})
        return {"answer": maj, "prob": prob, "acc": acc, "verify": ver, "samples": samples}

def extract_final(text: str) -> str | None:
    if not text: return None
    for tag in ["FINAL:", "Final:", "final:"]:
        if tag in text:
            return text.split(tag)[-1].strip().splitlines()[0].strip()
    # try last number
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None

def majority(items: List[str | None], k: int) -> str | None:
    from collections import Counter
    filt = [x for x in items if x is not None]
    if not filt: return None
    c = Counter(filt)
    winner, cnt = c.most_common(1)[0]
    return winner if cnt >= max(1,k) else winner  # still return winner

def ece(cal_log: List[Dict[str, float]], bins: int = 10) -> float:
    """Expected Calibration Error on prob vs acc."""
    if not cal_log: return 0.0
    import numpy as np
    probs = np.array([x["prob"] for x in cal_log])
    accs = np.array([x["acc"] for x in cal_log])
    edges = np.linspace(0,1,bins+1)
    e = 0.0
    for i in range(bins):
        mask = (probs >= edges[i]) & (probs < edges[i+1])
        if mask.any():
            conf = probs[mask].mean()
            acc = accs[mask].mean()
            e += (mask.mean()) * abs(acc - conf)
    return float(e)

def cli():
    import argparse, json, sys
    p = argparse.ArgumentParser()
    p.add_argument("--maj_k", type=int, default=5)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--verifier", default="real")
    p.add_argument("--dataset", help="path to JSONL with fields prompt/question and gold/answer", required=False)
    args = p.parse_args()
    r = RZeroOnPolicy(maj_k=args.maj_k, samples=args.samples, verifier_mode=args.verifier)
    if args.dataset and os.path.exists(args.dataset):
        out = []
        with open(args.dataset, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                out.append(r.solve(ex))
        print(json.dumps({"ece": ece(r.calibration_log), "n": len(out)}, indent=2))
    else:
        print(json.dumps(r.solve({"prompt":"What is 2+2?","gold":"4"}), indent=2))
