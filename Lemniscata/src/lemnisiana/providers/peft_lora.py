from __future__ import annotations
import os, json, time, pathlib
from typing import Optional

def train_lora_from_failures(failures_path: str, base_model: str, out_dir: str = "./adapters/lora_out"):
    """One-click LoRA stub: ingests fail-cases and prepares a dataset.
    The actual heavy training is left to user's GPU env (kept minimal here for plug-and-play).
    """
    os.makedirs(out_dir, exist_ok=True)
    ds_out = pathlib.Path(out_dir)/"train.jsonl"
    n = 0
    with open(ds_out, "w", encoding="utf-8") as w:
        with open(failures_path, "r", encoding="utf-8") as r:
            for line in r:
                try:
                    ex = json.loads(line)
                    prompt = ex.get("prompt") or ex.get("input") or ""
                    target = ex.get("gold") or ex.get("expected") or ex.get("reference") or ""
                    if prompt and target:
                        w.write(json.dumps({"prompt": prompt, "response": target}, ensure_ascii=False)+"\n")
                        n += 1
                except Exception:
                    continue
    meta = {"base_model": base_model, "created": time.time(), "examples": n}
    with open(os.path.join(out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
    return {"prepared_examples": n, "dataset_path": str(ds_out), "meta": meta}

def cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--failures", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--out_dir", default="./adapters/lora_out")
    args = p.parse_args()
    info = train_lora_from_failures(args.failures, args.base_model, args.out_dir)
    print(json.dumps(info, indent=2))
