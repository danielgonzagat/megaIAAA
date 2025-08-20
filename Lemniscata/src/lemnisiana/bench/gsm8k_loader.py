import json, os
def iter_gsm8k(path):
    """Yields {'prompt','gold'} from GSM8K-style JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            yield {"prompt": ex.get("question") or ex.get("prompt"), "gold": str(ex.get("answer") or ex.get("gold"))}
