import csv
def iter_mmlu(path):
    """Yields {'prompt','gold'} from simple CSV (question,answer)."""
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        for row in r:
            if not row: continue
            q, a = row[0], row[1] if len(row)>1 else ""
            yield {"prompt": q, "gold": a}
