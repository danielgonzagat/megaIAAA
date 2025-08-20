from __future__ import annotations
from typing import List, Dict
import numpy as np

def ece_from_log(cal_log: List[Dict[str,float]], bins: int = 10) -> float:
    if not cal_log: return 0.0
    probs = np.array([x["prob"] for x in cal_log])
    accs  = np.array([x["acc"] for x in cal_log])
    edges = np.linspace(0,1,bins+1)
    e = 0.0
    for i in range(bins):
        mask = (probs >= edges[i]) & (probs < edges[i+1])
        if mask.any():
            e += mask.mean() * abs(probs[mask].mean() - accs[mask].mean())
    return float(e)
