from __future__ import annotations
import json, os
from .gating import AdapterMetrics, allow_promotion

def decide_promotion(shadow_metrics_path: str, canary_metrics_path: str, thresholds: dict) -> dict:
    if not (os.path.exists(shadow_metrics_path) and os.path.exists(canary_metrics_path)):
        return {"promote": False, "reason": "missing_metrics"}
    s = json.load(open(shadow_metrics_path))
    c = json.load(open(canary_metrics_path))
    # require shadow pass and canary stable
    s_ok = allow_promotion(AdapterMetrics(**s), thresholds)
    c_ok = allow_promotion(AdapterMetrics(**c), thresholds)
    return {"promote": bool(s_ok and c_ok), "shadow_ok": s_ok, "canary_ok": c_ok}
