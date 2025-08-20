from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AdapterMetrics:
    ece: float
    latency_p95_ms: float
    cost_per_1k: float

def allow_promotion(m: AdapterMetrics, thr: dict) -> bool:
    return (m.ece <= thr.get('ece', 0.05) and
            m.latency_p95_ms <= thr.get('latency_p95_ms', 1500) and
            m.cost_per_1k <= thr.get('cost_per_1k', 0.05))
