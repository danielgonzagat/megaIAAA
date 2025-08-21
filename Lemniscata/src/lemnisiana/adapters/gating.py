from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AdapterMetrics:
    ece: float
    latency_p95_ms: float
    cost_per_1k: float
    lyapunov_deriv: float = 0.0
    oci: float = 1.0
    gpu_mem_gb: float = 0.0
    token_usage: int = 0
    hourly_cost: float = 0.0

def allow_promotion(m: AdapterMetrics, thr: dict, guards: dict, budgets: dict) -> bool:
    return (
        m.cost_per_1k <= thr.get('cost_per_1k', 0.05) and
        m.ece <= guards.get('ece_limit', 0.05) and
        m.latency_p95_ms <= guards.get('latency_p95_ms', 1500) and
        m.lyapunov_deriv <= guards.get('lyapunov_deriv_max', 0.0) and
        m.oci >= guards.get('oci_minimum', 0.0) and
        m.gpu_mem_gb <= budgets.get('gpu_memory_gb', float('inf')) and
        m.token_usage <= budgets.get('token_usage', float('inf')) and
        m.hourly_cost <= budgets.get('hourly_cost', float('inf'))
    )
