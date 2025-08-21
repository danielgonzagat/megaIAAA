import numpy as np

from autopoietic.metrics import compute_indicators, compute_oci
from autopoietic.orchestrator import AutopoieticOrchestrator
from autopoietic.lyapunov import V, vdot


def test_compute_indicators_and_oci():
    history = np.array([
        [10, 2, 3, 1],
        [8, 3, 4, 2],
    ])
    indicators = compute_indicators(history)
    assert indicators.production == 9.0
    assert indicators.decay == 2.5
    assert indicators.maintenance == 3.5
    assert indicators.consumption == 1.5
    oci = compute_oci(indicators)
    assert oci == (9.0 + 3.5) / (2.5 + 1.5)


def test_orchestrator_step_and_promotion():
    history = np.array([[10, 2, 3, 1]])
    orch = AutopoieticOrchestrator(start_server=False)
    orch.step(history, state=1.0, dstate_dt=-0.1)
    # vdot should be negative indicating stability
    assert orch.last_vdot == -0.1
    # mutation rate should decrease due to negative vdot
    assert orch.mutation_rate < 0.01
    # OCI should be large enough for promotion
    assert orch.check_promotion(oci_threshold=1.0)


def test_lyapunov_functions():
    x = 2.0
    dx = -0.5
    assert V(x) == 2.0
    assert vdot(x, dx) == -1.0
