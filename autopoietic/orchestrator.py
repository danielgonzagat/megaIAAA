from __future__ import annotations

from prometheus_client import Gauge, start_http_server

from .metrics import compute_indicators, compute_oci
from .lyapunov import vdot


class AutopoieticOrchestrator:
    """Simple orchestrator managing mutation/learning rates.

    It evaluates OCI and the Lyapunov derivative ``vdot`` for a given
    window of data. The values are exported as Prometheus metrics and
    the rates are adjusted based on the sign of ``vdot``.
    """

    def __init__(
        self,
        mutation_rate: float = 0.01,
        learning_rate: float = 0.01,
        port: int | None = 8000,
        start_server: bool = True,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.last_oci = 0.0
        self.last_vdot = 0.0

        self.oci_gauge = Gauge("oci", "Organizational Complexity Index")
        self.vdot_gauge = Gauge("vdot", "Derivative of Lyapunov function")

        if start_server and port is not None:
            start_http_server(port)

    # ------------------------------------------------------------------
    def step(self, history, state: float, dstate_dt: float) -> None:
        """Process a new window of history and update metrics."""
        indicators = compute_indicators(history)
        self.last_oci = compute_oci(indicators)
        self.oci_gauge.set(self.last_oci)

        self.last_vdot = vdot(state, dstate_dt)
        self.vdot_gauge.set(self.last_vdot)

        self._adjust_rates()

    # ------------------------------------------------------------------
    def _adjust_rates(self) -> None:
        """Adjust mutation and learning rates based on ``vdot``."""
        factor = 1.1 if self.last_vdot > 0 else 0.9
        self.mutation_rate *= factor
        self.learning_rate *= factor

    # ------------------------------------------------------------------
    def check_promotion(self, oci_threshold: float, vdot_threshold: float = 0.0) -> bool:
        """Check promotion gate conditions.

        Promotion is granted only if OCI exceeds ``oci_threshold`` and
        the system is stable (``vdot`` below ``vdot_threshold``).
        """
        return self.last_oci >= oci_threshold and self.last_vdot <= vdot_threshold
