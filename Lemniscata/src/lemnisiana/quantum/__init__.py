"""Adapter for quantum inspired agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class QuantumCircuit:
    """Minimal wrapper around a quantum style backend."""

    backend: Callable[..., Any] | None = None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the circuit via the backend."""

        if self.backend is None:
            raise RuntimeError("Quantum backend is not available")
        return self.backend(*args, **kwargs)


__all__ = ["QuantumCircuit"]

