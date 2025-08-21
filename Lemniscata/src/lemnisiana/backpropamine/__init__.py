"""Adapter for the Backpropamine neuromodulated network project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class NeuromodulatedNetwork:
    """Thin wrapper around a neuromodulated network backend.

    The heavy Backpropamine dependency is optional; a custom ``backend``
    callable implementing the update logic can be supplied instead.
    """

    backend: Callable[..., Any] | None = None

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Run a single update step via the backend."""

        if self.backend is None:
            raise RuntimeError("Backpropamine backend is not available")
        return self.backend(*args, **kwargs)


__all__ = ["NeuromodulatedNetwork"]

