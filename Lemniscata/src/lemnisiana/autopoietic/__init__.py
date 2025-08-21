"""Adapter for autopoietic neural network experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AutopoieticNetwork:
    """Expose an autopoietic network through a simple interface."""

    backend: Callable[..., Any] | None = None

    def evolve(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate evolution to the backend."""

        if self.backend is None:
            raise RuntimeError("Autopoietic backend is not available")
        return self.backend(*args, **kwargs)


__all__ = ["AutopoieticNetwork"]

