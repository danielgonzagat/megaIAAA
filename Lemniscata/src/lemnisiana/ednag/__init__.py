"""Adapters for the EDNAG neural architecture search project.

This module provides a light-weight wrapper so that importing
``lemnisiana.ednag`` does not require the heavy EDNAG dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ArchitectureGenerator:
    """Minimal adapter exposing an EDNAG style generator.

    The real EDNAG project is optional; users can pass in a custom
    ``backend`` callable implementing the generation logic. This keeps
    the dependency optional while offering a uniform interface.
    """

    backend: Callable[..., Any] | None = None

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the underlying backend if available."""

        if self.backend is None:
            raise RuntimeError("EDNAG backend is not available")
        return self.backend(*args, **kwargs)


__all__ = ["ArchitectureGenerator"]

