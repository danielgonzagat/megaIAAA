"""Adapter for the RDIS project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RDISPipeline:
    """Wrapper around an RDIS processing pipeline.

    Users may supply a ``backend`` callable that performs the actual
    computation, keeping the RDIS dependency optional.
    """

    backend: Callable[..., Any] | None = None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the pipeline via the backend."""

        if self.backend is None:
            raise RuntimeError("RDIS backend is not available")
        return self.backend(*args, **kwargs)


__all__ = ["RDISPipeline"]

