"""Common adapter interface for legacy modules."""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Adapter(ABC):
    """Abstract base class specifying the adapter interface."""

    @abstractmethod
    def propose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a proposal for a new candidate solution."""

    @abstractmethod
    def train(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Train the candidate produced by :meth:`propose`."""

    @abstractmethod
    def optimize(self, trained: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimisation steps to the trained candidate."""

    @abstractmethod
    def grade(self, optimised: Dict[str, Any]) -> float:
        """Return a numeric score representing quality of the candidate."""
