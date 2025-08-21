"""Adapter for the TTD-DR legacy module."""
from typing import Any, Dict
from .base import Adapter


class TTDDRAdapter(Adapter):
    def propose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"candidate": f"ttddr_proposal_{context.get('seed', 0)}"}

    def train(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        proposal["trained"] = True
        return proposal

    def optimize(self, trained: Dict[str, Any]) -> Dict[str, Any]:
        trained["optimized"] = True
        return trained

    def grade(self, optimised: Dict[str, Any]) -> float:
        return 0.7 if optimised.get("optimized") else 0.0
