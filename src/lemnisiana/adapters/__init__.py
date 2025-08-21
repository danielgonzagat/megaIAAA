"""Adapters for legacy modules providing a unified interface."""
from .base import Adapter
from .ednag import EDNAGAdapter
from .backpropamine import BackpropamineAdapter
from .rdis import RDISAdapter
from .ttd_dr import TTDDRAdapter
from .r_zero import RZeroAdapter

__all__ = [
    "Adapter",
    "EDNAGAdapter",
    "BackpropamineAdapter",
    "RDISAdapter",
    "TTDDRAdapter",
    "RZeroAdapter",
]
