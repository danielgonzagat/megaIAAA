from dataclasses import dataclass
import numpy as np


@dataclass
class MaintenanceIndicators:
    """Container for maintenance indicators.

    Parameters
    ----------
    production: float
        Average production value.
    decay: float
        Average decay value.
    maintenance: float
        Average maintenance value.
    consumption: float
        Average consumption value.
    """

    production: float
    decay: float
    maintenance: float
    consumption: float


def compute_indicators(history: np.ndarray) -> MaintenanceIndicators:
    """Compute maintenance indicators from a history matrix.

    The ``history`` array must have four columns representing
    Production, Decay, Maintenance and Consumption values respectively.
    The function returns the mean value for each indicator.
    """
    history = np.asarray(history)
    if history.ndim != 2 or history.shape[1] != 4:
        raise ValueError("history must be a 2D array with 4 columns")

    prod = float(np.mean(history[:, 0]))
    decay = float(np.mean(history[:, 1]))
    maint = float(np.mean(history[:, 2]))
    cons = float(np.mean(history[:, 3]))
    return MaintenanceIndicators(prod, decay, maint, cons)


def compute_oci(indicators: MaintenanceIndicators) -> float:
    """Compute the Organizational Complexity Index (OCI).

    OCI is defined as (production + maintenance) / (decay + consumption).
    The denominator is clamped to a small value to avoid division by zero.
    """
    denom = indicators.decay + indicators.consumption
    if abs(denom) < 1e-9:
        denom = 1e-9
    return (indicators.production + indicators.maintenance) / denom
