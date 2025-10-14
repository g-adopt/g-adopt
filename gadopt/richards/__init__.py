"""
Richards equation module for unsaturated flow modeling.

This module provides implementations for solving the Richards equation
for variably saturated groundwater flow, including soil curve models
and numerical solvers.
"""

from .soil_curves import (
    SoilCurve,
    HaverkampCurve,
    VanGenuchtenCurve,
    ExponentialCurve,
)

__all__ = [
    "SoilCurve",
    "HaverkampCurve",
    "VanGenuchtenCurve",
    "ExponentialCurve",
]
