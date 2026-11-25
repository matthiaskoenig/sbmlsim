"""Create SBML interpolation rules and use them to drive models with data."""

from sbmlsim.interpolation._core import (
    INTERPOLATION_CONSTANT,
    INTERPOLATION_CUBIC_SPLINE,
    INTERPOLATION_LINEAR,
    Interpolation,
    InterpolationMethod,
    Interpolator,
)

__all__ = [
    "INTERPOLATION_CONSTANT",
    "INTERPOLATION_CUBIC_SPLINE",
    "INTERPOLATION_LINEAR",
    "Interpolation",
    "InterpolationMethod",
    "Interpolator",
]
