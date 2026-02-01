"""Package for parameter fitting and parameter optimization.

For additional resources see for instance
https://petab.readthedocs.io/en/latest/index.html
https://pyabc.readthedocs.io/en/latest/index.html
"""

from .objects import FitMapping, FitData, FitExperiment, FitParameter

__all__ = [
    "FitMapping",
    "FitData",
    "FitExperiment",
    "FitParameter",
]
