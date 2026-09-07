"""Package for parameter fitting and parameter optimization.

For additional resources see for instance
https://petab.readthedocs.io/en/latest/index.html
https://pyabc.readthedocs.io/en/latest/index.html
"""

from .objects import FitData, FitExperiment, FitMapping, FitParameter

__all__ = [
    "FitData",
    "FitExperiment",
    "FitMapping",
    "FitParameter",
]
