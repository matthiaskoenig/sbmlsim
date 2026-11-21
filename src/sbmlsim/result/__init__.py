"""Results of simulations and simulation experiments."""

from .scan import ScanResult
from .timecourse import TimecourseResult, TimecourseSensitivities

__all__ = ["ScanResult", "TimecourseResult", "TimecourseSensitivities"]
