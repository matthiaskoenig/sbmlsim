"""Evaluation of a fit: acceptance criteria and their aggregates.

The functions work on the table of the data points of a fit,
`FitMetrics.datapoints_df`, i.e. on the columns `DV` (the measured value),
`IPRED` (the prediction) and `DV_SD` (the standard deviation of the measured
value, `NaN` without one), see `sbmlsim.fit.metrics`.

A criterion decides per data point whether its prediction is acceptable, and
it is applicable only to some points: a fold criterion to the points whose
value and prediction are positive and finite, which have a ratio, an
uncertainty criterion to the points with a standard deviation. A criterion is
reported as the percentage of the applicable points it accepts.

`GMFE`, the geometric mean fold error `10**mean(|log10(f/y)|)`, is how far the
predictions are from the data as a fold factor whatever the direction, `AFE`,
the average fold error `10**mean(log10(f/y))`, is the bias as a fold factor:
1 for no bias, above 1 for an overprediction.

The criterion of Guest et al. (2011) is defined for ratios, e.g. the ratio of
the AUC of a victim drug with and without a perpetrator, and is the published
one, `delta = 1`: `guest_limits` and `within_guest`. It is not a criterion of
the data points of a fit: applied to an absolute value its limits depend on
the unit of the value (an AUC of 50 in one unit is 0.05 in another).

Guest EJ, Aarons L, Houston JB, Rostami-Hodjegan A, Galetin A. Critique of the
two-fold measure of prediction success for ratios: application for the
assessment of drug-drug interactions. Drug Metab Dispos. 2011;39(2):170-173.
https://doi.org/10.1124/dmd.110.036103
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class Criterion(Protocol):
    """A criterion of the acceptance of the prediction of a data point."""

    @property
    def name(self) -> str:
        """Get the name of the criterion, the column of its percentage."""
        ...

    def accepted(self, points: pd.DataFrame) -> pd.Series:
        """Get whether the criterion accepts the prediction of every point.

        Args:
            points: the data points, see the module.

        Returns:
            `1.0` for an accepted point, `0.0` for a rejected one and `NaN`
            where the criterion is not applicable, with the index of `points`.
        """
        ...


def _ratio_mask(points: pd.DataFrame) -> np.ndarray:
    """Get the mask of the points with a ratio, a positive and finite value and prediction."""
    dv = np.asarray(points["DV"], dtype=float)
    ipred = np.asarray(points["IPRED"], dtype=float)
    return np.isfinite(dv) & np.isfinite(ipred) & (dv > 0) & (ipred > 0)


@dataclass(frozen=True)
class Fold:
    """The prediction is within a fold factor of the data, `y/fold <= f <= y*fold`.

    Attributes:
        fold: the fold factor, larger than 1.
    """

    fold: float = 2.0

    def __post_init__(self) -> None:
        """Check the fold factor.

        Raises:
            ValueError: if the fold factor is not larger than 1.
        """
        if not self.fold > 1.0:
            raise ValueError(
                f"The fold factor must be larger than 1, but '{self.fold}' given."
            )

    @property
    def name(self) -> str:
        """Get the name of the criterion, e.g. `within_2fold`."""
        return f"within_{self.fold:g}fold"

    def accepted(self, points: pd.DataFrame) -> pd.Series:
        """Get whether the prediction of every point is within the fold factor.

        The bounds are products, so that twofold is exact in floating point.
        """
        dv = np.asarray(points["DV"], dtype=float)
        ipred = np.asarray(points["IPRED"], dtype=float)
        inside = (ipred >= dv / self.fold) & (ipred <= dv * self.fold)
        values = np.where(_ratio_mask(points), inside.astype(float), np.nan)
        return pd.Series(values, index=points.index, dtype=float)


@dataclass(frozen=True)
class Uncertainty:
    """The prediction is within `k` standard deviations of the data.

    Attributes:
        k: the number of standard deviations.
    """

    k: float = 2.0

    @property
    def name(self) -> str:
        """Get the name of the criterion, e.g. `within_2sd`."""
        return f"within_{self.k:g}sd"

    def accepted(self, points: pd.DataFrame) -> pd.Series:
        """Get whether the prediction of every point is within `k` standard deviations."""
        dv = np.asarray(points["DV"], dtype=float)
        ipred = np.asarray(points["IPRED"], dtype=float)
        sd = np.asarray(points["DV_SD"], dtype=float)
        applicable = np.isfinite(sd) & (sd > 0) & np.isfinite(ipred) & np.isfinite(dv)
        with np.errstate(invalid="ignore"):
            inside = np.abs(dv - ipred) <= self.k * sd
        values = np.where(applicable, inside.astype(float), np.nan)
        return pd.Series(values, index=points.index, dtype=float)


#: the criteria of the metrics of a fit
DEFAULT_CRITERIA: tuple[Criterion, ...] = (
    Fold(1.25),
    Fold(1.5),
    Fold(2.0),
    Uncertainty(1.0),
    Uncertainty(2.0),
)


def _log_ratios(points: pd.DataFrame) -> np.ndarray:
    """Get `log10(f/y)` of the points with a ratio."""
    mask = _ratio_mask(points)
    dv = np.asarray(points["DV"], dtype=float)[mask]
    ipred = np.asarray(points["IPRED"], dtype=float)[mask]
    return np.log10(ipred / dv)


def gmfe(points: pd.DataFrame) -> float:
    """Get the geometric mean fold error `10**mean(|log10(f/y)|)`.

    Args:
        points: the data points, see the module.

    Returns:
        The fold error of the points with a ratio, `nan` without one.
    """
    ratios = _log_ratios(points)
    return float(10 ** np.mean(np.abs(ratios))) if ratios.size else float("nan")


def afe(points: pd.DataFrame) -> float:
    """Get the average fold error `10**mean(log10(f/y))`, the bias as a fold factor.

    Args:
        points: the data points, see the module.

    Returns:
        The fold error of the points with a ratio, `nan` without one.
    """
    ratios = _log_ratios(points)
    return float(10 ** np.mean(ratios)) if ratios.size else float("nan")


def criteria_metrics(
    points: pd.DataFrame, criteria: Sequence[Criterion] = DEFAULT_CRITERIA
) -> dict[str, float]:
    """Get the fold errors and the percentage of the points every criterion accepts.

    Args:
        points: the data points, see the module.
        criteria: the criteria.

    Returns:
        `GMFE`, `AFE` and the percentage (0 to 100) of the applicable points
        every criterion accepts by its name, `nan` for a criterion which is
        applicable to no point.
    """
    metrics: dict[str, float] = {"GMFE": gmfe(points), "AFE": afe(points)}
    for criterion in criteria:
        accepted = np.asarray(criterion.accepted(points), dtype=float)
        accepted = accepted[np.isfinite(accepted)]
        metrics[criterion.name] = (
            100.0 * float(np.mean(accepted)) if accepted.size else float("nan")
        )
    return metrics


def guest_limits(observed: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Get the limits of the criterion of Guest et al. for observed ratios.

    With the observed ratio `R`, inverted when it is below 1, the limit is
    `(1 + 2 (R - 1)) / R` (`delta = 1`, as published) and a predicted ratio
    is accepted between `R / limit` and `R * limit`. The limit is 1 at
    `R = 1` and tends to 2 for large ratios. The limits of a ratio below 1
    are the inverted limits of its inverse.

    Args:
        observed: the observed ratios, positive.

    Returns:
        The lower and the upper limit of the predicted ratio of every
        observed ratio.
    """
    ratio = np.asarray(observed, dtype=float)
    inverted = ratio < 1.0
    r = np.where(inverted, 1.0 / ratio, ratio)
    limit = (1.0 + 2.0 * (r - 1.0)) / r
    low, high = r / limit, r * limit
    return np.where(inverted, 1.0 / high, low), np.where(inverted, 1.0 / low, high)


def within_guest(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Get the percentage of the predicted ratios within the limits of Guest et al.

    Args:
        observed: the observed ratios, positive.
        predicted: the predicted ratios.

    Returns:
        The percentage (0 to 100) of the predicted ratios within the limits of
        `guest_limits`, the limits included, `nan` without a ratio.
    """
    predicted = np.asarray(predicted, dtype=float)
    if predicted.size == 0:
        return float("nan")
    low, high = guest_limits(observed)
    # a tolerance of a few units in the last place, so that a limit is in
    rtol = 1e-12
    inside = (predicted >= low * (1 - rtol)) & (predicted <= high * (1 + rtol))
    return 100.0 * float(np.mean(inside))
