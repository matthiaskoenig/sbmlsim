"""Evaluation of a fit: acceptance criteria and their aggregates.

The functions work on the table of the data points of a fit,
`FitMetrics.datapoints_df`, i.e. on the columns `DV` (the measured value),
`IPRED` (the prediction) and `DV_SD` (the standard deviation of the measured
value, `NaN` without one), see `sbmlsim.fit.metrics`.

The points are evaluated per subset of the data with a `Grouping`: by the
values of columns of the table, e.g. the tissue or the health of the metadata
of the mappings, or by named subsets of fit mappings, which are selected with
the `MappingFilter` of `sbmlsim.fit.helpers` a fit selects its data with.

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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import ArrayLike

from sbmlsim.fit.objects import EVALUATED_KINDS

if TYPE_CHECKING:
    from sbmlsim.fit.helpers import MappingFilter
    from sbmlsim.fit.objects import FitMapping

#: the label of a value of a column which a mapping does not report, see
#: `NOT_REPORTED_VALUES`
NOT_REPORTED = "not reported"

#: the values of a column which are not reported: the empty value of a field a
#: mapping does not have and `NR`, the value of pkpdsim for a value which a
#: study does not report; they are one group
NOT_REPORTED_VALUES: tuple[str, ...] = ("", "NR")


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


def _sd_mask(points: pd.DataFrame) -> np.ndarray:
    """Get the mask of the points with a standard deviation, a finite value and prediction."""
    dv = np.asarray(points["DV"], dtype=float)
    ipred = np.asarray(points["IPRED"], dtype=float)
    sd = np.asarray(points["DV_SD"], dtype=float)
    return np.isfinite(sd) & (sd > 0) & np.isfinite(ipred) & np.isfinite(dv)


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
        applicable = _sd_mask(points)
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
        observed: the observed ratios, positive and finite.
        predicted: the predicted ratios; a ratio which is not finite, e.g. of
            a failed simulation, is not counted.

    Returns:
        The percentage (0 to 100) of the finite predicted ratios within the
        limits of `guest_limits`, the limits included, `nan` without one.

    Raises:
        ValueError: if an observed ratio is not positive and finite, or the
            lengths of the ratios differ.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError(
            f"The observed and the predicted ratios have different lengths, "
            f"{observed.size} and {predicted.size}."
        )
    if not np.all(np.isfinite(observed) & (observed > 0)):
        raise ValueError(
            f"The observed ratios must be positive and finite: {observed.tolist()}."
        )
    finite = np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    if predicted.size == 0:
        return float("nan")
    low, high = guest_limits(observed)
    # a tolerance of a few units in the last place, so that a limit is in
    rtol = 1e-12
    inside = (predicted >= low * (1 - rtol)) & (predicted <= high * (1 + rtol))
    return 100.0 * float(np.mean(inside))


#: the fit mapping of an experiment and a mapping key, see `Grouping.groups`
MappingLookup = Callable[[str, str], "FitMapping | Any"]


@dataclass(frozen=True)
class Subset:
    """A named subset of the fit mappings.

    Attributes:
        label: the name of the subset.
        select: the filter of the mappings, called with the key and the fit
            mapping, see `sbmlsim.fit.helpers.MappingFilter`.
    """

    label: str
    select: "MappingFilter"


@dataclass(frozen=True)
class Grouping:
    """A grouping of the data points of a fit, by columns or by subsets.

    Attributes:
        name: the name of the grouping, the name of its figures and of its rows
            in the metrics.
        columns: the columns of the table of the data points; one group per
            combination of their values.
        subsets: the subsets of the fit mappings, one group per subset; the
            subsets may overlap.
    """

    name: str
    columns: tuple[str, ...] = ()
    subsets: tuple[Subset, ...] = ()

    def __post_init__(self) -> None:
        """Check that the grouping has columns or subsets.

        Raises:
            ValueError: if it has both or neither.
        """
        if bool(self.columns) == bool(self.subsets):
            raise ValueError(
                f"The grouping '{self.name}' groups by columns or subsets, "
                f"exactly one of them, not {list(self.columns)} and "
                f"{[subset.label for subset in self.subsets]}."
            )

    @classmethod
    def by(cls, *columns: str) -> "Grouping":
        """Get the grouping by the values of columns, named after them.

        Args:
            columns: the columns of the table of the data points, e.g. the
                fields of the metadata of the mappings.

        Returns:
            The grouping, e.g. `tissue_health` for `by("tissue", "health")`.
        """
        return cls(name="_".join(columns), columns=tuple(columns))

    @classmethod
    def of_subsets(cls, name: str, *subsets: Subset) -> "Grouping":
        """Get the grouping by named subsets of the fit mappings.

        Args:
            name: the name of the grouping.
            subsets: the subsets.

        Returns:
            The grouping.
        """
        return cls(name=name, subsets=tuple(subsets))

    def groups(
        self, points: pd.DataFrame, fit_mappings: MappingLookup
    ) -> list[tuple[str, pd.DataFrame]]:
        """Get the groups of the data points with their labels.

        Args:
            points: the data points with the columns `experiment` and
                `mapping` and the columns of the grouping.
            fit_mappings: the fit mapping of an experiment and a mapping key,
                which the filter of a subset is called with.

        Returns:
            The label and the points of every group: by columns in the order
            their values appear, the values joined by `, ` and the value `""`
            labelled `NOT_REPORTED`; by subsets in the order of the subsets,
            including a subset without points.

        Raises:
            ValueError: if a column of the grouping is not a column of the
                points.
        """
        if self.columns:
            missing = [c for c in self.columns if c not in points.columns]
            if missing:
                raise ValueError(
                    f"The grouping '{self.name}' groups by {missing}, which are "
                    f"no columns of the data points: {list(points.columns)}."
                )
            # the values which are not reported are one value
            keys = pd.DataFrame(
                {
                    column: [
                        NOT_REPORTED if str(v) in NOT_REPORTED_VALUES else str(v)
                        for v in points[column]
                    ]
                    for column in self.columns
                },
                index=points.index,
            )
            groups: list[tuple[str, pd.DataFrame]] = []
            for values, of_keys in keys.groupby(
                list(self.columns), sort=False, dropna=False
            ):
                values = values if isinstance(values, tuple) else (values,)
                label = ", ".join(str(value) for value in values)
                groups.append((label, points.loc[of_keys.index]))
            return groups
        keys = points[["experiment", "mapping"]].drop_duplicates()
        result: list[tuple[str, pd.DataFrame]] = []
        for subset in self.subsets:
            selected = {
                (experiment, mapping)
                for experiment, mapping in keys.itertuples(index=False)
                if subset.select(mapping, fit_mappings(experiment, mapping))
            }
            mask = [
                (experiment, mapping) in selected
                for experiment, mapping in zip(
                    points.experiment, points.mapping, strict=True
                )
            ]
            result.append((subset.label, points[mask]))
        return result


#: the columns of `group_metrics`
GROUP_METRICS_COLUMNS: tuple[str, ...] = (
    "grouping",
    "group",
    "kind",
    "n",
    "n_ratio",
    "n_sd",
    "NRMSE",
    "GMFE",
    "AFE",
    *(criterion.name for criterion in DEFAULT_CRITERIA),
)


def _rms(values: pd.Series) -> float:
    """Get the root mean square, `nan` without a value or with one which is not finite.

    As `sbmlsim.fit.metrics.rmse`: a residual which is not finite, e.g. of a
    failed simulation, makes the metric undefined.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan")
    return float(np.sqrt(np.mean(array**2)))


def group_metrics(
    points: pd.DataFrame, grouping: Grouping, fit_mappings: MappingLookup
) -> pd.DataFrame:
    """Get the metrics of every group and kind of fit mapping.

    Args:
        points: the data points, see `Grouping.groups`, with the columns
            `kind`, `DV`, `IPRED`, `IRES`, `NRES` and `DV_SD`.
        grouping: the grouping.
        fit_mappings: the fit mapping of an experiment and a mapping key.

    Returns:
        One row per group and kind, the kinds in the order of
        `EVALUATED_KINDS`, a group without points of a kind has no row of it;
        the columns `grouping`, `group`, `kind`, `n`, `n_ratio` (the points
        with a ratio, which the fold criteria count), `n_sd` (the points with
        a standard deviation, which the SD criteria count), `NRMSE` and the
        metrics of `criteria_metrics`. There is no RMSE, a group mixes the
        units of its observables; the NRMSE is scale free.
    """
    rows: list[dict[str, Any]] = []
    for label, of_group in grouping.groups(points, fit_mappings):
        for kind in EVALUATED_KINDS:
            of_kind = of_group[of_group.kind == kind.value]
            if of_kind.empty:
                continue
            rows.append(
                {
                    "grouping": grouping.name,
                    "group": label,
                    "kind": kind.value,
                    "n": len(of_kind),
                    "n_ratio": int(np.sum(_ratio_mask(of_kind))),
                    "n_sd": int(np.sum(_sd_mask(of_kind))),
                    "NRMSE": _rms(of_kind.NRES),
                    **criteria_metrics(of_kind),
                }
            )
    if not rows:
        return pd.DataFrame(columns=pd.Index(GROUP_METRICS_COLUMNS))
    return pd.DataFrame(rows)


#: the points with a ratio a subset needs for its own limits of agreement, see
#: `subset_agreement`
MIN_POINTS_SUBSET_AGREEMENT = 10


def subset_agreement(
    points: pd.DataFrame,
) -> tuple[float, float, float, float] | None:
    """Get the agreement of a subset of the data points with its confidence.

    The bias `mean(log10(f/y))` and the half width `1.96 SD` of the limits of
    agreement of the points with a ratio, and the half widths of the 95%
    confidence intervals of the bias, `t SD / sqrt(n)`, and of a limit,
    `t sqrt(3 SD^2 / n)` (Bland and Altman 1999), with `t` the 97.5% quantile
    of the t distribution of `n - 1` degrees of freedom: with few points a
    limit is uncertain, and the interval says how much.

    Bland JM, Altman DG. Measuring agreement in method comparison studies.
    Stat Methods Med Res. 1999;8(2):135-160.

    Args:
        points: the data points, see the module.

    Returns:
        The bias, the half width of the limits and the half widths of the
        intervals of the bias and of a limit, in decades; `None` for fewer
        than `MIN_POINTS_SUBSET_AGREEMENT` points with a ratio.
    """
    ratios = _log_ratios(points)
    n = ratios.size
    if n < MIN_POINTS_SUBSET_AGREEMENT:
        return None
    sd = float(np.std(ratios, ddof=1))
    t = float(scipy.stats.t.ppf(0.975, n - 1))
    return (
        float(np.mean(ratios)),
        1.96 * sd,
        t * sd / np.sqrt(n),
        t * np.sqrt(3.0 * sd**2 / n),
    )
