"""The variability between the subjects: the visual predictive check of the decisive points.

The adopted fits predict every curve with the typical subject of its study
group. A model declares how its subjects differ (`Credibility.variability`, a
`sbmlsim.fit.variability.Variability`: a relative distribution per target,
e.g. `LogNormal(cv=0.3)` of a clearance or of the body weight), and
`predictive_checks` evaluates every adopted fit for `VariabilitySettings.subjects`
subjects around the inputs of every study group
(`sbmlsim.fit.variability.predictive_check`): a data point of a group of `N`
subjects is compared with the means of virtual groups of `N` subjects, a
point of an individual with the subjects. With the propagation of the
parameter uncertainty (`sbmlsim.credibility.uncertainty`) the subjects are
nested in the draws of the parameters of the fit; without it they vary around
the adopted parameters.

Per context of use, `variability_record` gives the coverage of the evaluable
decisive observations by their bands (`VariabilityRecord`) with its Wilson
interval and whether the expected coverage, the probability between the
quantiles of the bands, lies in it (`consistent`), below it (`wide`: the bands
are wider than the data scatter) or above it (`narrow`: the declared
variability, with the parameter uncertainty, does not explain the scatter of
the data, or the typical predictions are biased). The verdict does not change.
`variability_record_from_dict` reads the record of pkpdsim 0.19 as well.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd
from scipy import stats

from sbmlsim.credibility.selection import normalize_observable
from sbmlsim.fit.result import fit_subset

if TYPE_CHECKING:
    from sbmlsim.credibility.declaration import QuantityOfInterest
    from sbmlsim.credibility.records import FitTable
    from sbmlsim.credibility.uncertainty import FitPropagation
    from sbmlsim.fit.report import FitReport
    from sbmlsim.fit.variability import PredictiveCheck, Variability
    from sbmlsim.project import Project

__all__ = [
    "VPC_COLUMNS",
    "PredictiveChecks",
    "QuantityCoverage",
    "VariabilityRecord",
    "VariabilitySettings",
    "VpcPoint",
    "predictive_checks",
    "variability_record",
    "variability_record_from_dict",
    "variability_statement",
]

logger = logging.getLogger(__name__)

#: the columns of the points of the predictive checks, see `VpcPoint`
VPC_COLUMNS = (
    "fit",
    "experiment",
    "mapping",
    "x",
    "DV",
    "N",
    "statistic",
    "PRED",
    "low",
    "median",
    "high",
    "inside",
)


@dataclass(frozen=True)
class VariabilitySettings:
    """The settings of the predictive checks.

    Attributes:
        subjects: the subjects of every study of a parameter vector of a fit.
        studies: the studies of every parameter vector of a variability
            between the studies (`Variability.between`), else one.
        groups: the virtual groups of a data point of a group per study.
        seed: the seed of the subjects and the groups.
        quantiles: the quantiles of the bands.
        level: the level of the Wilson interval of the coverage.
    """

    subjects: int = 64
    studies: int = 16
    groups: int = 1000
    seed: int = 1234
    quantiles: tuple[float, float] = (0.05, 0.95)
    level: float = 0.95

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            ValueError: for fewer than two subjects, no study, no group, quantiles which
                are not increasing in (0, 1) or a level outside (0, 1).
        """
        low, high = self.quantiles
        problems = [
            *(["fewer than two subjects"] if self.subjects < 2 else []),
            *(["no study"] if self.studies < 1 else []),
            *(["no virtual group"] if self.groups < 1 else []),
            *(
                []
                if 0.0 < low < high < 1.0
                else [f"the quantiles {self.quantiles} are not increasing in (0, 1)"]
            ),
            *(
                []
                if 0.0 < self.level < 1.0
                else [f"the level {self.level} is outside (0, 1)"]
            ),
        ]
        if problems:
            raise ValueError(
                f"The settings of the variability are not valid: {'; '.join(problems)}."
            )


class VpcPoint(TypedDict):
    """A data point of an adopted fit with the band of its predictive check.

    Attributes:
        fit: the subset of the fit.
        experiment: the experiment.
        mapping: the key of the mapping.
        x: the x of the point, `None` for a value per simulation.
        DV: the observation.
        N: the subjects of the observation, 1 for an individual.
        statistic: `individual`, `mean` or `gmean`.
        PRED: the prediction of the typical subject.
        low: the low quantile of the band.
        median: the median of the band.
        high: the high quantile of the band.
        inside: the observation within the band, `None` without a band.
    """

    fit: str
    experiment: str
    mapping: str
    x: float | None
    DV: float
    N: int
    statistic: str
    PRED: float
    low: float | None
    median: float | None
    high: float | None
    inside: bool | None


class QuantityCoverage(TypedDict):
    """The coverage of the observations of a quantity of interest.

    Attributes:
        quantity: the quantity.
        n_points: its evaluable decisive points with a band.
        coverage: the share inside their band, `None` without a point.
    """

    quantity: str
    n_points: int
    coverage: float | None


class VariabilityRecord(TypedDict):
    """The predictive check of the decisive points of a context of use.

    Attributes:
        subjects: the subjects of every study of a parameter vector.
        studies: the studies of every parameter vector, 1 without a
            variability between the studies.
        vectors: the parameter vectors of the fits, the draws of the
            uncertainty or 1 for the adopted parameters.
        groups: the virtual groups of a data point of a group.
        seed: the seed.
        targets: the target of every factor and its distribution.
        between: the target of every factor between the studies and its
            distribution, empty without.
        quantiles: the quantiles of the bands.
        expected: the probability between them, the coverage of a model whose
            variability is the one of the data.
        n_points: the evaluable decisive points with a band.
        n_groups: those of a group of more than one subject.
        coverage: the share of them inside their band, `None` without one.
        low: the low end of the Wilson interval of the coverage.
        high: the high end.
        level: the level of the interval.
        agreement: `consistent` (the expected coverage in the interval),
            `narrow` (above it: the bands miss the data) or `wide` (below it),
            `None` without a point.
        quantities: the coverage of every quantity of interest which is no
            ratio.
        failed: the subjects whose simulation failed, over the fits.
    """

    subjects: int
    studies: int
    vectors: int
    groups: int
    seed: int
    targets: dict[str, str]
    between: dict[str, str]
    quantiles: list[float]
    expected: float
    n_points: int
    n_groups: int
    coverage: float | None
    low: float | None
    high: float | None
    level: float
    agreement: str | None
    quantities: list[QuantityCoverage]
    failed: int


@dataclass(frozen=True)
class PredictiveChecks:
    """The predictive checks of the adopted fits of a model.

    Attributes:
        settings: the settings.
        variability: the declared variability.
        points: the band of every row of the table of the fits, indexed as it,
            with the columns of `VPC_COLUMNS` but `fit`, `experiment`,
            `mapping`, `x` and `DV`; no row for a fit without a check.
        vectors: the parameter vectors of every fit, by its subset.
        studies: the studies of every parameter vector of every fit, by its
            subset.
        failed: the failed subjects of every fit, by its subset.
    """

    settings: VariabilitySettings
    variability: Variability
    points: pd.DataFrame
    vectors: Mapping[str, int]
    studies: Mapping[str, int]
    failed: Mapping[str, int]


def _vectors(
    report: FitReport, subset: str, propagation: FitPropagation | None
) -> np.ndarray:
    """Get the parameter vectors of a fit: its draws, else its reference."""
    from sbmlsim.fit.parameters import ParameterSet

    problem = report.problem
    reference = problem.bind_parameter_set(report.reference_set).x(problem.pids)
    draws = None if propagation is None else propagation.draws.get(subset)
    if draws is None:
        return np.atleast_2d(reference)
    vectors = []
    for k, row in enumerate(draws.values):
        pset = ParameterSet(
            sid=f"draw_{k}",
            values=dict(zip(draws.pids, row.tolist(), strict=True)),
            units=dict(draws.units),
            provenance=f"draw {k} of the uncertainty of {draws.fit_id}",
        )
        vectors.append(problem.bind_parameter_set(pset).x(problem.pids))
    return np.asarray(vectors, dtype=float)


def _aligned(check: PredictiveCheck, reference: pd.DataFrame) -> bool:
    """Get whether the points of a check are the points of the fit in order."""
    points = check.points
    return len(points) == len(reference) and bool(
        (
            (points["experiment"].to_numpy() == reference["experiment"].to_numpy())
            & (points["mapping"].to_numpy() == reference["mapping"].to_numpy())
            & np.isclose(
                points["DV"].to_numpy(dtype=float),
                reference["DV"].to_numpy(dtype=float),
                rtol=1e-12,
                atol=0.0,
                equal_nan=True,
            )
        ).all()
    )


def predictive_checks(
    project: Project,
    fits: FitTable,
    variability: Variability,
    settings: VariabilitySettings,
    propagation: FitPropagation | None = None,
) -> PredictiveChecks:
    """Get the predictive check of every adopted fit of a project.

    Args:
        project: the project with its adopted fits.
        fits: the fresh evaluation of the fits with their reports, see
            `sbmlsim.credibility.records.evaluate_fits`.
        variability: the variability of the subjects.
        settings: the settings.
        propagation: the draws of the parameters of the fits, see
            `sbmlsim.credibility.uncertainty.propagate_fits`; the subjects vary
            around the adopted parameters without.

    Returns:
        The bands of the points of the fits.

    Raises:
        ValueError: if a target of the variability cannot be scaled in a
            simulation of a fit, or the points of a check are not the points
            of the fit.
    """
    from sbmlsim.credibility.records import adopted_ids
    from sbmlsim.fit.variability import predictive_check, quantile_column

    low_q, high_q = settings.quantiles
    table = fits.table
    frames: list[pd.DataFrame] = []
    vectors: dict[str, int] = {}
    studies: dict[str, int] = {}
    failed: dict[str, int] = {}
    for fit_id in adopted_ids(project):
        subset = fit_subset(fit_id)
        report = fits.reports.get(subset)
        if report is None:
            continue
        problem = report.problem
        rows = table.index[table["fit"] == subset]
        values = _vectors(report, subset, propagation)
        start = time.perf_counter()
        check = predictive_check(
            problem,
            values,
            variability,
            settings.subjects,
            indices=problem.indices(),
            quantiles=(low_q, 0.5, high_q),
            groups=settings.groups,
            seed=settings.seed,
            studies=settings.studies,
        )
        logger.info(
            "The predictive check of the fit '%s' with %s subjects at %s vectors "
            "in %.1f s",
            subset,
            settings.subjects,
            len(values),
            time.perf_counter() - start,
        )
        if not _aligned(check, table.loc[rows]):
            raise ValueError(
                f"The points of the predictive check of the fit '{subset}' are not "
                f"the points of the fit."
            )
        points = check.points
        frames.append(
            pd.DataFrame(
                {
                    "N": points["N"].to_numpy(),
                    "statistic": points["statistic"].to_numpy(),
                    "PRED": points["PRED"].to_numpy(dtype=float),
                    "low": points[quantile_column(low_q)].to_numpy(dtype=float),
                    "median": points[quantile_column(0.5)].to_numpy(dtype=float),
                    "high": points[quantile_column(high_q)].to_numpy(dtype=float),
                    "inside": points["inside"].to_numpy(),
                },
                index=rows,
            )
        )
        vectors[subset] = len(values)
        studies[subset] = check.n_studies
        failed[subset] = check.failed
    points = (
        pd.concat(frames)
        if frames
        else pd.DataFrame(
            columns=["N", "statistic", "PRED", "low", "median", "high", "inside"]
        )
    )
    return PredictiveChecks(
        settings=settings,
        variability=variability,
        points=points,
        vectors=vectors,
        studies=studies,
        failed=failed,
    )


def _wilson(inside: int, n: int, level: float) -> tuple[float, float]:
    """Get the Wilson interval of a share of `n` at a level."""
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    share = inside / n
    denominator = 1.0 + z**2 / n
    center = (share + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(share * (1 - share) / n + z**2 / (4 * n**2)) / denominator
    return max(center - half, 0.0), min(center + half, 1.0)


def _describe(variability: Variability) -> dict[str, str]:
    """Get the distribution of every factor of a variability in words."""
    words: dict[str, str] = {}
    for target, distribution in variability.factors.items():
        cv = getattr(distribution, "cv", None)
        name = type(distribution).__name__
        words[target] = f"{name}(cv={cv:g})" if cv is not None else repr(distribution)
    return words


def variability_record(
    checks: PredictiveChecks,
    decisive: pd.DataFrame,
    evaluable: pd.Index,
    quantities: Sequence[QuantityOfInterest],
) -> VariabilityRecord:
    """Get the predictive check of the decisive points of a context of use.

    Args:
        checks: the predictive checks of the fits.
        decisive: the decisive points of the context, indexed as the table
            of the fits.
        evaluable: the index of the evaluable decisive points.
        quantities: the quantities of interest of the context.

    Returns:
        The coverage of the evaluable decisive points with a band.
    """
    settings = checks.settings
    low_q, high_q = settings.quantiles
    expected = high_q - low_q
    index = evaluable.intersection(checks.points.index)
    banded = checks.points.loc[index].dropna(subset=["inside"])
    inside = banded["inside"].astype(bool).to_numpy()
    n = len(banded)
    coverage = float(inside.mean()) if n else None
    low = high = None
    agreement = None
    if n:
        low, high = _wilson(int(inside.sum()), n, settings.level)
        agreement = (
            "consistent"
            if low <= expected <= high
            else "narrow"
            if expected > high
            else "wide"
        )
    observables = (
        decisive.loc[banded.index, "observable"].astype(str).map(normalize_observable)
    )
    covered: list[QuantityCoverage] = []
    for q in quantities:
        if q.ratio:
            continue
        mask = (observables == normalize_observable(q.id)).to_numpy()
        covered.append(
            QuantityCoverage(
                quantity=q.id,
                n_points=int(mask.sum()),
                coverage=float(inside[mask].mean()) if mask.any() else None,
            )
        )
    fits = set(decisive["fit"].astype(str)) if "fit" in decisive.columns else set()
    between = checks.variability.between
    return VariabilityRecord(
        subjects=settings.subjects,
        studies=max((v for f, v in checks.studies.items() if f in fits), default=1),
        vectors=max((v for f, v in checks.vectors.items() if f in fits), default=1),
        groups=settings.groups,
        seed=settings.seed,
        targets=_describe(checks.variability),
        between={} if between is None else _describe(between),
        quantiles=[low_q, high_q],
        expected=expected,
        n_points=n,
        n_groups=int((banded["N"].astype(int) > 1).sum()),
        coverage=coverage,
        low=low,
        high=high,
        level=settings.level,
        agreement=agreement,
        quantities=covered,
        failed=sum(f for subset, f in checks.failed.items() if subset in fits),
    )


def variability_statement(record: VariabilityRecord) -> str:
    """Get the statement of the predictive check of a context of use.

    Args:
        record: the predictive check of the context.

    Returns:
        The statement: the variability, the subjects and the coverage with its
        interval and agreement.
    """
    targets = ", ".join(f"{t} {d}" for t, d in record["targets"].items())
    between = ", ".join(f"{t} {d}" for t, d in record["between"].items())
    nested = (
        f" nested in {record['vectors']} draws of the parameters"
        if record["vectors"] > 1
        else " at the adopted parameters"
    )
    subjects = (
        f"{record['studies']} studies of {record['subjects']} subjects"
        if record["between"]
        else f"{record['subjects']} subjects"
    )
    head = (
        f"The variability of the subjects ({targets})"
        + (f" and between the studies ({between})" if between else "")
        + f": {subjects}{nested}"
        + (f" ({record['failed']} failed)" if record["failed"] else "")
        + "."
    )
    if record["coverage"] is None:
        return f"{head} No evaluable decisive point has a band."
    percent = 100 * (record["quantiles"][1] - record["quantiles"][0])
    words = {
        "consistent": "consistent with",
        "narrow": "below",
        "wide": "above",
    }[str(record["agreement"])]
    return (
        f"{head} The {percent:g} % bands of the predictive check hold "
        f"{100 * record['coverage']:.0f} % ({100 * (record['low'] or 0):.0f} to "
        f"{100 * (record['high'] or 0):.0f} %) of the {record['n_points']} evaluable "
        f"decisive observations ({record['n_groups']} means of groups, compared "
        f"with virtual groups of their size), {words} the {percent:g} % of a "
        f"variability which is the one of the data."
    )


def variability_record_from_dict(
    data: Mapping[str, Any] | None,
) -> VariabilityRecord | None:
    """Get the predictive check of a context of use from plain data.

    The record of pkpdsim 0.19 is read as well; a record of pkpdsim 0.18 or
    earlier has none.

    Args:
        data: the record, `None` for none.

    Returns:
        The record, `None` without.
    """
    if data is None:
        return None
    return VariabilityRecord(
        subjects=int(data["subjects"]),
        studies=int(data["studies"]),
        vectors=int(data["vectors"]),
        groups=int(data["groups"]),
        seed=int(data["seed"]),
        targets=dict(data["targets"]),
        between=dict(data["between"]),
        quantiles=[float(q) for q in data["quantiles"]],
        expected=float(data["expected"]),
        n_points=int(data["n_points"]),
        n_groups=int(data["n_groups"]),
        coverage=data["coverage"],
        low=data["low"],
        high=data["high"],
        level=float(data["level"]),
        agreement=data["agreement"],
        quantities=[QuantityCoverage(**q) for q in data["quantities"]],
        failed=int(data["failed"]),
    )
