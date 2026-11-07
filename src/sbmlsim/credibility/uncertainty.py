"""The parameter uncertainty of the decisive predictions (VV&UQ step 4).

The adopted fits estimate their parameters with an uncertainty, the profile
likelihood interval or the standard error of the Fisher information of their
record (`sbmlsim.fit.records.AdoptedRecord`). A context of use whose acceptance is
met at the adopted parameters only may fail for parameters which the data fit
equally well; the propagation of the parameter uncertainty to the decisive
predictions shows how robust the acceptance is and how precise the
predictions of the quantities of interest are.

The draws of a fit (`parameter_draws`) have the marginals the releases draw
(`ParameterUncertainty.distribution`) and the correlation of the Fisher
information of the fit, in the space of the fit, as the correlation of the
normal scores of a Gaussian copula (`sbmlsim.uncertainty.draws.draw`): for a
normal or log-normal marginal in the scale of the fit this is the multivariate
normal of the Fisher covariance, with the widths of the profile. A Fisher
information which is not of full rank, or none, gives independent draws. The
fits are drawn independently, and the parameters a fit fixes from another
adopted fit stay at their adopted values.

`propagate_fits` evaluates every adopted fit at its draws with the report of
the fresh evaluation (`FitTable.reports`); the assessment reads the predictions
per draw (`FitPropagation`) and records per context of use the prediction bands
of the decisive points, their coverage of the observations and the
probability that every acceptance rule is met, decided with the anytime-valid
statements of `sbmlsim.uncertainty.testing` (`UncertaintyRecord`,
`uncertainty_record_from_dict` reads the one of a record of pkpdsim 0.19 as
well).

The names differ from the ones of pkpdsim, which clashed with the propagation
of `sbmlsim.uncertainty` and the settings of the draws of the experiments
(`sbmlsim.experiment.draws.UncertaintySettings`): `propagate_fits` was
`propagate`, `FitPropagation` was `Propagation` and `PropagationSettings` was
`UncertaintySettings`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd

from sbmlsim.fit.result import fit_subset

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from sbmlsim.credibility.records import FitTable
    from sbmlsim.fit.fisher import FisherInformation
    from sbmlsim.fit.records import ParameterUncertainty
    from sbmlsim.fit.report import FitReport
    from sbmlsim.project import Project

logger = logging.getLogger(__name__)

__all__ = [
    "FitDraws",
    "FitPropagation",
    "FitUncertainty",
    "PropagationSettings",
    "QuantityBand",
    "RuleProbability",
    "SensitivityIndex",
    "UncertaintyRecord",
    "parameter_draws",
    "propagate_fits",
    "rule_probability",
    "sensitivity_indices",
    "uncertainty_record_from_dict",
]


@dataclass(frozen=True, kw_only=True)
class PropagationSettings:
    """The settings of the propagation of the parameter uncertainty.

    Attributes:
        draws: the number of draws of every fit, a power of two keeps the
            balance of the Sobol points; 128 decide a rule met on every draw
            at the threshold 0.9 for up to about ten rules of a context.
        seed: the seed of the draws.
        quantiles: the quantiles of the prediction bands.
        threshold: the probability of the statement that a rule is met.
        level: the level of the decisions of the rules of a context of use,
            together (Bonferroni over its rules).
    """

    draws: int = 128
    seed: int = 1234
    quantiles: tuple[float, float] = (0.05, 0.95)
    threshold: float = 0.9
    level: float = 0.95

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            ValueError: for fewer than two draws, quantiles which are not
                increasing in (0, 1), a threshold or a level outside (0, 1).
        """
        low, high = self.quantiles
        problems = [
            *(["fewer than two draws"] if self.draws < 2 else []),
            *(
                []
                if 0.0 < low < high < 1.0
                else [f"the quantiles {self.quantiles} are not increasing in (0, 1)"]
            ),
            *(
                []
                if 0.0 < self.threshold < 1.0
                else [f"the threshold {self.threshold} is outside (0, 1)"]
            ),
            *(
                []
                if 0.0 < self.level < 1.0
                else [f"the level {self.level} is outside (0, 1)"]
            ),
        ]
        if problems:
            raise ValueError(
                f"The settings of the uncertainty are not valid: {'; '.join(problems)}."
            )


@dataclass(frozen=True)
class FitDraws:
    """The draws of the fitted parameters of an adopted fit.

    Attributes:
        fit_id: the id of the fit.
        pids: the fitted parameters, in the order of the columns of `values`.
        values: one row per draw, in the units of the model.
        units: the unit of every parameter of the model, `None` if unknown.
        correlated: whether the draws have the correlation of the Fisher
            information.
        note: where the dependence comes from, or why there is none.
    """

    fit_id: str
    pids: tuple[str, ...]
    values: np.ndarray = field(repr=False)
    units: Mapping[str, str | None]
    correlated: bool
    note: str

    @property
    def subset(self) -> str:
        """The subset of the fit."""
        return fit_subset(self.fit_id)


def parameter_draws(
    fit_id: str,
    parameters: Sequence[ParameterUncertainty],
    fisher: FisherInformation | None,
    settings: PropagationSettings,
) -> FitDraws:
    """Draw the fitted parameters of an adopted fit, see the module.

    Args:
        fit_id: the id of the fit.
        parameters: the uncertainty of every fitted parameter.
        fisher: the Fisher information of the fit, `None` without.
        settings: the number of draws and the seed.

    Returns:
        The draws.
    """
    from sbmlsim.simulation.sampling import distribution_from_formula
    from sbmlsim.uncertainty import Covariance
    from sbmlsim.uncertainty.draws import draw
    from sbmlsim.uncertainty.sources import Variable

    pids = tuple(p.pid for p in parameters)
    variables = [
        Variable(
            key=p.pid,
            targets=(p.pid,),
            distribution=distribution_from_formula(p.distribution()),
            kind="uncertainty",
            source="parameter",
        )
        for p in parameters
    ]
    covariance: list[Covariance] = []
    if fisher is None:
        note = "independent draws, the fit has no Fisher information"
    elif not fisher.is_identifiable:
        note = (
            f"independent draws, the Fisher information has rank {fisher.rank} "
            f"of {fisher.k}"
        )
    elif set(fisher.pids) != set(pids):
        note = "independent draws, the Fisher information has other parameters"
    elif len(pids) == 1:
        note = "nothing to correlate"
    else:
        matrix = fisher.correlation.loc[list(pids), list(pids)].to_numpy(dtype=float)
        covariance.append(
            Covariance.correlation(matrix=matrix, targets=pids, nearest=True)
        )
        note = "the correlation of the Fisher information in the space of the fit"
    values = draw(variables, covariance, settings.draws, seed=settings.seed)
    matrix = np.column_stack(
        [
            np.asarray(getattr(values[pid], "magnitude", values[pid]), dtype=float)
            for pid in pids
        ]
    )
    return FitDraws(
        fit_id=fit_id,
        pids=pids,
        values=matrix,
        units={p.pid: p.unit for p in parameters},
        correlated=bool(covariance),
        note=note,
    )


@dataclass(frozen=True)
class FitPropagation:
    """The decisive predictions at the draws of the parameters of the fits.

    Attributes:
        settings: the settings of the propagation.
        draws: the draws of every fit, by its subset.
        predictions: one column per draw, the rows of the table of the fits;
            `NaN` for a draw which failed and for a fit without draws.
        failed: the failed draws of every fit, by its subset.
    """

    settings: PropagationSettings
    draws: Mapping[str, FitDraws]
    predictions: pd.DataFrame = field(repr=False)
    failed: Mapping[str, int]


def _read_fisher(directory: Path) -> FisherInformation | None:
    """Read the Fisher information of the record of a fit, `None` without."""
    from sbmlsim.fit.fisher import FisherInformation

    path = directory / "fisher.json"
    if not path.is_file():
        return None
    return FisherInformation.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _same_points(points: pd.DataFrame, reference: pd.DataFrame) -> bool:
    """Get whether the points of an evaluation are the reference points in order."""
    return len(points) == len(reference) and bool(
        (
            (points["experiment"].to_numpy() == reference["experiment"].to_numpy())
            & (points["mapping"].to_numpy() == reference["mapping"].to_numpy())
            & (
                points["x"].to_numpy(dtype=float)
                == reference["x"].to_numpy(dtype=float)
            )
        ).all()
    )


def _evaluate(
    report: FitReport, draws: FitDraws, reference: pd.DataFrame
) -> tuple[np.ndarray, int]:
    """Evaluate a fit at its draws.

    Returns:
        The predictions, one column per draw, and the number of failed draws.

    Raises:
        ValueError: if the points of a draw are not the reference points.
    """
    from sbmlsim.fit.parameters import ParameterSet

    predictions = np.full((len(reference), len(draws.values)), np.nan)
    failed = 0
    for k, row in enumerate(draws.values):
        pset = ParameterSet(
            sid=f"draw_{k}",
            values=dict(zip(draws.pids, row.tolist(), strict=True)),
            units=dict(draws.units),
            provenance=f"draw {k} of the uncertainty of {draws.fit_id}",
        )
        try:
            points = report.metrics(pset).datapoints_df()
        except (RuntimeError, ValueError) as err:
            # sbmlsim raises a `ValueError` for the data of a failed simulation
            logger.warning(
                "The draw %s of the fit '%s' failed: %s", k, draws.subset, err
            )
            failed += 1
            continue
        if not _same_points(points, reference):
            raise ValueError(
                f"The points of the draw {k} of the fit '{draws.subset}' are not the "
                f"points of its reference."
            )
        predictions[:, k] = points["IPRED"].to_numpy(dtype=float)
    return predictions, failed


def propagate_fits(
    project: Project, fits: FitTable, settings: PropagationSettings
) -> FitPropagation:
    """Evaluate the adopted fits of a project at the draws of their parameters.

    Args:
        project: the project with its adopted fits.
        fits: the fresh evaluation of the fits with their reports, see
            `sbmlsim.credibility.records.evaluate_fits`.
        settings: the settings of the propagation.

    Returns:
        The predictions per draw.

    Raises:
        ValueError: if the points of a draw are not the reference points.
    """
    from sbmlsim.credibility.records import adopted_ids
    from sbmlsim.fit.records import AdoptedRecord

    table = fits.table
    columns = list(range(settings.draws))
    predictions = pd.DataFrame(np.nan, index=table.index, columns=columns)
    found: dict[str, FitDraws] = {}
    failed: dict[str, int] = {}
    for fit_id in adopted_ids(project):
        subset = fit_subset(fit_id)
        report = fits.reports.get(subset)
        if report is None:
            continue
        record = AdoptedRecord.load(project.adopted_record(fit_id))
        draws = parameter_draws(
            fit_id,
            list(record.parameters.values()),
            _read_fisher(record.directory),
            settings,
        )
        rows = table.index[table["fit"] == subset]
        start = time.perf_counter()
        values, failed[subset] = _evaluate(report, draws, table.loc[rows])
        logger.info(
            "Evaluate the fit '%s' at %s draws in %.1f s",
            subset,
            settings.draws,
            time.perf_counter() - start,
        )
        predictions.loc[rows, columns] = values
        found[subset] = draws
    return FitPropagation(
        settings=settings, draws=found, predictions=predictions, failed=failed
    )


class FitUncertainty(TypedDict):
    """The draws of a fit of the decisive points of a context of use.

    Attributes:
        fit: the subset of the fit.
        fit_id: the id of the fit.
        n_parameters: the fitted parameters which are drawn.
        correlated: whether the draws have the Fisher correlation.
        note: where the dependence comes from, or why there is none.
        n_failed: the draws whose evaluation failed.
    """

    fit: str
    fit_id: str
    n_parameters: int
    correlated: bool
    note: str
    n_failed: int


class RuleProbability(TypedDict):
    """The probability that an acceptance rule is met over the parameter draws.

    Attributes:
        name: the rule.
        limit: its limit.
        estimate: the fraction of the draws on which it is met, `None` if it
            is evaluable on none.
        low: the lower end of the anytime-valid confidence sequence.
        high: the upper end.
        n: the draws on which the rule is evaluable.
        n_failed: the draws on which it is not (a failed evaluation).
        decision: `accepted`, `rejected`, `equivalent`, `undecided` of the
            statement `P(met) >= threshold`, or `not_evaluable`.
        n_needed: the draws an undecided statement needs at its estimate.
    """

    name: str
    limit: str
    estimate: float | None
    low: float | None
    high: float | None
    n: int
    n_failed: int
    decision: str
    n_needed: int | None


class QuantityBand(TypedDict):
    """The prediction band of a quantity of interest over the parameter draws.

    Attributes:
        quantity: the quantity of interest.
        n_points: its evaluable decisive points with a band.
        band_fold: the median of `upper / lower` of the bands of its points,
            the precision of its predictions; `None` without a point.
        coverage: the share of its points whose observation is inside the band.
    """

    quantity: str
    n_points: int
    band_fold: float | None
    coverage: float | None


class SensitivityIndex(TypedDict):
    """The first-order sensitivity index of a target to a fitted parameter.

    Attributes:
        target: a quantity of interest or an acceptance rule.
        kind: `quantity` or `rule`.
        fit: the subset of the fit of the parameter.
        parameter: the fitted parameter.
        index: the share of the variance of the target over the draws which
            the parameter explains alone, given data (Plischke 2013), `None`
            without variance or with fewer than 8 draws.
    """

    target: str
    kind: str
    fit: str
    parameter: str
    index: float | None


def sensitivity_indices(
    target: str,
    kind: str,
    y: np.ndarray,
    draws: Mapping[str, FitDraws],
) -> list[SensitivityIndex]:
    """Get the first-order indices of a target over the draws of the parameters.

    The indices are estimated from the draws of the propagation, without
    further simulations (`sbmlsim.uncertainty.statistics.first_order_given_data`);
    the index of a parameter correlated with another includes the effect of
    the other, so the indices of correlated parameters do not add up.

    Args:
        target: the quantity or the rule.
        kind: `quantity` or `rule`.
        y: the value of the target on every draw, `NaN` for a failed draw.
        draws: the draws of the fits of the target, by subset; the draw `k` of
            every fit belongs to the value `k`.

    Returns:
        One index per parameter of the fits, in their order.
    """
    from sbmlsim.uncertainty.statistics import first_order_given_data

    indices: list[SensitivityIndex] = []
    for fit, fit_draws in draws.items():
        for i, pid in enumerate(fit_draws.pids):
            # the estimate bins by the ranks of the input, any scale is one
            value = first_order_given_data(fit_draws.values[:, i], y)
            indices.append(
                SensitivityIndex(
                    target=target,
                    kind=kind,
                    fit=fit,
                    parameter=pid,
                    index=None if math.isnan(value) else value,
                )
            )
    return indices


class UncertaintyRecord(TypedDict):
    """The parameter uncertainty of the decisive predictions of a context of use.

    Attributes:
        n_draws: the draws of every fit.
        seed: the seed.
        quantiles: the quantiles of the bands.
        threshold: the probability of the statements of the rules.
        level: the level of the decisions of the rules together.
        fits: the draws of the fits of the decisive points.
        n_points: the evaluable decisive points with a band.
        coverage: the share of them whose observation is inside the band.
        quantities: the bands of the quantities which are no ratio.
        rules: the probability of every acceptance rule, the ratio rule last.
        sensitivity: the first-order indices of every quantity (the mean log
            ratio of its predictions to the reference predictions) and of the
            value of every rule to every fitted parameter of the draws.
    """

    n_draws: int
    seed: int
    quantiles: list[float]
    threshold: float
    level: float
    fits: list[FitUncertainty]
    n_points: int
    coverage: float | None
    quantities: list[QuantityBand]
    rules: list[RuleProbability]
    sensitivity: list[SensitivityIndex]


def rule_probability(
    name: str,
    limit: str,
    met: Sequence[bool | None],
    settings: PropagationSettings,
    m: int,
) -> RuleProbability:
    """Get the probability that a rule is met from its results on the draws.

    Args:
        name: the rule.
        limit: its limit.
        met: whether the rule is met on every draw, `None` where it is not
            evaluable.
        settings: the threshold and the level.
        m: the number of rules decided together, for the level of Bonferroni.

    Returns:
        The probability with the decision of `P(met) >= threshold`.
    """
    import xarray as xr

    from sbmlsim.uncertainty.testing import probability

    values = np.array([np.nan if v is None else float(v) for v in met], dtype=float)
    n_failed = int(np.isnan(values).sum())
    if n_failed == len(values):
        return RuleProbability(
            name=name,
            limit=limit,
            estimate=None,
            low=None,
            high=None,
            n=0,
            n_failed=n_failed,
            decision="not_evaluable",
            n_needed=None,
        )
    result = probability(
        xr.DataArray(values, dims=("draw",), name=name),
        between=(1.0, 1.0),
        at_least=settings.threshold,
        level=1.0 - (1.0 - settings.level) / max(m, 1),
    )
    return RuleProbability(
        name=name,
        limit=limit,
        estimate=float(result.estimate),
        low=float(result.low),
        high=float(result.high),
        n=int(result.n),
        n_failed=n_failed,
        decision=str(result.decision),
        n_needed=None if result.n_needed is None else int(result.n_needed),
    )


def uncertainty_record_from_dict(
    data: Mapping[str, Any] | None,
) -> UncertaintyRecord | None:
    """Get the uncertainty record of a context of use from plain data.

    The record of pkpdsim 0.19 is read as well; a record of pkpdsim 0.18 or
    earlier has none, and a record without the sensitivity has none.

    Args:
        data: the record, `None` for none.

    Returns:
        The record, `None` without.
    """
    if data is None:
        return None
    return UncertaintyRecord(
        n_draws=int(data["n_draws"]),
        seed=int(data["seed"]),
        quantiles=[float(q) for q in data["quantiles"]],
        threshold=float(data["threshold"]),
        level=float(data["level"]),
        fits=[FitUncertainty(**f) for f in data["fits"]],
        n_points=int(data["n_points"]),
        coverage=data["coverage"],
        quantities=[QuantityBand(**q) for q in data["quantities"]],
        rules=[RuleProbability(**r) for r in data["rules"]],
        sensitivity=[SensitivityIndex(**i) for i in data.get("sensitivity", ())],
    )
