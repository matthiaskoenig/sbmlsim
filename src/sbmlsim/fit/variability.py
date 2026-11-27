"""The variability between the subjects of a fit problem and its predictive check.

A fit predicts the data of a mapping with one subject per simulation, the
typical subject of its study group. The subjects of a study differ from it:
`Variability` declares how, as relative distributions of targets of the model,
e.g. `LogNormal(cv=0.3)` of a clearance, whose factor has the median 1 and a
coefficient of variation of 30 %. A subject has every target at its factor
times the value its simulation gives the target (the value the experiment
sets for the study group, the value of a fitted parameter or the value of the
model), so the inputs of the study groups stay what they are and the subjects
vary around them (`OptimizationProblem.evaluations(..., factors=)`, see
`sbmlsim.simulator.plan.Plan.scaled`). The factors are correlated by
`Covariance`s of the targets: a correlation of their normal scores, or the
covariance of the logarithms of log-normal factors (`scale="log"`, the omega
matrix of a population analysis), see `sbmlsim.uncertainty.draws`. The
studies differ as well, in their populations, assays and protocols:
`Variability(between=Variability(...))` is the variability between the
studies, the factors of a study, which all its subjects share and multiply
with their own.

`subject_predictions` evaluates the mappings of a problem for `n` subjects of
a variability at one parameter vector. `predictive_check` nests the subjects
in the parameter vectors it is given, e.g. draws of the uncertainty of the
fitted parameters (uncertainty outside, variability inside), and compares
every observation with the distribution its prediction has over the
subjects: with a variability between the studies the subjects of a vector are
`studies` studies of `n_subjects` subjects each. A data point of a group of `N` subjects (the count of the data
point) is a mean of `N` subjects, so it is compared with the distribution of
the mean of virtual groups of `N` subjects drawn from the subjects of a
study (the geometric mean for a mapping of geometric standard
deviations); a data point without a count is one subject. The distributions
of all parameter vectors are pooled, the predictive distribution of the
observation, and its quantiles are the band of a visual predictive check
(Holford 2005; Bergstrand et al. 2011). The share of the observations inside
the outer quantiles is the coverage, about the probability between them for
a model whose variability is that of the data.

Holford N (2005). The visual predictive check: superiority to standard
diagnostic (Rorschach) plots. PAGE 14, Abstr 738. Bergstrand M, Hooker AC,
Wallin JE, Karlsson MO (2011). Prediction-corrected visual predictive checks
for diagnosing nonlinear mixed-effects models. AAPS J 13:143-151.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from sbmlsim.fit.pool import chunks, map_evaluations
from sbmlsim.simulation.sampling.distributions import (
    EPS,
    Distribution,
    LogNormal,
    Normal,
)
from sbmlsim.uncertainty.covariance import Covariance
from sbmlsim.uncertainty.draws import _factor, normal_score_correlation, unit_points
from sbmlsim.uncertainty.sources import Variable

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.fit.options import FitSettings
    from sbmlsim.fit.pool import EvaluationPool

__all__ = [
    "PredictiveCheck",
    "SubjectPredictions",
    "Variability",
    "predictive_check",
    "quantile_column",
    "subject_predictions",
]

logger = logging.getLogger(__name__)

#: the statistic of a group of subjects: the mean, or the geometric mean of a
#: mapping of geometric standard deviations
Statistic = Literal["mean", "gmean"]


def _located(distribution: Distribution) -> Distribution:
    """Get a relative normal or log-normal distribution of a factor with its location 1.

    A covariance of the factors needs the moments of their marginals, which
    a relative distribution has only with a reference.
    """
    if isinstance(distribution, LogNormal):
        return LogNormal(median=1.0, cv=distribution.cv)
    if isinstance(distribution, Normal):
        return Normal(mean=1.0, sd=distribution.sd, cv=distribution.cv)
    return distribution


@dataclass(frozen=True)
class Variability:
    """The variability between the subjects of a fit problem, see the module.

    Attributes:
        factors: target -> the relative distribution of its factor, e.g.
            `LogNormal(cv=0.3)`; a distribution with a location is refused,
            since a subject varies around the value of its simulation.
        covariance: the correlations or covariances of the factors, by
            their targets.
        between: the variability between the studies, whose factors the
            subjects of a study share; `None` for studies which differ in the
            inputs of their simulations only.

    Raises:
        ValueError: for a distribution which is not relative, a covariance of
            targets which are no factors, or a variability between the studies
            which has one between its studies.
    """

    factors: Mapping[str, Distribution]
    covariance: Sequence[Covariance] = field(default_factory=tuple)
    between: Variability | None = None

    def __post_init__(self) -> None:
        """Check the distributions and the covariances."""
        located = [t for t, d in self.factors.items() if not d.is_relative]
        if located:
            raise ValueError(
                f"The factors of {located} have distributions with a location; a "
                f"subject varies relative to the value of its simulation, give a "
                f"relative distribution, e.g. LogNormal(cv=0.3)."
            )
        if self.covariance and not self.factors:
            raise ValueError("A variability without factors has no covariance.")
        if self.between is not None and self.between.between is not None:
            raise ValueError(
                "The variability between the studies has no variability between "
                "studies of its own."
            )
        if self.factors:
            # the correlation is checked once, named by the targets
            normal_score_correlation(self._variables(), self.covariance)

    @property
    def targets(self) -> tuple[str, ...]:
        """Get the targets of the factors, in their order."""
        return tuple(self.factors)

    @property
    def all_targets(self) -> tuple[str, ...]:
        """Get the targets of the factors and of the factors between the studies."""
        between = () if self.between is None else self.between.targets
        return tuple(dict.fromkeys([*self.targets, *between]))

    def _variables(self) -> list[Variable]:
        """Get the factors as the variables of a draw."""
        return [
            Variable(
                key=target,
                targets=(target,),
                distribution=_located(distribution),
                kind="variability",
                source="parameter",
            )
            for target, distribution in self.factors.items()
        ]

    def draw(
        self,
        n: int,
        *,
        seed: int,
        design: Literal["sobol", "lhs", "random"] = "sobol",
        start: int = 0,
    ) -> np.ndarray:
        """Get the factors of `n` subjects.

        Args:
            n: the number of subjects.
            seed: the seed.
            design: `sobol`, `lhs` or `random`, see
                `sbmlsim.uncertainty.draws.unit_points`.
            start: the index of the first subject of the sequence, so that
                the subjects of several parameter vectors are one sequence.

        Returns:
            The factors, `n` rows of one column per target.
        """
        if not self.factors:
            return np.ones((n, 0))
        variables = self._variables()
        r = normal_score_correlation(variables, self.covariance)
        u = unit_points(len(variables), n, design=design, seed=seed, start=start)
        if not np.allclose(r, np.eye(len(variables))):
            z = stats.norm.ppf(np.clip(u, EPS, 1.0 - EPS)) @ _factor(r).T
            u = stats.norm.cdf(z)
        return np.column_stack(
            [
                np.asarray(distribution.ppf(u[:, k], 1.0), dtype=float)
                for k, distribution in enumerate(self.factors.values())
            ]
        )


@dataclass(frozen=True)
class SubjectPredictions:
    """The predictions of the mappings of a problem for subjects of a variability.

    Attributes:
        targets: the targets of the factors.
        factors: the factors of every subject, one row per subject and one
            column per target, the factors of its study included.
        predictions: the index of a mapping -> its predictions at its data,
            one row per subject; `NaN` for a subject whose simulation failed.
        failed: the subjects whose simulation failed.
        study: the factors of the study of the subjects by target, empty
            without a variability between the studies.
    """

    targets: tuple[str, ...]
    factors: np.ndarray
    predictions: Mapping[int, np.ndarray]
    failed: int
    study: Mapping[str, float] = field(default_factory=dict)


def _check_factors(
    problem: OptimizationProblem,
    x: np.ndarray,
    indices: Sequence[int],
    targets: Sequence[str],
) -> None:
    """Check that every group of the mappings can scale the targets.

    Raises:
        ValueError: see `OptimizationProblem.evaluated_plan`.
    """
    ones = dict.fromkeys(targets, 1.0)
    wanted = set(indices)
    for k_group, group in enumerate(problem.mapping_groups):
        if not wanted.isdisjoint(group):
            problem.evaluated_plan(k_group, x, factors=ones)


def subject_predictions(
    problem: OptimizationProblem,
    x: np.ndarray,
    variability: Variability,
    n: int,
    *,
    indices: Sequence[int] | None = None,
    seed: int,
    design: Literal["sobol", "lhs", "random"] = "sobol",
    start: int = 0,
    study: Mapping[str, float] | None = None,
) -> SubjectPredictions:
    """Evaluate the mappings of a problem for `n` subjects of a variability.

    Args:
        problem: the initialized problem.
        x: the values of the parameters in the units of the model.
        variability: the variability of the subjects.
        n: the number of subjects.
        indices: the indices of the mappings, every mapping by default.
        seed: the seed of the factors.
        design: the design of the factors, see `Variability.draw`.
        start: the index of the first subject of the sequence.
        study: the factors of the study of the subjects by target, which
            multiply the factors of every subject; `None` for none.

    Returns:
        The factors and the predictions of every subject.

    Raises:
        ValueError: if `n` is not positive, or a target of the variability
            cannot be scaled in a simulation of the mappings, see
            `sbmlsim.simulator.plan.Plan.scaled`.
    """
    return _subject_range(
        problem,
        x,
        variability,
        n,
        range(n),
        indices=indices,
        seed=seed,
        design=design,
        start=start,
        study=study,
    )


def _subject_range(
    problem: OptimizationProblem,
    x: np.ndarray,
    variability: Variability,
    n: int,
    subjects: range,
    *,
    indices: Sequence[int] | None = None,
    seed: int,
    design: Literal["sobol", "lhs", "random"] = "sobol",
    start: int = 0,
    study: Mapping[str, float] | None = None,
) -> SubjectPredictions:
    """Evaluate some of the `n` subjects of `subject_predictions`.

    The factors are drawn for all `n` subjects, as `subject_predictions` draws
    them, and the subjects of `subjects` are evaluated: the parts of a
    sequence of subjects, concatenated in their order (`_concatenated`), are
    the subjects of the whole sequence, number for number, so the subjects of
    a study can be evaluated in several tasks of a pool.

    Returns:
        The factors and the predictions of the subjects of `subjects`, see
        `subject_predictions`.
    """
    if n < 1:
        raise ValueError(f"The number of subjects must be positive, not {n}.")
    values = np.asarray(x, dtype=float)
    chosen = list(problem.indices() if indices is None else indices)
    shared = dict(study or {})
    targets = tuple(dict.fromkeys([*variability.targets, *shared]))
    _check_factors(problem, values, chosen, targets)
    own = variability.draw(n, seed=seed, design=design, start=start)
    factors = (
        np.column_stack(
            [
                (
                    own[:, variability.targets.index(target)]
                    if target in variability.factors
                    else np.ones(n)
                )
                * shared.get(target, 1.0)
                for target in targets
            ]
        )
        if targets
        else np.ones((n, 0))
    )[subjects.start : subjects.stop]
    m = len(factors)
    sizes = {k: len(problem.y_references[k]) for k in chosen}
    predictions = {k: np.full((m, sizes[k]), np.nan) for k in chosen}
    failed = 0
    for row, i in enumerate(subjects):
        subject = {target: float(factors[row, j]) for j, target in enumerate(targets)}
        try:
            evaluated = problem.predictions(values, chosen, factors=subject)
        except ValueError as err:
            # a failed integration of a subject, the factors were checked
            logger.warning("The subject %s failed: %s", i, err)
            failed += 1
            continue
        for k, prediction in evaluated.items():
            predictions[k][row] = prediction
    return SubjectPredictions(
        targets=targets,
        factors=factors,
        predictions=predictions,
        failed=failed,
        study=shared,
    )


def _concatenated(parts: Sequence[SubjectPredictions]) -> SubjectPredictions:
    """Get the subjects of the parts of a sequence of subjects, in their order."""
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    return SubjectPredictions(
        targets=first.targets,
        factors=np.concatenate([part.factors for part in parts], axis=0),
        predictions={
            k: np.concatenate([part.predictions[k] for part in parts], axis=0)
            for k in first.predictions
        },
        failed=sum(part.failed for part in parts),
        study=first.study,
    )


def _pieces(tasks: Sequence[range], n_subjects: int) -> list[list[tuple[int, range]]]:
    """Split the subjects of the units of a predictive check into the tasks of a pool.

    The subjects of all units are one sequence of `n_units * n_subjects`, and a
    task is a range of it (`chunks`), so the tasks are of one size whatever the
    number of units is: a unit of a study is a sequence of its own and a pool
    whose units were its tasks waited for its last units, e.g. 16 studies on
    5 workers ran in 4 rounds of which the last kept one worker busy.

    A piece draws the factors of all subjects of its unit and evaluates the
    ones of its range (`_subject_range`), so the pieces of a unit are its
    subjects, whatever the design of the factors.

    Args:
        tasks: the ranges of the subjects of all units.
        n_subjects: the subjects of every unit.

    Returns:
        Per task its pieces, each a unit and the range of its subjects.
    """
    pieces: list[list[tuple[int, range]]] = []
    for task in tasks:
        parts: list[tuple[int, range]] = []
        first, last = task.start, task.stop
        for u in range(first // n_subjects, (last - 1) // n_subjects + 1):
            lo = max(first, u * n_subjects) - u * n_subjects
            hi = min(last, (u + 1) * n_subjects) - u * n_subjects
            parts.append((u, range(lo, hi)))
        pieces.append(parts)
    return pieces


def _subjects_of(
    problem: OptimizationProblem,
    shared: tuple[
        Variability,
        int,
        list[int],
        int,
        Literal["sobol", "lhs", "random"],
        list[tuple[np.ndarray, int, dict[str, float] | None]],
    ],
    task: list[tuple[int, range]],
) -> list[tuple[int, SubjectPredictions]]:
    """Get the subjects of some pieces of the units of a predictive check.

    A task of `sbmlsim.fit.pool.map_evaluations`: a unit is a vector and a
    study, with the first subject of its sequence and the factors of its
    study, and a piece the subjects of a unit in a range, see `_pieces`.
    """
    variability, n_subjects, chosen, seed, design, units = shared
    return [
        (
            u,
            _subject_range(
                problem,
                units[u][0],
                variability,
                n_subjects,
                subjects,
                indices=chosen,
                seed=seed,
                design=design,
                start=units[u][1],
                study=units[u][2],
            ),
        )
        for u, subjects in task
    ]


def _group_means(
    values: np.ndarray,
    size: int,
    groups: int,
    statistic: Statistic,
    rng: np.random.Generator,
) -> np.ndarray:
    """Get the statistic of virtual groups of subjects.

    Args:
        values: the predictions of the subjects, one row per subject and one
            column per data point, the finite rows only.
        size: the subjects of a group.
        groups: the number of groups.
        statistic: `mean` or `gmean`.
        rng: the generator of the groups.

    Returns:
        One row per group and one column per data point.
    """
    if size == 1:
        return values
    members = rng.integers(0, values.shape[0], size=(groups, size))
    chosen = values[members]
    if statistic == "gmean":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.exp(np.mean(np.log(chosen), axis=1))
    return np.mean(chosen, axis=1)


@dataclass(frozen=True)
class PredictiveCheck:
    """The predictive check of the mappings of a problem, see the module.

    Attributes:
        quantiles: the quantiles of the bands, ascending.
        n_subjects: the subjects of every study of a parameter vector.
        n_vectors: the parameter vectors.
        n_studies: the studies of every parameter vector, 1 without a
            variability between the studies.
        groups: the virtual groups of a data point of a group per study.
        seed: the seed of the subjects and the groups.
        targets: the targets of the variability.
        failed: the failed subjects over all parameter vectors.
        points: one row per data point of the checked mappings: `index` of the
            mapping, `experiment`, `mapping`, `kind`, `x`, `DV`, `N` (the
            subjects of the data point, 1 without a count), `statistic`
            (`individual`, `mean` or `gmean`), `PRED` (the prediction of the
            typical subject at the first parameter vector), `n` (the finite
            values of its distribution), a column per quantile (`q<percent>`,
            e.g. `q5`, `q50`, `q95`) and `inside` (the observation between
            the outer quantiles, `None` without a band).
        subjects: the predictions of the subjects of every study of every
            parameter vector, the studies of a vector one after the other.
    """

    quantiles: tuple[float, ...]
    n_subjects: int
    n_vectors: int
    n_studies: int
    groups: int
    seed: int
    targets: tuple[str, ...]
    failed: int
    points: pd.DataFrame
    subjects: tuple[SubjectPredictions, ...]

    @property
    def coverage(self) -> float | None:
        """Get the share of the data points inside their band, `None` without one."""
        inside = self.points["inside"].dropna()
        return float(inside.astype(bool).mean()) if len(inside) else None

    def coverage_by_mapping(self) -> pd.DataFrame:
        """Get the coverage of every mapping.

        Returns:
            One row per mapping with the columns `experiment`, `mapping`,
            `n` (the data points with a band) and `coverage`.
        """
        banded = self.points.dropna(subset=["inside"])
        inside = banded["inside"].astype(bool)
        summary = (
            banded.assign(inside=inside)
            .groupby(["experiment", "mapping"], sort=False)["inside"]
            .agg(n="size", coverage="mean")
            .reset_index()
        )
        return summary.astype({"n": int, "coverage": float})


def quantile_column(q: float) -> str:
    """Get the column of a quantile, e.g. `q5` of 0.05 and `q2.5` of 0.025."""
    return f"q{100 * q:g}"


def predictive_check(
    problem: OptimizationProblem,
    parameters: Sequence[np.ndarray] | np.ndarray,
    variability: Variability,
    n_subjects: int,
    *,
    indices: Sequence[int] | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    groups: int = 1000,
    seed: int | None = None,
    design: Literal["sobol", "lhs", "random"] = "sobol",
    studies: int = 16,
    workers: int = 1,
    settings: FitSettings | None = None,
    pool: EvaluationPool | None = None,
    key: str | None = None,
) -> PredictiveCheck:
    """Get the predictive check of the mappings of a problem, see the module.

    Args:
        problem: the initialized problem.
        parameters: the parameter vectors in the units of the model, one row
            per vector, e.g. draws of the uncertainty of the fitted
            parameters; one vector for the variability alone.
        variability: the variability of the subjects.
        n_subjects: the subjects of every parameter vector, of every study
            with a variability between the studies.
        indices: the indices of the mappings, every mapping by default.
        quantiles: the quantiles of the bands, at least two, in `(0, 1)`.
        groups: the virtual groups of a data point of a group per study.
        seed: the seed, `None` draws one, which the result keeps.
        design: the design of the factors, see `Variability.draw`; the
            subjects of the vectors (and of their studies) are consecutive
            points of one sequence.
        studies: the studies of every parameter vector with a variability
            between the studies (`Variability.between`), one without.
        workers: the processes the subjects of the vectors and studies are
            evaluated in, see `sbmlsim.fit.pool.map_evaluations`; a vector
            and a study is a sequence of its own, so the predictions are the
            ones of one process.
        settings: the settings the problem is initialized with in a worker,
            the settings of the problem by default.
        pool: a started pool which holds the problem as `key`, whose workers
            evaluate the subjects instead of a pool of this call, see
            `sbmlsim.fit.pool.EvaluationPool`.
        key: the key of the problem in `pool`.

    Returns:
        The bands and the coverage of the data points.

    Raises:
        ValueError: for no parameter vector, quantiles which are not in
            `(0, 1)` or fewer than two, a number of groups or of studies
            which is not positive, see `subject_predictions`.
    """
    vectors = np.atleast_2d(np.asarray(parameters, dtype=float))
    if vectors.shape[0] == 0:
        raise ValueError("A predictive check needs at least one parameter vector.")
    qs = tuple(sorted(float(q) for q in quantiles))
    if len(qs) < 2 or not all(0.0 < q < 1.0 for q in qs):
        raise ValueError(
            f"The quantiles of a band are at least two in (0, 1), not {list(quantiles)}."
        )
    if groups < 1:
        raise ValueError(
            f"The number of virtual groups must be positive, not {groups}."
        )
    if studies < 1:
        raise ValueError(f"The number of studies must be positive, not {studies}.")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    between = variability.between
    n_studies = 1 if between is None else studies
    # the factors of the studies, a sequence of their own
    study_factors = (
        np.ones((len(vectors) * n_studies, 0))
        if between is None
        else between.draw(len(vectors) * n_studies, seed=seed + 1, design=design)
    )
    chosen = list(problem.indices() if indices is None else indices)
    units = [
        (
            vector,
            (o * n_studies + s) * n_subjects,
            None
            if between is None
            else {
                target: float(study_factors[o * n_studies + s, j])
                for j, target in enumerate(between.targets)
            },
        )
        for o, vector in enumerate(vectors)
        for s in range(n_studies)
    ]
    shared = (variability, n_subjects, chosen, seed, design, units)
    n_workers = workers if pool is None else pool.workers
    tasks = _pieces(chunks(len(units) * n_subjects, n_workers), n_subjects)
    parts: dict[int, list[SubjectPredictions]] = {}
    for chunk in map_evaluations(
        problem,
        settings if settings is not None else problem.settings,
        _subjects_of,
        tasks,
        workers,
        shared,
        pool=pool,
        key=key,
    ):
        for u, part in chunk:
            parts.setdefault(u, []).append(part)
    subjects = tuple(_concatenated(parts[u]) for u in range(len(units)))
    typical = problem.predictions(vectors[0], chosen)
    rng = np.random.default_rng(seed)
    columns = [quantile_column(q) for q in qs]
    rows: list[dict[str, object]] = []
    for k in chosen:
        dv = np.asarray(problem.y_references[k], dtype=float)
        counts = problem.y_counts[k]
        sizes = (
            np.ones(dv.size, dtype=int)
            if counts is None
            else np.where(
                np.isfinite(np.asarray(counts, dtype=float))
                & (np.asarray(counts, dtype=float) >= 1),
                np.asarray(counts, dtype=float),
                1,
            ).astype(int)
        )
        group_statistic: Statistic = (
            "gmean" if problem.y_errors_log[k] is not None else "mean"
        )
        bands = np.full((dv.size, len(qs)), np.nan)
        n_values = np.zeros(dv.size, dtype=int)
        for size in np.unique(sizes):
            points = np.flatnonzero(sizes == size)
            pooled: list[np.ndarray] = []
            for subject in subjects:
                values = subject.predictions[k][:, points]
                values = values[np.all(np.isfinite(values), axis=1)]
                if len(values):
                    pooled.append(
                        _group_means(values, int(size), groups, group_statistic, rng)
                    )
            if not pooled:
                continue
            distribution = np.concatenate(pooled, axis=0)
            for j, point in enumerate(points):
                column = distribution[:, j]
                column = column[np.isfinite(column)]
                n_values[point] = column.size
                if column.size:
                    bands[point] = np.quantile(column, qs)
        x = problem.x_references[k]
        x_values = np.full(dv.size, np.nan) if x is None else np.asarray(x, dtype=float)
        for j in range(dv.size):
            banded = bool(np.all(np.isfinite(bands[j])))
            rows.append(
                {
                    "index": k,
                    "experiment": problem.experiment_keys[k],
                    "mapping": problem.mapping_keys[k],
                    "kind": str(problem.mapping_kinds[k].value),
                    "x": float(x_values[j]) if x_values.size > j else np.nan,
                    "DV": float(dv[j]),
                    "N": int(sizes[j]),
                    "statistic": "individual" if sizes[j] == 1 else group_statistic,
                    "PRED": float(typical[k][j]),
                    "n": int(n_values[j]),
                    **{c: float(bands[j, i]) for i, c in enumerate(columns)},
                    "inside": (
                        bool(bands[j, 0] <= dv[j] <= bands[j, -1]) if banded else None
                    ),
                }
            )
    points = pd.DataFrame(
        rows,
        columns=[
            "index",
            "experiment",
            "mapping",
            "kind",
            "x",
            "DV",
            "N",
            "statistic",
            "PRED",
            "n",
            *columns,
            "inside",
        ],
    )
    return PredictiveCheck(
        quantiles=qs,
        n_subjects=n_subjects,
        n_vectors=len(vectors),
        n_studies=n_studies,
        groups=groups,
        seed=seed,
        targets=variability.targets,
        failed=sum(s.failed for s in subjects),
        points=points,
        subjects=subjects,
    )
