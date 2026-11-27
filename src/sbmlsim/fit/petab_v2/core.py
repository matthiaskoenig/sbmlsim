"""A PEtab v2 problem read, simulated and evaluated in sbmlsim-core.

sbmlsim-core reads a problem of PEtab v2 in Rust, `sbmlsim_core.petab.Problem`
(the crate `sbmlsim-petab`): the tables and their math are parsed there, every
experiment of a model is one simulation of the engine and the log-likelihood,
`chi2`, the log priors and the simulation of every measurement are the ones of
the specification. This module is the side of `sbmlsim`: a problem on disk is
evaluated by the core, `evaluate`, which `log_likelihood`, `chi2`, `log_prior`
and `unnorm_log_posterior` call for the path of a problem and which the runners
of the PEtab test suite and of the benchmark collection use.

What stays in python is what is a problem of `sbmlsim`: `from_petab` builds a
`SimulationExperiment` and an `OptimizationProblem` from the tables, whose fit
has the objective, the kinds of the data, the figures and the report of
`sbmlsim`, and the log-likelihood of an `OptimizationProblem`, which need not
be a problem of PEtab at all.

The core reads of the extension `sbmlsim` what changes the simulation of a
measurement: `data_at_change`, the tolerances of the integrator and the kind
of every observable, of which only the training data enter the likelihood
(and the validation data of the validation problem of a fit). Its
objective, the weighted least squares of a fit of `sbmlsim`, is not the
objective of PEtab and is not read.

The python reader of `from_petab` evaluates a problem of PEtab only in the
runner of the benchmark collection, for the problems the core does not run as
it does yet (`sbmlsim.fit.petab_v2.benchmark.CORE_GAPS`).
"""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sbmlsim.fit.options import FitSettings
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.model.tolerances import AbsoluteTolerance


def read_problem(
    yaml_file: str | os.PathLike[str],
    settings: FitSettings | None = None,
    *,
    extension: str = "read",
) -> Any:
    """Read a problem with sbmlsim-core.

    Args:
        yaml_file: the YAML file of the problem, with the files it names.
        settings: the settings whose tolerances of the integrator and
            `data_at_change` the problem is simulated with, the ones of its
            extension `sbmlsim` (PEtab's without it) by default. The smallest
            tolerance of the kinds of an `AbsoluteTolerance` is the absolute
            tolerance, the core has one for every state.
        extension: `read` reads the settings of the simulation of the
            extension `sbmlsim`, `ignore` reads the problem as a tool which
            does not know it.

    Returns:
        The `sbmlsim_core.petab.Problem`.

    Raises:
        ValueError: if the core cannot read the problem.
    """
    from sbmlsim_core import petab

    options: dict[str, Any] = {"extension": extension}
    if settings is not None:
        options["relative_tolerance"] = float(settings.relative_tolerance)
        options["absolute_tolerance"] = AbsoluteTolerance.of(
            settings.absolute_tolerance
        ).smallest()
        options["data_at_change"] = settings.data_at_change.value
    return petab.Problem.from_yaml(Path(yaml_file), **options)


@dataclass(frozen=True)
class PetabEvaluation:
    """A problem of PEtab evaluated at a point.

    Attributes:
        parameters: the values of the estimated parameters, by their ids.
        llh: the log-likelihood of the training data.
        chi2: the sum of the squared residuals of the training data in units
            of their noise.
        log_priors: the log prior of every estimated parameter, by its id.
        log_prior: the sum of the log priors.
        unnorm_log_posterior: the log-likelihood plus the log prior.
        simulations: the simulation table, a row per measurement in the order
            of the measurement table, `observableId`, `experimentId` (empty
            for a measurement without one), `time` and `simulation`, with
            `modelId` first for a problem of several models.
        sigma: the scale of the noise of every measurement, in that order.
        failures: the messages of the simulations which failed, whose
            measurements are `NaN`.
    """

    parameters: dict[str, float]
    llh: float
    chi2: float
    log_priors: dict[str, float]
    log_prior: float
    unnorm_log_posterior: float
    simulations: pd.DataFrame
    sigma: np.ndarray
    failures: list[str] = field(default_factory=list)


def _point(
    problem: Any, parameters: Mapping[str, float] | ParameterSet | None
) -> list[float]:
    """Get the point of the estimated parameters of a problem of the core.

    Raises:
        KeyError: if a parameter is no estimated parameter of the problem.
    """
    point = [float(value) for value in problem.nominal]
    if parameters is None:
        return point
    values = parameters.values if isinstance(parameters, ParameterSet) else parameters
    index = {pid: k for k, pid in enumerate(problem.parameters)}
    unknown = sorted(set(values) - set(index))
    if unknown:
        raise KeyError(
            f"The parameters {unknown} are no estimated parameters of the PEtab "
            f"problem, which are {list(problem.parameters)}."
        )
    for pid, value in values.items():
        point[index[pid]] = float(value)
    return point


def evaluate(
    problem: str | os.PathLike[str] | Any,
    parameters: Mapping[str, float] | ParameterSet | None = None,
    settings: FitSettings | None = None,
) -> PetabEvaluation:
    """Evaluate a problem of PEtab in sbmlsim-core.

    Args:
        problem: the YAML file of the problem, or a problem of the core of
            `read_problem`, which is read once for many points.
        parameters: values of estimated parameters by their ids, in the units
            of the parameter table, e.g. a `ParameterSet` of a fit of the
            problem; a parameter which is not given keeps its nominal value.
        settings: see `read_problem`, only for a file.

    Returns:
        The evaluation.

    Raises:
        KeyError: if a parameter is no estimated parameter of the problem.
        ValueError: if the core cannot read the problem, or if settings are
            given with a problem which is read already.
    """
    if isinstance(problem, (str, os.PathLike)):
        problem = read_problem(problem, settings)
    elif settings is not None:
        raise ValueError(
            "The settings of a problem are the ones it was read with, see "
            "`read_problem`."
        )
    point = _point(problem, parameters)
    evaluation = problem.evaluate(point)
    simulations = pd.read_csv(
        io.StringIO(evaluation["simulation_table"]),
        sep="\t",
        dtype={"observableId": str, "experimentId": str, "modelId": str},
        keep_default_na=False,
        na_values={"time": ["NaN", "nan"], "simulation": ["NaN", "nan"]},
    )
    return PetabEvaluation(
        parameters=dict(zip(problem.parameters, point, strict=True)),
        llh=float(evaluation["llh"]),
        chi2=float(evaluation["chi2"]),
        log_priors={pid: float(v) for pid, v in evaluation["log_priors"].items()},
        log_prior=float(evaluation["log_prior"]),
        unnorm_log_posterior=float(evaluation["unnorm_log_posterior"]),
        simulations=simulations,
        sigma=np.asarray(evaluation["sigma"], dtype=float),
        failures=[
            f"simulation {index}: {message}"
            for index, message in evaluation["failures"]
        ],
    )
