"""A PEtab problem read, simulated and evaluated in sbmlsim-core.

The values of the core are the ones of the python reader at tight tolerances
of the integrator, where the two integrators agree; without the reader of
PEtab in the installed sbmlsim-core the runners fall back to python and the
functions of a file raise.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit.options import DataAtChangeType, FitSettings, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2 import (
    chi2,
    core,
    log_likelihood,
    log_prior,
    unnorm_log_posterior,
)
from sbmlsim.fit.petab_v2.core import (
    CORE_PETAB_RELEASE,
    CorePetabUnavailable,
    Evaluator,
    resolve_evaluator,
)
from sbmlsim.fit.petab_v2.reader import PetabReader
from tests.engines import require_core_petab
from tests.fit.test_petab_v2_reader import write_problem

#: the integrator of a comparison of the core with python, where both agree
TIGHT = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-10,
    data_at_change=DataAtChangeType.AFTER,
)

#: the relative difference of a value of the core and of python at `TIGHT`
RTOL = 1e-8


def _problem(directory: Path) -> Path:
    """Write a problem of two experiments, one of them of a condition."""
    return write_problem(
        directory,
        observables={"prey_o": "prey", "predator_o": "predator"},
        experiments={"e1": "cond1", "e2": None},
        conditions=[("cond1", "delta", "1.8")],
    )


def _python(path: Path, values: dict[str, float] | None = None) -> dict[str, float]:
    """Get the values of a problem read with `PetabReader`, at `TIGHT`."""
    reader = PetabReader.from_yaml(path)
    problem = reader.to_optimization_problem()
    problem.initialize(TIGHT)
    pset = reader.nominal_parameters(problem)
    if values is not None:
        pset = ParameterSet(
            pset.sid, {**pset.values, **values}, pset.units, provenance="test"
        )
    evaluations = problem.evaluations(pset.x(problem.pids), problem.indices())
    return {
        "llh": log_likelihood(problem, pset, evaluations),
        "chi2": chi2(problem, pset, evaluations),
        "log_prior": float(sum(log_prior(problem, pset).values())),
        "unnorm_log_posterior": unnorm_log_posterior(problem, pset, evaluations),
    }


def test_without_the_reader_of_the_core_python_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older sbmlsim-core runs the suites in python and refuses a file."""
    monkeypatch.setattr(core, "_core_petab", lambda: None)
    assert not core.has_core_petab()
    assert resolve_evaluator() is Evaluator.PYTHON
    assert resolve_evaluator("python") is Evaluator.PYTHON
    with pytest.raises(CorePetabUnavailable, match=CORE_PETAB_RELEASE):
        resolve_evaluator("core")
    path = _problem(tmp_path)
    with pytest.raises(CorePetabUnavailable, match="from_petab"):
        log_likelihood(path)
    with pytest.raises(CorePetabUnavailable):
        core.evaluate(path)


def test_an_evaluator_is_core_or_python() -> None:
    with pytest.raises(ValueError, match="amici"):
        resolve_evaluator("amici")


def test_the_core_evaluates_a_problem_as_python(tmp_path: Path) -> None:
    """llh, chi2, the priors and the posterior of the core are the ones of python."""
    require_core_petab()
    path = _problem(tmp_path)
    expected = _python(path)

    evaluation = core.evaluate(path, settings=TIGHT)

    assert evaluation.failures == []
    assert evaluation.parameters == {"alpha": 1.3, "beta": 0.9}
    for name, value in expected.items():
        assert getattr(evaluation, name) == pytest.approx(value, rel=RTOL), name
    assert evaluation.log_priors == pytest.approx(
        {"alpha": -np.log(15.0), "beta": -np.log(15.0)}
    )
    measurements = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    table = evaluation.simulations
    assert list(table.columns) == ["observableId", "experimentId", "time", "simulation"]
    assert table["observableId"].tolist() == measurements["observableId"].tolist()
    assert table["experimentId"].tolist() == measurements["experimentId"].tolist()
    assert evaluation.sigma == pytest.approx(np.full(len(measurements), 0.05))


def test_the_functions_of_a_file_evaluate_it_in_the_core(tmp_path: Path) -> None:
    """`log_likelihood` and its siblings take the YAML file of a problem."""
    require_core_petab()
    path = _problem(tmp_path)
    evaluation = core.evaluate(path)

    assert log_likelihood(path) == evaluation.llh
    assert log_likelihood(str(path)) == evaluation.llh
    assert chi2(path) == evaluation.chi2
    assert log_prior(path) == evaluation.log_priors
    assert unnorm_log_posterior(path) == evaluation.unnorm_log_posterior


def test_the_core_evaluates_at_the_parameters(tmp_path: Path) -> None:
    """A set gives the estimated parameters, the others keep their nominal value."""
    require_core_petab()
    path = _problem(tmp_path)
    expected = _python(path, {"alpha": 1.5})
    pset = ParameterSet("x", {"alpha": 1.5}, {"alpha": "dimensionless"})

    evaluation = core.evaluate(path, pset, settings=TIGHT)

    assert evaluation.parameters == {"alpha": 1.5, "beta": 0.9}
    assert evaluation.llh == pytest.approx(expected["llh"], rel=RTOL)
    assert evaluation.llh != pytest.approx(core.evaluate(path, settings=TIGHT).llh)
    assert core.evaluate(path, {"alpha": 1.5}).llh == log_likelihood(path, pset)


def test_a_problem_is_read_once_for_many_points(tmp_path: Path) -> None:
    require_core_petab()
    path = _problem(tmp_path)
    problem = core.read_problem(path, TIGHT)

    first = core.evaluate(problem, {"alpha": 1.4})

    assert first.llh == core.evaluate(path, {"alpha": 1.4}, settings=TIGHT).llh
    with pytest.raises(ValueError, match="read_problem"):
        core.evaluate(problem, settings=TIGHT)


def test_a_parameter_the_problem_does_not_estimate_is_refused(tmp_path: Path) -> None:
    require_core_petab()
    with pytest.raises(KeyError, match="gamma"):
        core.evaluate(_problem(tmp_path), {"gamma": 1.0})


def test_the_evaluations_of_a_problem_are_none_of_a_file(tmp_path: Path) -> None:
    """A file is simulated by the core, evaluations of python do not enter."""
    path = _problem(tmp_path)
    reader = PetabReader.from_yaml(path)
    problem = reader.to_optimization_problem()
    problem.initialize(TIGHT)
    evaluations = problem.evaluations(problem.xmodel, problem.indices())
    with pytest.raises(ValueError, match="evaluations"):
        log_likelihood(path, evaluations=evaluations)
