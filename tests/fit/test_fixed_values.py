"""The values a fit problem keeps fixed: simulated whatever the model holds (#663).

A fit of one subset of a project keeps the adopted fits of the other subsets
fixed. A factory writes their values into the models of the project, but a
fit may run before it did, right after an adoption: the problem simulates the
fixed values as pre-initialization changes, so its cost is the one of the
rebuilt model, on the path of python and in the request of sbmlsim-core.

The model of the tests is a copy of the HCTZ model whose partition coefficient
`Kp_hctz` differs from the packaged one (the "stale" model), the problem the
iv data of Beermann1976 of the HCTZ example.
"""

import copy
import dataclasses
import logging
import pickle
from pathlib import Path

import libsbml
import numpy as np
import pytest

import examples.hctz_fitting.experiments.base_experiment as base_experiment
from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.engine import Engine
from sbmlsim.fit import FitParameter, FitSettings, project_fits
from sbmlsim.fit.cli import FitDefinition, load_fit_records, problem_of_records, run_fit
from sbmlsim.fit.fit_core import engine_problem, fit_request
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.project_fits import fixed_values
from sbmlsim.project import Project
from sbmlsim.resources import HCTZ_SBML
from sbmlsim.simulator import Simulator
from tests.engines import require_core_fit
from tests.fit.test_project_fits import OTHER, adopted_other, toy_fit  # noqa: F401
from tests.project_toy import PROJECT

#: the partition coefficient of the stale model, the packaged one is about 1
STALE_KP = 2.0


def _value(path: Path, pid: str) -> float:
    """Get the value of a parameter of a model."""
    model = libsbml.readSBMLFromFile(str(path)).getModel()
    return float(model.getParameter(pid).getValue())


#: the value of the packaged model, which the "adopted fit" gives
KP = _value(HCTZ_SBML, "Kp_hctz")


@pytest.fixture
def stale_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Let the experiments of the HCTZ example simulate a stale copy of its model."""
    doc = libsbml.readSBMLFromFile(str(HCTZ_SBML))
    doc.getModel().getParameter("Kp_hctz").setValue(STALE_KP)
    path = tmp_path / "hctz_stale.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    monkeypatch.setattr(base_experiment, "MODEL_PATH", path)
    return path


def _problem(
    opid: str, settings: FitSettings, fixed: dict[str, float] | None = None
) -> OptimizationProblem:
    """Get the initialized problem of the iv data."""
    problem = FIT_DEFINITIONS["IV"].problem(opid=opid, fixed_values=fixed)
    problem.initialize(settings)
    return problem


def _residuals(problem: OptimizationProblem, factor: float = 1.0) -> np.ndarray:
    """Get the residuals of a problem around the values of its models."""
    x = problem.to_scale(problem.xmodel * factor)
    return np.asarray(problem.residuals(x), dtype=float)


def _rebuilt(
    settings: FitSettings, monkeypatch: pytest.MonkeyPatch
) -> OptimizationProblem:
    """Get the problem of the iv data on the packaged model, the rebuilt one."""
    stale = base_experiment.MODEL_PATH
    monkeypatch.setattr(base_experiment, "MODEL_PATH", HCTZ_SBML)
    problem = FIT_DEFINITIONS["IV"].problem(opid="rebuilt")
    problem.initialize(settings)
    monkeypatch.setattr(base_experiment, "MODEL_PATH", stale)
    return problem


def test_a_fit_simulates_its_fixed_values_whatever_the_model_holds(
    fit_settings: FitSettings, stale_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduction of #663: a fit after an adoption, before the factory."""
    rebuilt = _rebuilt(fit_settings, monkeypatch)
    definition = dataclasses.replace(FIT_DEFINITIONS["IV"], settings=fit_settings)
    runs = {
        name: run_fit(
            definition,
            opid=name,
            size=1,
            seed=1,
            fixed_values=fixed,
            engine_fit=False,
            max_nfev=1,
        )[name]
        for name, fixed in (("stale", None), ("held", {"Kp_hctz": KP}))
    }
    stale, held = runs["stale"].problem, runs["held"].problem
    assert stale.models[0].source.path == stale_model

    for factor in (1.0, 1.3):
        expected = _residuals(rebuilt, factor)
        # the stale model is another model
        assert not np.allclose(_residuals(stale, factor), expected)
        # the fixed value makes it the rebuilt one
        np.testing.assert_allclose(
            _residuals(held, factor), expected, rtol=1e-9, atol=1e-12
        )
    assert runs["held"].result.provenance is not None
    assert runs["held"].result.provenance.fixed_values == {"Kp_hctz": KP}


def test_the_request_of_the_engine_carries_the_fixed_values(
    fit_settings: FitSettings, stale_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fit of sbmlsim-core simulates the fixed values, not the model's."""
    require_core_fit()
    rebuilt = _rebuilt(fit_settings, monkeypatch)
    held = _problem("held", fit_settings, fixed={"Kp_hctz": KP})
    held.set_simulator(Simulator(n_workers=1, engine=Engine.CORE))
    assert all(s.preinit_changes["Kp_hctz"] == KP for s in held.simulations)
    core = engine_problem(held, fit_request(held), threads=1)
    for factor in (1.0, 1.3):
        x = held.to_scale(held.xmodel * factor)
        np.testing.assert_allclose(
            np.asarray(core.residuals(x), dtype=float),
            _residuals(rebuilt, factor),
            rtol=1e-9,
            atol=1e-12,
        )


def test_a_value_of_a_parameter_of_the_fit_is_not_fixed(
    fit_settings: FitSettings,
) -> None:
    definition = dataclasses.replace(FIT_DEFINITIONS["IV"], settings=fit_settings)
    run_ = run_fit(
        definition,
        opid="IV_held",
        size=1,
        seed=1,
        fixed_values={"Kp_hctz": KP, "Ka_dis_hctz": 1.0},
        max_nfev=1,
    )["IV_held"]
    # the fit sets its parameters, the record holds what was fixed
    assert run_.problem.fixed_values == {"Kp_hctz": KP}
    assert run_.result.provenance is not None
    assert run_.result.provenance.fixed_values == {"Kp_hctz": KP}


def test_the_fixed_values_of_a_problem(fit_settings: FitSettings) -> None:
    problem = FIT_DEFINITIONS["IV"].problem(
        opid="iv",
        fixed_values={"Kp_hctz": 2.0, "KI__HCTZEX_k": 0.1, "no_entity": 1.0},
    )
    # the target of a parameter of the problem is set by the parameter
    assert problem.fixed_values == {"Kp_hctz": 2.0, "no_entity": 1.0}
    assert problem.to_dict()["fixed_values"] == problem.fixed_values
    # a worker of a pool and a report get them with the problem
    assert pickle.loads(pickle.dumps(problem)).fixed_values == problem.fixed_values
    assert copy.deepcopy(problem).fixed_values == problem.fixed_values

    problem.initialize(fit_settings)
    model = problem.models[0]
    # a value of an entity the model does not have is not simulated
    assert problem.fixed_changes(model) == {"Kp_hctz": 2.0}
    assert FIT_DEFINITIONS["IV"].problem(opid="iv").fixed_values == {}


def test_a_change_of_the_model_wins_over_a_fixed_value(
    fit_settings: FitSettings,
) -> None:
    problem = FIT_DEFINITIONS["IV"].problem(opid="iv", fixed_values={"Kp_hctz": 2.0})
    problem.initialize(fit_settings)
    model = problem.models[0]
    model.changes["Kp_hctz"] = 3.0
    problem._simulation_cache.clear()
    experiment = next(iter(problem.runner_initialized.experiments.values()))
    task = next(iter(experiment._tasks.values()))
    simulation, _ = problem._simulation_of(
        "x", task.model_id, task.simulation_id, experiment, model
    )
    assert simulation.preinit_changes["Kp_hctz"] == 3.0


# ----------------------------------------------------------------------------
# the fixed values of a project and of a record
# ----------------------------------------------------------------------------
def test_a_record_reproduces_the_problem_with_its_fixed_values(
    toy_fit: Path,  # noqa: F811
) -> None:
    from sbmlsim.fit.cli import recorded_fixed_values

    records = load_fit_records(toy_fit)
    assert recorded_fixed_values(records) == {}
    assert records.opt_result is not None
    opt_result = copy.deepcopy(records.opt_result)
    assert opt_result.provenance is not None
    opt_result.provenance = dataclasses.replace(
        opt_result.provenance, fixed_values={"Vc": 2.0}
    )
    recorded = dataclasses.replace(records, opt_result=opt_result)
    assert recorded_fixed_values(recorded) == {"Vc": 2.0}

    definition = PROJECT.fit("DECAY")
    # the problem of the record keeps the values the fit kept fixed
    problem = problem_of_records(definition, recorded, opid="DECAY")
    assert problem.fixed_values == {"Vc": 2.0}
    # an assessment of the model gives the values it keeps fixed now
    problem = problem_of_records(definition, recorded, opid="DECAY", fixed_values={})
    assert problem.fixed_values == {}


def _without_parameters(project: Project) -> FitDefinition:
    """Get the fit DECAY of a project as the fit of a parameter other than `k`."""
    return dataclasses.replace(
        project.fit("DECAY"),
        parameters=[
            FitParameter("Vc", start_value=1, lower_bound=0.1, upper_bound=10, unit="l")
        ],
    )


def test_a_model_which_holds_other_values_than_the_adopted_fits_is_named(
    toy_fit: Path,  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sbmlsim.fit.project_fits import stale_values

    project = adopted_other(toy_fit, tmp_path)
    k = project.fit_records().records[OTHER].parameters["k"].value
    definition = _without_parameters(project)
    # the variant `slow` holds the rate 0.1, the default variant 0.2
    stale = stale_values(project, {"k": k})
    assert [line.split(" in ")[1].split(",")[0] for line in stale] == [
        "toy.xml",
        "toy_slow.xml",
    ]
    assert stale_values(project, {"k": 0.2}) == ["'k' is 0.1 in toy_slow.xml, not 0.2"]
    assert stale_values(project, {"no_parameter": 1.0}) == []

    project_fits._warned.clear()
    with caplog.at_level(logging.WARNING, logger=project_fits.__name__):
        assert fixed_values(project, "DECAY", definition) == ([OTHER], {"k": k})
        fixed_values(project, "DECAY", definition)
    warnings = [r.getMessage() for r in caplog.records]
    # once, though the plan of a fit reads them more than once
    assert len(warnings) == 1
    assert OTHER in warnings[0]
    assert "The factory has not rebuilt the models" in warnings[0]


def test_a_rebuilt_model_is_not_named(
    toy_fit: Path,  # noqa: F811
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sbmlsim.fit.project_fits import stale_values

    project = adopted_other(toy_fit, tmp_path)
    k = project.fit_records().records[OTHER].parameters["k"].value
    for path in project.variants.values():
        doc = libsbml.readSBMLFromFile(str(path))
        doc.getModel().getParameter("k").setValue(k)
        local = tmp_path / path.name
        libsbml.writeSBMLToFile(doc, str(local))
    rebuilt = project.replace(
        variants={name: tmp_path / p.name for name, p in project.variants.items()}
    )
    assert stale_values(rebuilt, {"k": k}) == []
    project_fits._warned.clear()
    with caplog.at_level(logging.WARNING, logger=project_fits.__name__):
        fixed_values(rebuilt, "DECAY", _without_parameters(rebuilt))
    assert caplog.records == []
