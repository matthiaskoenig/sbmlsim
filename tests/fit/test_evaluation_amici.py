"""The evaluation of a fit problem with the engine AMICI.

A problem evaluates its residuals, metrics and reports with the engine of
its simulator, as with COPASI, see `tests.fit.test_evaluation_copasi`: a
problem of the engine AMICI loads its models in AMICI at `initialize`, and
`set_simulator` with a simulator of AMICI loads them again in AMICI. The
models of the HCTZ problem are one SBML, which AMICI compiles once into the
cache of the test session.
"""

import dataclasses
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("amici")

from sbmlsim.engine import Engine
from sbmlsim.fit import (
    FitMappingCollection,
    FitMetrics,
    FitParameter,
    FitSettings,
    cli,
    methods,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.report import FitReport
from sbmlsim.model.model_amici import AmiciSBMLModel
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.simulator import Simulator
from tests.fit.scalar_experiment import ScalarStudy

#: the integrator settings of a comparison of the engines, where both
#: integrate far below the differences the comparison allows
SETTINGS_TIGHT: dict[str, Any] = {
    "absolute_tolerance": 1e-12,
    "relative_tolerance": 1e-10,
}


@pytest.fixture
def settings_tight(fit_settings: FitSettings) -> FitSettings:
    """Get the settings of the fit with a tight integrator."""
    return dataclasses.replace(fit_settings, **SETTINGS_TIGHT)


def _amici_problem(problem: OptimizationProblem) -> OptimizationProblem:
    """Get the definition of a problem with the engine AMICI."""
    return OptimizationProblem(
        opid=problem.opid,
        mapping_collections=problem.mapping_collections,
        fit_parameters=problem.parameters,
        base_path=problem.base_path,
        data_path=problem.data_path,
        engine="amici",
    )


def _unique_models(problem: OptimizationProblem) -> list[Any]:
    return list({id(model): model for model in problem.models}.values())


def _relative_tolerance(model: AmiciSBMLModel) -> float:
    """Get the relative tolerance of CVODES of a model of AMICI."""
    return float(model.solver.get_relative_tolerance())


def test_residuals_with_amici_equal_roadrunner(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The residuals of AMICI are the ones of roadrunner.

    The 186 weighted residuals of the training data are at most 4.8 in
    magnitude, 21 of them 0 in both engines. Measured at
    `absolute_tolerance=1e-12` and `relative_tolerance=1e-10`: the largest
    absolute difference of a residual is 4.2e-9 and the largest relative
    difference 1.5e-7. The largest difference is 1.1 % of what `rtol=1e-5`
    and `atol=1e-8` allow, `atol + rtol * |r|`, a margin of a factor 90. At
    the settings of the fit, `1e-6` each, the relative difference is up to
    3.4e-3, the error of the integrators.
    """
    op_hctz_pk.initialize(settings_tight)
    assert op_hctz_pk.engine is Engine.ROADRUNNER
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    r = np.asarray(op_hctz_pk.residuals(x), dtype=float)

    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))
    assert op_hctz_pk.engine is Engine.AMICI
    assert op_hctz_pk.runner_initialized.engine is Engine.AMICI
    r_amici = np.asarray(op_hctz_pk.residuals(x), dtype=float)
    assert r_amici.shape == r.shape
    assert np.all(np.isfinite(r_amici))
    np.testing.assert_allclose(r_amici, r, rtol=1e-5, atol=1e-8)


def test_set_simulator_of_amici_loads_the_models_again(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The models of the problem are the ones of the runner, in AMICI.

    The models of the mappings keep their grouping: one model of roadrunner
    is one model of AMICI.
    """
    op_hctz_pk.initialize(fit_settings)
    groups = [list(group) for group in op_hctz_pk.mapping_groups]
    identity = [
        [k for k, other in enumerate(op_hctz_pk.models) if other is model]
        for model in op_hctz_pk.models
    ]
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))

    assert all(isinstance(model, AmiciSBMLModel) for model in op_hctz_pk.models)
    runner = op_hctz_pk.runner_initialized
    for k, model in enumerate(op_hctz_pk.models):
        experiment = runner.experiments[op_hctz_pk.experiment_keys[k]]
        assert experiment._models[op_hctz_pk.model_keys[k]] is model
    assert [
        [k for k, other in enumerate(op_hctz_pk.models) if other is model]
        for model in op_hctz_pk.models
    ] == identity
    assert [list(group) for group in op_hctz_pk.mapping_groups] == groups
    # the models integrate with the settings of the fit
    for model in _unique_models(op_hctz_pk):
        assert _relative_tolerance(model) == fit_settings.relative_tolerance


def test_a_problem_of_amici_loads_its_models_in_amici_only(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`initialize` loads the models in AMICI, not in roadrunner first."""
    problem = _amici_problem(op_hctz_pk)
    assert problem.engine is Engine.AMICI

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a model of roadrunner was loaded")

    monkeypatch.setattr(RoadrunnerSBMLModel, "__init__", refuse)
    problem.initialize(fit_settings)
    assert all(isinstance(model, AmiciSBMLModel) for model in problem.models)
    runner = problem.runner_initialized
    assert runner.engine is Engine.AMICI
    simulator = runner.simulator
    assert simulator is not None
    assert simulator.engine is Engine.AMICI
    assert simulator.integrator_settings["relative_tolerance"] == (
        fit_settings.relative_tolerance
    )
    for model in _unique_models(problem):
        assert _relative_tolerance(model) == fit_settings.relative_tolerance


def test_problem_pickles_with_its_engine(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A worker of a parallel fit evaluates with the engine of the problem."""
    problem = _amici_problem(op_hctz_pk)
    assert pickle.loads(pickle.dumps(problem)).engine is Engine.AMICI

    # the engine a simulator set is the engine of the problem
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))
    copy = pickle.loads(pickle.dumps(op_hctz_pk))
    assert copy.engine is Engine.AMICI
    assert not copy.is_initialized
    copy.initialize(fit_settings)
    assert all(isinstance(model, AmiciSBMLModel) for model in copy.models)


def test_metrics_ipred_with_amici(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The DV/PRED/IPRED table of AMICI is the one of roadrunner.

    Measured at the settings of `test_residuals_with_amici_equal_roadrunner`:
    the largest relative difference of a prediction of the 255 data points
    is 6.4e-9, the largest absolute difference 6.4e-11 of at most 0.41; the
    largest difference is 0.025 % of what the tolerances allow.
    """
    op_hctz_pk.initialize(settings_tight)
    pset = op_hctz_pk.parameter_set_model()
    roadrunner = FitMetrics(op_hctz_pk, pset).datapoints_df()

    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))
    assert all(isinstance(model, AmiciSBMLModel) for model in op_hctz_pk.models)
    amici = FitMetrics(op_hctz_pk, pset).datapoints_df()

    assert list(amici.columns) == list(roadrunner.columns)
    assert len(amici) == len(roadrunner)
    np.testing.assert_array_equal(amici["DV"], roadrunner["DV"])
    for column in ("PRED", "IPRED"):
        values = amici[column].to_numpy(dtype=float)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(
            values, roadrunner[column].to_numpy(dtype=float), rtol=1e-5, atol=1e-8
        )


def test_fit_report_of_stored_parameters_with_amici(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
) -> None:
    """A report of stored parameters evaluates them with AMICI."""
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    # the report evaluates in a runtime of its own, in the engine of the problem
    assert report.problem.engine is Engine.AMICI
    assert all(isinstance(model, AmiciSBMLModel) for model in report.problem.models)
    index = report.create(tmp_path, name="amici") / "index.html"
    assert index.is_file()
    assert "hctz_pk" in index.read_text()


def _scalar_problem(tmp_path: Path, engine: str = "roadrunner") -> OptimizationProblem:
    """Get the problem of the cmax of a PK observable, see `ScalarStudy`."""
    return OptimizationProblem(
        "scalar",
        [FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"])],
        [
            FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
        engine=engine,
    )


def test_the_residuals_of_a_pk_observable_with_amici_equal_roadrunner(
    tmp_path: Path,
) -> None:
    """A group which keeps the output of its simulation evaluates with AMICI.

    The cmax of `ScalarStudy` is the maximum over the steps of the
    integrator, of AMICI a grid with the events, see
    `sbmlsim.simulator.executor_amici`, evaluated at the start value of the
    volume, 30 l, where the residual is -4.46. Measured at
    `absolute_tolerance=1e-12` and `relative_tolerance=1e-10`: the relative
    difference of the residual is 1.4e-10.
    """
    settings = FitSettings(**SETTINGS_TIGHT)
    problem = _scalar_problem(tmp_path)
    problem.initialize(settings)
    assert all(graph is not None for graph in problem._group_graphs)
    x = problem.to_scale(np.asarray(problem.x0, dtype=float))
    r = np.asarray(problem.residuals(x), dtype=float)
    assert np.all(np.abs(r) > 1.0)

    amici = _scalar_problem(tmp_path, engine="amici")
    amici.initialize(settings)
    assert all(isinstance(model, AmiciSBMLModel) for model in amici.models)
    r_amici = np.asarray(amici.residuals(x), dtype=float)
    assert np.all(np.isfinite(r_amici))
    np.testing.assert_allclose(r_amici, r, rtol=1e-5, atol=1e-8)


def test_an_amici_evaluation_logs_no_warning(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The settings of roadrunner only are not given to the models of AMICI.

    `FitSettings` always has a variable step size and an initial time step,
    which AMICI has no equivalent of; a model of AMICI would report them as
    ignored on `set_simulator`, in the copy of a report and in every worker.
    """
    op_hctz_pk.initialize(fit_settings)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    with caplog.at_level(logging.WARNING):
        op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))
        op_hctz_pk.residuals(x)
        report = FitReport(
            problem=op_hctz_pk,
            settings=fit_settings,
            parameter_sets=op_hctz_pk.parameter_set_model(),
            mapping_figures=False,
        )
        report.metrics(report.parameter_sets[0]).datapoints_df()
    assert [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ] == []
    for problem in (op_hctz_pk, report.problem):
        simulator = problem.runner_initialized.simulator
        assert simulator is not None
        assert "variable_step_size" not in simulator.integrator_settings
        assert "initial_time_step" not in simulator.integrator_settings


def test_the_engine_is_in_the_provenance(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dictionary, the text, the console and the report name the engine."""
    problem = _amici_problem(op_hctz_pk)
    assert problem.to_dict()["engine"] == "amici"
    problem.initialize(fit_settings)
    assert "\nEngine: amici\n" in problem.report(print_output=False)

    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=problem.parameter_set_model(),
        mapping_figures=False,
    )
    assert report.fit_info()["engine"] == "amici"
    html = (report.create(tmp_path, name="amici") / "index.html").read_text()
    assert '<dt>engine</dt><dd class="mono">amici</dd>' in html
    # the methods of the report name the engine and its integrator
    assert "AMICI" in html and "its integrator CVODES" in html

    shown: list[dict[str, Any]] = []
    monkeypatch.setattr(cli.display, "key_values", shown.append)
    monkeypatch.setattr(cli.display, "print_parameters", lambda *a, **k: None)
    monkeypatch.setattr(cli.display, "print_settings", lambda *a: None)
    cli._display_records(report, "PK", [], tmp_path)
    assert shown[0]["engine"] == "amici"


def test_the_methods_name_the_engine(fit_settings: FitSettings) -> None:
    """AMICI integrates with CVODES, without the settings of roadrunner."""
    from sbmlsim.report.document import Document, Section
    from sbmlsim.report.render import HTMLRenderer

    def html(paragraph: Any) -> str:
        section = Section("Methods", (paragraph,))
        return HTMLRenderer().render(Document("t", (section,)))

    amici = html(methods._integrator(fit_settings, Engine.AMICI))
    assert "AMICI" in amici and "CVODES" in amici
    assert "libRoadRunner" not in amici and "LSODA" not in amici
    assert "variable step size" not in amici
    # the tolerance AMICI integrates with, the one of the fit times the factor
    assert AbsoluteTolerance.of(fit_settings.absolute_tolerance).amount == 1e-6
    assert "<mn>9</mn>" in amici
    assert "the smallest absolute tolerance of the fit times 0.001" in amici


def test_initialize_after_set_simulator_keeps_the_engine(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Other settings initialize the problem again, in its engine."""
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))
    other = dataclasses.replace(fit_settings, relative_tolerance=1e-8)
    op_hctz_pk.initialize(other)
    assert op_hctz_pk.engine is Engine.AMICI
    assert op_hctz_pk.runner_initialized.engine is Engine.AMICI
    assert all(isinstance(model, AmiciSBMLModel) for model in op_hctz_pk.models)
    for model in _unique_models(op_hctz_pk):
        assert _relative_tolerance(model) == 1e-8


def test_the_complete_data_with_amici_equal_roadrunner(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The curves and the predictions of an evaluation agree.

    The curves of the 35 mappings are output at the same times by both
    engines, 47565 points. Measured at the settings of
    `test_residuals_with_amici_equal_roadrunner`: the largest difference of
    a curve relative to its largest value is 7.2e-9, the two curves which
    are 0 are 0 in both engines; the comparison allows `1e-6` of the largest
    value of a curve and `1e-14`. The predictions at the data differ by at
    most 6.4e-9 relative and the costs of the mappings by 5.5e-8 relative.
    """
    op_hctz_pk.initialize(settings_tight)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    roadrunner = op_hctz_pk.residuals(x, complete_data=True)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="amici"))
    amici = op_hctz_pk.residuals(x, complete_data=True)
    assert isinstance(roadrunner, dict) and isinstance(amici, dict)
    assert sorted(amici) == sorted(roadrunner)
    curves = zip(
        roadrunner["x_obs"],
        roadrunner["y_obs"],
        amici["x_obs"],
        amici["y_obs"],
        strict=True,
    )
    for x_rr, y_rr, x_am, y_am in curves:
        np.testing.assert_array_equal(np.asarray(x_am), np.asarray(x_rr))
        y_rr, y_am = np.asarray(y_rr, float), np.asarray(y_am, float)
        assert np.all(np.isfinite(y_am))
        largest = float(np.max(np.abs(y_rr)))
        np.testing.assert_allclose(y_am, y_rr, rtol=0, atol=1e-6 * largest + 1e-14)
    for key in ("y_obsip", "res_abs"):
        for a, b in zip(amici[key], roadrunner[key], strict=True):
            np.testing.assert_allclose(
                np.asarray(a, float), np.asarray(b, float), rtol=1e-5, atol=1e-12
            )
    np.testing.assert_allclose(amici["cost"], roadrunner["cost"], rtol=1e-5)
