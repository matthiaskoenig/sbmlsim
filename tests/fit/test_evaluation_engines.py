"""The evaluation of a fit problem with the engines COPASI and AMICI (#219).

A problem evaluates its residuals, metrics and reports with the engine of
its simulator: a problem of an engine loads its models in the engine at
`initialize`, and `set_simulator` with a simulator of another engine loads
them again in its engine. Optimizing with another engine is not a goal, the
evaluation of stored parameters is, e.g. the prediction of an adopted fit
recomputed with a second engine. Every test runs with every engine besides
roadrunner, see `tests.engines`.

The agreement with roadrunner of the 186 weighted residuals of the training
data of `op_hctz_pk` (at most 4.8 in magnitude, 21 of them 0 in every
engine), measured at `absolute_tolerance=1e-12` and
`relative_tolerance=1e-10`: COPASI 4.3e-9 absolute and 1.4e-7 relative,
AMICI 4.2e-9 and 1.5e-7, 1.3 % and 1.1 % of what `rtol=1e-5` and `atol=1e-8`
allow, `atol + rtol * |r|`. At the settings of the fit, `1e-6` each, the
relative difference is up to 3.8e-3 (COPASI) and 3.4e-3 (AMICI), the error
of the integrators.
"""

import dataclasses
import importlib.metadata
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import FitMetrics, FitSettings, cli, methods
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.provenance import collect_versions
from sbmlsim.fit.report import FitReport
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.report.document import Document, Paragraph, Section, p
from sbmlsim.report.render import HTMLRenderer
from sbmlsim.simulator import Simulator
from tests.engines import (
    INTEGRATORS,
    PACKAGES,
    model_class,
    relative_tolerance,
    require,
)
from tests.fit.scalar_experiment import scalar_problem

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


def engine_problem(problem: OptimizationProblem, engine: Engine) -> OptimizationProblem:
    """Get the definition of a problem with an engine."""
    return OptimizationProblem(
        opid=problem.opid,
        mapping_collections=problem.mapping_collections,
        fit_parameters=problem.parameters,
        base_path=problem.base_path,
        data_path=problem.data_path,
        engine=engine,
    )


def _unique_models(problem: OptimizationProblem) -> list[Any]:
    return list({id(model): model for model in problem.models}.values())


def test_residuals_equal_roadrunner(
    other_engine: Engine, op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The residuals of the engine are the ones of roadrunner, see the module."""
    op_hctz_pk.initialize(settings_tight)
    assert op_hctz_pk.engine is Engine.ROADRUNNER
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    r = np.asarray(op_hctz_pk.residuals(x), dtype=float)

    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))
    assert op_hctz_pk.engine is other_engine
    assert op_hctz_pk.runner_initialized.engine is other_engine
    r_engine = np.asarray(op_hctz_pk.residuals(x), dtype=float)
    assert r_engine.shape == r.shape
    assert np.all(np.isfinite(r_engine))
    np.testing.assert_allclose(r_engine, r, rtol=1e-5, atol=1e-8)


def test_set_simulator_of_another_engine_loads_the_models_again(
    other_engine: Engine, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The models of the problem are the ones of the runner, in the engine.

    The models of the mappings keep their grouping: one model of roadrunner
    is one model of the engine.
    """
    op_hctz_pk.initialize(fit_settings)
    groups = [list(group) for group in op_hctz_pk.mapping_groups]
    identity = [
        [k for k, other in enumerate(op_hctz_pk.models) if other is model]
        for model in op_hctz_pk.models
    ]
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))

    cls = model_class(other_engine)
    assert all(isinstance(model, cls) for model in op_hctz_pk.models)
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
        assert relative_tolerance(model) == fit_settings.relative_tolerance


def test_a_problem_of_the_engine_loads_its_models_in_the_engine_only(
    other_engine: Engine,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`initialize` loads the models in the engine, not in roadrunner first."""
    problem = engine_problem(op_hctz_pk, other_engine)
    assert problem.engine is other_engine

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a model of roadrunner was loaded")

    monkeypatch.setattr(RoadrunnerSBMLModel, "__init__", refuse)
    problem.initialize(fit_settings)
    assert all(isinstance(m, model_class(other_engine)) for m in problem.models)
    runner = problem.runner_initialized
    assert runner.engine is other_engine
    simulator = runner.simulator
    assert simulator is not None
    assert simulator.engine is other_engine
    assert simulator.integrator_settings["relative_tolerance"] == (
        fit_settings.relative_tolerance
    )
    for model in _unique_models(problem):
        assert relative_tolerance(model) == fit_settings.relative_tolerance


def test_problem_pickles_with_its_engine(
    other_engine: Engine, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A worker of a parallel fit evaluates with the engine of the problem."""
    assert pickle.loads(pickle.dumps(op_hctz_pk)).engine is Engine.ROADRUNNER
    problem = engine_problem(op_hctz_pk, other_engine)
    assert pickle.loads(pickle.dumps(problem)).engine is other_engine

    # the engine a simulator set is the engine of the problem
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))
    copy = pickle.loads(pickle.dumps(op_hctz_pk))
    assert copy.engine is other_engine
    assert not copy.is_initialized
    copy.initialize(fit_settings)
    assert all(isinstance(m, model_class(other_engine)) for m in copy.models)


def test_metrics_ipred(
    other_engine: Engine, op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The DV/PRED/IPRED table of the engine is the one of roadrunner.

    Measured at the settings of `test_residuals_equal_roadrunner`: the
    largest relative difference of a prediction of the 255 data points is
    6.8e-9 (COPASI) and 6.4e-9 (AMICI), the largest absolute difference
    6.3e-11 and 6.4e-11 of at most 0.41; the largest difference is 0.026 %
    of what the tolerances allow.
    """
    op_hctz_pk.initialize(settings_tight)
    pset = op_hctz_pk.parameter_set_model()
    roadrunner = FitMetrics(op_hctz_pk, pset).datapoints_df()

    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))
    assert all(isinstance(m, model_class(other_engine)) for m in op_hctz_pk.models)
    engine = FitMetrics(op_hctz_pk, pset).datapoints_df()

    assert list(engine.columns) == list(roadrunner.columns)
    assert len(engine) == len(roadrunner)
    np.testing.assert_array_equal(engine["DV"], roadrunner["DV"])
    for column in ("PRED", "IPRED"):
        values = engine[column].to_numpy(dtype=float)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(
            values, roadrunner[column].to_numpy(dtype=float), rtol=1e-5, atol=1e-8
        )


def test_fit_report_of_stored_parameters(
    other_engine: Engine,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    tmp_path: Path,
) -> None:
    """A report of stored parameters evaluates them with the engine."""
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    # the report evaluates in a runtime of its own, in the engine of the problem
    assert report.problem.engine is other_engine
    cls = model_class(other_engine)
    assert all(isinstance(model, cls) for model in report.problem.models)
    index = report.create(tmp_path, name=other_engine.value) / "index.html"
    assert index.is_file()
    assert "hctz_pk" in index.read_text()


def test_the_residuals_of_a_pk_observable_equal_roadrunner(
    other_engine: Engine, tmp_path: Path
) -> None:
    """A group which keeps the output of its simulation evaluates with the engine.

    The cmax of `ScalarStudy` is the maximum over the steps of the
    integrator, evaluated at the start value of the volume, 30 l, three
    times the volume of the data, where the residual is -4.46. Measured at
    `absolute_tolerance=1e-12` and `relative_tolerance=1e-10`: the relative
    difference of the residual is 1.6e-10 (COPASI) and 1.4e-10 (AMICI).
    """
    settings = FitSettings(**SETTINGS_TIGHT)
    problem = scalar_problem(tmp_path, Engine.ROADRUNNER)
    problem.initialize(settings)
    assert all(graph is not None for graph in problem._group_graphs)
    x = problem.to_scale(np.asarray(problem.x0, dtype=float))
    r = np.asarray(problem.residuals(x), dtype=float)
    assert np.all(np.abs(r) > 1.0)

    engine = scalar_problem(tmp_path, other_engine)
    engine.initialize(settings)
    assert all(isinstance(m, model_class(other_engine)) for m in engine.models)
    r_engine = np.asarray(engine.residuals(x), dtype=float)
    assert np.all(np.isfinite(r_engine))
    np.testing.assert_allclose(r_engine, r, rtol=1e-5, atol=1e-8)


def test_an_evaluation_logs_no_warning(
    other_engine: Engine,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The settings of roadrunner only are not given to the models of the engine.

    `FitSettings` always has a variable step size and an initial time step,
    which COPASI and AMICI have no equivalent of; a model of the engine would
    report them as ignored on `set_simulator`, in the copy of a report and in
    every worker.
    """
    op_hctz_pk.initialize(fit_settings)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    with caplog.at_level(logging.WARNING):
        op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))
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


@pytest.mark.parametrize("engine", list(Engine), ids=str)
def test_the_engine_is_in_the_provenance(
    engine: Engine,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dictionary, the text, the console and the report name the engine.

    The versions of a fit name the package of its engine, the installed one.
    """
    if engine is not Engine.ROADRUNNER:
        require(engine)
    problem = engine_problem(op_hctz_pk, engine)
    assert problem.to_dict()["engine"] == engine.value
    problem.initialize(fit_settings)
    assert f"\nEngine: {engine.value}\n" in problem.report(print_output=False)

    report = FitReport(
        problem=problem,
        settings=fit_settings,
        parameter_sets=problem.parameter_set_model(),
        mapping_figures=False,
    )
    assert report.fit_info()["engine"] == engine.value
    html = (report.create(tmp_path, name=engine.value) / "index.html").read_text()
    assert f'<dt>engine</dt><dd class="mono">{engine.value}</dd>' in html
    # the methods of the report name the engine and its integrator
    assert f"its integrator {INTEGRATORS[engine]} " in html

    shown: list[dict[str, Any]] = []
    monkeypatch.setattr(cli.display, "key_values", shown.append)
    monkeypatch.setattr(cli.display, "print_parameters", lambda *a, **k: None)
    monkeypatch.setattr(cli.display, "print_settings", lambda *a: None)
    cli._display_records(report, "PK", [], tmp_path)
    assert shown[0]["engine"] == engine.value

    if engine is not Engine.ROADRUNNER:
        package = PACKAGES[engine]
        versions = collect_versions(engine)
        assert versions[package] == importlib.metadata.version(package)


def test_initialize_after_set_simulator_keeps_the_engine(
    other_engine: Engine, op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Other settings initialize the problem again, in its engine."""
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))
    other = dataclasses.replace(fit_settings, relative_tolerance=1e-8)
    op_hctz_pk.initialize(other)
    assert op_hctz_pk.engine is other_engine
    assert op_hctz_pk.runner_initialized.engine is other_engine
    assert all(isinstance(m, model_class(other_engine)) for m in op_hctz_pk.models)
    for model in _unique_models(op_hctz_pk):
        assert relative_tolerance(model) == 1e-8


def test_the_complete_data_equal_roadrunner(
    other_engine: Engine, op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The curves and the predictions of an evaluation agree.

    The curves of the 35 mappings are output at the same times by every
    engine, 47565 points. Measured at the settings of
    `test_residuals_equal_roadrunner`: the largest difference of a curve
    relative to its largest value is 7.0e-9 (COPASI) and 7.2e-9 (AMICI), the
    two curves which are 0 differ by at most 2e-16; the comparison allows
    `1e-6` of the largest value of a curve and `1e-14`. The predictions at
    the data differ by at most 6.8e-9 relative and the costs of the mappings
    by 5.5e-8 relative.
    """
    op_hctz_pk.initialize(settings_tight)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    roadrunner = op_hctz_pk.residuals(x, complete_data=True)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine=other_engine))
    engine = op_hctz_pk.residuals(x, complete_data=True)
    assert isinstance(roadrunner, dict) and isinstance(engine, dict)
    assert sorted(engine) == sorted(roadrunner)
    curves = zip(
        roadrunner["x_obs"],
        roadrunner["y_obs"],
        engine["x_obs"],
        engine["y_obs"],
        strict=True,
    )
    for x_rr, y_rr, x_e, y_e in curves:
        np.testing.assert_array_equal(np.asarray(x_e), np.asarray(x_rr))
        y_rr, y_e = np.asarray(y_rr, float), np.asarray(y_e, float)
        assert np.all(np.isfinite(y_e))
        largest = float(np.max(np.abs(y_rr)))
        np.testing.assert_allclose(y_e, y_rr, rtol=0, atol=1e-6 * largest + 1e-14)
    for key in ("y_obsip", "res_abs"):
        for a, b in zip(engine[key], roadrunner[key], strict=True):
            np.testing.assert_allclose(
                np.asarray(a, float), np.asarray(b, float), rtol=1e-5, atol=1e-12
            )
    np.testing.assert_allclose(engine["cost"], roadrunner["cost"], rtol=1e-5)


def _html(paragraph: Paragraph) -> str:
    """Render a paragraph of the methods as HTML."""
    section = Section("Methods", (paragraph,))
    return HTMLRenderer().render(Document("t", (section,)))


def _rendered(value: float) -> str:
    """Get a number as the methods render it within a paragraph."""
    html = _html(p("<<", methods._number(value), ">>"))
    return html[html.index("&lt;&lt;") + 8 : html.index("&gt;&gt;")]


def test_the_methods_name_the_engine(
    other_engine: Engine, fit_settings: FitSettings
) -> None:
    """The engine integrates with its integrator, without the settings of CVODE."""
    roadrunner = _html(methods._integrator(fit_settings))
    assert roadrunner == _html(methods._integrator(fit_settings, Engine.ROADRUNNER))
    assert "libRoadRunner" in roadrunner and "its integrator CVODE " in roadrunner

    name = {Engine.COPASI: "COPASI", Engine.AMICI: "AMICI"}[other_engine]
    if other_engine is Engine.COPASI:
        from sbmlsim.model.model_copasi import COPASI_ATOL_FACTOR as factor
    else:
        from sbmlsim.model.model_amici import AMICI_ATOL_FACTOR as factor
    text = _html(methods._integrator(fit_settings, other_engine))
    assert name in text and f"its integrator {INTEGRATORS[other_engine]} " in text
    assert "libRoadRunner" not in text and "variable step size" not in text
    others = set(INTEGRATORS.values()) - {INTEGRATORS[other_engine]}
    assert not any(f"integrator {integrator} " in text for integrator in others)
    # the tolerance the engine integrates with, the one of the fit times the
    # factor of the engine
    smallest = AbsoluteTolerance.of(fit_settings.absolute_tolerance).smallest()
    assert smallest == 1e-6
    assert _rendered(smallest * factor) in text
    assert f"the smallest absolute tolerance of the fit times {factor:g}" in text
