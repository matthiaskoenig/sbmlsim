"""The evaluation of a fit problem with the engine COPASI (#219).

A problem evaluates its residuals, metrics and reports with the engine of
its simulator: a problem of the engine COPASI loads its models in COPASI at
`initialize`, and `set_simulator` with a simulator of another engine loads
them again in its engine. Optimizing with COPASI is not a goal, the
evaluation of stored parameters is, e.g. the prediction of an adopted fit
recomputed with a second engine.
"""

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import FitMappingCollection, FitMetrics, FitParameter, FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.report import FitReport
from sbmlsim.model.model_copasi import CopasiSBMLModel
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
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


def _copasi_problem(problem: OptimizationProblem) -> OptimizationProblem:
    """Get the definition of a problem with the engine COPASI."""
    return OptimizationProblem(
        opid=problem.opid,
        mapping_collections=problem.mapping_collections,
        fit_parameters=problem.parameters,
        base_path=problem.base_path,
        data_path=problem.data_path,
        engine="copasi",
    )


def _unique_models(problem: OptimizationProblem) -> list[Any]:
    return list({id(model): model for model in problem.models}.values())


def _relative_tolerance(model: CopasiSBMLModel) -> float:
    """Get the relative tolerance of LSODA of a model of COPASI."""
    method = model.dm.getTask("Time-Course").getMethod()
    return float(method.getParameter("Relative Tolerance").getDblValue())


def test_residuals_with_copasi_equal_roadrunner(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The residuals of COPASI are the ones of roadrunner.

    The 186 weighted residuals of the training data are at most 4.8 in
    magnitude, 21 of them 0 in both engines. Measured at
    `absolute_tolerance=1e-12` and `relative_tolerance=1e-10`: the largest
    absolute difference of a residual is 4.3e-9 and the largest relative
    difference 1.4e-7, within `rtol=1e-5` and `atol=1e-8` with a margin of
    two orders of magnitude. At the settings of the fit, `1e-6` each, the
    relative difference is up to 3.8e-3, the error of the integrators.
    """
    op_hctz_pk.initialize(settings_tight)
    assert op_hctz_pk.engine is Engine.ROADRUNNER
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)
    r = np.asarray(op_hctz_pk.residuals(x), dtype=float)

    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="copasi"))
    assert op_hctz_pk.engine is Engine.COPASI
    assert op_hctz_pk.runner_initialized.engine is Engine.COPASI
    r_copasi = np.asarray(op_hctz_pk.residuals(x), dtype=float)
    assert r_copasi.shape == r.shape
    assert np.all(np.isfinite(r_copasi))
    np.testing.assert_allclose(r_copasi, r, rtol=1e-5, atol=1e-8)


def test_set_simulator_of_another_engine_loads_the_models_again(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The models of the problem are the ones of the runner, in the engine.

    The models of the mappings keep their grouping: one model of roadrunner
    is one model of COPASI.
    """
    op_hctz_pk.initialize(fit_settings)
    groups = [list(group) for group in op_hctz_pk.mapping_groups]
    identity = [
        [k for k, other in enumerate(op_hctz_pk.models) if other is model]
        for model in op_hctz_pk.models
    ]
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="copasi"))

    assert all(isinstance(model, CopasiSBMLModel) for model in op_hctz_pk.models)
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


def test_set_simulator_of_the_same_engine_keeps_the_models(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    op_hctz_pk.initialize(fit_settings)
    models = list(op_hctz_pk.models)
    plans = list(op_hctz_pk.plans)
    simulator = Simulator(n_workers=1)
    op_hctz_pk.set_simulator(simulator)
    assert op_hctz_pk.engine is Engine.ROADRUNNER
    assert all(a is b for a, b in zip(op_hctz_pk.models, models, strict=True))
    assert all(a is b for a, b in zip(op_hctz_pk.plans, plans, strict=True))
    assert op_hctz_pk.runner_initialized.simulator is simulator


def test_a_problem_of_copasi_loads_its_models_in_copasi_only(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`initialize` loads the models in COPASI, not in roadrunner first."""
    problem = _copasi_problem(op_hctz_pk)
    assert problem.engine is Engine.COPASI

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a model of roadrunner was loaded")

    monkeypatch.setattr(RoadrunnerSBMLModel, "__init__", refuse)
    problem.initialize(fit_settings)
    assert all(isinstance(model, CopasiSBMLModel) for model in problem.models)
    runner = problem.runner_initialized
    assert runner.engine is Engine.COPASI
    simulator = runner.simulator
    assert simulator is not None
    assert simulator.engine is Engine.COPASI
    assert simulator.integrator_settings["relative_tolerance"] == (
        fit_settings.relative_tolerance
    )
    for model in _unique_models(problem):
        assert _relative_tolerance(model) == fit_settings.relative_tolerance


def test_problem_pickles_with_its_engine(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A worker of a parallel fit evaluates with the engine of the problem."""
    assert pickle.loads(pickle.dumps(op_hctz_pk)).engine is Engine.ROADRUNNER
    problem = _copasi_problem(op_hctz_pk)
    assert pickle.loads(pickle.dumps(problem)).engine is Engine.COPASI

    # the engine a simulator set is the engine of the problem
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="copasi"))
    copy = pickle.loads(pickle.dumps(op_hctz_pk))
    assert copy.engine is Engine.COPASI
    assert not copy.is_initialized
    copy.initialize(fit_settings)
    assert all(isinstance(model, CopasiSBMLModel) for model in copy.models)


def test_metrics_ipred_with_copasi(
    op_hctz_pk: OptimizationProblem, settings_tight: FitSettings
) -> None:
    """The DV/PRED/IPRED table of COPASI is the one of roadrunner.

    Measured at the settings of `test_residuals_with_copasi_equal_roadrunner`:
    the largest relative difference of a prediction of the 255 data points
    is 6.8e-9, the largest absolute difference 6.3e-11 of at most 0.41.
    """
    op_hctz_pk.initialize(settings_tight)
    pset = op_hctz_pk.parameter_set_model()
    roadrunner = FitMetrics(op_hctz_pk, pset).datapoints_df()

    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="copasi"))
    assert all(isinstance(model, CopasiSBMLModel) for model in op_hctz_pk.models)
    copasi = FitMetrics(op_hctz_pk, pset).datapoints_df()

    assert list(copasi.columns) == list(roadrunner.columns)
    assert len(copasi) == len(roadrunner)
    np.testing.assert_array_equal(copasi["DV"], roadrunner["DV"])
    for column in ("PRED", "IPRED"):
        values = copasi[column].to_numpy(dtype=float)
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(
            values, roadrunner[column].to_numpy(dtype=float), rtol=1e-5, atol=1e-8
        )


def test_fit_report_of_stored_parameters_with_copasi(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings, tmp_path: Path
) -> None:
    """A report of stored parameters evaluates them with COPASI."""
    op_hctz_pk.initialize(fit_settings)
    op_hctz_pk.set_simulator(Simulator(n_workers=1, engine="copasi"))
    report = FitReport(
        problem=op_hctz_pk,
        settings=fit_settings,
        parameter_sets=op_hctz_pk.parameter_set_model(),
        mapping_figures=False,
    )
    # the report evaluates in a runtime of its own, in the engine of the problem
    assert report.problem.engine is Engine.COPASI
    assert all(isinstance(model, CopasiSBMLModel) for model in report.problem.models)
    index = report.create(tmp_path, name="copasi") / "index.html"
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


def test_the_residuals_of_a_pk_observable_with_copasi_equal_roadrunner(
    tmp_path: Path,
) -> None:
    """A group which keeps the output of its simulation evaluates with COPASI.

    The cmax of `ScalarStudy` is the maximum over the steps of the
    integrator, evaluated at the start value of the volume, 30 l, three
    times the volume of the data, where the residual is -4.46. Measured at
    `absolute_tolerance=1e-12` and `relative_tolerance=1e-10`: the relative
    difference of the residual is 1.6e-10.
    """
    settings = FitSettings(**SETTINGS_TIGHT)
    problem = _scalar_problem(tmp_path)
    problem.initialize(settings)
    assert all(graph is not None for graph in problem._group_graphs)
    x = problem.to_scale(np.asarray(problem.x0, dtype=float))
    r = np.asarray(problem.residuals(x), dtype=float)
    assert np.all(np.abs(r) > 1.0)

    copasi = _scalar_problem(tmp_path, engine="copasi")
    copasi.initialize(settings)
    assert all(isinstance(model, CopasiSBMLModel) for model in copasi.models)
    r_copasi = np.asarray(copasi.residuals(x), dtype=float)
    assert np.all(np.isfinite(r_copasi))
    np.testing.assert_allclose(r_copasi, r, rtol=1e-5, atol=1e-8)


def test_an_unsupported_feature_raises_instead_of_failing_the_evaluation(
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature the engine does not simulate is raised, not a failed point.

    `NotImplementedError` is a `RuntimeError`, which an evaluation turns into
    the residuals of a failed integration; a feature COPASI does not simulate
    fails for every parameter, so it is raised. Both kinds of groups raise
    it: the groups of the HCTZ problem output the times of their data
    (`_simulate_at_data`), the group of the cmax keeps the output of its
    simulation (`_simulate_points`).
    """
    import sbmlsim.fit.optimization as optimization

    op_hctz_pk.initialize(fit_settings)
    assert all(graph is None for graph in op_hctz_pk._group_graphs)
    scalar = _scalar_problem(tmp_path)
    scalar.initialize(fit_settings)
    assert all(graph is not None for graph in scalar._group_graphs)

    def unsupported(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("the engine does not simulate it")

    monkeypatch.setattr(optimization, "execute_plan", unsupported)
    for problem in (op_hctz_pk, scalar):
        x = problem.to_scale(problem.xmodel)
        with pytest.raises(NotImplementedError, match="does not simulate"):
            problem.residuals(x)
        with pytest.raises(NotImplementedError, match="does not simulate"):
            problem.residuals(x, complete_data=True)
