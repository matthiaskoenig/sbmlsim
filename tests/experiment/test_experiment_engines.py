"""Simulation experiments run with the engine of their simulator.

Every test runs with every engine besides roadrunner, see `tests.engines`.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from examples.demo.demo import DemoExperiment
from examples.repressilator.repressilator import RepressilatorExperiment
from sbmlsim import parallel
from sbmlsim.engine import Engine
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.experiment.runner import _run_job, model_key, run_experiments
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.resources import DEMO_SBML, REPRESSILATOR_SBML
from sbmlsim.result.scan import ScanResult
from sbmlsim.simulator import Simulator
from tests.engines import model_class

SID = "RepressilatorExperiment"

#: the options of `SimulationExperiment.run` of a run of the runner
OPTIONS = {
    "show_figures": False,
    "save_results": False,
    "reduced_selections": True,
    "keep_results": False,
    "figure_settings": None,
}


def _runner(engine: Engine | str | None) -> ExperimentRunner:
    return ExperimentRunner(
        RepressilatorExperiment,
        simulator=None if engine is None else Simulator(n_workers=1, engine=engine),
        on_error="raise",
    )


def _results(engine: Engine, path: Path) -> dict[str, ScanResult]:
    runner = _runner(engine)
    experiment = runner.experiments[SID]
    experiment.run(runner.simulator, output_path=path / engine.value)
    return experiment.results


def test_repressilator_experiment_runs_with_the_engine(
    other_engine: Engine, tmp_path: Path
) -> None:
    roadrunner = _results(Engine.ROADRUNNER, tmp_path)
    engine = _results(other_engine, tmp_path)

    assert set(engine) == set(roadrunner)
    for key, r in roadrunner.items():
        e = engine[key]
        assert r.ds.attrs["engine"] == "roadrunner"
        assert e.ds.attrs["engine"] == other_engine.value
        assert set(e.ds.data_vars) == set(r.ds.data_vars)
        assert e.ds.sizes == r.ds.sizes
        np.testing.assert_array_equal(e.ds["time"].values, r.ds["time"].values)
        for name in r.ds.data_vars:
            np.testing.assert_allclose(
                e.ds[name].values,
                r.ds[name].values,
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"{key}: {name}",
            )
    # the JSON of the experiment records the engine which ran its tasks
    for compared in (Engine.ROADRUNNER, other_engine):
        written = json.loads((tmp_path / compared.value / f"{SID}.json").read_text())
        assert written["engine"] == compared.value


def test_the_runner_loads_its_models_in_the_engine_of_its_simulator(
    other_engine: Engine,
) -> None:
    assert _runner(None).engine is Engine.ROADRUNNER
    assert _runner(Engine.ROADRUNNER).engine is Engine.ROADRUNNER
    runner = _runner(other_engine)

    assert runner.engine is other_engine
    assert len(runner.models) == 2
    assert all(isinstance(m, model_class(other_engine)) for m in runner.models.values())
    assert all(key.engine is other_engine for key in runner.models)
    # the changes of a model are kept
    assert runner.experiments[SID]._models["model2"].changes


def test_the_model_key_includes_the_engine(other_engine: Engine) -> None:
    model = AbstractModel(REPRESSILATOR_SBML)

    assert model_key(model) == model_key(model, Engine.ROADRUNNER)
    assert model_key(model, other_engine) != model_key(model)
    assert model_key(model, other_engine)._replace(
        engine=Engine.ROADRUNNER
    ) == model_key(model)


def test_set_simulator_with_another_engine_reloads_the_models(
    other_engine: Engine,
) -> None:
    runner = _runner(Engine.ROADRUNNER)
    experiment = runner.experiments[SID]
    assert all(isinstance(m, RoadrunnerSBMLModel) for m in runner.models.values())
    changes = dict(experiment._models["model2"].changes)
    assert changes

    runner.set_simulator(Simulator(n_workers=1, engine=other_engine))

    cls = model_class(other_engine)
    assert runner.engine is other_engine
    assert len(runner.models) == 2
    assert all(isinstance(m, cls) for m in runner.models.values())
    assert all(key.engine is other_engine for key in runner.models)
    # the experiment simulates the models of the runner
    assert {id(m) for m in experiment._models.values()} == {
        id(m) for m in runner.models.values()
    }
    assert experiment._models["model2"].changes == changes
    assert experiment.simulator is runner.simulator

    # a simulator of the same engine keeps the models
    loaded = dict(runner.models)
    runner.set_simulator(Simulator(n_workers=1, engine=other_engine))
    assert all(runner.models[k] is m for k, m in loaded.items())

    runner.set_simulator(Simulator(n_workers=1))
    assert all(isinstance(m, RoadrunnerSBMLModel) for m in runner.models.values())
    assert all(isinstance(m, RoadrunnerSBMLModel) for m in experiment._models.values())


def test_a_worker_loads_the_models_in_the_engine_of_the_run(
    other_engine: Engine, tmp_path: Path
) -> None:
    runner = _runner(other_engine)
    job = runner._jobs(tmp_path, ["svg"], None, OPTIONS)[SID]
    assert job.engine is other_engine

    outcome = _run_job(job)

    assert outcome.error is None
    models = parallel.worker_cache(
        ("experiment models", other_engine, job.jit_optimization), dict
    )
    assert models
    assert all(isinstance(m, model_class(other_engine)) for m in models.values())
    written = json.loads((tmp_path / SID / f"{SID}.json").read_text())
    assert written["engine"] == other_engine.value


def test_run_experiments_in_a_pool_with_the_engine(
    other_engine: Engine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    classes: list[type[SimulationExperiment]] = [
        RepressilatorExperiment,
        DemoExperiment,
    ]
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.experiment.runner"):
        results = run_experiments(
            classes,
            output_path=tmp_path,
            simulator=Simulator(engine=other_engine),
            n_workers=2,
            progress=False,
        )

    assert [r.error for r in results] == [None, None]
    assert [r.failed_figures for r in results] == [{}, {}]
    for cls in classes:
        sid = cls.__name__
        # the experiment ran in a worker
        assert any(
            r.getMessage() == f"Initialize SimulationExperiment: {sid}"
            and r.process != os.getpid()
            for r in caplog.records
        )
        # the engine of the simulator of the worker; the models the worker
        # loads are tested on `_run_job` in this process, see
        # `test_a_worker_loads_the_models_in_the_engine_of_the_run`
        written = json.loads((tmp_path / sid / f"{sid}.json").read_text())
        assert written["engine"] == other_engine.value
        assert (tmp_path / sid / f"{sid}_fig1.svg").exists()


def test_the_worker_simulates_with_the_settings_of_the_run(
    other_engine: Engine, tmp_path: Path
) -> None:
    """A worker builds its simulator with the engine and settings of the run."""
    runner = ExperimentRunner(
        [RepressilatorExperiment, DemoExperiment],
        simulator=Simulator(n_workers=1, engine=other_engine, relative_tolerance=1e-8),
    )
    results = runner.run_experiments(
        output_path=tmp_path, save_results=True, n_workers=2, progress=False
    )

    assert [r.error for r in results] == [None, None]
    files = sorted(tmp_path.glob("*/*.nc"))
    assert {f.parent.name for f in files} == {
        "RepressilatorExperiment",
        "DemoExperiment",
    }
    for path in files:
        attrs = ScanResult.from_netcdf(path).ds.attrs
        assert attrs["engine"] == other_engine.value
        assert attrs["integrator_settings"]["relative_tolerance"] == 1e-8


def test_a_model_the_experiment_loaded_is_loaded_again_by_its_description(
    other_engine: Engine,
) -> None:
    """The runner loads it with its own registry, the defaults of its engine."""
    for engine in (Engine.ROADRUNNER, other_engine):
        runner = ExperimentRunner(
            DemoExperiment,
            simulator=Simulator(n_workers=1, engine=engine),
            on_error="raise",
        )
        model = runner.experiments["DemoExperiment"]._models["model"]
        assert model.engine is engine
        # the experiment gives no selections: the model has the defaults of
        # its engine, and roadrunner's are not given to another engine
        assert model.selections_given is False
        assert model.source.path == DEMO_SBML.resolve()
