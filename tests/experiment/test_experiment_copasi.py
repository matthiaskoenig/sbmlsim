"""Simulation experiments run with the engine of their simulator."""

import contextlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.engine import Engine, import_copasi

with contextlib.suppress(ImportError):
    # COPASI imported with the locale of the process kept, see `import_copasi`
    import_copasi()
COPASI = pytest.importorskip("COPASI")

from examples.demo.demo import DemoExperiment  # noqa: E402
from examples.repressilator.repressilator import RepressilatorExperiment  # noqa: E402
from sbmlsim import parallel  # noqa: E402
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment  # noqa: E402
from sbmlsim.experiment.runner import _run_job, model_key, run_experiments  # noqa: E402
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel  # noqa: E402
from sbmlsim.model.model_copasi import CopasiSBMLModel  # noqa: E402
from sbmlsim.resources import REPRESSILATOR_SBML  # noqa: E402
from sbmlsim.result.scan import ScanResult  # noqa: E402
from sbmlsim.simulator import Simulator  # noqa: E402

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


def test_repressilator_experiment_runs_with_copasi(tmp_path: Path) -> None:
    roadrunner = _results(Engine.ROADRUNNER, tmp_path)
    copasi = _results(Engine.COPASI, tmp_path)

    assert set(copasi) == set(roadrunner)
    for key, r in roadrunner.items():
        c = copasi[key]
        assert r.ds.attrs["engine"] == "roadrunner"
        assert c.ds.attrs["engine"] == "copasi"
        assert set(c.ds.data_vars) == set(r.ds.data_vars)
        assert c.ds.sizes == r.ds.sizes
        np.testing.assert_array_equal(c.ds["time"].values, r.ds["time"].values)
        for name in r.ds.data_vars:
            np.testing.assert_allclose(
                c.ds[name].values,
                r.ds[name].values,
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"{key}: {name}",
            )
    # the JSON of the experiment records the engine which ran its tasks
    for engine in Engine:
        written = json.loads((tmp_path / engine.value / f"{SID}.json").read_text())
        assert written["engine"] == engine.value


def test_the_runner_loads_its_models_in_the_engine_of_its_simulator() -> None:
    assert _runner(None).engine is Engine.ROADRUNNER
    assert _runner(Engine.ROADRUNNER).engine is Engine.ROADRUNNER
    runner = _runner(Engine.COPASI)

    assert runner.engine is Engine.COPASI
    assert len(runner.models) == 2
    assert all(isinstance(m, CopasiSBMLModel) for m in runner.models.values())
    assert all(key.engine is Engine.COPASI for key in runner.models)
    # the changes of a model are kept
    assert runner.experiments[SID]._models["model2"].changes


def test_the_model_key_includes_the_engine() -> None:
    model = AbstractModel(REPRESSILATOR_SBML)

    assert model_key(model) == model_key(model, Engine.ROADRUNNER)
    assert model_key(model, Engine.COPASI) != model_key(model)
    assert model_key(model, Engine.COPASI)._replace(
        engine=Engine.ROADRUNNER
    ) == model_key(model)


def test_set_simulator_with_another_engine_reloads_the_models() -> None:
    runner = _runner(Engine.ROADRUNNER)
    experiment = runner.experiments[SID]
    assert all(isinstance(m, RoadrunnerSBMLModel) for m in runner.models.values())
    changes = dict(experiment._models["model2"].changes)
    assert changes

    runner.set_simulator(Simulator(n_workers=1, engine="copasi"))

    assert runner.engine is Engine.COPASI
    assert len(runner.models) == 2
    assert all(isinstance(m, CopasiSBMLModel) for m in runner.models.values())
    assert all(key.engine is Engine.COPASI for key in runner.models)
    # the experiment simulates the models of the runner
    assert {id(m) for m in experiment._models.values()} == {
        id(m) for m in runner.models.values()
    }
    assert experiment._models["model2"].changes == changes
    assert experiment.simulator is runner.simulator

    # a simulator of the same engine keeps the models
    loaded = dict(runner.models)
    runner.set_simulator(Simulator(n_workers=1, engine="copasi"))
    assert all(runner.models[k] is m for k, m in loaded.items())

    runner.set_simulator(Simulator(n_workers=1))
    assert all(isinstance(m, RoadrunnerSBMLModel) for m in runner.models.values())
    assert all(isinstance(m, RoadrunnerSBMLModel) for m in experiment._models.values())


def test_a_worker_loads_the_models_in_the_engine_of_the_run(tmp_path: Path) -> None:
    runner = _runner(Engine.COPASI)
    job = runner._jobs(tmp_path, ["svg"], None, OPTIONS)[SID]
    assert job.engine is Engine.COPASI

    outcome = _run_job(job)

    assert outcome.error is None
    models = parallel.worker_cache(
        ("experiment models", Engine.COPASI, job.jit_optimization), dict
    )
    assert models
    assert all(isinstance(m, CopasiSBMLModel) for m in models.values())
    written = json.loads((tmp_path / SID / f"{SID}.json").read_text())
    assert written["engine"] == "copasi"


def test_run_experiments_in_a_pool_with_copasi(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    classes: list[type[SimulationExperiment]] = [
        RepressilatorExperiment,
        DemoExperiment,
    ]
    with caplog.at_level(logging.DEBUG, logger="sbmlsim.experiment.runner"):
        results = run_experiments(
            classes,
            output_path=tmp_path,
            simulator=Simulator(engine="copasi"),
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
        written = json.loads((tmp_path / sid / f"{sid}.json").read_text())
        assert written["engine"] == "copasi"
        assert (tmp_path / sid / f"{sid}_fig1.svg").exists()
