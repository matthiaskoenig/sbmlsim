"""Simulation experiments with the engine AMICI.

The tests AMICI shares with COPASI are in `test_experiment_engines.py`.
"""

import json
from collections import OrderedDict
from pathlib import Path

import pytest

pytest.importorskip("amici")

from examples.repressilator.repressilator import RepressilatorExperiment
from sbmlsim import parallel
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.experiment.runner import _run_job
from sbmlsim.model.model_amici import AmiciSBMLModel, cache_root
from sbmlsim.simulator import Simulator
from tests.engines import another_cache

SID = "RepressilatorExperiment"


def test_a_worker_loads_the_models_from_the_cache_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker loads the models from the directory of compiled models of the run.

    The workers of a pool, and the forkserver they start from, keep the
    environment of the time they started; a job carries the directory the
    models of the runner were loaded from, so a worker compiles nothing
    after `SBMLSIM_AMICI_CACHE` changed.
    """
    runner = ExperimentRunner(
        RepressilatorExperiment,
        simulator=Simulator(n_workers=1, engine="amici"),
        on_error="raise",
    )
    (model, *_) = runner.models.values()
    assert isinstance(model, AmiciSBMLModel)
    options = {
        "show_figures": False,
        "save_results": False,
        "reduced_selections": True,
        "keep_results": False,
        "figure_settings": None,
    }
    job = runner._jobs(tmp_path / "out", ["svg"], None, options)[SID]
    assert job.amici_cache == str(model.cache_root)
    assert job.amici_cache == str(cache_root())

    empty = another_cache(monkeypatch, tmp_path / "other")
    # the worker loads the models, none of an earlier test of the process
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    outcome = _run_job(job)
    assert parallel._CACHE

    assert outcome.error is None
    assert not any(empty.iterdir())
    written = json.loads(
        (tmp_path / "out" / SID / f"{SID}.json").read_text(encoding="utf-8")
    )
    assert written["engine"] == "amici"
