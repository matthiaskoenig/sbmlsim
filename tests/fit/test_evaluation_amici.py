"""The evaluation of a fit problem with the engine AMICI.

The tests AMICI shares with COPASI are in `test_evaluation_engines.py`.
"""

import pickle
from pathlib import Path

import pytest

pytest.importorskip("amici")

from sbmlsim.engine import Engine
from sbmlsim.fit import FitSettings
from sbmlsim.model.model_amici import AmiciSBMLModel, cache_root
from tests.engines import another_cache
from tests.fit.scalar_experiment import scalar_problem


def test_a_worker_of_a_fit_loads_the_models_from_the_cache_of_the_parent(
    fit_settings: FitSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pickled problem loads its models from the directory of its parent.

    The workers of a fit, and the forkserver they start from, keep the
    environment of the time they started; the pickled definition carries the
    directory of the compiled models of the parent, so a worker compiles
    nothing after `SBMLSIM_AMICI_CACHE` changed.
    """
    problem = scalar_problem(tmp_path, Engine.AMICI)
    problem.initialize(fit_settings)
    (model, *_) = problem.models
    assert isinstance(model, AmiciSBMLModel)
    assert model.cache_root == cache_root()
    # the definition a worker gets, of an initialized and of a new problem
    copies = [
        pickle.loads(pickle.dumps(problem)),
        pickle.loads(pickle.dumps(scalar_problem(tmp_path, Engine.AMICI))),
    ]
    # of another engine, nothing
    assert pickle.loads(pickle.dumps(scalar_problem(tmp_path)))._amici_cache is None

    empty = another_cache(monkeypatch, tmp_path / "other")
    for copy in copies:
        assert copy._amici_cache == str(model.cache_root)
        copy.initialize(fit_settings)
        (loaded, *_) = copy.models
        assert isinstance(loaded, AmiciSBMLModel)
        assert loaded.cache_root == model.cache_root
    assert not any(empty.iterdir())
