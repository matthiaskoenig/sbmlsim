"""The propagation of uncertain values with the engines COPASI and AMICI (#219).

The draws of a propagation are inputs of its scans: the references of a
relative distribution are read with roadrunner whatever the engine of the
simulator (`sbmlsim.simulation.sampling.references`), so every engine
simulates the same points. Every test runs with every engine besides
roadrunner, see `tests.engines`.
"""

from typing import Any

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.engine import Engine
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.simulation import PK, Change, Simulation
from sbmlsim.simulation.sampling import LogNormal, references
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import Uncertain, UncertaintyModel, propagate
from tests.engines import model_class
from tests.simulator.models import sbml_pk

OBSERVABLES = [PK("pk", "[C]", dose="PODOSE", route="oral")]
OUTPUTS = ["pk.cmax", "pk.auc_last"]


def _simulation(volume: float = 10.0) -> Simulation:
    return Simulation(
        end=48,
        times=np.linspace(0, 48, 481).tolist(),
        preinit_changes={"V": volume},
        changes=[Change(0, {"PODOSE": Uncertain(Q(100, "mg"), cv=0.1)})],
    )


def _propagate(simulator: Simulator) -> Any:
    return propagate(
        sbml_pk(),
        {"light": _simulation(), "heavy": _simulation(volume=14.0)},
        # a relative distribution, whose references are read with roadrunner
        UncertaintyModel(parameters={"ke": LogNormal(cv=0.2)}),
        OBSERVABLES,
        outputs=OUTPUTS,
        n=256,
        seed=3,
        simulator=simulator,
    )


def test_propagate_with_the_engine_draws_the_same_points(other_engine: Engine) -> None:
    """The engine simulates the points of roadrunner, the summaries agree.

    The points are equal bitwise, the statistics of both engines are within
    the Monte Carlo errors of roadrunner (the outputs of a point differ by
    the tolerances of the integrators, far below the errors of 256 draws).
    """
    rr = _propagate(Simulator(n_workers=1))
    cp = _propagate(Simulator(n_workers=1, engine=other_engine))
    assert rr.scenarios == cp.scenarios
    for name in rr.scenarios:
        ds_rr, ds_cp = rr.results[name].ds, cp.results[name].ds
        assert sorted(ds_rr.coords) == sorted(ds_cp.coords)
        for coordinate in ds_rr.coords:
            np.testing.assert_array_equal(
                ds_cp[coordinate].values, ds_rr[coordinate].values
            )
        assert ds_cp.attrs["engine"] == other_engine.value
    s_rr = rr.summary(OUTPUTS).set_index(["scenario", "output"])
    s_cp = cp.summary(OUTPUTS).set_index(["scenario", "output"])
    assert (s_cp["n_failed"] == 0).all()
    assert (s_cp["mean"] - s_rr["mean"]).abs().le(s_rr["mean_se"]).all()
    assert (s_cp["sd"] - s_rr["sd"]).abs().le(s_rr["sd_se"]).all()


def test_the_references_of_a_model_of_the_engine_are_the_ones_of_roadrunner(
    other_engine: Engine,
) -> None:
    simulation = _simulation(volume=14.0)
    rr = Simulator(n_workers=1).load(sbml_pk())
    cp = Simulator(n_workers=1, engine=other_engine).load(sbml_pk())
    assert isinstance(cp, model_class(other_engine))
    targets = ["ke", "ka", "V"]
    assert references(cp, targets, simulation) == references(rr, targets, simulation)


def test_propagate_with_the_engine_loads_the_references_model_once(
    other_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model of roadrunner of the references is loaded once per propagation.

    Not once per batch and scenario.
    """
    loads: list[int] = []
    load = RoadrunnerSBMLModel.from_abstract_model

    def counted(*args: Any, **kwargs: Any) -> RoadrunnerSBMLModel:
        loads.append(1)
        return load(*args, **kwargs)

    monkeypatch.setattr(RoadrunnerSBMLModel, "from_abstract_model", counted)
    result = _propagate(Simulator(n_workers=1, engine=other_engine))
    assert len(result.scenarios) == 2
    assert len(loads) == 1


def test_propagate_with_the_engine_without_relative_distributions_loads_no_roadrunner(
    other_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a relative distribution no reference is read."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a model of roadrunner was loaded")

    monkeypatch.setattr(RoadrunnerSBMLModel, "from_abstract_model", refuse)
    monkeypatch.setattr(RoadrunnerSBMLModel, "__init__", refuse)
    result = propagate(
        sbml_pk(),
        _simulation(),
        UncertaintyModel(parameters={"ke": Uncertain(0.2, cv=0.2)}),
        OBSERVABLES,
        outputs=OUTPUTS,
        n=16,
        seed=3,
        simulator=Simulator(n_workers=1, engine=other_engine),
    )
    assert result.summary(OUTPUTS)["n_failed"].eq(0).all()
