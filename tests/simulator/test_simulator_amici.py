"""`Simulator(engine="amici")` runs simulations and scans with AMICI.

The tests AMICI shares with COPASI are in `test_simulator_engines.py`.
AMICI compiles a model once into the cache of the test session, see
`tests.conftest._amici_cache`, which the workers of a pool share.
"""

from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pytest

pytest.importorskip("amici")

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.model_amici import AmiciSBMLModel, cache_root
from sbmlsim.resources import HCTZ_SBML
from sbmlsim.result.scan import ScanResult
from sbmlsim.simulation import (
    Change,
    Dimension,
    Scan,
    Simulation,
)
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.observables import identity_graph
from sbmlsim.simulator.plan import compile_simulation
from sbmlsim.simulator.worker import (
    Chunk,
    ModelSpec,
    OnError,
    run_chunk,
)
from tests.engines import another_cache
from tests.simulator.models import sbml


def _dose_scan() -> Scan:
    """Three oral doses of HCTZ at the times of a grid."""
    return Scan(
        Simulation(start=0, end=100, times=np.linspace(0, 100, 51).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )


@pytest.fixture(scope="module")
def amici_doses() -> ScanResult:
    """Get the scan of the doses with AMICI, serially."""
    return Simulator(n_workers=1, engine="amici").run(HCTZ_SBML, _dose_scan())


def test_amici_scan_in_a_pool_equals_serial(amici_doses: ScanResult) -> None:
    """The workers of a pool import the model the parent compiled.

    A worker loads the model from the cache of the compiled models, so the
    results do not depend on the number of workers, bitwise.
    """
    pooled = Simulator(n_workers=2, engine="amici").run(HCTZ_SBML, _dose_scan())
    assert pooled.ds.attrs["engine"] == "amici"
    assert set(pooled.ds.data_vars) == set(amici_doses.ds.data_vars)
    for name in amici_doses.ds.data_vars:
        np.testing.assert_array_equal(
            pooled.ds[name].values, amici_doses.ds[name].values, err_msg=str(name)
        )


def _nan_compartment() -> str:
    """Get the probe model with a compartment whose size is not a number.

    AMICI does not simulate a change of such a compartment, see
    `executor_amici.check_supported`.
    """
    doc = libsbml.readSBMLFromString(sbml())
    doc.getModel().getCompartment("C").setSize(float("nan"))
    doc.getModel().getSpecies("A").setInitialAmount(1.0)
    doc.getModel().getSpecies("B").setInitialAmount(1.0)
    return libsbml.writeSBMLToString(doc)


def _no_point_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test in which a run runs a chunk or starts a pool."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a point ran")

    monkeypatch.setattr(simulator_module, "run_chunk", fail)
    monkeypatch.setattr(parallel, "pool", fail)


@pytest.mark.parametrize("n_workers", [1, 2])
@pytest.mark.parametrize("on_error", ["raise", "flag"])
@pytest.mark.parametrize(
    "scan",
    [
        Scan(
            Simulation(end=4, steps=8, changes=[Change(2, {"C": 2.0})]),
            dimensions=[Dimension("k", values={"k1": [0.5, 1.0]})],
        ),
        # the change of a dimension with `at`, of the plans of the points
        Scan(
            Simulation(end=4, steps=8),
            dimensions=[Dimension("c", values={"C": [1.0, 2.0]}, at=2)],
        ),
    ],
    ids=["simulation", "at"],
)
def test_an_unsupported_feature_raises_before_a_point_runs(
    on_error: OnError, n_workers: int, scan: Scan, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature AMICI does not simulate fails every point alike, the run raises.

    It raises when the scan is compiled, also with `on_error="flag"`, before
    a point runs or a pool starts.
    """
    _no_point_runs(monkeypatch)
    with pytest.raises(NotImplementedError, match="not a number"):
        Simulator(n_workers=n_workers, engine="amici").run(
            _nan_compartment(), scan, on_error=on_error
        )
    assert not parallel._POOLS


def test_a_chunk_of_an_unsupported_feature_raises_also_when_flagged() -> None:
    """The executor checks every plan, a chunk raises it as it is."""
    model = AmiciSBMLModel(source=_nan_compartment())
    plan = compile_simulation(
        Simulation(end=4, steps=8, changes=[Change(2, {"C": 2.0})]),
        model.symbols,
        model.uinfo,
    )
    chunk = Chunk(
        indices=np.arange(2),
        plan=plan,
        model=0,
        graph=identity_graph(("A", "B")),
        values={"k1": np.array([0.5, 1.0])},
        timed={},
        time=None,
        on_error="flag",
    )
    with pytest.raises(NotImplementedError, match="not a number"):
        run_chunk(chunk, model)


def test_a_worker_loads_the_model_from_the_cache_of_the_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker loads a model from the directory of compiled models of its parent.

    A kept pool, and the forkserver its workers start from, keep the
    environment of the time they started; `ModelSpec.load` ignores it, so a
    worker compiles nothing after the parent changed `SBMLSIM_AMICI_CACHE`.
    """
    model = AmiciSBMLModel(source=sbml())
    spec = ModelSpec.of(model)
    assert spec.cache == str(model.cache_root)
    assert ModelSpec.of(RoadrunnerSBMLModel(source=sbml())).cache is None

    empty = another_cache(monkeypatch, tmp_path / "other")
    loaded = spec.load()
    assert isinstance(loaded, AmiciSBMLModel)
    assert loaded.cache_root == model.cache_root
    assert not any(empty.iterdir())
    # the directory is the one of the load only
    assert cache_root() == empty
