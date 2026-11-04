"""`Simulator(engine="copasi")` runs simulations and scans with COPASI."""

import contextlib
from typing import Any

import numpy as np
import pytest

from sbmlsim.engine import Engine, import_copasi

with contextlib.suppress(ImportError):
    # COPASI imported with the locale of the process kept, see `import_copasi`
    import_copasi()
COPASI = pytest.importorskip("COPASI")

import sbmlsim.simulator.simulator as simulator_module  # noqa: E402
from sbmlsim import parallel  # noqa: E402
from sbmlsim.model import RoadrunnerSBMLModel  # noqa: E402
from sbmlsim.model.model_copasi import CopasiSBMLModel  # noqa: E402
from sbmlsim.resources import HCTZ_SBML  # noqa: E402
from sbmlsim.simulation import (  # noqa: E402
    Change,
    Dimension,
    Scan,
    Simulation,
)
from sbmlsim.simulator import Simulator  # noqa: E402
from sbmlsim.simulator.engines import check_plan, execute_plan  # noqa: E402
from sbmlsim.simulator.observables import identity_graph  # noqa: E402
from sbmlsim.simulator.plan import compile_simulation  # noqa: E402
from sbmlsim.simulator.worker import (  # noqa: E402
    Chunk,
    OnError,
    run_chunk,
)
from tests.simulator.models import PREINIT_FORMULAS, sbml  # noqa: E402

#: the delayed event of the model fires across a change after the start,
#: which COPASI does not simulate, see `executor_copasi.check_supported`
DELAYED = """
model delayed
  X = 0
  k = 1
  X' = k
  n = 0
  E: at 0.5 after (X >= 1): n = n + 1
end
"""


def _dose_scan() -> Scan:
    """Three oral doses of HCTZ at the times of a grid."""
    return Scan(
        Simulation(start=0, end=100, times=np.linspace(0, 100, 51).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )


def test_copasi_scan_in_a_pool_agrees_with_serial() -> None:
    """The workers of a pool load the model of the parent in COPASI.

    The results agree within the integration and not bitwise: COPASI sums
    the rates of the reactions of a species in an order which depends on the
    memory the process allocated before, so the rates of two loads of a model
    differ in the last bit and the integration amplifies it, see the module of
    the simulator. Measured on this scan at the defaults of `Simulator`, 20
    loads in one process with allocations between them, three times: at most
    4.5e-7 relative beyond an absolute 1e-12 (6.2e-7 relative to a value
    above 1e-6 of the largest value of its variable, 4.2e-8 relative to the
    largest value), so the tolerance is about four times the largest
    measured scatter.
    """
    serial = Simulator(n_workers=1, engine="copasi").run(HCTZ_SBML, _dose_scan())
    pooled = Simulator(n_workers=2, engine="copasi").run(HCTZ_SBML, _dose_scan())
    assert pooled.ds.attrs["engine"] == "copasi"
    assert set(pooled.ds.data_vars) == set(serial.ds.data_vars)
    for name in serial.ds.data_vars:
        np.testing.assert_allclose(
            pooled.ds[name].values,
            serial.ds[name].values,
            rtol=2e-6,
            atol=1e-12,
            err_msg=str(name),
        )


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
            Simulation(end=4, steps=8, changes=[Change(2, {"X": 0})]),
            dimensions=[Dimension("k0", values={"k": [1, 2]})],
        ),
        # the change of a dimension with `at`, of the plans of the points
        Scan(
            Simulation(end=4, steps=8),
            dimensions=[Dimension("x", values={"X": [0, 0.5]}, at=2)],
        ),
    ],
    ids=["simulation", "at"],
)
def test_an_unsupported_feature_raises_before_a_point_runs(
    on_error: OnError, n_workers: int, scan: Scan, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature COPASI does not simulate fails every point alike, the run raises.

    It raises when the scan is compiled, also with `on_error="flag"`, before
    a point runs or a pool starts.
    """
    _no_point_runs(monkeypatch)
    with pytest.raises(NotImplementedError, match="delayed events"):
        Simulator(n_workers=n_workers, engine="copasi").run(
            sbml(DELAYED), scan, on_error=on_error
        )
    assert not parallel._POOLS


def test_a_chunk_of_an_unsupported_feature_raises_also_when_flagged() -> None:
    """The executor checks every plan, a chunk raises it as it is."""
    model = CopasiSBMLModel(source=sbml(DELAYED))
    plan = compile_simulation(
        Simulation(end=4, steps=8, changes=[Change(2, {"X": 0})]),
        model.symbols,
        model.uinfo,
    )
    chunk = Chunk(
        indices=np.arange(2),
        plan=plan,
        model=0,
        graph=identity_graph(("X", "n")),
        values={"k": np.array([1.0, 2.0])},
        timed={},
        time=None,
        on_error="flag",
    )
    with pytest.raises(NotImplementedError, match="delayed events"):
        run_chunk(chunk, model)


def test_the_executor_of_the_engine_of_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RoadrunnerSBMLModel(source=sbml())
    plan = compile_simulation(Simulation(end=1, steps=2), model.symbols, model.uinfo)
    check_plan(plan, model)
    monkeypatch.setattr(model, "engine", Engine.COPASI)
    with pytest.raises(TypeError, match="is no `CopasiSBMLModel`"):
        execute_plan(plan, model, ["time"])
    monkeypatch.setattr(model, "engine", "other")
    with pytest.raises(ValueError, match="The engine 'other' has no executor"):
        execute_plan(plan, model, ["time"])
    with pytest.raises(ValueError, match="The engine 'other' has no executor"):
        check_plan(plan, model)


def test_a_preinit_formula_reading_a_rule_while_a_value_outside_the_state_is_set() -> (
    None
):
    """COPASI keeps a parameter whose value is not a number out of its state."""
    source = sbml(PREINIT_FORMULAS.replace("  u = 7\n", "  u = NaN\n  ru := 3 * u\n"))
    simulation = Simulation(end=10, preinit_changes={"u": 5, "y": "ru"})
    Simulator().simulate(source, simulation)
    with pytest.raises(
        NotImplementedError, match=r"reads the assignment rules \['ru'\]"
    ):
        Simulator(engine="copasi").simulate(source, simulation)
    # a formula which reads no rule reads the value which is set
    simulation = Simulation(end=10, times=[0, 10], preinit_changes={"u": 5, "y": "u"})
    r = Simulator().simulate(source, simulation)
    c = Simulator(engine="copasi").simulate(source, simulation)
    np.testing.assert_allclose(c["y"], r["y"])
