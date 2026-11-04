"""`Simulator(engine="copasi")` runs simulations and scans with COPASI."""

import contextlib
from pathlib import Path
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
from sbmlsim.comparison.diff import DataSetsComparison  # noqa: E402
from sbmlsim.model import RoadrunnerSBMLModel  # noqa: E402
from sbmlsim.model.model_copasi import CopasiSBMLModel  # noqa: E402
from sbmlsim.resources import HCTZ_SBML  # noqa: E402
from sbmlsim.result.scan import ScanResult  # noqa: E402
from sbmlsim.simulation import (  # noqa: E402
    PK,
    Change,
    Dimension,
    Formula,
    Scan,
    Simulation,
)
from sbmlsim.simulator import Simulator  # noqa: E402
from sbmlsim.simulator.engines import check_plan, execute_plan  # noqa: E402
from sbmlsim.simulator.observables import identity_graph  # noqa: E402
from sbmlsim.simulator.plan import compile_simulation  # noqa: E402
from sbmlsim.simulator.worker import (  # noqa: E402
    Chunk,
    ModelSpec,
    OnError,
    run_chunk,
)
from tests.simulator.models import sbml  # noqa: E402

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


class _Comparison(DataSetsComparison):
    """The comparison of the engines, within the agreement of their integrators."""

    tol_abs = 1e-10
    tol_rel = 1e-6


def _dose_scan() -> Scan:
    """Three oral doses of HCTZ at the times of a grid."""
    return Scan(
        Simulation(start=0, end=100, times=np.linspace(0, 100, 51).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )


def _assert_points_agree(r: ScanResult, c: ScanResult) -> None:
    """Every point of a scan over `dose` agrees between the engines."""
    for k in range(r.ds.sizes["dose"]):
        comparison = _Comparison(
            {
                "roadrunner": r.ds.isel(dose=k).drop_vars("PODOSE_hctz"),
                "copasi": c.ds.isel(dose=k).drop_vars("PODOSE_hctz"),
            }
        )
        assert comparison.is_equal(), comparison.report_str()


def _scaled_error(result: ScanResult, reference: ScanResult) -> float:
    """Get the largest error of a variable relative to its largest value."""
    errors = [0.0]
    for name in reference.ds.data_vars:
        expected = reference.ds[name].values
        scale = np.nanmax(np.abs(expected))
        if scale > 1e-12:
            difference = np.abs(result.ds[name].values - expected)
            errors.append(float(np.nanmax(difference) / scale))
    return max(errors)


def test_copasi_at_the_defaults_is_as_accurate_as_roadrunner() -> None:
    """COPASI integrates at least as accurately as roadrunner at equal settings.

    See `COPASI_ATOL_FACTOR`: the reference is roadrunner at tight
    tolerances, the error of a variable is relative to its largest value.
    """
    scan = Scan(
        Simulation(end=24 * 60, times=np.linspace(0, 24 * 60, 241).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )
    reference = Simulator(absolute_tolerance=1e-16, relative_tolerance=1e-12).run(
        HCTZ_SBML, scan
    )
    r = Simulator().run(HCTZ_SBML, scan)
    c = Simulator(engine="copasi").run(HCTZ_SBML, scan)
    assert _scaled_error(c, reference) <= _scaled_error(r, reference)


def test_simulator_engine_switch_gives_the_same_result() -> None:
    r = Simulator().run(HCTZ_SBML, _dose_scan())
    c = Simulator(engine="copasi").run(HCTZ_SBML, _dose_scan())
    assert r.ds.attrs["engine"] == "roadrunner"
    assert c.ds.attrs["engine"] == "copasi"
    assert set(c.ds.data_vars) == set(r.ds.data_vars)
    assert c.ds.sizes == r.ds.sizes
    _assert_points_agree(r, c)


def test_simulator_engine() -> None:
    assert Simulator().engine is Engine.ROADRUNNER
    simulator = Simulator(engine="copasi")
    assert simulator.engine is Engine.COPASI
    assert "engine=copasi" in repr(simulator)
    with pytest.raises(ValueError, match="Unknown engine 'nope'"):
        Simulator(engine="nope")


def test_simulate_with_copasi() -> None:
    simulation = Simulation(
        end=100,
        times=np.linspace(0, 100, 11).tolist(),
        changes=[Change(0, {"PODOSE_hctz": 25})],
    )
    model = Simulator(engine="copasi").load(HCTZ_SBML)
    assert isinstance(model, CopasiSBMLModel)
    c = Simulator(engine="copasi").simulate(model, simulation)
    r = Simulator().simulate(HCTZ_SBML, simulation)
    # the default selections of the engines are in another order
    assert sorted(c.columns) == sorted(r.columns)
    for name in r.columns:
        np.testing.assert_allclose(c[name], r[name], rtol=1e-6, atol=1e-10)


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


def test_copasi_observables_formula_and_pk() -> None:
    observables = [
        Formula("cp", "[Cve_hctz] * Mr_hctz", unit="ng/ml"),
        Formula("cpmax", "max(cp)"),
        PK("pk", "cp", dose="PODOSE_hctz", route="oral"),
    ]
    # the output times of both engines, the steps of the integrators differ
    scan = Scan(
        Simulation(end=24 * 60, times=np.linspace(0, 24 * 60, 24 * 60 + 1).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )
    r = Simulator().run(HCTZ_SBML, scan, observables)
    c = Simulator(engine="copasi").run(HCTZ_SBML, scan, observables)
    for name in ("cpmax", "pk.cmax", "pk.auc_inf_obs"):
        assert r.ds.attrs["units"][name] == c.ds.attrs["units"][name]
        np.testing.assert_allclose(c.ds[name].values, r.ds[name].values, rtol=1e-6)


def test_a_roadrunner_model_given_to_a_copasi_simulator_is_reloaded() -> None:
    model = RoadrunnerSBMLModel(source=HCTZ_SBML, selections=["time", "[Cve_hctz]"])
    loaded = Simulator(engine="copasi").load(model)
    assert isinstance(loaded, CopasiSBMLModel)
    assert loaded.selections == ["time", "[Cve_hctz]"]
    # the model of the engine is used as it is
    assert Simulator(engine="copasi").load(loaded) is loaded
    assert Simulator().load(model) is model


def test_engine_attribute_survives_netcdf(tmp_path: Path) -> None:
    c = Simulator(engine="copasi").run(HCTZ_SBML, _dose_scan())
    path = tmp_path / "copasi.nc"
    c.to_netcdf(path)
    assert ScanResult.from_netcdf(path).ds.attrs["engine"] == "copasi"
    assert c.to({"time": "hr"}).ds.attrs["engine"] == "copasi"


def test_the_spec_of_a_copasi_model_loads_it_with_the_settings() -> None:
    model = CopasiSBMLModel(source=sbml())
    model.set_integrator_settings(relative_tolerance=1e-8, maximum_num_steps=50)
    spec = ModelSpec.of(model)
    assert spec.engine is Engine.COPASI
    loaded = spec.load()
    assert isinstance(loaded, CopasiSBMLModel)
    assert loaded.integrator_state() == model.integrator_state()
    assert ModelSpec.of(loaded) == spec
    # the engine is part of the key
    other = ModelSpec.of(RoadrunnerSBMLModel(source=sbml()))
    assert other.engine is Engine.ROADRUNNER
    assert other.key != spec.key


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
