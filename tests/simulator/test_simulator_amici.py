"""`Simulator(engine="amici")` runs simulations and scans with AMICI.

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
from sbmlsim.comparison.diff import DataSetsComparison
from sbmlsim.engine import Engine
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.model_amici import AmiciSBMLModel
from sbmlsim.resources import HCTZ_SBML
from sbmlsim.result.scan import ScanResult
from sbmlsim.simulation import (
    PK,
    Change,
    Dimension,
    Formula,
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
from tests.simulator.models import PREINIT_FORMULAS, sbml


class _Comparison(DataSetsComparison):
    """The comparison of the engines, within the agreement of their integrators.

    Measured on `_dose_scan` at the defaults of `Simulator`: the largest
    relative difference beyond an absolute 1e-10 is 7.9e-8, a factor 12
    below `tol_rel`.
    """

    tol_abs = 1e-10
    tol_rel = 1e-6


def _dose_scan() -> Scan:
    """Three oral doses of HCTZ at the times of a grid."""
    return Scan(
        Simulation(start=0, end=100, times=np.linspace(0, 100, 51).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )


@pytest.fixture(scope="module")
def roadrunner_doses() -> ScanResult:
    """Get the scan of the doses with roadrunner."""
    return Simulator().run(HCTZ_SBML, _dose_scan())


@pytest.fixture(scope="module")
def amici_doses() -> ScanResult:
    """Get the scan of the doses with AMICI, serially."""
    return Simulator(n_workers=1, engine="amici").run(HCTZ_SBML, _dose_scan())


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


@pytest.mark.slow
def test_amici_at_the_defaults_is_as_accurate_as_roadrunner() -> None:
    """AMICI integrates at least as accurately as roadrunner at equal settings.

    See `AMICI_ATOL_FACTOR`: the reference is roadrunner at tight
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
    a = Simulator(engine="amici").run(HCTZ_SBML, scan)
    assert _scaled_error(a, reference) <= _scaled_error(r, reference)


def test_simulator_engine_switch_gives_the_same_result(
    roadrunner_doses: ScanResult, amici_doses: ScanResult
) -> None:
    r, a = roadrunner_doses, amici_doses
    assert r.ds.attrs["engine"] == "roadrunner"
    assert a.ds.attrs["engine"] == "amici"
    assert set(a.ds.data_vars) == set(r.ds.data_vars)
    assert a.ds.sizes == r.ds.sizes
    np.testing.assert_array_equal(a.ds["time"].values, r.ds["time"].values)
    for k in range(r.ds.sizes["dose"]):
        comparison = _Comparison(
            {
                "roadrunner": r.ds.isel(dose=k).drop_vars("PODOSE_hctz"),
                "amici": a.ds.isel(dose=k).drop_vars("PODOSE_hctz"),
            }
        )
        assert comparison.is_equal(), comparison.report_str()


def test_simulator_engine() -> None:
    simulator = Simulator(engine="amici")
    assert simulator.engine is Engine.AMICI
    assert "engine=amici" in repr(simulator)
    assert Simulator(engine=Engine.AMICI).engine is Engine.AMICI


def test_simulate_with_amici() -> None:
    simulation = Simulation(
        end=100,
        times=np.linspace(0, 100, 11).tolist(),
        changes=[Change(0, {"PODOSE_hctz": 25})],
    )
    model = Simulator(engine="amici").load(HCTZ_SBML)
    assert isinstance(model, AmiciSBMLModel)
    a = Simulator(engine="amici").simulate(model, simulation)
    r = Simulator().simulate(HCTZ_SBML, simulation)
    assert sorted(a.columns) == sorted(r.columns)
    for name in r.columns:
        np.testing.assert_allclose(a[name], r[name], rtol=1e-6, atol=1e-10)


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


def test_amici_observables_formula_and_pk() -> None:
    observables = [
        Formula("cp", "[Cve_hctz] * Mr_hctz", unit="ng/ml"),
        Formula("cpmax", "max(cp)"),
        PK("pk", "cp", dose="PODOSE_hctz", route="oral"),
    ]
    # the output times of both engines
    scan = Scan(
        Simulation(end=24 * 60, times=np.linspace(0, 24 * 60, 24 * 60 + 1).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )
    r = Simulator().run(HCTZ_SBML, scan, observables)
    a = Simulator(engine="amici").run(HCTZ_SBML, scan, observables)
    for name in ("cpmax", "pk.cmax", "pk.auc_inf_obs"):
        assert r.ds.attrs["units"][name] == a.ds.attrs["units"][name]
        np.testing.assert_allclose(a.ds[name].values, r.ds[name].values, rtol=1e-6)


def test_a_roadrunner_model_given_to_an_amici_simulator_is_reloaded() -> None:
    model = RoadrunnerSBMLModel(source=HCTZ_SBML, selections=["time", "[Cve_hctz]"])
    loaded = Simulator(engine="amici").load(model)
    assert isinstance(loaded, AmiciSBMLModel)
    assert loaded.selections == ["time", "[Cve_hctz]"]
    # the model of the engine is used as it is
    assert Simulator(engine="amici").load(loaded) is loaded
    assert Simulator().load(model) is model


def test_engine_attribute_survives_netcdf(
    amici_doses: ScanResult, tmp_path: Path
) -> None:
    path = tmp_path / "amici.nc"
    amici_doses.to_netcdf(path)
    assert ScanResult.from_netcdf(path).ds.attrs["engine"] == "amici"
    assert amici_doses.to({"time": "hr"}).ds.attrs["engine"] == "amici"


def test_the_spec_of_an_amici_model_loads_it_with_the_settings() -> None:
    model = AmiciSBMLModel(source=sbml())
    model.set_integrator_settings(relative_tolerance=1e-8, maximum_num_steps=50)
    spec = ModelSpec.of(model)
    assert spec.engine is Engine.AMICI
    assert spec.jit_optimization is None
    loaded = spec.load()
    assert isinstance(loaded, AmiciSBMLModel)
    assert loaded.integrator_state() == model.integrator_state()
    assert ModelSpec.of(loaded) == spec
    # the engine is part of the key
    other = ModelSpec.of(RoadrunnerSBMLModel(source=sbml()))
    assert other.engine is Engine.ROADRUNNER
    assert other.key != spec.key


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


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("q", id="a target"),
        pytest.param("p", id="a dependent of a target"),
        pytest.param("u", id="an unrelated parameter"),
        pytest.param("rq", id="a rule of a target"),
        pytest.param("rp", id="a rule of a dependent of a target"),
        pytest.param("rc + ra", id="rules of a species in a set compartment"),
        pytest.param("2 * q + p - u", id="several parameters"),
    ],
)
def test_preinit_formulas_read_what_they_read_in_roadrunner(formula: str) -> None:
    """A formula reads the values set and as loaded, as in roadrunner (PEtab v2)."""
    simulation = Simulation(
        end=10,
        times=[0, 5, 10],
        preinit_changes={"q": 5, "c": 2, "y": formula},
    )
    source = sbml(PREINIT_FORMULAS)
    r = Simulator().simulate(source, simulation)
    a = Simulator(engine="amici").simulate(source, simulation)
    assert sorted(a.columns) == sorted(r.columns)
    for name in r.columns:
        np.testing.assert_allclose(a[name], r[name], rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("formula", ["S", "[S]"], ids=["an amount", "a concentration"])
def test_a_preinit_formula_reading_a_species_raises(formula: str) -> None:
    """Nothing but a parameter has a value before the initialization."""
    simulation = Simulation(end=10, preinit_changes={"q": 5, "y": formula})
    with pytest.raises(ValueError, match="which is not a parameter"):
        Simulator(engine="amici").simulate(sbml(PREINIT_FORMULAS), simulation)
