"""`Simulator(engine=...)` runs simulations and scans with COPASI and AMICI.

Every test runs with every engine besides roadrunner, see `tests.engines`,
and compares it with roadrunner; the tests of one engine are in
`test_simulator_copasi.py` and `test_simulator_amici.py`.
"""

from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.comparison.diff import DataSetsComparison
from sbmlsim.engine import Engine
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import HCTZ_SBML
from sbmlsim.result.scan import ScanResult
from sbmlsim.simulation import (
    PK,
    Change,
    Custom,
    Dimension,
    Formula,
    Scan,
    Simulation,
)
from sbmlsim.simulation.observables import ObservableKind
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.worker import ModelSpec
from tests.engines import model_class
from tests.simulator.models import (
    PREINIT_FORMULAS,
    auc_of_c,
    doubled,
    sbml,
    sbml_pk,
)


class _Comparison(DataSetsComparison):
    """The comparison of the engines, within the agreement of their integrators.

    Measured on `_dose_scan` at the defaults of `Simulator`, the largest
    relative difference beyond an absolute 1e-10 of AMICI is 7.9e-8, a factor
    12 below `tol_rel`.
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
def engine_doses(other_engine: Engine) -> ScanResult:
    """Get the scan of the doses with the engine, serially."""
    return Simulator(n_workers=1, engine=other_engine).run(HCTZ_SBML, _dose_scan())


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


def test_the_engine_at_the_defaults_is_as_accurate_as_roadrunner(
    other_engine: Engine,
) -> None:
    """The engine integrates at least as accurately as roadrunner at equal settings.

    See `COPASI_ATOL_FACTOR` and `AMICI_ATOL_FACTOR`: the reference is
    roadrunner at tight tolerances, the error of a variable is relative to
    its largest value.
    """
    scan = Scan(
        Simulation(end=24 * 60, times=np.linspace(0, 24 * 60, 241).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )
    reference = Simulator(absolute_tolerance=1e-16, relative_tolerance=1e-12).run(
        HCTZ_SBML, scan
    )
    r = Simulator().run(HCTZ_SBML, scan)
    e = Simulator(engine=other_engine).run(HCTZ_SBML, scan)
    assert _scaled_error(e, reference) <= _scaled_error(r, reference)


def test_simulator_engine_switch_gives_the_same_result(
    other_engine: Engine, roadrunner_doses: ScanResult, engine_doses: ScanResult
) -> None:
    r, e = roadrunner_doses, engine_doses
    assert r.ds.attrs["engine"] == "roadrunner"
    assert e.ds.attrs["engine"] == other_engine.value
    assert set(e.ds.data_vars) == set(r.ds.data_vars)
    assert e.ds.sizes == r.ds.sizes
    np.testing.assert_array_equal(e.ds["time"].values, r.ds["time"].values)
    for k in range(r.ds.sizes["dose"]):
        comparison = _Comparison(
            {
                "roadrunner": r.ds.isel(dose=k).drop_vars("PODOSE_hctz"),
                other_engine.value: e.ds.isel(dose=k).drop_vars("PODOSE_hctz"),
            }
        )
        assert comparison.is_equal(), comparison.report_str()


def test_simulator_engine(other_engine: Engine) -> None:
    assert Simulator().engine is Engine.ROADRUNNER
    simulator = Simulator(engine=other_engine.value)
    assert simulator.engine is other_engine
    assert f"engine={other_engine.value}" in repr(simulator)
    with pytest.raises(ValueError, match="Unknown engine 'nope'"):
        Simulator(engine="nope")


def test_simulate_with_the_engine(other_engine: Engine) -> None:
    simulation = Simulation(
        end=100,
        times=np.linspace(0, 100, 11).tolist(),
        changes=[Change(0, {"PODOSE_hctz": 25})],
    )
    model = Simulator(engine=other_engine).load(HCTZ_SBML)
    assert isinstance(model, model_class(other_engine))
    e = Simulator(engine=other_engine).simulate(model, simulation)
    r = Simulator().simulate(HCTZ_SBML, simulation)
    # the default selections of the engines may be in another order
    assert sorted(e.columns) == sorted(r.columns)
    for name in r.columns:
        np.testing.assert_allclose(e[name], r[name], rtol=1e-6, atol=1e-10)


def test_observables_formula_and_pk(other_engine: Engine) -> None:
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
    e = Simulator(engine=other_engine).run(HCTZ_SBML, scan, observables)
    for name in ("cpmax", "pk.cmax", "pk.auc_inf_obs"):
        assert r.ds.attrs["units"][name] == e.ds.attrs["units"][name]
        np.testing.assert_allclose(e.ds[name].values, r.ds[name].values, rtol=1e-6)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_custom_observables(other_engine: Engine, n_workers: int) -> None:
    """A `Custom` observable, a value and a timecourse, serially and in a pool.

    The workers of a pool import the function of a `Custom` by its module.
    """
    observables = [
        Formula("c", "[C]"),
        Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
        Custom(
            "twice", doubled, "mg/l", symbols=["[C]"], kind=ObservableKind.TIMECOURSE
        ),
    ]
    scan = Scan(
        Simulation(end=48, times=np.linspace(0, 48, 97).tolist()),
        [Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})],
    )
    r = Simulator(n_workers=1).run(sbml_pk(), scan, observables)
    e = Simulator(n_workers=n_workers, engine=other_engine).run(
        sbml_pk(), scan, observables
    )
    assert e.ds.attrs["engine"] == other_engine.value
    np.testing.assert_allclose(
        e["auc"].values, np.trapezoid(e["c"].values, e["time"].values, axis=1)
    )
    np.testing.assert_allclose(e["twice"].values, 2.0 * e["c"].values)
    for name in ("auc", "twice"):
        assert e.ds.attrs["units"][name] == r.ds.attrs["units"][name]
        np.testing.assert_allclose(
            e[name].values, r[name].values, rtol=1e-6, atol=1e-10
        )


def test_a_roadrunner_model_given_to_the_simulator_is_reloaded(
    other_engine: Engine,
) -> None:
    model = RoadrunnerSBMLModel(source=HCTZ_SBML, selections=["time", "[Cve_hctz]"])
    loaded = Simulator(engine=other_engine).load(model)
    assert isinstance(loaded, model_class(other_engine))
    assert loaded.selections == ["time", "[Cve_hctz]"]
    # the model of the engine is used as it is
    assert Simulator(engine=other_engine).load(loaded) is loaded
    assert Simulator().load(model) is model


def test_engine_attribute_survives_netcdf(
    other_engine: Engine, engine_doses: ScanResult, tmp_path: Path
) -> None:
    path = tmp_path / "engine.nc"
    engine_doses.to_netcdf(path)
    assert ScanResult.from_netcdf(path).ds.attrs["engine"] == other_engine.value
    assert engine_doses.to({"time": "hr"}).ds.attrs["engine"] == other_engine.value


def test_the_spec_of_a_model_loads_it_with_the_settings(other_engine: Engine) -> None:
    model = model_class(other_engine)(source=sbml())
    model.set_integrator_settings(relative_tolerance=1e-8, maximum_num_steps=50)
    spec = ModelSpec.of(model)
    assert spec.engine is other_engine
    assert spec.jit_optimization is None
    loaded = spec.load()
    assert isinstance(loaded, model_class(other_engine))
    assert loaded.integrator_state() == model.integrator_state()
    assert ModelSpec.of(loaded) == spec
    # the engine is part of the key
    other = ModelSpec.of(RoadrunnerSBMLModel(source=sbml()))
    assert other.engine is Engine.ROADRUNNER
    assert other.key != spec.key


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
def test_preinit_formulas_read_what_they_read_in_roadrunner(
    other_engine: Engine, formula: str
) -> None:
    """A formula reads the values set and as loaded, as in roadrunner (PEtab v2)."""
    simulation = Simulation(
        end=10,
        times=[0, 5, 10],
        preinit_changes={"q": 5, "c": 2, "y": formula},
    )
    source = sbml(PREINIT_FORMULAS)
    r = Simulator().simulate(source, simulation)
    e = Simulator(engine=other_engine).simulate(source, simulation)
    assert sorted(e.columns) == sorted(r.columns)
    for name in r.columns:
        np.testing.assert_allclose(e[name], r[name], rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("formula", ["S", "[S]"], ids=["an amount", "a concentration"])
def test_a_preinit_formula_reading_a_species_raises(
    other_engine: Engine, formula: str
) -> None:
    """Nothing but a parameter has a value before the initialization."""
    simulation = Simulation(end=10, preinit_changes={"q": 5, "y": formula})
    for engine in (Engine.ROADRUNNER, other_engine):
        with pytest.raises(ValueError, match="which is not a parameter"):
            Simulator(engine=engine).simulate(sbml(PREINIT_FORMULAS), simulation)
