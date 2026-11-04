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
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
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
from tests.engines import OTHER_ENGINES, model_class, require
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


#: a dose by an event whose trigger is false before the time 0, which a
#: pre-initialization of `dose` makes true; `RESET` is the assignment of
#: `dose` back to 0, which sets the trigger to its threshold
DOSE_EVENT = """
model dose_event
  dose = 0
  total = 0
  total' = -0.1 * total
  E: at (dose > 0), t0=false: total = total + 10 RESET
end
"""

#: an event of the start, `time >= tn` true at the start and false before it;
#: `RESET` moves `tn`, which the trigger reads
TIME_EVENT = """
model time_event
  tn = 0
  total = 0
  total' = -0.1 * total
  E: at (time >= tn), t0=false: total = total + 10 RESET
end
"""


def _engine_run(engine: Engine, model: str, simulation: Simulation) -> ScanResult:
    """Run a simulation of an antimony model with an engine, serially."""
    simulator = Simulator(n_workers=1, engine=engine)
    return simulator.run(simulator.load(AbstractModel(source=sbml(model))), simulation)


@pytest.mark.parametrize("times", [[0, 1, 2], None], ids=["times", "steps"])
@pytest.mark.parametrize("reset", ["", ", dose = 0"], ids=["kept", "reset"])
def test_an_event_a_pre_initialization_makes_true_fires_at_the_start(
    other_engine: Engine, reset: str, times: list[float] | None
) -> None:
    """A dose of the pre-initialization fires at the start, its row has the state after it.

    COPASI and AMICI fire it as raw libroadrunner after `resetAll` and the
    SBML Test Suite do; the executor of roadrunner does not fire it (#439).
    """
    res = _engine_run(
        other_engine,
        DOSE_EVENT.replace("RESET", reset),
        Simulation(end=2, times=times, preinit_changes={"dose": 10}),
    )
    time = res.ds["time"].values
    total = res.ds["total"].values
    # one row at the start, the state after the dose
    assert np.count_nonzero(time == 0.0) == 1
    assert total[0] == 10.0
    np.testing.assert_allclose(total, 10.0 * np.exp(-0.1 * time), rtol=1e-6)
    if times is not None:
        assert np.allclose(total, 10.0 * np.exp(-0.1 * np.asarray(times)))


@pytest.mark.parametrize("times", [[0, 1, 2], None], ids=["times", "steps"])
@pytest.mark.parametrize("reset", ["", ", tn = tn + 5"], ids=["kept", "moved"])
def test_an_event_at_the_start_agrees_with_roadrunner(
    other_engine: Engine, reset: str, times: list[float] | None
) -> None:
    """An event of the start without a pre-initialization has the state after it at the start, as in roadrunner."""
    model = TIME_EVENT.replace("RESET", reset)
    simulation = Simulation(end=2, times=times)
    res = _engine_run(other_engine, model, simulation)
    reference = Simulator().run(RoadrunnerSBMLModel(source=sbml(model)), simulation)
    for result in (res, reference):
        assert np.count_nonzero(result.ds["time"].values == 0.0) == 1
        assert result.ds["total"].values[0] == 10.0
    assert res.ds["total"].values[-1] == pytest.approx(
        reference.ds["total"].values[-1], rel=1e-6
    )
    if times is not None:
        np.testing.assert_allclose(
            res.ds["total"].values, reference.ds["total"].values, rtol=1e-6
        )


@pytest.mark.parametrize("times", [[0, 0.5, 1, 2], None], ids=["times", "steps"])
def test_copasi_and_amici_agree_on_a_dose_of_the_pre_initialization(
    times: list[float] | None,
) -> None:
    """COPASI and AMICI have the same states at the start and at the end of a dose of the pre-initialization."""
    for engine in OTHER_ENGINES:
        require(engine)
    simulation = Simulation(end=2, times=times, preinit_changes={"dose": 10})
    model = DOSE_EVENT.replace("RESET", ", dose = 0")
    copasi, amici = (_engine_run(engine, model, simulation) for engine in OTHER_ENGINES)
    for name in ("total", "dose"):
        values = copasi.ds[name].values, amici.ds[name].values
        assert values[0][0] == values[1][0]
        assert values[0][-1] == pytest.approx(values[1][-1], rel=1e-6)
        if times is not None:
            np.testing.assert_allclose(*values, rtol=1e-6)
