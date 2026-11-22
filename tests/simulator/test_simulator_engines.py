"""`Simulator(engine=...)` runs simulations and scans with COPASI, AMICI and core.

Every test runs with every engine besides roadrunner, see `tests.engines`,
and compares it with roadrunner; the tests of one engine are in
`test_simulator_copasi.py`, `test_simulator_amici.py` and
`test_simulator_core.py`.
"""

from pathlib import Path

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.comparison.diff import DataSetsComparison
from sbmlsim.engine import DEFAULT_ENGINE, Engine, UnsupportedModelError
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
    SteadyState,
)
from sbmlsim.simulation.observables import ObservableKind
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.worker import ModelSpec
from tests.engines import OTHER_ENGINES, model_class, require
from tests.simulator.models import (
    PREINIT_FORMULAS,
    auc_of_c,
    doubled,
    roadrunner_simulator,
    sbml,
    sbml_pk,
)


class _Comparison(DataSetsComparison):
    """The comparison of the engines, within the agreement of their integrators.

    Measured on `_dose_scan` at the defaults of `Simulator`, the largest
    relative difference beyond an absolute 1e-10 of AMICI is 7.9e-8, a factor
    12 below `tol_rel`, and the one of the engine core 8.8e-9.
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
    return roadrunner_simulator().run(HCTZ_SBML, _dose_scan())


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


#: how much larger than roadrunner's the error of an engine may be at the
#: defaults of `Simulator`: COPASI and AMICI integrate with one absolute
#: tolerance, whose factor was measured so that they are at least as accurate
#: (`COPASI_ATOL_FACTOR`, `AMICI_ATOL_FACTOR`), while the engine core
#: integrates with the absolute tolerance of every state of roadrunner and has
#: no factor to measure, so its error is roadrunner's up to the steps its own
#: CVODE chooses and is not bounded by it, the known difference 9 of
#: `sbmlsim.simulator.executor_core`, which carries the measurements: 1.0003
#: of roadrunner's error on the model of this test at these settings, 0.98 and
#: 1.03 of it on the other packaged models with an error at all and 1.237 of
#: it one pair of tolerances away. The margin is therefore no bound of the
#: agreement of the two integrators but a check of the class of the error, a
#: core which does not receive the tolerance vector of roadrunner being wrong
#: by orders of magnitude
ACCURACY_MARGIN: dict[Engine, float] = {
    Engine.COPASI: 1.0,
    Engine.AMICI: 1.0,
    Engine.CORE: 1.1,
}


def test_the_engine_at_the_defaults_is_as_accurate_as_roadrunner(
    other_engine: Engine,
) -> None:
    """The engine integrates at least as accurately as roadrunner at equal settings.

    See `COPASI_ATOL_FACTOR`, `AMICI_ATOL_FACTOR` and `ACCURACY_MARGIN`: the
    reference is roadrunner at tight tolerances, the error of a variable is
    relative to its largest value.
    """
    scan = Scan(
        Simulation(end=24 * 60, times=np.linspace(0, 24 * 60, 241).tolist()),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10, 25, 50]})],
    )
    reference = roadrunner_simulator(
        absolute_tolerance=1e-16, relative_tolerance=1e-12
    ).run(HCTZ_SBML, scan)
    r = roadrunner_simulator().run(HCTZ_SBML, scan)
    e = Simulator(engine=other_engine).run(HCTZ_SBML, scan)
    margin = ACCURACY_MARGIN[other_engine]
    assert _scaled_error(e, reference) <= margin * _scaled_error(r, reference)


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
    assert Simulator().engine is DEFAULT_ENGINE is Engine.CORE
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
    r = roadrunner_simulator().simulate(HCTZ_SBML, simulation)
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
    r = roadrunner_simulator().run(HCTZ_SBML, scan, observables)
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
    r = roadrunner_simulator(n_workers=1).run(sbml_pk(), scan, observables)
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
    assert roadrunner_simulator().load(model) is model


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


#: what the engine core says when it refuses a formula before the
#: initialization: a value of a request is a number and the model of the engine
#: is stateless, so the executor folds only a symbol whose value it knows
#: without the model, one the same pre-initialization sets to a number and one
#: which is a constant parameter of the model, the variable of no rule and of
#: no initial assignment, which nothing the pre-initialization sets can move;
#: a known difference of `sbmlsim.simulator.executor_core` which
#: `tests.simulator.test_simulator_core` pins
PREINIT_FORMULA_REFUSED: str = (
    "does not simulate the formula .* before the initialization"
)


@pytest.mark.parametrize(
    ("formula", "folded"),
    [
        pytest.param("q", True, id="a target"),
        pytest.param("p", False, id="a dependent of a target"),
        pytest.param("u", True, id="an unrelated parameter"),
        pytest.param("q + u", True, id="a target and an unrelated parameter"),
        pytest.param("rq", False, id="a rule of a target"),
        pytest.param("rp", False, id="a rule of a dependent of a target"),
        pytest.param("rc + ra", False, id="rules of a species in a set compartment"),
        pytest.param("2 * q + p - u", False, id="several parameters"),
    ],
)
def test_preinit_formulas_read_what_they_read_in_roadrunner(
    other_engine: Engine, formula: str, folded: bool
) -> None:
    """A formula reads the values set and as loaded, as in roadrunner (PEtab v2).

    `folded` is whether the engine core can fold the formula, i.e. whether it
    knows the value of every symbol of it without the model: one the same
    pre-initialization sets to a number, and a constant parameter of the
    model, whose value nothing the pre-initialization sets can move. It
    refuses the others, see `PREINIT_FORMULA_REFUSED`. A folded formula is
    compared with roadrunner, which is what holds the fold to its values.
    """
    simulation = Simulation(
        end=10,
        times=[0, 5, 10],
        preinit_changes={"q": 5, "c": 2, "y": formula},
    )
    source = sbml(PREINIT_FORMULAS)
    if other_engine is Engine.CORE and not folded:
        with pytest.raises(UnsupportedModelError, match=PREINIT_FORMULA_REFUSED):
            Simulator(engine=other_engine).simulate(source, simulation)
        return
    r = roadrunner_simulator().simulate(source, simulation)
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
#: `dose` back to 0, which sets the trigger to its threshold, `n` counts the
#: executions of the event across a presimulation
DOSE_EVENT = """
model dose_event
  dose = 0
  n = 0
  total = 0
  total' = -0.1 * total
  E: at (dose > 0), t0=false: total = total + 10, n = n + 1 RESET
end
"""

#: an event of the start, `time >= tn` true at the start and false before it;
#: `RESET` moves `tn`, which the trigger reads
TIME_EVENT = """
model time_event
  tn = 0
  n = 0
  total = 0
  total' = -0.1 * total
  E: at (time >= tn), t0=false: total = total + 10, n = n + 1 RESET
end
"""

#: the simulations of the start: from 0, after a presimulation, from a
#: negative start and with the steady state after the end (output times only)
STARTS: dict[str, dict] = {
    "start": {"end": 2},
    "presimulation": {"end": 2, "presimulation": SteadyState()},
    "negative start": {"start": -1, "end": 2},
    "steady state after the end": {"end": 2},
}

#: the cases of `STARTS` with output times and with the steps of the
#: integrator, the steady state after the end with output times only, it is
#: an output time
START_CASES = [
    (case, times)
    for case in STARTS
    for times in (True, False)
    if times or case != "steady state after the end"
]

#: the ids of `START_CASES`
START_IDS = [f"{case}-{'times' if times else 'steps'}" for case, times in START_CASES]


def _start_simulation(
    case: str, times: bool, preinit_changes: dict[str, float] | None = None
) -> Simulation:
    """Get a simulation of `STARTS` with output times or the steps of the integrator."""
    kwargs = dict(STARTS[case])
    if case == "steady state after the end":
        assert times, "the steady state after the end is an output time"
        kwargs["times"] = [0, 1, 2, float("inf")]
    elif times:
        start = kwargs.get("start", 0)
        kwargs["times"] = [start, *(t for t in (0, 1, 2) if t > start)]
    return Simulation(preinit_changes=preinit_changes or {}, **kwargs)


def _engine_run(engine: Engine, model: str, simulation: Simulation) -> ScanResult:
    """Run a simulation of an antimony model with an engine, serially."""
    simulator = Simulator(n_workers=1, engine=engine)
    return simulator.run(simulator.load(AbstractModel(source=sbml(model))), simulation)


def _first_row(result: ScanResult) -> dict[str, float]:
    """Get the state at the start, which must be one row."""
    time = result.ds["time"].values
    assert np.count_nonzero(time == time[0]) == 1
    return {name: float(result.ds[name].values[0]) for name in ("total", "n")}


#: every engine, roadrunner first
ENGINES = [Engine.ROADRUNNER, *OTHER_ENGINES]


def _engine(engine: Engine) -> Engine:
    """Skip a test of an engine whose extra is not installed."""
    if engine is not Engine.ROADRUNNER:
        require(engine)
    return engine


@pytest.mark.parametrize(("case", "times"), START_CASES, ids=START_IDS)
@pytest.mark.parametrize("reset", ["", ", dose = 0"], ids=["kept", "reset"])
@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_an_event_a_pre_initialization_makes_true_fires_at_the_start(
    engine: Engine, reset: str, case: str, times: bool
) -> None:
    """A dose of the pre-initialization fires at the start, its row has the state after it.

    Every engine fires it as raw libroadrunner after `resetAll` and the SBML
    Test Suite do, with a presimulation at its start, the time 0 (#439).
    """
    engine = _engine(engine)
    model = DOSE_EVENT.replace("RESET", reset)
    simulation = _start_simulation(case, times, {"dose": 10})
    res = _engine_run(engine, model, simulation)
    first = _first_row(res)
    assert first["n"] == 1.0
    assert res.ds["n"].values[-1] == 1.0
    time = res.ds["time"].values
    total = res.ds["total"].values
    if case == "presimulation":
        # the dose decays to the steady state before the start
        np.testing.assert_allclose(total, 0.0, atol=1e-6)
    else:
        finite = np.isfinite(time)
        expected = 10.0 * np.exp(-0.1 * (time[finite] - time[0]))
        assert first["total"] == 10.0
        np.testing.assert_allclose(total[finite], expected, rtol=1e-5)
        if not np.all(finite):
            assert total[-1] == pytest.approx(0.0, abs=1e-6)


#: a dose every `tau` from `tn`, of the amount `dose`
PERIODIC_DOSE = """
model periodic_dose
  tn = 0
  tau = 5
  dose = 100
  n = 0
  total = 0
  total' = -0.1 * total
  E: at (time >= tn), t0=false: total = total + dose, tn = tn + tau, n = n + 1
end
"""


@pytest.mark.parametrize("times", [True, False], ids=["times", "steps"])
@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_a_pre_initialization_moves_an_event_of_the_start(
    engine: Engine, times: bool
) -> None:
    """An event of the start fires with the values of the pre-initialization.

    roadrunner fired the events of the time 0 with the values as loaded
    before the pre-initialization was set, i.e. the first dose at 0 instead
    of at the moved time (#439).
    """
    res = _engine_run(
        _engine(engine), PERIODIC_DOSE, _start_simulation("start", times, {"tn": 1})
    )
    assert _first_row(res) == {"total": 0.0, "n": 0.0}
    time = res.ds["time"].values
    n = res.ds["n"].values
    assert n[-1] == 1.0
    # the rows of the event at 1 aside, which the engines place differently
    np.testing.assert_array_equal(n[time < 1 - 1e-6], 0.0)
    after = time > 1 + 1e-6
    np.testing.assert_allclose(
        res.ds["total"].values[after],
        100.0 * np.exp(-0.1 * (time[after] - 1)),
        rtol=1e-5,
    )


@pytest.mark.parametrize("engine", ENGINES, ids=str)
def test_an_event_of_the_start_assigns_the_values_of_the_pre_initialization(
    engine: Engine,
) -> None:
    """The dose of an event of the start is the one of the pre-initialization (#439)."""
    res = _engine_run(
        _engine(engine),
        PERIODIC_DOSE,
        _start_simulation("presimulation", True, {"dose": 0}),
    )
    np.testing.assert_array_equal(res.ds["n"].values, 1.0)
    np.testing.assert_allclose(res.ds["total"].values, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("reset", "case", "times"),
    [
        pytest.param(reset, case, times, id=f"{name}-{label}")
        for reset, name in (("", "kept"), (", tn = tn + 5", "moved"))
        for (case, times), label in zip(START_CASES, START_IDS, strict=True)
        # a dose every 5 time units has no steady state
        if not reset or case not in ("presimulation", "steady state after the end")
    ],
)
def test_an_event_at_the_start_agrees_with_roadrunner(
    other_engine: Engine, reset: str, case: str, times: bool
) -> None:
    """An event of the start without a pre-initialization has the state after it at the start, as in roadrunner.

    roadrunner evaluates the triggers at the time 0 also from a negative
    start, COPASI and AMICI at the start, so from a negative start the event
    fires when the time reaches 0, see the known differences.
    """
    model = TIME_EVENT.replace("RESET", reset)
    simulation = _start_simulation(case, times)
    res = _engine_run(other_engine, model, simulation)
    first = _first_row(res)
    time = res.ds["time"].values
    if case == "negative start":
        assert first == {"total": 0.0, "n": 0.0}
        assert res.ds["n"].values[-1] == 1.0
        # after the event at 0, whose state before it the steps have at 0 too
        finite = time > 0
        np.testing.assert_allclose(
            res.ds["total"].values[finite],
            10.0 * np.exp(-0.1 * time[finite]),
            rtol=1e-5,
        )
        return
    reference = roadrunner_simulator().run(
        RoadrunnerSBMLModel(source=sbml(model)), simulation
    )
    expected = _first_row(reference)
    assert first["n"] == expected["n"] == 1.0
    assert first["total"] == pytest.approx(expected["total"], rel=1e-6, abs=1e-9)
    for name in ("total", "n"):
        assert res.ds[name].values[-1] == pytest.approx(
            reference.ds[name].values[-1], rel=1e-6, abs=1e-9
        )
        if times:
            np.testing.assert_allclose(
                res.ds[name].values, reference.ds[name].values, rtol=1e-6, atol=1e-9
            )


@pytest.mark.parametrize(("case", "times"), START_CASES, ids=START_IDS)
def test_copasi_and_amici_agree_on_a_dose_of_the_pre_initialization(
    case: str, times: bool
) -> None:
    """COPASI and AMICI have the same states at the start and at the end of a dose of the pre-initialization.

    The two engines are named here and not read from `OTHER_ENGINES`, which
    has every engine besides roadrunner: this test compares exactly these two
    with each other, so a third engine of that list must neither skip it nor
    be unpacked into its two names.
    """
    pair = (Engine.COPASI, Engine.AMICI)
    for engine in pair:
        require(engine)
    simulation = _start_simulation(case, times, {"dose": 10})
    model = DOSE_EVENT.replace("RESET", ", dose = 0")
    copasi, amici = (_engine_run(engine, model, simulation) for engine in pair)
    first = _first_row(copasi), _first_row(amici)
    assert first[0]["n"] == first[1]["n"] == 1.0
    assert first[0]["total"] == pytest.approx(first[1]["total"], rel=1e-6, abs=1e-9)
    for name in ("total", "n", "dose"):
        values = copasi.ds[name].values, amici.ds[name].values
        assert values[0][-1] == pytest.approx(values[1][-1], rel=1e-6, abs=1e-9)
        if times:
            np.testing.assert_allclose(*values, rtol=1e-6, atol=1e-9)


#: a simulation with changes after the start and at the end, from a negative
#: start, with a time shift and the steady state after the end
LATER_TIME = Simulation(
    start=-1, end=3, changes=[Change([1.5, 3], {"k1": 2.0})], time_shift=10.0
)


@pytest.mark.parametrize("times", [None, [-1, 0, 1.5, 3, float("inf")]])
def test_a_later_time_column_is_the_time_of_the_rows(
    other_engine: Engine, times: list[float] | None
) -> None:
    """A selection `time` after the first column is the time of every row in every engine (#445)."""
    model = AbstractModel(source=sbml(), selections=["A", "time"])
    simulation = (
        LATER_TIME
        if times is None
        else Simulation(
            start=-1,
            end=3,
            changes=list(LATER_TIME.changes),
            time_shift=10.0,
            times=times,
        )
    )
    # the last of the given times is the steady state after the end, which the
    # engine core continues from the state the timecourse ended in
    results = {
        engine: Simulator(engine=engine).simulate(model, simulation)
        for engine in (Engine.ROADRUNNER, other_engine)
    }
    for result in results.values():
        assert result.columns == ("time", "A", "time")
        assert np.count_nonzero(result.time == 11.5) == 2
        np.testing.assert_array_equal(result.values[:, 2], result.time)
    if times is not None:
        np.testing.assert_array_equal(
            results[other_engine].time, results[Engine.ROADRUNNER].time
        )


#: an event of the time at `T` which counts its firings, `OPERATOR` the
#: comparison of its trigger
TIME_TRIGGER = """
model time_trigger
  X = 1
  X' = -0.1*X
  n = 0
  T = 1
  E: at (time OPERATOR T), t0=false: n = n + 1
end
"""

#: the start of the steady state of a simulation in roadrunner: a
#: presimulation from 0, and the steady state after the end of a simulation
#: from 0 to 1 from the resolution of the time after the end
HORIZON_STARTS = {"presimulation": 0.0, "steady state after the end": 1.0 + 1e-9}


@pytest.mark.parametrize("operator", [">", ">="])
@pytest.mark.parametrize("output", HORIZON_STARTS)
def test_a_trigger_of_the_time_at_the_horizons_of_a_steady_state_fires_once(
    other_engine: Engine, output: str, operator: str
) -> None:
    """A trigger of the time at or a little after the end of a horizon fires once in every engine (#444).

    The horizons of a steady state of roadrunner end at 1, 10 and 100 after
    its start.
    """
    model = AbstractModel(
        source=sbml(TIME_TRIGGER.replace("OPERATOR", operator)),
        selections=["time", "n", "X"],
    )
    for after in (0.0, 1e-3, 0.1):
        time = HORIZON_STARTS[output] + 10.0 + after
        if output == "presimulation":
            simulation = Simulation(
                end=0.5, presimulation=SteadyState(), preinit_changes={"T": time}
            )
        else:
            simulation = Simulation(
                end=1, times=[0, 1, float("inf")], preinit_changes={"T": time}
            )
        for engine in (Engine.ROADRUNNER, other_engine):
            simulator = Simulator(engine=engine)
            result = simulator.simulate(model, simulation)
            assert result["n"][-1] == 1.0, (engine, after)
            assert result["X"][-1] == pytest.approx(0.0, abs=1e-6)
