"""`Simulator(engine="core")` runs a simulation with sbmlsim-core.

Every test compares the engine with roadrunner, the reference of every engine,
on the same simulation, see `test_simulator_engines.py` for COPASI and AMICI;
the requests of the contract `simulation/1` a plan becomes are read as they
are, so the fields of the contract are pinned.
"""

from dataclasses import replace
from typing import Final

import numpy as np
import pytest

pytest.importorskip("sbmlsim_core")

from sbmlsim import parallel
from sbmlsim.engine import Engine, UnsupportedModelError, import_core
from sbmlsim.model import RoadrunnerSBMLModel, load_model
from sbmlsim.model import model_core as core_module
from sbmlsim.model.model_core import CoreSBMLModel
from sbmlsim.model.symbols import TargetKind
from sbmlsim.resources import HCTZ_SBML
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulation import (
    Change,
    Dimension,
    Scan,
    Simulation,
    SteadyState,
)
from sbmlsim.simulator import IntegratorSettings, Simulator, executor_core
from sbmlsim.simulator.executor import SteadyStateError
from sbmlsim.simulator.executor_core import (
    CONTRACT,
    INTEGRATOR_STEPS,
    STEADY_STATE_FLAG,
    _failed,
    request_of,
)
from sbmlsim.simulator.plan import Assignment, Plan, PlanEvent, compile_simulation
from sbmlsim.simulator.worker import ModelSpec, core_threads
from tests.simulator.models import (
    ALGEBRAIC,
    INDIRECT_VOLUME,
    PROBE,
    TOLERANCE_PROBE,
    roadrunner_simulator,
    sbml,
    sbml_pk,
)

#: a model whose parameter `tt` the test renames to `time`, an entity of the
#: name of the clock, as in case 01820 of the SBML Test Suite
ENTITY_TIME = """
model entity_time
  A = 1; k1 = 0.8
  J1: A -> ; k1*A
  tt = 10
end
"""

#: a parameter `p` which a relative change adds to, the parameter `q` another
#: formula reads and `X`, whose rate is `p`: the timecourse of `X` is the one
#: of the segments between the changes
RELATIVE = """
model relative
  p = 0
  q = 3
  X = 0
  X' = p
end
"""

#: the dose state of a model of pkdb_models, the parameter `PODOSE_ra` which a
#: repeated dose adds to and whose absorption fills the central compartment,
#: the smallest model of the protocol of issue #548
DOSE_STATE = """
model dose_state
  compartment Vc = 1
  species Ac in Vc = 0
  PODOSE_ra = 0
  ka = 0.1
  ke = 0.3
  PODOSE_ra' = -ka * PODOSE_ra
  Ac' = ka * PODOSE_ra - ke * Ac
end
"""

#: an event of the time which adds to the state `x`, the model of the
#: disagreement a formula at a change would hide: the executor of roadrunner
#: reads `x` before the event of the time of the change fires and a request of
#: the engine answers it after
EVENT_AT_FIVE = """
model event_at_five
  x = 0
  k = 0.1
  x' = -k*x + 0.01
  p = 0
  E1: at (time >= 5), t0=false: x = x + 10
end
"""

#: the tolerances at which the engines agree, see the section Engines of
#: `docs/simulation.md`
TIGHT: Final[IntegratorSettings] = {
    "absolute_tolerance": 1e-12,
    "relative_tolerance": 1e-10,
}

#: the timecourses the engines are compared on, by the id of their model: the
#: probe model of the simulation core and an oral dose of HCTZ, the reference
#: model of the package, over a day in its time unit, the minute
TIMECOURSES: Final[dict[str, Simulation]] = {
    "probe": Simulation(end=20, times=np.linspace(0, 20, 41).tolist()),
    "hctz": Simulation(
        end=24 * 60,
        preinit_changes={"PODOSE_hctz": 25.0},
        times=np.linspace(0, 24 * 60, 49).tolist(),
    ),
}

#: the largest error of a column relative to its largest magnitude at which
#: the engines agree, see `_compare`: 180 times the worst error measured there,
#: so an accuracy regression of the engine does not pass
TOLERANCE: Final[float] = 1e-8


def _integration(message: str, flag: int) -> Exception:
    """Get the `IntegrationError` of the bindings of a message and a flag of CVODE."""
    error = import_core().IntegrationError(message)
    error.flag = flag
    error.time = 0.0
    return error


@pytest.fixture(scope="module")
def probe_sbml() -> str:
    """Get the SBML of the probe model, which every engine simulates."""
    return sbml(PROBE)


@pytest.fixture(scope="module")
def sources(probe_sbml: str) -> dict[str, str]:
    """Get the SBML of the models the pre-initialization is measured on."""
    return {
        "probe": probe_sbml,
        "tolerances": sbml(TOLERANCE_PROBE),
        "volume": sbml(INDIRECT_VOLUME),
    }


def _orders(values: dict[str, float]) -> list[dict[str, float]]:
    """Get a mapping of values in both orders, the one it is written in and the reverse.

    The engine applies the values of a pre-initialization one after the other,
    so a pair of them is measured in both orders, which roadrunner answers
    alike.
    """
    if len(values) < 2:
        return [values]
    return [values, dict(reversed(list(values.items())))]


@pytest.fixture
def model_core(probe_sbml: str) -> CoreSBMLModel:
    """Get the probe model in sbmlsim-core."""
    model = load_model(probe_sbml, Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    return model


@pytest.fixture
def model_rr(probe_sbml: str) -> RoadrunnerSBMLModel:
    """Get the probe model in roadrunner, the reference of the engine."""
    model = load_model(probe_sbml, Engine.ROADRUNNER)
    assert isinstance(model, RoadrunnerSBMLModel)
    return model


def _compare(core: TimecourseResult, roadrunner: TimecourseResult) -> None:
    """Compare the output of sbmlsim-core with the one of roadrunner.

    The error of a column is relative to its largest magnitude, as the tests
    of COPASI and AMICI measure it (`_scaled_error` of
    `test_simulator_engines.py`): the absolute tolerance of the integrator
    bounds the error of a variable and not its relative error, so a value
    which passes through a tiny magnitude, e.g. a concentration right after a
    dose, differs by more relative to itself. At the tolerances of `TIGHT` the
    engines agree to 5.7e-11 of the scale of a column on these simulations,
    measured on the probe model (6.8e-12) and on the dose of HCTZ (5.6e-11).
    """
    assert core.columns == roadrunner.columns
    assert core.values.shape == roadrunner.values.shape
    # a `NaN` of a point which did not integrate is no error of the metric
    # below, which skips it, so the two outputs are finite in the same places
    np.testing.assert_array_equal(
        np.isfinite(core.values), np.isfinite(roadrunner.values)
    )
    difference = np.abs(core.values - roadrunner.values)
    scale = np.maximum(np.nanmax(np.abs(roadrunner.values), axis=0), 1e-12)
    errors = np.nanmax(difference / scale, axis=0)
    worst = int(np.argmax(errors))
    assert errors[worst] <= TOLERANCE, (
        f"the column '{core.columns[worst]}' differs by {errors[worst]:.3g} of its "
        f"largest magnitude"
    )


@pytest.mark.parametrize("model_id", TIMECOURSES)
def test_a_timecourse_matches_roadrunner(probe_sbml: str, model_id: str) -> None:
    """The timecourse of the engine is the one of roadrunner at tight tolerances."""
    model = probe_sbml if model_id == "probe" else HCTZ_SBML
    simulation = TIMECOURSES[model_id]
    core = Simulator(engine="core", **TIGHT).simulate(model, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model, simulation)
    _compare(core, roadrunner)


def test_a_change_gives_two_rows_at_its_time(model_core: CoreSBMLModel) -> None:
    """A change after the start has the state before it and the state after it."""
    result = Simulator(engine="core").simulate(
        model_core, Simulation(end=20, changes=[Change(10, {"X": 5.0})], steps=20)
    )
    times = result.values[:, 0]
    assert (times == 10.0).sum() == 2
    before, after = result["X"][times == 10.0]
    assert before != after
    assert after == 5.0


def test_the_changes_of_a_plan_match_roadrunner(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """The rows of changes at the start, after it and at the end are roadrunner's."""
    simulation = Simulation(
        end=20,
        changes=[Change(0, {"k1": 0.5}), Change([5, 20], {"X": 3.0})],
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    _compare(core, roadrunner)


def test_the_steps_of_the_integrator_are_the_output_without_times(
    model_core: CoreSBMLModel,
) -> None:
    """A simulation without times or steps is the output of the integrator."""
    result = Simulator(engine="core").simulate(model_core, Simulation(end=20))
    assert result.values.shape[0] > 21
    times = result.values[:, 0]
    assert times[0] == 0.0
    assert times[-1] == 20.0
    assert np.all(np.diff(times) >= 0.0)


def test_a_steady_state_before_the_start(model_core: CoreSBMLModel) -> None:
    """A presimulation integrates to the steady state the simulation starts from."""
    result = Simulator(engine="core").simulate(
        model_core, Simulation(end=1, presimulation=SteadyState(), steps=1)
    )
    assert np.isfinite(result.values).all()
    # the net rate of `A` vanishes in the steady state, up to the criterion of
    # the presimulation (3.0e-11 relative here, bounded by `1e-8 + 1e-6 |A|`)
    assert result["J1"][0] == pytest.approx(result["J2"][0], rel=1e-5)


def test_a_steady_state_matches_roadrunner(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """The state a presimulation starts the simulation from is roadrunner's."""
    simulation = Simulation(
        end=5,
        presimulation=SteadyState(preinit_changes={"a0": 2.0}),
        times=np.linspace(0, 5, 11).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    _compare(core, roadrunner)


def test_an_integration_which_fails_raises_as_in_roadrunner(
    probe_sbml: str, capfd: pytest.CaptureFixture[str]
) -> None:
    """A simulation the engine fails is a `RuntimeError` and no error of the bindings."""
    simulation = Simulation(end=1000, steps=2)
    with pytest.raises(RuntimeError, match="failed to simulate the model") as failure:
        Simulator(engine="core", maximum_num_steps=1).simulate(probe_sbml, simulation)
    # the error of an executor and not the `IntegrationError` of the bindings,
    # with the message of the engine and the flag of CVODE, here -1
    assert type(failure.value) is RuntimeError
    assert "(flag -1)" in str(failure.value)
    # the engine writes nothing of its own, its message carries the failure
    assert capfd.readouterr() == ("", "")
    # roadrunner reports a failed integration the same way, the contract of
    # `execute_plan`, and logs the error of CVODE on its own console
    with pytest.raises(RuntimeError):
        Simulator(maximum_num_steps=1).simulate(probe_sbml, simulation)


def test_the_error_of_a_simulation_is_the_error_of_its_cause(
    model_core: CoreSBMLModel,
) -> None:
    """Every error of a simulation of the engine maps onto the one of its cause.

    The four errors of the bindings map onto the three a run tells apart, see
    `_failed`: a feature the engine does not support is raised for the whole
    scan (`UnsupportedModelError` is a `NotImplementedError`), a presimulation
    without a steady state is a `SteadyStateError` and everything else is a
    point a run flags.
    """
    core = import_core()
    cases: list[tuple[Exception, type[Exception], str]] = [
        (core.UnsupportedError("a fast reaction"), UnsupportedModelError, "fast"),
        (
            _integration("no steady state", STEADY_STATE_FLAG),
            SteadyStateError,
            "steady",
        ),
        (_integration("mxstep steps taken", -1), RuntimeError, "mxstep"),
        (core.RequestError("a value which is not finite"), RuntimeError, "finite"),
        (core.InternalError("the model trapped"), RuntimeError, "internally"),
    ]
    for error, expected, match in cases:
        failed = _failed(error, model_core, core)
        assert type(failed) is expected, error
        assert match in str(failed)
        assert model_core.label in str(failed)
    # a `SteadyStateError` is a `RuntimeError`, the error of a failed point,
    # while an unsupported feature is a `NotImplementedError`, which a run
    # raises instead of flagging the point, see `worker.run_chunk`
    assert issubclass(UnsupportedModelError, NotImplementedError)


def test_a_value_which_is_not_finite_is_a_point_a_run_flags() -> None:
    """A request the engine refuses is a `RuntimeError`, the point roadrunner flags too.

    A `NaN` of a dimension reaches the engine as a request it refuses; with
    roadrunner the point fails its integration (`CV_CONV_FAILURE`). Both are a
    point a run flags, so a user sees the same of both engines, see `_failed`.
    """
    scan = Scan(
        Simulation(end=10, steps=5),
        dimensions=[Dimension("dose", values={"PODOSE_hctz": [10.0, float("nan")]})],
    )
    result = Simulator(engine="core", n_workers=1).run(HCTZ_SBML, scan, on_error="flag")
    assert result["status"].values.tolist() == [0, 1]
    (error,) = result.ds.attrs["errors"]
    assert "RuntimeError" in error
    assert np.isfinite(result["[Cve_hctz]"].values[0]).all()
    assert np.isnan(result["[Cve_hctz]"].values[1]).all()


def test_a_presimulation_without_a_steady_state_raises(probe_sbml: str) -> None:
    """A presimulation which runs out of time is a `SteadyStateError`, as elsewhere."""
    with pytest.raises(SteadyStateError, match="steady state"):
        Simulator(engine="core").simulate(
            probe_sbml,
            Simulation(end=1, presimulation=SteadyState(max_time=0.1), steps=1),
        )


def test_the_pre_initialization_matches_roadrunner(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """The values before the initialization are set and the initial assignments see them."""
    simulation = Simulation(end=10, preinit_changes={"f": 3.0, "a0": 2.0}, steps=10)
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    # `pinit` follows `f` by an initial assignment and `X` follows `pinit`
    assert core["X"][0] == pytest.approx(18.0)
    _compare(core, roadrunner)


def test_a_formula_at_a_change_of_a_symbol_the_model_has_not_is_refused(
    model_core: CoreSBMLModel,
) -> None:
    """A formula at a change which reads no output of the engine is refused.

    `compile_simulation` refuses a formula of a symbol which is neither the
    time nor an entity of the model, so only a plan built by hand carries one;
    the executor refuses it as well, since a request of the engine carries
    numbers and the engine answers no value of such a symbol.
    """
    plan = compile_simulation(
        Simulation(end=10, times=[0.0, 10.0]), model_core.symbols, model_core.uinfo
    )
    event = PlanEvent(
        5.0, (Assignment("X", TargetKind.SPECIES_AMOUNT, formula="2 * nothing"),)
    )
    with pytest.raises(UnsupportedModelError, match="formula") as failure:
        request_of(replace(plan, events=(event,)), model_core, ["time", "X"])
    message = str(failure.value)
    # the target, the time and the formula, and what a request takes
    assert "'X'" in message
    assert "2 * nothing" in message
    assert "5.0" in message
    assert "'nothing'" in message
    assert "is a number" in message


def test_a_relative_change_matches_roadrunner() -> None:
    """A change which adds to its target reads the state at its time, as roadrunner.

    `p` is 0, 2 and 4 over the three segments of the simulation and `X`, whose
    rate is `p`, follows them; the extra outputs the executor requests of the
    engine are its own and no column of the result.
    """
    source = sbml(RELATIVE)
    simulation = Simulation(
        end=30,
        changes=[Change([10, 20], {"p": "p + 2"})],
        times=np.linspace(0, 30, 31).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    assert core.columns == roadrunner.columns
    times = core.values[:, 0]
    assert core["p"][times == 5.0].tolist() == [0.0]
    assert core["p"][times == 15.0].tolist() == [2.0]
    assert core["p"][times == 25.0].tolist() == [4.0]
    # `X` integrates `p`: 0 until 10, then 2 per unit of time, then 4
    assert core["X"][times == 30.0][-1] == pytest.approx(60.0)
    _compare(core, roadrunner)


def test_a_formula_at_a_change_reads_another_parameter() -> None:
    """A formula of a change reads a parameter which is not its target."""
    source = sbml(RELATIVE)
    simulation = Simulation(
        end=20,
        changes=[Change(10, {"p": "2 * q + 1"})],
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    assert core["p"][core.values[:, 0] == 15.0].tolist() == [7.0]
    _compare(core, roadrunner)


def test_a_formula_at_a_change_at_the_start_reads_its_target() -> None:
    """A change at the start folds with the values after the initialization."""
    source = sbml(RELATIVE)
    simulation = Simulation(
        end=10,
        changes=[Change(0, {"q": "q + 2", "p": "q + 1"})],
        times=np.linspace(0, 10, 11).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    # the values of a change are evaluated before any of them is set, so `p`
    # reads `q` as it is after the initialization, 3
    assert core["q"][0] == 5.0
    assert core["p"][0] == 4.0
    _compare(core, roadrunner)


@pytest.mark.parametrize("symbol", ["A", "[A]"], ids=["an amount", "a concentration"])
def test_a_formula_at_a_change_reads_a_species(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel, symbol: str
) -> None:
    """A formula which reads the amount or the concentration of a species is folded.

    The outputs of the engine are the selections of roadrunner, `A` the amount
    of a species and `[A]` its concentration, which is what `apply_changes`
    reads of the model of roadrunner: both are folded with the state at the
    time of the change. The compartment of the probe model has the size 2, so
    the two differ by that factor and the value of the change says which of
    them was read.
    """
    simulation = Simulation(
        end=20,
        changes=[Change(5, {"X": f"{symbol} + 1"})],
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    at_five = core.values[:, 0] == 5.0
    read = core[symbol][at_five][0]
    other = core["[A]" if symbol == "A" else "A"][at_five][0]
    # the value of the change is the symbol it reads and not the other one, up
    # to the error of the integration of the segment it is folded with
    assert core["X"][at_five][-1] == pytest.approx(read + 1.0, rel=1e-9)
    assert read == pytest.approx(2.0 * other if symbol == "A" else other / 2.0)
    _compare(core, roadrunner)


def test_a_repeated_dose_of_a_dose_state_matches_roadrunner() -> None:
    """Two doses a day apart on a dose state run and match roadrunner, see #548.

    The protocol of every multiple-dose experiment of pkdb_models: the change
    `PODOSE_ra: PODOSE_ra + 2.0` adds to the dose state instead of setting it,
    so a protocol does not have to know what the state holds.
    """
    source = sbml(DOSE_STATE)
    simulation = Simulation(
        end=48,
        changes=[Change([0, 24], {"PODOSE_ra": "PODOSE_ra + 2.0"})],
        times=np.linspace(0, 48, 49).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    times = core.values[:, 0]
    # the first dose is the whole state, the second one adds to what is left
    assert core["PODOSE_ra"][times == 0.0][-1] == 2.0
    rest = core["PODOSE_ra"][times == 24.0]
    assert rest[1] == pytest.approx(rest[0] + 2.0)
    assert rest[0] == pytest.approx(2.0 * np.exp(-0.1 * 24.0))
    _compare(core, roadrunner)


def test_ten_relative_doses_match_roadrunner() -> None:
    """Ten doses a day apart match roadrunner, a protocol towards a steady state.

    Every segment of the fold resumes from the state of the one before it, so a
    protocol of many doses costs one pass over the simulation and not one from
    the start per dose, see `_state_at` of the executor.
    """
    source = sbml(DOSE_STATE)
    doses = [24.0 * k for k in range(10)]
    simulation = Simulation(
        end=240,
        changes=[Change(doses, {"PODOSE_ra": "PODOSE_ra + 2.0"})],
        times=np.linspace(0, 240, 121).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    times = core.values[:, 0]
    # the doses accumulate towards the steady state of the protocol
    first = core["PODOSE_ra"][times == 24.0]
    last = core["PODOSE_ra"][times == 216.0]
    assert first[1] == pytest.approx(first[0] + 2.0)
    assert last[1] == pytest.approx(last[0] + 2.0)
    assert last[1] > first[1]
    _compare(core, roadrunner)


def test_the_fold_of_a_state_the_executor_cannot_set_matches_roadrunner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model whose state the executor cannot hand to the engine folds from the start.

    Every segment then starts at the start of the simulation with the
    pre-initialization and the presimulation of the plan, which is correct and
    costs one request from the start per change, see `_state_targets`: no
    packaged model has such a state, so the path is measured here with the
    targets taken away.
    """
    source = sbml(DOSE_STATE)
    monkeypatch.setattr(executor_core, "_state_targets", lambda model: None)
    simulation = Simulation(
        end=72,
        changes=[Change([0, 24, 48], {"PODOSE_ra": "PODOSE_ra + 2.0"})],
        times=np.linspace(0, 72, 37).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    rest = core["PODOSE_ra"][core.values[:, 0] == 48.0]
    assert rest[1] == pytest.approx(rest[0] + 2.0)
    _compare(core, roadrunner)


def test_a_relative_change_on_the_steps_of_the_integrator_matches_roadrunner() -> None:
    """A simulation without times or steps folds on the steps of the integrator.

    Every simulation of pkdb_models takes this path, a `Simulation` without
    `steps`: the segment of a change is then the steps of the integrator up to
    the change, whose last row is the state there.
    """
    source = sbml(RELATIVE)
    simulation = Simulation(end=30, changes=[Change([10, 20], {"p": "p + 2"})])
    model = load_model(source, Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    # the output of the request and of every segment of the fold is the steps
    # of the integrator and no grid of times
    assert request_of(plan, model, ["time", "p"])["output"] == INTEGRATOR_STEPS
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    times = core.values[:, 0]
    assert core["p"][times == 10.0].tolist() == [0.0, 2.0]
    assert core["p"][times == 20.0].tolist() == [2.0, 4.0]
    # `X` integrates `p`: 0 until 10, then 2 per unit of time, then 4
    assert core["X"][-1] == pytest.approx(60.0)
    assert core.columns == roadrunner.columns
    np.testing.assert_allclose(
        core.values[-1], roadrunner.values[-1], rtol=1e-6, atol=1e-9
    )


def test_a_change_which_reads_the_state_of_a_model_with_an_event_folds_it() -> None:
    """A formula at a change reads the state before the events of its time.

    The model `EVENT_AT_FIVE` adds 10 to `x` at the time 5, which is also the
    time of the change: the executor of roadrunner evaluates the triggers and
    applies the change before it fires the events of that time, so the formula
    reads about 0.0067 and not 10.0067. The probe of the engine asks for the
    same state with `before_events` of `simulation/1` (sbmlsim-core#19), where
    it ended one unit in the last place before the time of the change and
    refused such a formula before 0.18.0.
    """
    source = sbml(EVENT_AT_FIVE)
    simulation = Simulation(
        end=10, changes=[Change(5, {"p": "x"})], times=[0.0, 5.0, 10.0]
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    folded = core["p"][core.values[:, 0] == 5.0][-1]
    # the state before the event of the time, which the event raises by 10
    assert folded < 1.0
    assert folded == pytest.approx(
        roadrunner["p"][roadrunner.values[:, 0] == 5.0][-1], rel=1e-6
    )
    # the event of the time fired, the change did not take its state
    assert core["x"][-1] == pytest.approx(roadrunner["x"][-1], rel=1e-6)


def test_a_change_of_numbers_runs_on_a_model_with_an_event() -> None:
    """A model with an event and a change of numbers is no refusal."""
    source = sbml(EVENT_AT_FIVE)
    simulation = Simulation(
        end=10,
        changes=[Change(3, {"p": 1.0})],
        times=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    times = core.values[:, 0]
    assert core["p"][-1] == 1.0
    # the event of the time fired at 5, which added 10 to `x`
    assert core["x"][times == 4.0][-1] < 1.0
    assert core["x"][times == 6.0][-1] > 9.0
    _compare(core, roadrunner)


def test_a_relative_change_after_a_presimulation_matches_roadrunner(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """A change is folded with the state the presimulation starts the simulation in.

    The segment of a change carries the pre-equilibration of the plan, so the
    state of a change at the start is the one of the steady state and not the
    one of the initialization, as in roadrunner.
    """
    simulation = Simulation(
        end=10,
        presimulation=SteadyState(preinit_changes={"a0": 2.0}),
        changes=[Change([0, 5], {"k1": "k1 + 0.1"})],
        times=np.linspace(0, 10, 11).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    times = core.values[:, 0]
    assert core["k1"][times == 0.0][-1] == pytest.approx(0.9)
    assert core["k1"][times == 5.0][-1] == pytest.approx(1.0)
    _compare(core, roadrunner)


def test_a_scan_of_a_relative_change_matches_roadrunner() -> None:
    """The points of a scan of a relative change are folded, each with its own state.

    A scan of the engine builds the request of every point of a chunk, see
    `sbmlsim.simulator.worker._run_chunk_core`, so the folding of a change
    runs on the plan of a point and not only on a single simulation.
    """
    source = sbml(DOSE_STATE)
    scan = Scan(
        Simulation(
            end=48,
            changes=[Change([0, 24], {"PODOSE_ra": "PODOSE_ra + 2.0"})],
            times=np.linspace(0, 48, 13).tolist(),
        ),
        dimensions=[Dimension("absorption", values={"ka": [0.05, 0.1, 0.2]})],
    )
    core = Simulator(engine="core", n_workers=1, **TIGHT).run(source, scan)
    roadrunner = Simulator(n_workers=1, **TIGHT).run(source, scan)
    # the dose state of every point is its own, the absorption of the point
    assert core["PODOSE_ra"].values[:, 0].tolist() == [2.0, 2.0, 2.0]
    assert len(set(core["PODOSE_ra"].values[:, -1].tolist())) == 3
    np.testing.assert_allclose(
        core["Ac"].values, roadrunner["Ac"].values, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize(
    "formula", ["2 * time", "5 + 5"], ids=["of the time", "of numbers"]
)
def test_a_formula_of_the_time_at_a_change_is_folded(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel, formula: str
) -> None:
    """A formula of the time and of numbers at a change is folded into its value.

    The time of a change is the one of the plan, so the value is the one the
    executor of roadrunner evaluates on the model at that time.
    """
    simulation = Simulation(
        end=20,
        changes=[Change(5, {"X": formula})],
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert core["X"][core.values[:, 0] == 5.0][-1] == 10.0
    _compare(core, roadrunner)


def test_a_pre_initialization_formula_is_folded_against_the_model_as_loaded(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """A pre-initialization of formulas alone reads the model as loaded.

    No value is set before the formulas are evaluated, so every symbol is the
    value of the model as loaded, which is what roadrunner reads: `kk` is the
    assignment rule `3*f`, 6 in the model as loaded.
    """
    simulation = Simulation(
        end=20,
        preinit_changes={"k1": "0.5 * kk"},
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert core["k1"][0] == 3.0
    _compare(core, roadrunner)


def test_a_pre_initialization_formula_of_the_values_it_sets_is_folded(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """A formula whose symbols the same pre-initialization sets reads those numbers."""
    simulation = Simulation(
        end=20,
        preinit_changes={"f": 3.0, "k1": "0.1 * f"},
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert core["k1"][0] == 0.1 * 3.0
    _compare(core, roadrunner)


def test_a_formula_of_a_steady_state_reads_the_whole_pre_initialization(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """The pre-initialization of a plan and of its steady state are one.

    Every executor applies them together (`preinit_targets`), so a formula of
    the steady state reads the numbers the plan sets as well.
    """
    simulation = Simulation(
        end=5,
        preinit_changes={"a0": 2.0},
        presimulation=SteadyState(preinit_changes={"k1": "0.1 * a0"}),
        times=np.linspace(0, 5, 11).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert core["k1"][0] == 0.1 * 2.0
    _compare(core, roadrunner)


def test_a_pre_initialization_formula_of_a_value_which_is_not_set_is_refused(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """A pre-initialization of values and a formula of another symbol is refused.

    `kk` is the assignment rule `3*f`, which roadrunner evaluates with the `f`
    the pre-initialization sets, `0.1 * 9`, and not with the 6 of the model as
    loaded, `0.1 * 6`: the engine refuses such a formula rather than folding
    the wrong value, since it cannot set the values of the pre-initialization
    on the model before it evaluates them.
    """
    simulation = Simulation(
        end=20, preinit_changes={"f": 3.0, "k1": "0.1 * kk"}, times=[0.0, 20.0]
    )
    with pytest.raises(UnsupportedModelError, match="formula") as failure:
        Simulator(engine="core").simulate(model_core, simulation)
    assert "'kk'" in str(failure.value)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert roadrunner["k1"][0] == 0.1 * 9.0
    assert roadrunner["k1"][0] != 0.1 * 6.0


@pytest.mark.parametrize("target", ["C", "X"], ids=["a compartment", "a species"])
def test_a_pre_initialization_formula_of_a_compartment_or_a_species_is_folded(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel, target: str
) -> None:
    """A formula of any target is folded, not only one of a parameter (#607).

    The fold gives the number the formula has, and the engine applies that
    number to the target with the bookkeeping of the volumes and the amounts
    of its initialization, exactly as it applies a number the
    pre-initialization states for the same target, so the answer is
    roadrunner's. A PEtab v2 condition which estimates a concentration needs
    it.
    """
    simulation = Simulation(
        end=10, preinit_changes={target: "2 * f"}, times=[0.0, 10.0]
    )

    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)

    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    for column in (target, "A", "X", "C"):
        assert core[column] == pytest.approx(roadrunner[column], rel=1e-6), column
    # the number of the formula reached the target and the model as loaded has
    # another one, so the fold is what the test sees
    plain = Simulation(end=10, preinit_changes={target: 2 * 2.0}, times=[0.0, 10.0])
    assert core[target][0] == pytest.approx(
        roadrunner_simulator(**TIGHT).simulate(model_rr, plain)[target][0], rel=1e-12
    )
    assert core[target][0] != pytest.approx(
        roadrunner_simulator(**TIGHT).simulate(
            model_rr, Simulation(end=10, times=[0.0, 10.0])
        )[target][0]
    )


def test_a_steady_state_after_the_end_continues_from_the_end(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """The steady state after the end is a second request from the state of the end.

    The `steady_state` of a request is the pre-equilibration before the start,
    so one request cannot continue from the state the timecourse ended in: the
    executor carries that state into the pre-initialization of a second
    request whose steady state is the criterion of the plan, and its row is
    the one of roadrunner at the output time `inf`.
    """
    simulation = Simulation(end=10, times=[0.0, 10.0, float("inf")])
    core = Simulator(engine="core").simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert core.values[-1, 0] == float("inf")
    assert roadrunner.values[-1, 0] == float("inf")
    assert core.columns == roadrunner.columns
    # the row of the steady state, which neither integrator carries an error
    # of the timecourse into: both equilibrate from the state of the end
    np.testing.assert_allclose(
        core.values[-1, 1:], roadrunner.values[-1, 1:], rtol=1e-6, atol=1e-9
    )


#: a model which relaxes to `target`, so the steady state after the end is
#: the value of a parameter and differs from point to point in a scan
RELAXATION: Final[str] = """
model relaxation
  k = 0.5; target = 7; x = 0
  x' = k*(target - x)
end
"""

#: `RELAXATION` with an event which raises its target once on the way there,
#: so the steady state after the end is 17 when the event fired once and 27
#: when it fired again: a model whose steady state after the end says whether
#: the triggers of the timecourse were carried into it
RELAXATION_EVENT: Final[str] = """
model relaxation_event
  k = 0.5; target = 7; x = 0
  x' = k*(target - x)
  E1: at (x >= 3), t0=false: target = target + 10
end
"""


def test_the_steady_state_of_a_point_of_a_scan_is_its_own() -> None:
    """Every point of a scan continues its steady state from its own end.

    The requests of the steady states of a chunk are simulated as one batch,
    as its folds are, so the state of every point must be carried into its
    own request: the model relaxes to `target`, which the scan sets, so a
    point which read the state of another would answer the target of that
    one. The values are the ones of roadrunner.
    """
    source = sbml(RELAXATION)
    simulation = Simulation(start=0, end=2, times=[0.0, 1.0, 2.0, float("inf")])
    targets = np.array([3.0, 7.0, 11.0])
    scan = Scan(simulation, [Dimension(id="d", values={"target": targets})])
    core = Simulator(engine="core", n_workers=1).run(source, scan)
    roadrunner = Simulator(engine="roadrunner", n_workers=1, **TIGHT).run(source, scan)
    # the last time of every point is the steady state, the target it relaxes to
    np.testing.assert_allclose(np.asarray(core.ds["x"])[:, -1], targets, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(core.ds["x"]), np.asarray(roadrunner.ds["x"]), rtol=1e-6, atol=1e-9
    )


def test_a_steady_state_after_the_end_of_a_model_with_events_keeps_their_triggers() -> (
    None
):
    """A model with events continues its steady state with the triggers as it left them.

    `steady_state_after` of `simulation/1` continues from the state the
    timecourse of the same request ended in, so an event which fired during
    the timecourse does not fire again at the same threshold
    (sbmlsim-core#60): the event of `EVENT_AT_FIVE` adds 10 to `x` at the time
    5, once, which is what roadrunner does when it continues the model of its
    timecourse. A second request which carried that state in would initialize
    the model and fire it again, which answered 2500 against roadrunner's 1500
    on case 0023 of the PEtab test suite, whose event adds 1000 to its state
    at the threshold its steady state is past.
    """
    source = sbml(RELAXATION_EVENT)
    simulation = Simulation(end=2, times=[0.0, 2.0, float("inf")])
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    # the event fired once, during the timecourse, so the target is 17 and not 27
    assert core["target"][-1] == 17.0
    assert core["x"][-1] == pytest.approx(17.0, rel=1e-6)
    assert core["x"][-1] == pytest.approx(roadrunner["x"][-1], rel=1e-6, abs=1e-9)


def test_a_pre_initialization_formula_of_a_target_set_by_a_formula_reads_it_as_loaded(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """A symbol the same pre-initialization sets by a formula is its value as loaded.

    Every executor evaluates the formulas of a pre-initialization against the
    model before it sets the result of any of them, so the `k1` of `2 * k1` is
    the 0.8 of the model as loaded and not the 6 the same pre-initialization
    sets it to. Only an assignment of a number is a known value of the fold
    (`_plain_values`), which is what keeps the two apart.
    """
    simulation = Simulation(
        end=20,
        preinit_changes={"k2": "2 * k1", "k1": "3 * f"},
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert core["k1"][0] == 3.0 * 2.0
    assert core["k2"][0] == 2.0 * 0.8
    _compare(core, roadrunner)


def test_a_pre_initialization_formula_of_a_target_set_by_a_formula_and_a_number(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel
) -> None:
    """One number of the same pre-initialization does not change the fold.

    `k1` is a constant parameter of the model, the variable of no rule and of
    no initial assignment, so nothing the pre-initialization sets can move it
    and the value of the model as loaded is the one roadrunner reads there
    too, whatever else is set (`executor_core._constant_parameter`). The
    number of `a0` is set before the formulas are evaluated and reaches none
    of them, so `k2` is `2 * 0.8` here as it is without it.
    """
    simulation = Simulation(
        end=20,
        preinit_changes={"k2": "2 * k1", "k1": "3 * f", "a0": 2.0},
        times=np.linspace(0, 20, 21).tolist(),
    )
    core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
    assert core["k2"][0] == 2.0 * 0.8
    _compare(core, roadrunner)


@pytest.mark.parametrize(
    ("model_id", "preinit"),
    [
        ("probe", {"[A]": 5.0, "C": 3.0}),
        ("volume", {"[A]": 5.0, "f": 3.0}),
        ("volume", {"[A]": 5.0, "g": 3.0}),
        ("volume", {"[N]": 5.0, "f": 3.0}),
        ("probe", {"a0": 2.0, "C": 3.0}),
        ("probe", {"[A]": 5.0, "k1": 0.5}),
        ("probe", {"C": 3.0}),
        ("probe", {"A": 5.0, "C": 3.0}),
        ("probe", {"X": 5.0, "C": 3.0}),
        ("tolerances", {"[A]": 2.0, "U": 2.0}),
        ("volume", {"A": 5.0, "f": 3.0}),
        ("volume", {"N": 5.0, "f": 3.0}),
        ("volume", {"A": 5.0, "g": 3.0}),
        ("volume", {"f": 3.0}),
        ("volume", {"g": 3.0}),
    ],
    ids=[
        "a concentration and its compartment",
        "a concentration and a parameter of its size",
        "a concentration and a parameter its size follows",
        "a substance species by its concentration",
        "a parameter and a compartment",
        "a concentration and a parameter",
        "a compartment alone",
        "an amount and its compartment",
        "a substance species and its compartment",
        "a concentration of another compartment",
        "an amount of a changed volume",
        "a substance species of a changed volume",
        "an amount two steps from a changed volume",
        "a changed volume alone",
        "a volume changed two steps away alone",
    ],
)
def test_a_pre_initialization_of_a_changed_volume_is_the_one_of_roadrunner(
    sources: dict[str, str], model_id: str, preinit: dict[str, float]
) -> None:
    """Every pre-initialization of a changed volume is the one of roadrunner.

    Measured here in either order: a concentration, which sbmlsim-core holds
    across the initialization since 0.7.0 and converts with the size its
    compartment has once the model is initialized, as roadrunner does (PEtab
    v2, sbmlsim-core#7); an amount, which is set as it is whenever the volume
    settles, the amount of a substance species among them; and a compartment
    whose size the pre-initialization changes without a species in it, whose
    concentrations the engine carries across the initialization as well.
    """
    source = sources[model_id]
    for values in _orders(preinit):
        simulation = Simulation(end=10, preinit_changes=values, times=[0.0, 10.0])
        core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
        roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
        _compare(core, roadrunner)


@pytest.mark.parametrize("target", ["[A]", "A"], ids=["a concentration", "an amount"])
def test_a_change_of_a_species_and_its_compartment_is_simultaneous(
    model_core: CoreSBMLModel, model_rr: RoadrunnerSBMLModel, target: str
) -> None:
    """A change of a request is simultaneous, unlike its pre-initialization.

    The engine applies the values of a change at once, as the executor of
    roadrunner does, so a change of a species and of its compartment is the
    one of roadrunner in either order and nothing is refused there.
    """
    for values in _orders({target: 5.0, "C": 3.0}):
        simulation = Simulation(
            end=10, changes=[Change(5, values)], times=[0.0, 5.0, 10.0]
        )
        core = Simulator(engine="core", **TIGHT).simulate(model_core, simulation)
        roadrunner = roadrunner_simulator(**TIGHT).simulate(model_rr, simulation)
        _compare(core, roadrunner)


def test_the_time_of_the_output_is_the_time_of_the_simulation(
    model_core: CoreSBMLModel,
) -> None:
    """The time is shifted and a later column `time` is a copy of it, see `output_time`."""
    model_core.set_selections(["time", "A", "time"])
    result = Simulator(engine="core").simulate(
        model_core, Simulation(end=4, steps=4, time_shift=100)
    )
    np.testing.assert_array_equal(
        result.values[:, 0], [100.0, 101.0, 102.0, 103.0, 104.0]
    )
    np.testing.assert_array_equal(result.values[:, 2], result.values[:, 0])


def test_a_later_time_column_of_an_entity_time_is_its_value() -> None:
    """A selection `time` of a model with an entity `time` is its value (case 01820)."""
    source = sbml(ENTITY_TIME).replace('id="tt"', 'id="time"')
    simulation = Simulation(end=3, times=[0, 1.5, 3])
    results = []
    for engine in ("core", "roadrunner"):
        model = Simulator(engine=engine).load(source)
        model.set_selections(["time", "time", "A"])
        results.append(Simulator(engine=engine, **TIGHT).simulate(model, simulation))
    core, roadrunner = results
    np.testing.assert_array_equal(core.values[:, 1], 10.0)
    _compare(core, roadrunner)


def test_a_selection_the_model_cannot_write_is_reported(
    model_core: CoreSBMLModel,
) -> None:
    """A kept selection the model has not is reported by its name."""
    scan = Scan(
        Simulation(end=10, steps=10),
        dimensions=[Dimension("dose", values={"a0": [1.0, 2.0]})],
    )
    with pytest.raises(ValueError, match="nonsense"):
        Simulator(engine="core", n_workers=1).run(model_core, scan, keep=["nonsense"])


def test_the_request_of_a_plan_of_times(model_core: CoreSBMLModel) -> None:
    """The request of a plan of output times carries them in the order of the plan."""
    simulator = Simulator(engine="core", relative_tolerance=1e-9)
    # the simulator sets its settings on the model it loads, which is this one
    assert simulator.load(model_core) is model_core
    plan = simulator.compile(model_core, Simulation(start=2, end=4, steps=2))
    request = request_of(plan, model_core, ["time", "A"])
    assert request["contract"] == CONTRACT
    assert request["start"] == 2.0
    assert request["end"] == 4.0
    assert request["output"] == {"times": [2.0, 3.0, 4.0]}
    assert request["preinit"] == []
    assert request["changes"] == []
    assert request["steady_state"] is None
    assert request["selections"] == ["time", "A"]
    assert request["settings"] == model_core.request_settings()
    assert request["settings"]["relative_tolerance"] == 1e-9
    # a plan has no seed, the engine orders simultaneous events with its own
    assert "seed" not in request


def test_the_request_of_the_steps_of_the_integrator(model_core: CoreSBMLModel) -> None:
    """The output of a plan without times is the unit variant of the contract."""
    plan = Simulator(engine="core").compile(model_core, Simulation(end=4))
    request = request_of(plan, model_core, ["time"])
    assert request["output"] == INTEGRATOR_STEPS


def test_the_settings_of_a_request_can_be_given(
    model_core: CoreSBMLModel,
) -> None:
    """A caller of many requests computes the settings once and passes them."""
    plan = Simulator(engine="core").compile(model_core, Simulation(end=4, steps=4))
    settings = model_core.request_settings()
    assert request_of(plan, model_core, ["time"], settings)["settings"] is settings


def test_the_request_carries_the_changes_and_the_steady_state(
    model_core: CoreSBMLModel,
) -> None:
    """The changes are the events of the plan and the steady state its presimulation."""
    simulation = Simulation(
        end=20,
        preinit_changes={"f": 3.0},
        presimulation=SteadyState(preinit_changes={"a0": 2.0}, max_time=1e6),
        changes=[Change([10, 5], {"X": 1.0}), Change(10, {"k1": 0.5})],
        steps=20,
    )
    plan = Simulator(engine="core").compile(model_core, simulation)
    request = request_of(plan, model_core, ["time"])
    assert request["preinit"] == [["f", 3.0]]
    assert request["changes"] == [
        {"time": 5.0, "values": [["X", 1.0]]},
        {"time": 10.0, "values": [["X", 1.0], ["k1", 0.5]]},
    ]
    assert request["steady_state"] == {
        "preinit": [["a0", 2.0]],
        "absolute_tolerance": SteadyState().absolute_tolerance,
        "relative_tolerance": SteadyState().relative_tolerance,
        "max_time": 1e6,
    }


def test_the_settings_of_a_request_are_computed_once(
    model_core: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulation computes the tolerances of the states of its request once."""
    computed = 0
    settings = model_core.request_settings

    def counted() -> dict[str, object]:
        nonlocal computed
        computed += 1
        return settings()

    monkeypatch.setattr(model_core, "request_settings", counted)
    Simulator(engine="core").simulate(model_core, Simulation(end=4, steps=4))
    assert computed == 1


def test_a_core_model_round_trips_through_a_model_spec(
    model_core: CoreSBMLModel,
) -> None:
    """A worker rebuilds a core model with its integrator and every setting."""
    model_core.set_integrator_settings(relative_tolerance=1e-9, maximum_num_steps=777)
    spec = ModelSpec.of(model_core)
    assert spec.engine is Engine.CORE
    assert spec.jit_optimization is None
    assert spec.cache is None
    assert spec.integrator == "cvode"
    rebuilt = spec.load()
    assert isinstance(rebuilt, CoreSBMLModel)
    assert rebuilt.integrator_state() == model_core.integrator_state()
    # the settings of its requests, the tolerance of every state among them,
    # so the rebuilt model integrates as the one of the parent does
    assert rebuilt.state_ids() == model_core.state_ids()
    assert rebuilt.request_settings() == model_core.request_settings()
    assert ModelSpec.of(rebuilt).key == spec.key


def test_the_key_of_a_core_model_covers_its_settings(
    model_core: CoreSBMLModel, probe_sbml: str
) -> None:
    """Two core models of other settings never share a cache entry of a worker."""
    keys = {ModelSpec.of(model_core).key}
    for settings in (
        {"relative_tolerance": 1e-3},
        {"absolute_tolerance": 1e-3},
        {"maximum_num_steps": 123},
    ):
        other = load_model(probe_sbml, Engine.CORE)
        other.set_integrator_settings(**settings)
        keys.add(ModelSpec.of(other).key)
    assert len(keys) == 4


def test_the_threads_of_a_batch_are_the_workers_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The threads of a batch are the processes a run of another engine would have.

    They are `parallel.resolve_workers` of the `n_workers` of the simulator
    and of the points of the run: a number is taken as given, and inside a
    worker, which starts no pool, a run is serial.
    """
    assert core_threads(1, n_points=1000) == 1
    assert core_threads(3, n_points=1) == 3
    monkeypatch.setattr(parallel, "in_worker", lambda: True)
    assert core_threads(None, n_points=1000) == 1


def test_an_algebraic_rule_is_solved_against_its_analytic_solution() -> None:
    """The engine integrates a DAE with IDA since sbmlsim-core 0.10.0.

    Every other test of this file compares the engine with roadrunner, which
    does not load a model with an algebraic rule, so the reference here is the
    solution of the equations: a rate rule takes `x` from 1 at the rate 1 and
    the constraint `y - x = 0` determines `y`, so `x(t) = 1 + t` and `y = x`.
    """
    result = Simulator(engine="core", **TIGHT).simulate(
        ALGEBRAIC, Simulation(end=10, steps=10)
    )
    time = result.values[:, 0]
    np.testing.assert_allclose(result["x"], 1.0 + time, rtol=1e-8)
    # the constraint holds at every output time and not only at the end
    np.testing.assert_allclose(result["y"], result["x"], rtol=1e-8)


#: a model whose rate reads the time, so its steady state after the end
#: depends on the time it is integrated from: `x` relaxes to `time`, which the
#: executor of roadrunner continues at the end of the timecourse (`local` of
#: `sbmlsim.simulator.executor`) and the engine at the `from` of
#: `steady_state_after`
TIME_RELAXATION: Final[str] = """
model time_relaxation
  k = 1; x = 0
  x' = k*(time - x)
end
"""


def test_a_steady_state_after_the_end_of_a_model_which_reads_the_time_continues_at_it() -> (
    None
):
    """A model whose math reads the time continues its steady state at the end.

    roadrunner integrates the steady state of such a model from the time its
    model is at, the end of the timecourse, and the one of every other model
    from 0; the `from` of `steady_state_after` is that time, so the two agree
    whatever the model (sbmlsim-core#60). `x` follows `time - 1` here, so a
    steady state from 0 instead of from the end of the timecourse is off by
    the length of the timecourse.
    """
    source = sbml(TIME_RELAXATION)
    for end in (5.0, 20.0):
        simulation = Simulation(end=end, times=[0.0, end, float("inf")])
        core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
        roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
        assert core["x"][-1] == pytest.approx(roadrunner["x"][-1], rel=1e-6)
        # the steady state is reached at the end of a horizon after the end
        assert core["x"][-1] > end


def test_a_steady_state_after_the_end_of_a_plan_with_a_change_formula() -> None:
    """A plan of both a change formula and a steady state after the end is the one of roadrunner.

    The fold of a change reads the state at the time of the change with a
    request of the segment which ends there, which carries no steady state of
    its own: the steady state after the end is the work of the request of the
    plan alone, and a segment which carried it would run it at every change.
    """
    source = sbml(RELAXATION_EVENT)
    simulation = Simulation(
        end=2,
        changes=[Change(1, {"target": "target + x"})],
        times=[0.0, 1.0, 2.0, float("inf")],
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    assert core.values.shape == roadrunner.values.shape
    np.testing.assert_allclose(core["x"], roadrunner["x"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        core["target"], roadrunner["target"], rtol=1e-6, atol=1e-9
    )


def test_a_formula_at_a_change_at_the_start_of_a_model_with_events() -> None:
    """A change at the start of a model with events folds as roadrunner does.

    The probe of a model with events carries `before_events` and the segment
    of a change at the start is an empty interval, where the field leaves the
    events of the start which have not fired yet alone: the row is the state
    after the initialization and after the events the change itself triggers,
    which is the state the executor of roadrunner folds.
    """
    source = sbml(RELAXATION_EVENT)
    simulation = Simulation(
        end=2, changes=[Change(0, {"target": "target + x"})], times=[0.0, 1.0, 2.0]
    )
    core = Simulator(engine="core", **TIGHT).simulate(source, simulation)
    roadrunner = roadrunner_simulator(**TIGHT).simulate(source, simulation)
    # `x` is 0 after the initialization, so the target keeps its 7
    assert core["target"][0] == 7.0
    np.testing.assert_allclose(core["x"], roadrunner["x"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        core["target"], roadrunner["target"], rtol=1e-6, atol=1e-9
    )


# ----------------------------------------------------------------------------
# the wall clock deadline of the runtime (#581)
# ----------------------------------------------------------------------------
def test_a_simulation_of_the_engine_has_no_wall_clock_deadline() -> None:
    """The runtime of the process has no deadline, so the load never interrupts.

    The result of a simulation does not depend on the load of the machine, and
    with a deadline whether it completes does: the same model with the same
    parameters answered on an idle machine and was interrupted next to other
    work, which is the one property a simulator must not have (#581).
    """
    assert core_module.TIMEOUT_MS == 0
    assert core_module.runtime_timeout_ms() == 0


@pytest.mark.parametrize(
    ("given", "expected"),
    [("", 0), ("   ", 0), ("0", 0), ("250", 250), ("30000", 30000)],
)
def test_the_deadline_of_the_runtime_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, given: str, expected: int
) -> None:
    """A caller which wants a deadline sets it in milliseconds, 0 for none."""
    monkeypatch.setenv(core_module.TIMEOUT_VARIABLE, given)

    assert core_module.runtime_timeout_ms() == expected


@pytest.mark.parametrize("given", ["abc", "1.5", "-5", "1e3"])
def test_a_deadline_which_is_no_milliseconds_is_refused(
    monkeypatch: pytest.MonkeyPatch, given: str
) -> None:
    """The variable names itself and what it takes, instead of being ignored."""
    monkeypatch.setenv(core_module.TIMEOUT_VARIABLE, given)

    with pytest.raises(ValueError, match=core_module.TIMEOUT_VARIABLE):
        core_module.runtime_timeout_ms()


def test_a_deadline_which_fires_names_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted execution says a deadline fired, how long it was and what to do.

    It surfaced as `wasm trap: interrupt` of the bindings, which does not tell
    a reader that a deadline fired (#581). The deadline of this test is one
    millisecond, so it fires inside the first simulation which is longer than
    that; the runtime of the process is created again for it and the models
    compiled into it, which the fixture of the module undoes.
    """
    pytest.importorskip("sbmlsim_core")
    monkeypatch.setenv(core_module.TIMEOUT_VARIABLE, "1")
    core_module.runtime.cache_clear()
    core_module.COMPILED.clear()
    model = load_model(HCTZ_SBML, Engine.CORE)
    # long enough that a deadline of one millisecond fires inside it
    simulation = Simulation(start=0, end=240, steps=200_000)

    with pytest.raises(RuntimeError, match="wall clock deadline") as caught:
        Simulator(engine=Engine.CORE, n_workers=1).simulate(model, simulation)

    message = str(caught.value)
    assert core_module.TIMEOUT_VARIABLE in message
    assert "1 ms" in message
    core_module.runtime.cache_clear()
    core_module.COMPILED.clear()


def _urine_plan(formulas: dict[float, str]) -> tuple[CoreSBMLModel, Plan]:
    """Get the PK model with urine and a plan which sets `Aurine` by formulas."""
    model = load_model(sbml_pk(urine=True), Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    simulation = Simulation(
        end=40,
        changes=[Change(t, {"Aurine": f}) for t, f in formulas.items()],
        times=np.linspace(0, 40, 41).tolist(),
    )
    return model, compile_simulation(simulation, model.symbols, model.uinfo)


def test_a_change_of_a_linear_formula_is_a_form_of_the_request() -> None:
    """A formula at a change is a form, and the request simulates nothing to build it."""
    model, plan = _urine_plan({10.0: "0.5 * Aurine"})
    request = request_of(plan, model, ["time", "Aurine"], forms=True)
    assert request["changes"][0]["forms"] == [
        {"target": "Aurine", "constant": 0.0, "terms": [["Aurine", 0.5]]}
    ]
    assert request["changes"][0]["values"] == []
    # nothing to fold: the engine is not asked for a single segment
    segments: list[dict[str, object]] = []

    def refuse(requests: list[dict[str, object]]) -> list[object]:
        segments.extend(requests)
        return []

    folded = executor_core.requests_of(
        [(plan, ["time", "Aurine"])], model, refuse, forms=True
    )
    assert segments == []
    assert not isinstance(folded[0], Exception)


def test_a_form_simulates_as_the_fold_it_replaces() -> None:
    """The result of a request of forms is the result of the folded request, bitwise."""
    model, plan = _urine_plan({10.0: "1 + 0.5 * Aurine", 25.0: "3"})
    columns = ["time", "Aurine", "Aelim", "[C]"]
    folded = request_of(plan, model, columns)
    formed = request_of(plan, model, columns, forms=True)
    assert formed["changes"][0].get("forms")
    assert not folded["changes"][0].get("forms")
    a = model.instance().simulate(folded)
    b = model.instance().simulate(formed)
    assert list(a.columns) == list(b.columns)
    assert np.array_equal(a.values, b.values)


def test_a_plan_of_a_formula_which_is_no_form_folds_everything() -> None:
    """A form and a fold never meet in one request."""
    model, plan = _urine_plan({10.0: "0.5 * Aurine", 20.0: "Aurine * Aurine"})
    request = request_of(plan, model, ["time", "Aurine"], forms=True)
    assert all(not change.get("forms") for change in request["changes"])
    assert all(change["values"] for change in request["changes"])


def test_a_formula_of_a_concentration_is_no_form_and_folds() -> None:
    """The parser of PEtab math reads no `[S]`, so such a formula folds."""
    model, plan = _urine_plan({10.0: "2 * [C] + 1"})
    request = request_of(plan, model, ["time", "Aurine"], forms=True)
    assert "forms" not in request["changes"][0]
    assert request["changes"][0]["values"]


@pytest.mark.parametrize(
    "formula",
    ["Aurine / 90 + Aelim * 3", "Aelim + Aurine + Aelim", "Aurine / 90"],
    ids=["many terms with a division", "a repeated id", "a division"],
)
def test_a_form_of_several_terms_simulates_as_the_fold(formula: str) -> None:
    """Forms with several terms, a repeated id and a division match the fold."""
    model, plan = _urine_plan({10.0: formula})
    columns = ["time", "Aurine", "Aelim"]
    folded = request_of(plan, model, columns)
    formed = request_of(plan, model, columns, forms=True)
    assert formed["changes"][0].get("forms")
    a = model.instance().simulate(folded)
    b = model.instance().simulate(formed)
    assert np.array_equal(a.values, b.values), np.max(np.abs(a.values - b.values))


def test_a_formula_of_a_reaction_is_no_form_and_folds() -> None:
    """A reaction is no entity the engine reads as a term, so such a formula folds."""
    model, plan = _urine_plan({10.0: "0.5 * Aurine"})
    event = PlanEvent(
        10.0,
        (Assignment("Aurine", TargetKind.SPECIES_AMOUNT, formula="0.5 * absorption"),),
    )
    # `compile_simulation` refuses a reaction, so the plan is built by hand
    plan = replace(plan, events=(event,))
    request = request_of(plan, model, ["time", "Aurine"], forms=True)
    assert "forms" not in request["changes"][0]
    assert request["changes"][0]["values"]
