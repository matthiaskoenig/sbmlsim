"""Tests of multiple dosing in the PEtab v2 layer.

A dosing protocol is a `Simulation` with a `Change` at the times of the doses,
which is an experiment of several periods in PEtab. The protocol of `Weir1998`
is the case this covers: 25 mg every 12 hours, eleven doses, and the data is
reported from the last dose, so the simulation starts at a negative time.
"""

from types import SimpleNamespace

import numpy as np
import petab.v2 as petab_v2
import pytest

from sbmlsim import Q
from sbmlsim.fit.petab_v2.export import PetabExporter, _table
from sbmlsim.fit.petab_v2.extension import simulation_of_timecourses
from sbmlsim.fit.petab_v2.reader import PetabReader, _to_float
from sbmlsim.model.symbols import ModelSymbols
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator.plan import Plan, compile_simulation
from sbmlsim.units import UnitsInformation, ureg

#: doses of the protocol
N_DOSES = 11

#: minutes between two doses
INTERVAL = 12 * 60

#: minutes the last period is observed
LAST_PERIOD = 60 * 60

#: the times of the doses, the last one is `time=0`
DOSE_TIMES = [-(N_DOSES - 1) * INTERVAL + k * INTERVAL for k in range(N_DOSES)]


@pytest.fixture(scope="module")
def dosing_simulation() -> Simulation:
    """Get the multiple dosing protocol of `Weir1998`."""
    return Simulation(
        time_unit="min",
        start=DOSE_TIMES[0],
        end=LAST_PERIOD,
        changes=[
            Change(DOSE_TIMES, {"PODOSE_hctz": Q(25, "mg")}),
            # the urine collection is reset with every dose but the first
            Change(DOSE_TIMES[1:], {"Aurine_hctz": Q(0, "mmole")}),
        ],
    )


def _plan(simulation: Simulation) -> Plan:
    """Compile the protocol against the entities it changes."""
    symbols = ModelSymbols(
        parameters=frozenset({"PODOSE_hctz"}),
        compartments=frozenset({"Vurine"}),
        species=frozenset({"Aurine_hctz"}),
        species_compartment={"Aurine_hctz": "Vurine"},
        only_substance=frozenset({"Aurine_hctz"}),
        initial_assignments=frozenset(),
        assignment_rules=frozenset(),
        rate_rules=frozenset(),
    )
    uinfo = UnitsInformation(
        udict={"time": "min", "PODOSE_hctz": "mg", "Aurine_hctz": "mmol"},
        ureg=ureg,
    )
    return compile_simulation(simulation, symbols, uinfo)


def _periods_of(simulation: Simulation) -> tuple[petab_v2.Problem, list]:
    """Get the PEtab periods and conditions of a simulation."""
    problem = petab_v2.Problem()
    exporter = PetabExporter.__new__(PetabExporter)
    # the period logic only reads the id, the parameter mapping and the plans
    # of the problem, and `group_index` is unused when there is no mapping
    exporter.problem = SimpleNamespace(  # ty: ignore[invalid-assignment]
        opid="dosing", parameter_mapping=None, plans=[_plan(simulation)]
    )
    exporter.sciml = None
    periods = exporter._periods(
        problem,
        experiment_id="dosing",
        group_index=0,
        simulation_key="dosing",
    )
    return problem, periods


def test_one_period_per_dose(dosing_simulation: Simulation) -> None:
    """Every dose is a period of the experiment."""
    _, periods = _periods_of(dosing_simulation)
    assert len(periods) == N_DOSES


def test_the_periods_are_the_times_of_the_doses(
    dosing_simulation: Simulation,
) -> None:
    """The time of a period is the time of its dose.

    The data of the study is reported from the last dose, so the simulation
    starts eleven doses earlier and the last period is `time=0`.
    """
    _, periods = _periods_of(dosing_simulation)
    assert [period.time for period in periods] == DOSE_TIMES
    assert periods[-1].time == 0.0


def test_every_dose_is_a_condition(dosing_simulation: Simulation) -> None:
    """The changes at the time of a dose are the condition of its period."""
    problem, periods = _periods_of(dosing_simulation)
    assert len(problem.conditions) == N_DOSES

    conditions = {condition.id: condition for condition in problem.conditions}
    for k, period in enumerate(periods):
        assert len(period.condition_ids) == 1
        changes = conditions[period.condition_ids[0]].changes
        targets = {
            change.target_id: _to_float(change.target_value) for change in changes
        }
        assert targets["PODOSE_hctz"] == pytest.approx(25.0)
        if k > 0:
            # the urine collection is reset with every dose but the first
            assert targets["Aurine_hctz"] == pytest.approx(0.0)
        else:
            assert "Aurine_hctz" not in targets


def test_petab_accepts_the_experiment(dosing_simulation: Simulation) -> None:
    """PEtab reads an experiment of eleven periods."""
    problem, periods = _periods_of(dosing_simulation)
    _table(problem, "experiment_tables").experiments.append(
        petab_v2.Experiment(id="dosing", periods=periods)
    )
    assert len(problem.experiments) == 1
    assert len(problem.experiments[0].periods) == N_DOSES

    # the periods of an experiment are ordered in time and unique
    times = [period.time for period in problem.experiments[0].periods]
    assert times == sorted(times)
    assert len(set(times)) == N_DOSES


def test_the_protocol_is_read_back(dosing_simulation: Simulation) -> None:
    """The reader rebuilds the protocol from the periods of the experiment.

    This is the problem of another tool, i.e. without the `sbmlsim` extension:
    the first period is the start of the simulation and its condition is
    applied before the initialization, every other period is a change at its
    time, and the simulation ends at the last measurement.
    """
    problem, periods = _periods_of(dosing_simulation)
    _table(problem, "experiment_tables").experiments.append(
        petab_v2.Experiment(id="dosing", periods=periods)
    )
    _table(problem, "observable_tables").observables.append(
        petab_v2.Observable(id="obs", formula="Cve_hctz", noise_formula="1.0")
    )
    for time in [0.0, 60.0, LAST_PERIOD]:
        _table(problem, "measurement_tables").measurements.append(
            petab_v2.Measurement(
                observable_id="obs",
                experiment_id="dosing",
                time=time,
                measurement=1.0,
                observable_parameters=[],
                noise_parameters=[],
            )
        )

    reader = PetabReader.__new__(PetabReader)
    reader.petab_problem = problem
    reader.extension = None
    reader.sciml = None
    simulation = reader._simulation_of_periods(problem.experiments[0])

    # the simulation starts at the first dose and not at zero
    assert simulation.start == pytest.approx(DOSE_TIMES[0])
    assert simulation.end == pytest.approx(LAST_PERIOD)
    assert simulation.preinit_changes == {"PODOSE_hctz": pytest.approx(25.0)}
    assert [change.times for change in simulation.changes] == [
        (float(t),) for t in DOSE_TIMES[1:]
    ]
    for change in simulation.changes:
        assert change.values == {
            "PODOSE_hctz": pytest.approx(25.0),
            "Aurine_hctz": pytest.approx(0.0),
        }


def test_the_extension_keeps_the_protocol(dosing_simulation: Simulation) -> None:
    """With the extension the simulation comes back as it was."""
    reader = PetabReader.__new__(PetabReader)
    reader.ureg = ureg
    simulation = reader._simulation_of_extension(
        {"simulation": dosing_simulation.to_dict()}
    )
    assert simulation.to_dict() == dosing_simulation.to_dict()


def test_the_timecourses_of_an_old_extension_are_converted() -> None:
    """An extension before 0.3.0 holds the timecourses of the protocol."""
    first = {
        "start": 0,
        "end": INTERVAL,
        "steps": 4,
        "discard": False,
        "changes": {"PODOSE_hctz": 25.0},
        "units": {"PODOSE_hctz": "mg"},
    }
    repeat = {
        **first,
        "changes": {"PODOSE_hctz": 25.0, "Aurine_hctz": 0.0},
        "units": {"PODOSE_hctz": "mg", "Aurine_hctz": "mmole"},
    }
    info = {
        "time_offset": DOSE_TIMES[0],
        "timecourses": [first] + [repeat] * (N_DOSES - 1),
    }
    simulation = simulation_of_timecourses(info, ureg)
    assert simulation.start == DOSE_TIMES[0]
    assert simulation.end == DOSE_TIMES[-1] + INTERVAL
    assert simulation.preinit_changes == {"PODOSE_hctz": Q(25.0, "mg")}
    assert [change.times for change in simulation.changes] == [
        (float(t),) for t in DOSE_TIMES[1:]
    ]
    assert simulation.times is not None
    np.testing.assert_allclose(
        np.asarray(simulation.times, dtype=float),
        np.linspace(DOSE_TIMES[0], DOSE_TIMES[-1] + INTERVAL, 4 * N_DOSES + 1),
    )
