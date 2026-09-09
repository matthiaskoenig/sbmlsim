"""Tests of multiple dosing in the PEtab v2 layer.

A dosing protocol is a `TimecourseSim` of several timecourses, which is an
experiment of several periods in PEtab. The protocol of `Weir1998` is the case
this covers: 25 mg every 12 hours, eleven doses, and the data is reported from
the last dose, so the simulation starts at a negative time.
"""

from types import SimpleNamespace

import numpy as np
import petab.v2 as petab_v2
import pytest

from sbmlsim.fit.petab_v2.export import PetabExporter, _table
from sbmlsim.fit.petab_v2.reader import PetabReader, _to_float
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.units import UnitRegistry

#: doses of the protocol
N_DOSES = 11

#: hours between two doses
INTERVAL = 12 * 60

#: minutes the last period is observed
LAST_PERIOD = 60 * 60


@pytest.fixture(scope="module")
def dosing_simulation() -> TimecourseSim:
    """Get the multiple dosing protocol of `Weir1998`."""
    ureg = UnitRegistry()
    q = ureg.Quantity
    first = Timecourse(
        start=0, end=INTERVAL, steps=500, changes={"PODOSE_hctz": q(25, "mg")}
    )
    repeat = Timecourse(
        start=0,
        end=INTERVAL,
        steps=500,
        # the urine collection is reset with every dose
        changes={"PODOSE_hctz": q(25, "mg"), "Aurine_hctz": q(0, "mmole")},
    )
    last = Timecourse(
        start=0,
        end=LAST_PERIOD,
        steps=500,
        changes={"PODOSE_hctz": q(25, "mg"), "Aurine_hctz": q(0, "mmole")},
    )
    return TimecourseSim(
        [first] + [repeat for _ in range(N_DOSES - 2)] + [last],
        # the data is reported from the last dose, which is `time=0`
        time_offset=-(N_DOSES - 1) * INTERVAL,
    )


def _periods_of(simulation: TimecourseSim) -> tuple[petab_v2.Problem, list]:
    """Get the PEtab periods and conditions of a simulation."""
    problem = petab_v2.Problem()
    exporter = PetabExporter.__new__(PetabExporter)
    # the period logic only reads the id of the problem, for its messages
    exporter.problem = SimpleNamespace(opid="dosing")  # ty: ignore[invalid-assignment]
    periods = exporter._periods(problem, experiment_id="dosing", simulation=simulation)
    return problem, periods


def test_one_period_per_dose(dosing_simulation: TimecourseSim) -> None:
    """Every timecourse of the protocol is a period of the experiment."""
    _, periods = _periods_of(dosing_simulation)
    assert len(periods) == N_DOSES


def test_the_periods_are_the_times_of_the_doses(
    dosing_simulation: TimecourseSim,
) -> None:
    """The time of a period is the time of the simulation it starts at.

    The data of the study is reported from the last dose, so the simulation
    starts eleven doses earlier and the last period is `time=0`.
    """
    _, periods = _periods_of(dosing_simulation)
    expected = [-(N_DOSES - 1) * INTERVAL + k * INTERVAL for k in range(N_DOSES)]
    assert [period.time for period in periods] == expected
    assert periods[-1].time == 0.0
    # which is where the simulation of sbmlsim runs
    assert dosing_simulation.time[0] == pytest.approx(expected[0])


def test_every_dose_is_a_condition(dosing_simulation: TimecourseSim) -> None:
    """The changes of a timecourse are the condition of its period."""
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


def test_petab_accepts_the_experiment(dosing_simulation: TimecourseSim) -> None:
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


def test_the_protocol_is_read_back(dosing_simulation: TimecourseSim) -> None:
    """The reader rebuilds the protocol from the periods of the experiment.

    This is the problem of another tool, i.e. without the `sbmlsim` extension:
    a period lasts until the next one starts, the last one until the last
    measurement, and the simulation starts where the first period does.
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
    simulation = reader._simulation_of_periods(problem.experiments[0])

    assert len(simulation.timecourses) == N_DOSES
    # the simulation starts at the first dose and not at zero
    assert simulation.time_offset == pytest.approx(-(N_DOSES - 1) * INTERVAL)
    assert simulation.time[0] == pytest.approx(-(N_DOSES - 1) * INTERVAL)
    # every dose but the last lasts until the next one
    assert [tc.end for tc in simulation.timecourses[:-1]] == [
        pytest.approx(INTERVAL)
    ] * (N_DOSES - 1)
    # and the last until the last measurement
    assert simulation.timecourses[-1].end == pytest.approx(LAST_PERIOD)
    assert simulation.time[-1] == pytest.approx(LAST_PERIOD)


def test_the_extension_keeps_the_protocol(dosing_simulation: TimecourseSim) -> None:
    """With the extension the timecourses come back as they were."""
    info = {
        "time_offset": dosing_simulation.time_offset,
        "reset": dosing_simulation.reset,
        "timecourses": [
            PetabExporter._timecourse_dict(tc) for tc in dosing_simulation.timecourses
        ],
    }
    reader = PetabReader.__new__(PetabReader)
    reader.ureg = UnitRegistry()
    simulation = reader._simulation_of_extension(info)

    assert len(simulation.timecourses) == N_DOSES
    assert simulation.time_offset == dosing_simulation.time_offset
    for tc, original in zip(
        simulation.timecourses, dosing_simulation.timecourses, strict=True
    ):
        assert tc.start == original.start
        assert tc.end == original.end
        assert tc.steps == original.steps
        assert set(tc.changes) == set(original.changes)
    assert np.allclose(simulation.time, dosing_simulation.time)
