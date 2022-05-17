"""Test simulations."""
import pandas as pd

from sbmlsim.model import ModelChange
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerialRR
from tests import MODEL_REPRESSILATOR


def test_timecourse_simulation(repressilator_model_state: str) -> None:
    """Run timecourse simulation."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)

    tc = Timecourse(start=0, end=100, steps=100)
    s = simulator.run_timecourse(TimecourseSim(tc))
    assert s is not None

    tc = Timecourse(start=0, end=100, steps=100, changes={"PX": 10.0})
    xres = simulator.run_timecourse(TimecourseSim(tc))
    assert xres is not None
    assert hasattr(xres, "_time")
    assert len(xres._time) == 101
    assert xres['[PX]'][0] == 10.0

    tcsim = TimecourseSim(
        timecourses=[Timecourse(start=0, end=100, steps=100, changes={"[X]": 10.0})]
    )
    xres = simulator.run_timecourse(tcsim)
    assert xres is not None


def test_timecourse_combined(repressilator_model_state: str) -> None:
    """Test timecourse combination."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)

    xres = simulator.run_timecourse(
        simulation=TimecourseSim(
            [
                Timecourse(start=0, end=100, steps=100),
                Timecourse(
                    start=0,
                    end=50,
                    steps=100,
                    model_changes={ModelChange.CLAMP_SPECIES: {"X": True}},
                ),
                Timecourse(
                    start=0,
                    end=100,
                    steps=100,
                    model_changes={ModelChange.CLAMP_SPECIES: {"X": False}},
                ),
            ]
        )
    )

    assert xres._time.values[-1] == 250.0


def test_timecourse_concat(repressilator_model_state: str) -> None:
    """Reuse of timecourses."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    tc = Timecourse(start=0, end=50, steps=100, changes={"X": 10})

    xres = simulator.run_timecourse(simulation=TimecourseSim([tc] * 3))
    assert xres._time.values[-1] == 150.0
    assert len(xres._time) == 3 * 101
    assert xres["[X]"].values[0] == 10.0
    assert xres["[X]"].values[101] == 10.0
    assert xres["[X]"].values[202] == 10.0


def test_timecourse_empty(repressilator_model_state: str) -> None:
    """Reuse of timecourses."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)
    tc = Timecourse(start=0, end=50, steps=100, changes={"X": 10})

    tcsim = TimecourseSim([None, tc, None])
    xres = simulator.run_timecourse(
        simulation=tcsim,
    )
    assert len(tcsim.timecourses) == 1
    assert xres._time.values[-1] == 50.0
    assert len(xres._time) == 101


def test_timecourse_discard(repressilator_model_state: str) -> None:
    """Test discarding pre-simulation."""
    simulator = SimulatorSerialRR()
    simulator.set_model(repressilator_model_state)

    xres = simulator.run_timecourse(
        simulation=TimecourseSim(
            [
                Timecourse(
                    start=0,
                    end=100,
                    steps=100,
                    discard=True,
                    changes={
                        "[X]": 20.0,
                        "[Y]": 20.0,
                        "[Z]": 20.0,
                    },
                ),
                Timecourse(start=0, end=100, steps=100),
            ]
        )
    )
    assert len(xres._time) == 101
    assert xres._time.values[0] == 0.0
    assert xres._time.values[-1] == 100.0
