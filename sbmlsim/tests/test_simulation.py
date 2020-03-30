import pandas as pd

from sbmlsim.simulator import SimulatorSerial as Simulator
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.result import XResult
from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def test_create_simulator():
    simulator = Simulator(MODEL_REPRESSILATOR)
    assert simulator


def test_create_simulator_strpath():
    simulator = Simulator(str(MODEL_REPRESSILATOR))
    assert simulator


def test_timecourse_simulation():
    simulator = Simulator(MODEL_REPRESSILATOR)

    s = simulator._timecourse(Timecourse(start=0, end=100, steps=100))
    assert s is not None

    s = simulator._timecourse(
        Timecourse(start=0, end=100, steps=100,
                   changes={"PX": 10.0})
    )
    assert s is not None
    assert isinstance(s, pd.DataFrame)
    assert "time" in s
    assert len(s.time) == 101
    assert s.PX[0] == 10.0

    s = simulator._timecourse(TimecourseSim(timecourses=[
        Timecourse(start=0, end=100, steps=100, changes={"[X]": 10.0})
    ])
                                  )
    assert s is not None


def test_timecourse_combined():
    simulator = Simulator(MODEL_REPRESSILATOR)

    s = simulator._timecourse(simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100),
            Timecourse(start=0, end=50, steps=100,
                       model_changes={"boundary_condition": {"X": True}}),
            Timecourse(start=0, end=100, steps=100,
                       model_changes={"boundary_condition": {"X": False}}),
        ])
    )
    assert isinstance(s, pd.DataFrame)
    assert "time" in s
    assert s.time.values[-1] == 250.0
