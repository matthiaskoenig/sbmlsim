import pandas as pd

from sbmlsim.simulation_serial import SimulatorSerial as Simulator
from sbmlsim.timecourse import Timecourse, TimecourseSim, ensemble
from sbmlsim.tests.constants import MODEL_REPRESSILATOR
from sbmlsim.result import Result


def test_create_simulator():
    simulator = Simulator(MODEL_REPRESSILATOR)
    assert simulator


def test_create_simulator_strpath():
    simulator = Simulator(str(MODEL_REPRESSILATOR))
    assert simulator


def test_timecourse_simulation():
    simulator = Simulator(MODEL_REPRESSILATOR)

    s = simulator.timecourse(Timecourse(start=0, end=100, steps=100))
    assert s is not None

    s = simulator.timecourse(
        Timecourse(start=0, end=100, steps=100,
                   changes={"PX": 10.0})
    )
    assert s is not None
    assert isinstance(s, pd.DataFrame)
    assert "time" in s
    assert len(s.time) == 101
    assert s.PX[0] == 10.0

    s = simulator.timecourse(TimecourseSim(timecourses=[
        Timecourse(start=0, end=100, steps=100, changes={"[X]": 10.0})
    ])
                                  )
    assert s is not None


def test_timecourse_combined():
    simulator = Simulator(MODEL_REPRESSILATOR)

    s = simulator.timecourse(simulation=TimecourseSim([
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


def test_timecourse_ensemble():
    changeset = [
         {"[X]": 10.0},
         {"[X]": 15.0},
         {"[X]": 20.0},
         {"[X]": 25.0},
    ]
    simulator = Simulator(MODEL_REPRESSILATOR)
    tcsims = ensemble(TimecourseSim([
            Timecourse(start=0, end=400, steps=400),
        ]), changeset=changeset)
    result = simulator.timecourses(tcsims)
    assert isinstance(result, Result)
