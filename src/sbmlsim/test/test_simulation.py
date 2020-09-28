import pandas as pd

from sbmlsim.model import ModelChange
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial as Simulator
from sbmlsim.test import MODEL_REPRESSILATOR


def test_create_simulator():
    simulator = Simulator(MODEL_REPRESSILATOR)
    assert simulator


def test_create_simulator_strpath():
    simulator = Simulator(str(MODEL_REPRESSILATOR))
    assert simulator


def test_timecourse_simulation():
    simulator = Simulator(MODEL_REPRESSILATOR)

    tc = Timecourse(start=0, end=100, steps=100)
    tc.normalize(udict=simulator.udict, ureg=simulator.ureg)
    s = simulator._timecourse(tc)
    assert s is not None

    tc = Timecourse(start=0, end=100, steps=100, changes={"PX": 10.0})
    tc.normalize(udict=simulator.udict, ureg=simulator.ureg)
    s = simulator._timecourse(tc)
    assert s is not None
    assert isinstance(s, pd.DataFrame)
    assert "time" in s
    assert len(s.time) == 101
    assert s.PX[0] == 10.0

    tcsim = TimecourseSim(
        timecourses=[Timecourse(start=0, end=100, steps=100, changes={"[X]": 10.0})]
    )
    tcsim.normalize(udict=simulator.udict, ureg=simulator.ureg)
    s = simulator._timecourse(tcsim)
    assert s is not None


def test_timecourse_combined():
    simulator = Simulator(MODEL_REPRESSILATOR)

    s = simulator._timecourse(
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
    assert isinstance(s, pd.DataFrame)
    assert "time" in s
    assert s.time.values[-1] == 250.0
