import os
import pandas as pd

import sbmlsim
from sbmlsim.tests.settings import DATA_PATH
from sbmlsim.simulation import Timecourse, TimecourseSimulation

REPRESSILATOR_PATH = os.path.join(DATA_PATH, 'models', 'repressilator.xml')


def test_simulate():
    model_path = os.path.join(DATA_PATH, 'models', 'body19_livertoy_flat.xml')
    r = sbmlsim.load_model(model_path)
    s = sbmlsim.simulate(r, start=0, end=100, steps=100)
    assert s is not None
    assert isinstance(s, pd.DataFrame)


def test_timecourse():
    r = sbmlsim.load_model(REPRESSILATOR_PATH)
    s = sbmlsim.timecourse(r, Timecourse(start=0, end=100, steps=100))
    assert s is not None

    s = sbmlsim.timecourse(r,
        Timecourse(start=0, end=100, steps=100,
                   changes={"PX": 10.0})
    )
    assert s is not None
    assert isinstance(s, pd.DataFrame)
    assert "time" in s
    assert len(s.time) == 101
    assert s.PX[0] == 10.0

    s = sbmlsim.timecourse(r, TimecourseSimulation(timecourses=[
        Timecourse(start=0, end=100, steps=100, changes={"[X]": 10.0})
    ])
                                  )
    assert s is not None


def test_timecourse_combined():
    r = sbmlsim.load_model(REPRESSILATOR_PATH)
    s = sbmlsim.timecourse(r, sim=TimecourseSimulation([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=50, steps=100,
                   model_changes={"boundary_condition": {"X": True}}),
        Timecourse(start=0, end=100, steps=100,
                   model_changes={"boundary_condition": {"X": False}}),
    ]))
    assert isinstance(s, pd.DataFrame)
    assert "time" in s
    assert s.time.values[-1] == 250.0


def test_timecourse_ensemble():
    changes = [
         {"[X]": 10.0},
         {"[X]": 15.0},
         {"[X]": 20.0},
         {"[X]": 25.0},
    ]
    assert 0