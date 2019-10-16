import pandas as pd

import sbmlsim
from sbmlsim.simulation import Timecourse, TimecourseSimulation, timecourses
from sbmlsim.tests.settings import MODEL_REPRESSILATOR
from sbmlsim.results import Result


def test_timecourse():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
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
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
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
    changeset = [
         {"[X]": 10.0},
         {"[X]": 15.0},
         {"[X]": 20.0},
         {"[X]": 25.0},
    ]
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    tc_sims = TimecourseSimulation(
        Timecourse(start=0, end=400, steps=400),
    ).ensemble(changeset)
    result = timecourses(r, tc_sims)
    assert isinstance(result, Result)
