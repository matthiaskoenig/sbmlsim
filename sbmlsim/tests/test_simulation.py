import os
import sbmlsim
from sbmlsim.tests.settings import DATA_PATH
from sbmlsim.simulation import TimecourseSimulation

REPRESSILATOR_PATH = os.path.join(DATA_PATH, 'models', 'repressilator.xml')


def test_simulate():
    model_path = os.path.join(DATA_PATH, 'models', 'body19_livertoy_flat.xml')
    r = sbmlsim.load_model(model_path)
    s = sbmlsim.simulate(r, start=0, end=100, steps=100)
    assert s is not None


def test_timecourse():
    r = sbmlsim.load_model(REPRESSILATOR_PATH)
    s = sbmlsim.timecourse(r, sim=TimecourseSimulation(tstart=0, tend=100, steps=100))
    assert s is not None

    s_result = sbmlsim.timecourse(r, sim=TimecourseSimulation(tstart=0, tend=100, steps=100,
                                                              changeset={
                                         "PX": 10.0
                                     }))
    assert s_result is not None

    s_result = sbmlsim.timecourse(r, sim=TimecourseSimulation(tstart=0, tend=100, steps=100,
                                                              changeset=[
                                         {"[X]": 10.0},
                                         {"[X]": 15.0},
                                         {"[X]": 20.0},
                                         {"[X]": 25.0},
                                     ])
                                  )
    assert s_result is not None


def test_timecourse_combined():
    r = sbmlsim.load_model(REPRESSILATOR_PATH)
    s = sbmlsim.timecourse(r, sim=TimecourseSimulation(tstart=0, tend=100, steps=100))
    assert s is not None

    tsim =TimecourseSimulation(tstart=0, tend=100, steps=100, changeset={"PX": 10.0})
    res1 = sbmlsim.timecourse(r, tsim)
    assert res1 is not None
    res2 = sbmlsim.timecourse(r, tsim)

    from sbmlsim.results import TimecourseResult

    TimecourseResult.append_results()
    assert s_result is not None