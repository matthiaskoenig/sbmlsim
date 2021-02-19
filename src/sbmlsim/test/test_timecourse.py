from sbmlsim.examples import example_timecourse
from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim


def test_timecourse():
    example_timecourse.run_timecourse_examples()


def test_serialization():
    tcsim = TimecourseSim(
        [
            Timecourse(0, 100, steps=101),
            Timecourse(0, 100, steps=101, changes={"k": 100, "p": 200}),
            Timecourse(0, 50, steps=51, changes={"k": 100, "p": 50}),
        ]
    )
    jsonstr = tcsim.to_json()
    assert jsonstr
    print(jsonstr)

    tcsim2 = TimecourseSim.from_json(jsonstr)
    assert tcsim2
    jsonstr2 = tcsim2.to_json()
    assert jsonstr2

    assert jsonstr == jsonstr2
