from sbmlsim.timecourse import Timecourse, TimecourseSim


def test_json():
    tcsim = TimecourseSim([
        Timecourse(0, 100, steps=101),
        Timecourse(0, 100, steps=101, changes={'k': 100, 'p': 200}),
        Timecourse(0, 50, steps=51, changes={'k': 100, 'p': 50}),
    ])
    jsonstr = tcsim.to_json()
    assert jsonstr

    tcsim2 = TimecourseSim.from_json(jsonstr)
    assert tcsim2
    jsonstr2 = tcsim2.to_json()
    assert jsonstr2

    assert jsonstr == jsonstr2
