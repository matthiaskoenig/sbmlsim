

def test_ensemble():
    tcsim = TimecourseSim([
        Timecourse(0, 100, steps=101),
        Timecourse(0, 100, steps=101, changes={'k': 100, 'p': 200}),
        Timecourse(0, 50, steps=51, changes={'k': 100, 'p': 50}),
    ])
    import numpy as np
    tcsims = ensemble(tcsim, changeset=ChangeSet.scan_changeset("k2", np.linspace(0, 4, num=5)))
    for tcs in tcsims:
        print(tcs)

