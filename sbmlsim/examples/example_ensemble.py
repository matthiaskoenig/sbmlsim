"""
Example shows basic model simulations and plotting.
"""
import os
import numpy as np
from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim import load_model, timecourse, TimecourseSimulation
from sbmlsim.simulation import timecourses, Timecourse
from sbmlsim.results import TimecourseResult

from sbmlsim.parametrization import ChangeSet
from sbmlsim.plotting import add_line
from sbmlsim.tests.settings import DATA_PATH

model_path = os.path.join(DATA_PATH, 'models', 'repressilator.xml')
r = load_model(model_path)


# [2] value scan
scan_changeset = ChangeSet.scan_changeset('n', values=np.linspace(start=2, stop=10, num=8))
tc_sims = TimecourseSimulation(
    Timecourse(start=0, end=100, steps=100)
).ensemble(changeset=scan_changeset)

for tc_sim in tc_sims:
    print(tc_sim)
s_results = timecourses(r, tc_sims)


# [3] parameter sensitivity
psensitivity_changeset = ChangeSet.parameter_sensitivity_changeset(r)
results = timecourses(r,
    TimecourseSimulation([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=200, steps=100, model_changes={"boundary_condition": {"X": True}}),
        Timecourse(start=0, end=100, steps=100, model_changes={"boundary_condition": {"X": False}}),
    ]
).ensemble(changeset=psensitivity_changeset))
results = TimecourseResult(results)

# create figure
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
fig.subplots_adjust(wspace=0.3, hspace=0.3)

add_line(ax=ax1, data=results,
         xid='time', yid="X", label="X")
add_line(ax=ax1, data=results,
         xid='time', yid="Y", label="Y", color="darkblue")
add_line(ax=ax1, data=results,
         xid='time', yid="Z", label="Z", color="darkorange")

ax1.legend()
plt.show()
