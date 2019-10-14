"""
Example shows basic model simulations and plotting.
"""
import os
import numpy as np
from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim import load_model, timecourse, TimecourseSimulation
from sbmlsim.parametrization import ChangeSet
from sbmlsim.plotting import add_line
from sbmlsim.tests.settings import DATA_PATH

model_path = os.path.join(DATA_PATH, 'models', 'repressilator.xml')
r = load_model(model_path)

# [1] simple timecourse simulation
s = timecourse(r, sim=TimecourseSimulation(tstart=0, tend=100, steps=100))

# [2] value scan
scan_changeset = ChangeSet.scan_changeset('n', values=np.linspace(start=2, stop=10, num=8))
s_result = timecourse(r,
                      TimecourseSimulation(tstart=0, tend=100, steps=100,
                                           changeset=scan_changeset)
                      )

# [3] parameter sensitivity
psensitivity_changeset = ChangeSet.parameter_sensitivity_changeset(r)
results = timecourse(r, sim=TimecourseSimulation(tstart=0, tend=100, steps=100,
                                                  changeset=psensitivity_changeset))


# create figure
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
fig.subplots_adjust(wspace=0.3, hspace=0.3)

add_line(ax=ax1, data=results,
         xid='time', yid="X", label="X")
add_line(ax=ax1, data=results,
         xid='time', yid="Y", label="Y", color="darkblue")

ax1.legend()
plt.show()
