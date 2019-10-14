"""
Example showing basic timecourse simulations and plotting.
"""
import os
import numpy as np
from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim import load_model, timecourse
from sbmlsim.simulation import Timecourse, TimecourseSimulation

from sbmlsim.parametrization import ChangeSet
from sbmlsim.plotting import add_line
from sbmlsim.tests.settings import DATA_PATH

model_path = os.path.join(DATA_PATH, 'models', 'repressilator.xml')
r = load_model(model_path)

# 1. simple timecourse simulation
s1 = timecourse(r, sim=TimecourseSimulation(
                    Timecourse(start=0, end=100, steps=100))
               )

# 2. timecourse with parameter changes
s2 = timecourse(r, sim=TimecourseSimulation(
                    Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 200}))
               )

# 3. combined timecourses
s3 = timecourse(r, sim=TimecourseSimulation([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
    ]))

# 4. combined timecourses with model_change
s4 = timecourse(r, sim=TimecourseSimulation([
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=50, steps=100, model_changes={"boundary_condition": {"X": True}}),
        Timecourse(start=0, end=100, steps=100, model_changes={"boundary_condition": {"X": False}}),
    ]))

# create figure
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
fig.subplots_adjust(wspace=0.3, hspace=0.3)

ax1.set_title("simple timecourse")
ax2.set_title("parameter change")
ax3.set_title("combined timecourse")
ax4.set_title("model change")

for s, ax in [(s1, ax1), (s2, ax2), (s3, ax3), (s4, ax4)]:
    ax.plot(s.time, s.X, label="X")
    ax.plot(s.time, s.Y, label="Y")
    ax.plot(s.time, s.Z, label="Z")

for ax in (ax1, ax2, ax3, ax4):
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("concentration")
plt.show()
