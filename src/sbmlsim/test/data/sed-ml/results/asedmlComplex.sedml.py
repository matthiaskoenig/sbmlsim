"""
####################################################################################################
                            tellurium 2.0.2
-+++++++++++++++++-         Python Environment for Modeling and Simulating Biological Systems
 .+++++++++++++++.
  .+++++++++++++.           Homepage:      http://tellurium.analogmachine.org/
-//++++++++++++/.   -:/-`   Documentation: https://tellurium.readthedocs.io/en/latest/index.html
.----:+++++++/.++  .++++/   Forum:         https://groups.google.com/forum/#!forum/tellurium-discuss
      :+++++:  .+:` .--++   Bug reports:   https://github.com/sys-bio/tellurium/issues
       -+++-    ./+:-://.   Repository:    https://github.com/sys-bio/tellurium
        .+.       `...`

SED-ML simulation experiments: http://www.sed-ml.org/
    sedmlDoc: L1V1  
    inputType:      'SEDML_FILE'
    workingDir:     '/home/mkoenig/git/tellurium/tellurium/tests/testdata/sedml/sed-ml'
    saveOutputs:    'False'
    outputDir:      'None'
    plottingEngine: '<PlotlyEngine>'

Linux-4.10.0-35-generic-x86_64-with-Ubuntu-16.04-xenial
python 2.7.12 (default, Nov 19 2016, 06:48:10) 
[GCC 5.4.0 20160609]
####################################################################################################
"""
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import tellurium as te
from roadrunner import Config
from tellurium.sedml.mathml import *
from tellurium.sedml.tesedml import fix_endpoints, process_trace, terminate_trace


try:
    import tesedml as libsedml
except ImportError:
    import libsedml

import os.path

import pandas


Config.LOADSBMLOPTIONS_RECOMPILE = True

workingDir = r"/home/mkoenig/git/tellurium/tellurium/tests/testdata/sedml/sed-ml"

# --------------------------------------------------------
# Models
# --------------------------------------------------------
# Model <Application0>
Application0 = te.loadSBMLModel(os.path.join(workingDir, "../models/asedmlComplex.xml"))
# Model <Application0_0>
Application0_0 = te.loadSBMLModel(
    os.path.join(workingDir, "../models/asedmlComplex.xml")
)
# /sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='s0'] 25.0
Application0_0["init([s0])"] = 25.0


# --------------------------------------------------------
# Tasks
# --------------------------------------------------------
# Task <task_0_0>
# not part of any DataGenerator: task_0_0
# Task <repeatedTask_0_0>

repeatedTask_0_0 = []
__range__range_0_0_s1_init_uM = np.linspace(start=5.0, stop=15.0, num=4)
for __k__range_0_0_s1_init_uM, __value__range_0_0_s1_init_uM in enumerate(
    __range__range_0_0_s1_init_uM
):
    Application0_0.reset()
    # Task: <task_0_0>
    task_0_0 = [None]
    Application0_0.setIntegrator("cvode")
    if Application0_0.conservedMoietyAnalysis == True:
        Application0_0.conservedMoietyAnalysis = False
    Application0_0["init([s1])"] = __value__range_0_0_s1_init_uM
    Application0_0.timeCourseSelections = ["time", "[s0]", "[s1]"]
    Application0_0.reset()
    task_0_0[0] = Application0_0.simulate(start=0.0, end=30.0, steps=1000)

    repeatedTask_0_0.extend(task_0_0)

# --------------------------------------------------------
# DataGenerators
# --------------------------------------------------------
# DataGenerator <time_repeatedTask_0_0>
__var__t = np.concatenate([process_trace(sim["time"]) for sim in repeatedTask_0_0])
if len(__var__t.shape) == 1:
    __var__t.shape += (1,)
time_repeatedTask_0_0 = __var__t
# DataGenerator <dataGen_repeatedTask_0_0_s0>
__var__s0 = np.concatenate([process_trace(sim["[s0]"]) for sim in repeatedTask_0_0])
if len(__var__s0.shape) == 1:
    __var__s0.shape += (1,)
dataGen_repeatedTask_0_0_s0 = __var__s0
# DataGenerator <dataGen_repeatedTask_0_0_s1>
__var__s1 = np.concatenate([process_trace(sim["[s1]"]) for sim in repeatedTask_0_0])
if len(__var__s1.shape) == 1:
    __var__s1.shape += (1,)
dataGen_repeatedTask_0_0_s1 = __var__s1

# --------------------------------------------------------
# Outputs
# --------------------------------------------------------
# Output <plot2d_Simulation1>
_stacked = False
_engine = te.getPlottingEngine()
if _stacked:
    tefig = _engine.newStackedFigure(
        title="plot2d_Simulation1 (Application0plots)",
        xtitle="time_repeatedTask_0_0 (time_repeatedTask_0_0)",
    )
else:
    tefig = _engine.newFigure(
        title="plot2d_Simulation1 (Application0plots)",
        xtitle="time_repeatedTask_0_0 (time_repeatedTask_0_0)",
    )

for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_s0 (curve_0)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_s0[:, k],
        color="C0",
        tag="tag0",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_s1 (curve_1)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_s1[:, k],
        color="C1",
        tag="tag1",
        **extra_args
    )
fig = tefig.render()

####################################################################################################
