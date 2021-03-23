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
# Model <repressor_activator_oscillations>
repressor_activator_oscillations = te.loadSBMLModel(
    os.path.join(workingDir, "../models/BioModel1_repressor_activator_oscillations.xml")
)


# --------------------------------------------------------
# Tasks
# --------------------------------------------------------
# Task <task_0_0>
# not part of any DataGenerator: task_0_0
# Task <repeatedTask_0_0>

repeatedTask_0_0 = []
__range__range_0_0_common_delta_A = np.linspace(start=0.2, stop=1.0, num=4)
for __k__range_0_0_common_delta_A, __value__range_0_0_common_delta_A in enumerate(
    __range__range_0_0_common_delta_A
):
    repressor_activator_oscillations.reset()
    # Task: <task_0_0>
    task_0_0 = [None]
    repressor_activator_oscillations.setIntegrator("cvode")
    if repressor_activator_oscillations.conservedMoietyAnalysis == True:
        repressor_activator_oscillations.conservedMoietyAnalysis = False
    repressor_activator_oscillations[
        "common_delta_A"
    ] = __value__range_0_0_common_delta_A
    repressor_activator_oscillations.timeCourseSelections = [
        "[PrmR]",
        "[R]",
        "time",
        "[A]",
        "[mRNA_R]",
        "[mRNA_A_]",
        "[PrmA_bound]",
        "[C]",
        "[PrmR_bound]",
        "[PrmA]",
    ]
    repressor_activator_oscillations.reset()
    task_0_0[0] = repressor_activator_oscillations.simulate(
        start=0.0, end=200.0, steps=400
    )

    repeatedTask_0_0.extend(task_0_0)

# --------------------------------------------------------
# DataGenerators
# --------------------------------------------------------
# DataGenerator <time_repeatedTask_0_0>
__var__t = np.concatenate([process_trace(sim["time"]) for sim in repeatedTask_0_0])
if len(__var__t.shape) == 1:
    __var__t.shape += (1,)
time_repeatedTask_0_0 = __var__t
# DataGenerator <dataGen_repeatedTask_0_0_mRNA_R>
__var__mRNA_R = np.concatenate(
    [process_trace(sim["[mRNA_R]"]) for sim in repeatedTask_0_0]
)
if len(__var__mRNA_R.shape) == 1:
    __var__mRNA_R.shape += (1,)
dataGen_repeatedTask_0_0_mRNA_R = __var__mRNA_R
# DataGenerator <dataGen_repeatedTask_0_0_A>
__var__A = np.concatenate([process_trace(sim["[A]"]) for sim in repeatedTask_0_0])
if len(__var__A.shape) == 1:
    __var__A.shape += (1,)
dataGen_repeatedTask_0_0_A = __var__A
# DataGenerator <dataGen_repeatedTask_0_0_R>
__var__R = np.concatenate([process_trace(sim["[R]"]) for sim in repeatedTask_0_0])
if len(__var__R.shape) == 1:
    __var__R.shape += (1,)
dataGen_repeatedTask_0_0_R = __var__R
# DataGenerator <dataGen_repeatedTask_0_0_PrmA>
__var__PrmA = np.concatenate([process_trace(sim["[PrmA]"]) for sim in repeatedTask_0_0])
if len(__var__PrmA.shape) == 1:
    __var__PrmA.shape += (1,)
dataGen_repeatedTask_0_0_PrmA = __var__PrmA
# DataGenerator <dataGen_repeatedTask_0_0_PrmR>
__var__PrmR = np.concatenate([process_trace(sim["[PrmR]"]) for sim in repeatedTask_0_0])
if len(__var__PrmR.shape) == 1:
    __var__PrmR.shape += (1,)
dataGen_repeatedTask_0_0_PrmR = __var__PrmR
# DataGenerator <dataGen_repeatedTask_0_0_C>
__var__C = np.concatenate([process_trace(sim["[C]"]) for sim in repeatedTask_0_0])
if len(__var__C.shape) == 1:
    __var__C.shape += (1,)
dataGen_repeatedTask_0_0_C = __var__C
# DataGenerator <dataGen_repeatedTask_0_0_PrmA_bound>
__var__PrmA_bound = np.concatenate(
    [process_trace(sim["[PrmA_bound]"]) for sim in repeatedTask_0_0]
)
if len(__var__PrmA_bound.shape) == 1:
    __var__PrmA_bound.shape += (1,)
dataGen_repeatedTask_0_0_PrmA_bound = __var__PrmA_bound
# DataGenerator <dataGen_repeatedTask_0_0_PrmR_bound>
__var__PrmR_bound = np.concatenate(
    [process_trace(sim["[PrmR_bound]"]) for sim in repeatedTask_0_0]
)
if len(__var__PrmR_bound.shape) == 1:
    __var__PrmR_bound.shape += (1,)
dataGen_repeatedTask_0_0_PrmR_bound = __var__PrmR_bound
# DataGenerator <dataGen_repeatedTask_0_0_mRNA_A_>
__var__mRNA_A_ = np.concatenate(
    [process_trace(sim["[mRNA_A_]"]) for sim in repeatedTask_0_0]
)
if len(__var__mRNA_A_.shape) == 1:
    __var__mRNA_A_.shape += (1,)
dataGen_repeatedTask_0_0_mRNA_A_ = __var__mRNA_A_

# --------------------------------------------------------
# Outputs
# --------------------------------------------------------
# Output <plot2d_scan_for_delta_A>
_stacked = False
_engine = te.getPlottingEngine()
if _stacked:
    tefig = _engine.newStackedFigure(
        title="plot2d_scan_for_delta_A (repressor_activator_oscillationsplots)",
        xtitle="time_repeatedTask_0_0 (time_repeatedTask_0_0)",
    )
else:
    tefig = _engine.newFigure(
        title="plot2d_scan_for_delta_A (repressor_activator_oscillationsplots)",
        xtitle="time_repeatedTask_0_0 (time_repeatedTask_0_0)",
    )

for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_mRNA_R (curve_0)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_mRNA_R[:, k],
        color="C0",
        tag="tag0",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_A (curve_1)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_A[:, k],
        color="C1",
        tag="tag1",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_R (curve_2)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_R[:, k],
        color="C2",
        tag="tag2",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_PrmA (curve_3)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_PrmA[:, k],
        color="C3",
        tag="tag3",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_PrmR (curve_4)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_PrmR[:, k],
        color="C4",
        tag="tag4",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_C (curve_5)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_C[:, k],
        color="C5",
        tag="tag5",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_PrmA_bound (curve_6)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_PrmA_bound[:, k],
        color="C6",
        tag="tag6",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_PrmR_bound (curve_7)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_PrmR_bound[:, k],
        color="C0",
        tag="tag7",
        **extra_args
    )
for k in range(time_repeatedTask_0_0.shape[1]):
    extra_args = {}
    if k == 0:
        extra_args["name"] = "dataGen_repeatedTask_0_0_mRNA_A_ (curve_8)"
    tefig.addXYDataset(
        time_repeatedTask_0_0[:, k],
        dataGen_repeatedTask_0_0_mRNA_A_[:, k],
        color="C1",
        tag="tag8",
        **extra_args
    )
fig = tefig.render()

####################################################################################################
