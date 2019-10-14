"""
Run typical simulation experiments on SBML models

TODO: implement timings (duration of interventions)
- timings of changes are necessary, i.e. when should change start and when end
Also what is the exact time point the change should be applied.
- different classes of changes:
    - initial changes (applied to the complete simulation)
    - timed changes (applied during the timecourse, start times and end times)

TODO: implement clamping of substances

"""
import logging
import json
from collections import namedtuple

import numpy as np
import pandas as pd
import roadrunner
from copy import deepcopy

from sbmlsim.results import TimecourseResult




class TimecourseSimulation(object):
    """ Simulation definition.

    Definition of all information necessary to run a single timecourse simulation.

    A single simulation consists of multiple changes which are applied,
    all simulations are performed and collected.

    Changesets and selections are deepcopied for persistance

    """
    def __init__(self, tstart: float, tend: float, steps: int,
                 changeset=None, selections=None, repeats: int=1):
        """ Create a time course definition for simulation.

        :param tstart:
        :param tend:
        :param changeset
        :param selections:
        :param repeats:
        """
        if changeset is None:
            changeset = [{}]  # add empty changeset

        if isinstance(changeset, dict):
            changeset = [changeset]

        self.tstart = tstart
        self.tend = tend
        self.steps = steps
        self.changeset = deepcopy(changeset)
        self.selections = deepcopy(selections)
        self.repeats = repeats

    def add_change(self, sid, value):
        self.changeset[sid] = value

    def remove_change(self, sid):
        del self.changeset[sid]



def timecourse(r, sim: TimecourseSimulation):
    """ Timecourse simulations based on timecourse_definition.

    :param r: Roadrunner model instance
    :param sim: Simulation definition
    :param reset_all: Reset model at the beginning
    :return:
    """
    # FIXME: support repeats
    # FIXME: handle model state persistance, i.e. the initial state of the
    # model should persist for all the simulations.

    # set selections
    model_selections = r.timeCourseSelections
    if sim.selections is not None:
        r.timeCourseSelections = sim.selections

    # empty array for storage
    columns = r.timeCourseSelections
    Nt = sim.steps + 1
    Ncol = len(columns)
    Nsim = len(sim.changeset)
    s_data = np.empty((Nt, Ncol, Nsim)) * np.nan

    for idx, changes in enumerate(sim.changeset):
        # ! parallelization of simulation and better data structures ! # FIXME

        # reset
        reset_all(r)

        # apply changes
        for key, value in changes.items():
            r[key] = value

        # run simulation
        s_data[:, :, idx] = r.simulate(start=0, end=sim.tend, steps=sim.steps)

    # reset selections
    r.timeCourseSelections = model_selections

    # postprocessing
    if Nsim > 2:
        return TimecourseResult(data=s_data, selections=columns,
                                changeset=sim.changeset)
    else:
        return pd.DataFrame(s_data[:, :, 0], columns=columns)


def simulate(r, start=None, end=None, steps=None, points=None, **kwargs):
    """ Simple simulation.

    :param r:
    :param start:
    :param end:
    :param steps:
    :param points:
    :param kwargs:
    :return:
    """
    s = r.simulate(start=start, end=end, steps=steps, points=points, **kwargs)
    return pd.DataFrame(s, columns=s.colnames)


def reset_all(r):
    """ Reset all model variables to CURRENT init(X) values.

    This resets all variables, S1, S2 etc to the CURRENT init(X) values. It also resets all
    parameters back to the values they had when the model was first loaded.
    """
    r.reset(roadrunner.SelectionRecord.TIME |
            roadrunner.SelectionRecord.RATE |
            roadrunner.SelectionRecord.FLOATING |
            roadrunner.SelectionRecord.GLOBAL_PARAMETER)
