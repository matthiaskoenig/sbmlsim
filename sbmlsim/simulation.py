"""
Run typical simulation experiments on SBML models
"""
import logging
import json
from collections import namedtuple

import numpy as np
import pandas as pd
import roadrunner


# TODO: parameter sensitivity
# TODO: initial condition sensitivity
# TODO: parameter scans


class Sim(object):
    """ Simulation definition."""

    def __init__(self, tstart, tend, steps,
                 changeset=None, selections=None, repeats=1):
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
        self.changesets = changeset
        self.selections = selections
        self.repeats = repeats


TimecourseResult = namedtuple("Result", ['mean', 'std', 'min', 'max'])


def timecourse(r, sim):
    """ Timecourse simulations based on timecourse_definition.

    :param r: Roadrunner model instance
    :param sim: Simulation definition
    :return:
    """
    # FIXME: support repeats

    # set selections
    model_selections = r.timeCourseSelections
    if sim.selections is not None:
        r.timeCourseSelections = sim.selections

    # empty array for storage
    columns = r.timeCourseSelections
    Nt = sim.steps + 1
    Ncol = len(columns)
    Nsim = len(sim.changesets)
    s_data = np.empty((Nt, Ncol, Nsim)) * np.nan

    for idx, changes in enumerate(sim.changesets):
        # ! parallelization of simulation and better data structures ! # FIXME

        # reset
        reset_all(r)

        # apply changes
        for key, value in changes.items():
            r[key] = value
        # TODO: handle the steps and points correctly

        # run simulation
        s_data[:, :, idx] = r.simulate(start=0, end=sim.tend, steps=sim.steps)

    # reset selections
    r.timeCourseSelections = model_selections

    # postprocessing
    if Nsim > 2:
        s_mean = pd.DataFrame(np.mean(s_data, axis=2), columns=columns)
        s_std = pd.DataFrame(np.std(s_data, axis=2), columns=columns)
        s_min = pd.DataFrame(np.min(s_data, axis=2), columns=columns)
        s_max = pd.DataFrame(np.max(s_data, axis=2), columns=columns)
        return TimecourseResult(mean=s_mean, std=s_std, min=s_min, max=s_max)
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


# TODO: create advanced changesets

def scan_sim():
    pass


def parameter_sensitivity_changeset(r, sensitivity=0.1):
    """ Create changeset to calculate parameter sensitivity.

    :param r: RoadRunner model
    :return: changeset
    """
    from sbmlsim.model import _parameters_for_sensitivity
    p_dict = _parameters_for_sensitivity(r)
    print(p_dict)

    # create parameter changeset
    changeset = []
    for pid, value in p_dict.items():
        for change in [1.0 + sensitivity, 1.0 - sensitivity]:
            changeset.append(
                {pid: change*value}
            )
    return changeset


if __name__ == "__main__":
    import os
    import sbmlsim
    from sbmlsim.tests.settings import DATA_PATH
    model_path = os.path.join(DATA_PATH, 'models', 'repressilator.xml')

    r = sbmlsim.load_model(model_path)
    s = timecourse(r, sim=Sim(tstart=0, tend=100, steps=100))
    psensitivity_changeset = parameter_sensitivity_changeset(r)

    s_result = timecourse(r, sim=Sim(tstart=0, tend=100, steps=100,
                                     changeset=psensitivity_changeset))
    print(s_result)
