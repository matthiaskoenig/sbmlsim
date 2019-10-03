"""
Run typical simulation experiments on SBML models
"""
import logging
import json
from collections import namedtuple

import numpy as np
import pandas as pd
import roadrunner

from sbmlsim.model import _parameters_for_sensitivity

# TODO: initial condition sensitivity


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
        self.changeset = changeset
        self.selections = selections
        self.repeats = repeats


class TimecourseResult(object):
    """Result of a timecourse simulation."""

    def __init__(self, data, selections, changeset):
        self.data = data
        self.changeset = changeset
        self.selections = selections

    @property
    def Nsel(self):
        return len(self.selections)

    @property
    def Nsim(self):
        return len(self.changeset)

    @property
    def mean(self):
        return pd.DataFrame(np.mean(self.data, axis=2), columns=self.selections)

    @property
    def std(self):
        return pd.DataFrame(np.std(self.data, axis=2), columns=self.selections)

    @property
    def min(self):
        return pd.DataFrame(np.min(self.data, axis=2), columns=self.selections)

    @property
    def max(self):
        return pd.DataFrame(np.max(self.data, axis=2), columns=self.selections)


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


def value_scan_changeset(selector, values):
    """Create changeset to scan parameter.

    :param r: RoadRunner model
    :param selector: selector in model
    :param values:
    :return: changeset
    """
    return [{selector: value} for value in values]


def parameter_sensitivity_changeset(r, sensitivity=0.1):
    """ Create changeset to calculate parameter sensitivity.

    :param r: RoadRunner model
    :return: changeset
    """
    p_dict = _parameters_for_sensitivity(r)
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

    scan_changeset = value_scan_changeset('n',
                                          values=np.linspace(start=2, stop=10, num=8))
    s_result = timecourse(r,
                          Sim(tstart=0, tend=100, steps=100,
                              changeset=scan_changeset)
                          )
