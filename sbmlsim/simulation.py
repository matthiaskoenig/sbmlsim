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
from sbmlsim.model import clamp_species, MODEL_CHANGE_BOUNDARY_CONDITION
import warnings
from typing import List, Union
from json import JSONEncoder



class Timecourse(JSONEncoder):
    """ Simulation definition.

    Definition of all information necessary to run a single timecourse simulation.

    A single simulation consists of multiple changes which are applied,
    all simulations are performed and collected.

    Changesets and selections are deepcopied for persistance

    """
    def __init__(self, start: float, end: float, steps: int,
                 changes: dict = None, model_changes: dict = None):
        """ Create a time course definition for simulation.

        :param start: start time
        :param end: end time
        :param steps: simulation steps
        :param changes: parameter and initial condition changes
        :param model_changes: changes to model structure
        """
        # Create empty changes and model changes for serialization
        if changes is None:
            changes = {}
        if model_changes is None:
            model_changes = {}

        self.start = start
        self.end = end
        self.steps = steps
        self.changes = deepcopy(changes)
        self.model_changes = deepcopy(model_changes)

    def default(self, o):
        """json encoder"""
        return o.__dict__

    def add_change(self, sid: str, value: float):
        self.changes[sid] = value

    def remove_change(self, sid: str):
        del self.changes[sid]

    def add_model_change(self, sid: str, change):
        self.model_changes[sid] = change

    def remove_model_change(self, sid: str):
        del self.model_changes[sid]


class ObjectJSONEncoder(JSONEncoder):
    def default(self, o):
        """json encoder"""
        return o.__dict__


class TimecourseSimulation(object):
    """ Complex timecourse simulation consisting of multiple
    concatenated timecourses.
    """
    def __init__(self, timecourses: List[Timecourse],
                 selections: list = None, reset: bool = True):
        """
        :param timecourses:
        :param selections:
        :param reset: resetToOrigin at beginning of simulation
        """
        if isinstance(timecourses, Timecourse):
            timecourses = [timecourses]


        self.id = None
        self.timecourses = timecourses
        self.selections = deepcopy(selections)
        self.reset = reset

    def clone(self):
        return deepcopy(self)

    def ensemble(self, changeset):
        """ Creates an ensemble of timecourse by mixin the changeset

        :return:
        """
        sims = []
        for changes in changeset:
            sim = self.clone()
            tc = sim.timecourses[0]
            for key, value in changes.items():
                tc.add_change(key, value)
            sims.append(sim)

        return sims

    def to_json(self):
        return json.dumps(self, cls=ObjectJSONEncoder, indent=2)

    def __str__(self):
        return "{}\n{}".format(type(self), self.to_json())


def timecourse(r: roadrunner.RoadRunner, sim: Union[TimecourseSimulation, Timecourse]) -> pd.DataFrame:
    """ Timecourse simulations based on timecourse_definition.

    :param r: Roadrunner model instance
    :param sim: Simulation definition(s)
    :param reset_all: Reset model at the beginning
    :return:
    """
    if isinstance(sim, Timecourse):
        sim = TimecourseSimulation(timecourses=[sim])

    if sim.reset:
        r.resetToOrigin()

    # selections backup
    model_selections = r.timeCourseSelections
    if sim.selections is not None:
        r.timeCourseSelections = sim.selections

    frames = []
    t_offset = 0.0
    for tc in sim.timecourses:

        # apply changes
        for key, value in tc.changes.items():
            r[key] = value

        for key, value in tc.model_changes.items():
            if key == MODEL_CHANGE_BOUNDARY_CONDITION:
                for sid, bc in value.items():
                    # setting boundary conditions
                    r = clamp_species(r, sid, boundary_condition=bc)
            else:
                logging.error("Unsupported model change: {}:{}".format(key, value))

        # run simulation
        s = r.simulate(start=tc.start, end=tc.end, steps=tc.steps)
        df = pd.DataFrame(s, columns=s.colnames)
        df.time = df.time + t_offset
        frames.append(df)
        t_offset += tc.end

    # reset selections
    r.timeCourseSelections = model_selections

    return pd.concat(frames)


def timecourses(r: roadrunner.RoadRunner, sims: List[TimecourseSimulation]) -> List[pd.DataFrame]:
    """ Run many timecourses."""
    # FIXME: parallelize
    results = []
    for sim in sims:
        df = timecourse(r, sim)
        results.append(df)

    return results


def timecourse_old(r, sim: TimecourseSimulation):
    """ Timecourse simulations based on timecourse_definition.

    :param r: Roadrunner model instance
    :param sim: Simulation definition(s)
    :param reset_all: Reset model at the beginning
    :return:
    """

    # selections backup
    model_selections = r.timeCourseSelections
    if sim.selections is not None:
        r.timeCourseSelections = sim.selections

    # empty array for storage
    columns: list = r.timeCourseSelections
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

        # TODO: run the full simulation experiment (?!)

        # run simulation
        s_data[:, :, idx] = r.simulate(start=0, end=sim.tend, steps=sim.steps)

    # reset selections
    r.timeCourseSelections = model_selections

    # Create a result structure
    if Nsim > 2:
        return TimecourseResult(data=s_data, selections=columns,
                                changeset=sim.changeset)
    else:
        return pd.DataFrame(s_data[:, :, 0], columns=columns)


def reset_all(r):
    """ Reset all model variables to CURRENT init(X) values.

    This resets all variables, S1, S2 etc to the CURRENT init(X) values. It also resets all
    parameters back to the values they had when the model was first loaded.
    """
    r.reset(roadrunner.SelectionRecord.TIME |
            roadrunner.SelectionRecord.RATE |
            roadrunner.SelectionRecord.FLOATING |
            roadrunner.SelectionRecord.GLOBAL_PARAMETER)


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
    warnings.warn(
        "simulate is deprecated, use timecourse instead",
        DeprecationWarning
    )
    s = r.simulate(start=start, end=end, steps=steps, points=points, **kwargs)
    return pd.DataFrame(s, columns=s.colnames)