"""
Run typical simulation experiments on SBML models


TODO: implement clamping of substances
TODO: better model changes
- timings of changes are necessary, i.e. when should change start and when end
Also what is the exact time point the change should be applied.
- different classes of changes:
    - initial changes (applied to the complete simulation)
    - timed changes (applied during the timecourse, start times and end times)
"""

import logging
from typing import List, Union

import pandas as pd
import roadrunner


from sbmlsim.model import clamp_species, MODEL_CHANGE_BOUNDARY_CONDITION
from sbmlsim.result import Result
from sbmlsim.timecourse import Timecourse, TimecourseSim


def timecourse(r: roadrunner.RoadRunner, sim: Union[TimecourseSim, Timecourse]) -> pd.DataFrame:
    """ Timecourse simulations based on timecourse_definition.

    :param r: Roadrunner model instance
    :param sim: Simulation definition(s)
    :param reset_all: Reset model at the beginning
    :return:
    """
    if isinstance(sim, Timecourse):
        sim = TimecourseSim(timecourses=[sim])

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


def timecourses(r: roadrunner.RoadRunner, sims: List[TimecourseSim]) -> List[pd.DataFrame]:
    """ Run many timecourses."""
    if isinstance(sims, TimecourseSim):
        sims = [sims]

    dfs = []
    for sim in sims:
        df = timecourse(r, sim)
        dfs.append(df)

    return Result(dfs)
