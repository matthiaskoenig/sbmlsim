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

from sbmlsim.model import clamp_species, MODEL_CHANGE_BOUNDARY_CONDITION, load_model
from sbmlsim.result import Result
from sbmlsim.timecourse import Timecourse, TimecourseSim

logger = logging.getLogger(__name__)


class SimulatorAbstract(object):
    def __init__(self, path, selections: List[str] = None, **kwargs):
        """ Must be implemented by simulator. """
        pass

    def timecourses(self):
        """ Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")


class SimulatorWorker(object):

    def timecourse(self, simulation: Union[TimecourseSim, Timecourse]) -> pd.DataFrame:
        """ Timecourse simulations based on timecourse_definition.

        :param r: Roadrunner model instance
        :param simulation: Simulation definition(s)
        :param reset_all: Reset model at the beginning
        :return:
        """

        if isinstance(simulation, Timecourse):
            simulation = TimecourseSim(timecourses=[simulation])

        if simulation.reset:
            self.r.resetToOrigin()

        # selections backup
        model_selections = self.r.timeCourseSelections
        if simulation.selections is not None:
            self.r.timeCourseSelections = simulation.selections

        frames = []
        t_offset = 0.0
        for tc in simulation.timecourses:

            # apply changes
            for key, value in tc.changes.items():
                self.r[key] = value

            # FIXME: model changes (make run in parallel, better handling in model)
            # logger.error("No support for model changes")

            if len(tc.model_changes) > 0:
                r_old = self.r
            for key, value in tc.model_changes.items():
                if key == MODEL_CHANGE_BOUNDARY_CONDITION:
                    for sid, bc in value.items():
                        # setting boundary conditions
                        r_new = clamp_species(r.sid, sid, boundary_condition=bc)
                else:
                    loggeself.r.error("Unsupported model change: {}:{}".format(key, value))


            # run simulation
            s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)
            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset
            frames.append(df)
            t_offset += tc.end

        # reset selections
        self.r.timeCourseSelections = model_selections

        return pd.concat(frames)
