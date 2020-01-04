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
from typing import List, Union, Dict

import pandas as pd
import roadrunner
import itertools
from copy import deepcopy
from pprint import pprint

from sbmlsim.model import clamp_species, MODEL_CHANGE_BOUNDARY_CONDITION
from sbmlsim.result import Result
from sbmlsim.timecourse import Timecourse, TimecourseSim, TimecourseScan

logger = logging.getLogger(__name__)

# --------------------------------
# Integrator settings
# --------------------------------
# FIXME: implement setting of ode solver properties: variable_step_size, stiff, absolute_tolerance, relative_tolerance
def set_integrator_settings(r: roadrunner.RoadRunner, **kwargs) -> None:
    """ Set integrator settings.

    Keys are:
        variable_step_size [boolean]
        stiff [boolean]
        absolute_tolerance [float]
        relative_tolerance [float]

    """
    integrator = r.getIntegrator()
    for key, value in kwargs.items():
        # adapt the absolute_tolerance relative to the amounts
        if key == "absolute_tolerance":
            value = value * min(r.model.getCompartmentVolumes())
        integrator.setValue(key, value)
    return integrator


def set_default_settings(r: roadrunner.RoadRunner, **kwargs):
    """ Set default settings of integrator. """
    set_integrator_settings(r,
            variable_step_size=True,
            stiff=True,
            absolute_tolerance=1E-8,
            relative_tolerance=1E-8
    )


class SimulatorAbstract(object):
    def __init__(self, path, selections: List[str] = None, **kwargs):
        """ Must be implemented by simulator. """
        pass

    def timecourses(self, simulations: List[TimecourseSim]) -> Result:
        """ Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")

    def scan(self, tcscan: TimecourseScan) -> Result:
        """ Timecourse simulations based on timecourse_definition.

        :param tcscan: Scan definition
        :param reset_all: Reset model at the beginning
        :return:
        """
        # Create all possible combinations of the scan

        # TODO: refactor on TimecourseScan
        keys = []
        vecs = []
        index_vecs = []
        for key, vec in tcscan.scan.items():
            keys.append(key)
            vecs.append(list(vec))
            index_vecs.append(range(len(vec)))

        indices = list(itertools.product(*index_vecs))

        # from pprint import pprint
        # pprint(keys)
        # pprint(changes_values)

        sims = []
        for index_list in indices:
            sim_new = deepcopy(tcscan.tcsim)
            # changes are mixed in the first timecourse
            tc = sim_new.timecourses[0]
            for k, pos_index in enumerate(index_list):
                key = keys[k]
                value = vecs[k][pos_index]
                tc.add_change(key, value)
            sims.append(sim_new)

        result = self.timecourses(sims)
        result.keys = keys
        result.vecs = vecs
        result.indices = indices

        return result


class SimulatorWorker(object):

    def timecourse(self, simulation: TimecourseSim) -> pd.DataFrame:
        """ Timecourse simulations based on timecourse_definition.

        :param simulation: Simulation definition(s)
        :return:
        """

        if isinstance(simulation, Timecourse):
            simulation = TimecourseSim(timecourses=[simulation])
            logger.warning("Default TimecourseSim created for Timecourse. Best practise is to"
                           "provide a TimecourseSim instance.")

        if simulation.reset:
            self.r.resetToOrigin()

        # selections backup
        model_selections = self.r.timeCourseSelections
        if simulation.selections is not None:
            self.r.timeCourseSelections = simulation.selections

        frames = []
        t_offset = simulation.time_offset
        for tc in simulation.timecourses:
            if not tc.normalized:
                tc.normalize(udict=self.udict, ureg=self.ureg)

            # apply changes
            for key, item in tc.changes.items():
                try:
                    self.r[key] = item.magnitude
                except AttributeError as err:
                    logger.error(f"Change is not a Quantity with unit: '{key} = {item}'")
                    raise err

            # FIXME: model changes (make run in parallel, better handling in model)
            if len(tc.model_changes) > 0:
                r_old = self.r
            for key, value in tc.model_changes.items():
                if key == MODEL_CHANGE_BOUNDARY_CONDITION:
                    for sid, bc in value.items():
                        # setting boundary conditions
                        r_new = clamp_species(self.r, sid, boundary_condition=bc)
                        self.r = r_new
                else:
                    logger.error("Unsupported model change: {}:{}".format(key, value))

            # run simulation
            s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)
            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset
            frames.append(df)
            t_offset += tc.end

        # reset selections
        self.r.timeCourseSelections = model_selections

        return pd.concat(frames)

