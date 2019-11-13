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
from pint.errors import DimensionalityError

import pandas as pd
import roadrunner

from sbmlsim.model import clamp_species, MODEL_CHANGE_BOUNDARY_CONDITION, load_model
from sbmlsim.result import Result
from sbmlsim.timecourse import Timecourse, TimecourseSim

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

    def timecourses(self):
        """ Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")


class SimulatorWorker(object):

    def timecourse(self, simulation: TimecourseSim) -> pd.DataFrame:
        """ Timecourse simulations based on timecourse_definition.

        :param r: Roadrunner model instance
        :param simulation: Simulation definition(s)
        :param reset_all: Reset model at the beginning
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

            # apply changes
            for key, item in tc.changes.items():
                if hasattr(item, "units"):
                    # pint
                    # perform unit conversion
                    if self.units:
                        try:
                            # FIXME: handle the conversion prefactors correctly
                            # logger.warning(self.units[key])
                            item_converted = item.to(self.units[key]) * item.magnitude
                            logger.info(f"Unit converted: {item} -> {item_converted}")
                            item = item_converted
                        except DimensionalityError as err:
                            logger.error(f"DimensionalityError "
                                         f"'{key} = {item}'. {err}")
                            raise err

                    else:
                        logger.warning(f"Not possible to check units for change: '{item}'")

                    value = item.magnitude
                else:
                    value = item

                self.r[key] = value

            # FIXME: model changes (make run in parallel, better handling in model)
            # logger.error("No support for model changes")

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
