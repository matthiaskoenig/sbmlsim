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
from typing import List
import pandas as pd
import roadrunner

from sbmlsim.result import Result
from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
from sbmlsim.simulation.scan import ScanSim

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

    def _run_timecourses(self, simulations: List[TimecourseSim]) -> Result:
        """ Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")

    def run_scan(self, scan: ScanSim) -> Result:
        """ Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")


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
                    logger.error(f"Change is not a Quantity with unit: '{key} = {item}'. "
                                 f"Add units to all changes.")
                    raise err

            # FIXME: implement model changes & also make run in parallel
            """
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
            """

            # run simulation
            s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)
            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset
            frames.append(df)
            t_offset += tc.end

        # reset selections
        self.r.timeCourseSelections = model_selections

        return pd.concat(frames)

