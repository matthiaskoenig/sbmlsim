"""
Classes for running simulations with SBML models.
"""
import logging

import pandas as pd

from sbmlsim.model import ModelChange
from sbmlsim.result import XResult
from sbmlsim.simulation import ScanSim, Timecourse, TimecourseSim


logger = logging.getLogger(__name__)


class SimulatorAbstract(object):
    def run_timecourse(self, simulation: TimecourseSim) -> XResult:
        """Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")

    def run_scan(self, scan: ScanSim) -> XResult:
        """Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")


class SimulatorWorker(object):
    def _timecourse(self, simulation: TimecourseSim) -> pd.DataFrame:
        """Timecourse simulation.

        Requires for all timecourse definitions to be unit NORMALIZED
        before being sent here ! The changes have no units any more
        for parallel simulations.
        You should never call this function directly!

        :param simulation: Simulation definition(s)
        :return:
        """
        if isinstance(simulation, Timecourse):
            simulation = TimecourseSim(timecourses=[simulation])

        if simulation.reset:
            self.r.resetToOrigin()

        frames = []
        t_offset = simulation.time_offset
        for tc in simulation.timecourses:

            # apply changes
            for key, item in tc.changes.items():
                try:
                    self.r[key] = item.magnitude
                except AttributeError as err:
                    self.r[key] = item

            # model changes are applied to model
            if len(tc.model_changes) > 0:
                for key, value in tc.model_changes.items():
                    if key == ModelChange.CLAMP_SPECIES:
                        for sid, formula in value.items():
                            ModelChange.clamp_species(self.r, sid, formula)
                    else:
                        raise ValueError(
                            f"Unsupported model change: "
                            f"'{key}': {value}. Supported changes are: "
                            f"['{ModelChange.CLAMP_SPECIES}']"
                        )

            # run simulation
            integrator = self.r.integrator
            if integrator.getValue("variable_step_size"):
                s = self.r.simulate(start=tc.start, end=tc.end)
            else:
                s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)

            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset
            frames.append(df)
            t_offset += tc.end

        return pd.concat(frames, sort=False)
