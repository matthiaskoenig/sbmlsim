"""
Classes for running simulations with SBML models.
"""
import logging

import pandas as pd
from roadrunner import RoadRunner, SelectionRecord

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

        Requires for all timecourse definitions in the timecourse simulation
        to be unit normalized. The changes have no units any more
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
        for k, tc in enumerate(simulation.timecourses):
            # print(tc.to_dict())

            if k == 0 and tc.model_changes:
                # [1] apply model changes of first simulation
                logger.warning("Applying model changes")
                for key, item in tc.model_changes.items():
                    try:
                        self.r[key] = item.magnitude
                    except AttributeError as err:
                        self.r[key] = item

                # [2] re-evaluate initial assignments
                # https://github.com/sys-bio/roadrunner/issues/710
                logger.warning("Reavaluate initial conditions")
                # FIXME: how to re-evalutate initial assignments
                # self.r.reset(SelectionRecord.INITIAL_GLOBAL_PARAMETER)
                # self.r.reset(SelectionRecord.DEPENDENT_INITIAL_GLOBAL_PARAMETER)

            # [3] apply model manipulations
            # model manipulations are applied to model
            if len(tc.model_manipulations) > 0:
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

            # [4] apply changes
            for key, item in tc.changes.items():
                try:
                    self.r[key] = item.magnitude
                    # logger.warning(f"Set: {key} = {item.magnitude}")
                except AttributeError as err:
                    self.r[key] = item
                    # logger.warning(f"Set: {key} = {item}")

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
