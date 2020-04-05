"""
Classes for running simulations with SBML models.
"""
import logging
from typing import List
import roadrunner
import pandas as pd
import xarray as xr

from sbmlsim.model import ModelChange
from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
from sbmlsim.simulation.scan import ScanSim

logger = logging.getLogger(__name__)


class SimulatorAbstract(object):
    def __init__(self, path, selections: List[str] = None, **kwargs):
        """ Must be implemented by simulator. """
        pass

    def run_timecourse(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """ Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")

    def run_scan(self, scan: ScanSim) -> xr.Dataset:
        """ Must be implemented by simulator.

        :return:
        """
        raise NotImplementedError("Use concrete implementation")


class SimulatorWorker(object):

    def _timecourse(self, simulation: TimecourseSim) -> pd.DataFrame:
        """ Timecourse simulation.

        Requires for all timecourse definitions to be unit normalized
        before being sent here !
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
                    logger.error(
                        f"Change is not a Quantity: '{key} = {item}'. "
                        f"Units are required for all changes.")
                    raise err

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
                            f"['{ModelChange.CLAMP_SPECIES}']")

            # run simulation
            s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)
            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset
            frames.append(df)
            t_offset += tc.end

        return pd.concat(frames, sort=False)
