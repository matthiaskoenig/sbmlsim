"""Classes for running simulations with SBML models."""
import logging

import pandas as pd
from roadrunner import RoadRunner, SelectionRecord

from sbmlsim.model import ModelChange
from sbmlsim.result import XResult
from sbmlsim.simulation import ScanSim, Timecourse, TimecourseSim


logger = logging.getLogger(__name__)


class SimulatorWorker:
    """Worker running simulations.

    Implements the timecourse simulation once which can be reused by
    the different simulators.
    """

    def __init__(self):
        """Init placeholder."""
        self.r = None

    def _timecourse(self, simulation: TimecourseSim) -> pd.DataFrame:
        """Timecourse simulation.

        Requires for all timecourse definitions in the timecourse simulation
        to be unit normalized. The changes have no units any more
        for parallel simulations.
        You should never call this function directly!

        :param simulation: Simulation definition(s)
        :return: DataFrame with results
        """
        if isinstance(simulation, Timecourse):
            simulation = TimecourseSim(timecourses=[simulation])

        if simulation.reset:
            self.r.resetToOrigin()

        frames = []
        t_offset = simulation.time_offset
        for k, tc in enumerate(simulation.timecourses):

            if k == 0 and tc.model_changes:
                # [1] apply model changes of first simulation
                logger.debug("Applying model changes")
                for key, item in tc.model_changes.items():
                    if key.startswith("init"):
                        logger.error(
                            f"Initial model changes should be provided "
                            f"without 'init': '{key} = {item}'"
                        )
                    # FIXME: implement model changes via init
                    # init_key = f"init({key})"
                    init_key = key
                    try:
                        value = item.magnitude
                    except AttributeError:
                        value = item

                    try:
                        self.r[init_key] = value
                    except RuntimeError:
                        logger.error(f"roadrunner RuntimeError: '{init_key} = {item}'")
                        # boundary condition=true species, trying direct fallback
                        # see https://github.com/sys-bio/roadrunner/issues/711
                        init_key = key
                        self.r[key] = value

                    logger.debug(f"\t{init_key} = {item}")

                # [2] re-evaluate initial assignments
                # https://github.com/sys-bio/roadrunner/issues/710
                logger.debug("Reavaluate initial conditions")
                # FIXME: support initial model changes
                # self.r.resetAll()
                self.r.reset(SelectionRecord.DEPENDENT_FLOATING_AMOUNT)
                self.r.reset(SelectionRecord.DEPENDENT_INITIAL_GLOBAL_PARAMETER)

            # [3] apply model manipulations
            # model manipulations are applied to model
            if len(tc.model_manipulations) > 0:
                # FIXME: update to support roadrunner model changes
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
                logger.debug("Applying simulation changes")
                try:
                    self.r[key] = item.magnitude
                except AttributeError:
                    self.r[key] = item
                logger.debug(f"\t{key} = {item}")

            # run simulation
            integrator = self.r.integrator
            if integrator.getValue("variable_step_size"):
                s = self.r.simulate(start=tc.start, end=tc.end)
            else:
                s = self.r.simulate(start=tc.start, end=tc.end, steps=tc.steps)

            df = pd.DataFrame(s, columns=s.colnames)
            df.time = df.time + t_offset

            if not tc.discard:
                # discard timecourses (pre-simulation)
                t_offset += tc.end
                frames.append(df)

        return pd.concat(frames, sort=False)
