"""Parallel simulation using ray."""

import logging
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd
import psutil
import ray
import roadrunner

from sbmlsim.simulation import TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation import SimulatorWorker


logger = logging.getLogger(__name__)

# start ray
ray.init(ignore_reinit_error=True)


def cpu_count() -> int:
    """Get physical CPU count."""
    return psutil.cpu_count(logical=False)


@ray.remote
class SimulatorActor(SimulatorWorker):
    """Ray actor to execute simulations.

    An actor is essentially a stateful worker
    """

    def __init__(self, path_state=None):
        """State contains model, integrator settings and selections."""
        self.r = None
        if path_state is not None:
            self.set_model(path_state)

    def set_model(self, path_state: Path) -> None:
        """Set model using the Path to a state file.

        Faster to load the state.
        """
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner()
        if path_state is not None:
            self.r.loadState(str(path_state))

    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set the timecourse selections."""
        try:
            if selections is None:
                r_model: roadrunner.ExecutableModel = self.r.model
                self.r.timeCourseSelections = (
                    ["time"]
                    + r_model.getFloatingSpeciesIds()
                    + r_model.getBoundarySpeciesIds()
                    + r_model.getGlobalParameterIds()
                    + r_model.getReactionIds()
                    + r_model.getCompartmentIds()
                )
                self.r.timeCourseSelections += [
                    f"[{key}]"
                    for key in (
                        r_model.getFloatingSpeciesIds()
                        + r_model.getBoundarySpeciesIds()
                    )
                ]
            else:
                self.r.timeCourseSelections = selections
        except RuntimeError as err:
            print(f"ERROR: {err}")
            raise (err)

    def work(self, simulations):
        """Run a bunch of simulations on a single worker."""
        results = []
        for tc_sim in simulations:
            results.append(self._timecourse(tc_sim))
        return results


class SimulatorParallel(SimulatorSerial):
    """Parallel simulator.

    The parallel simulator is a subclass of the SimulatorSerial reusing the
    logic for running simulations.
    """

    def __init__(self, model=None, **kwargs):
        """Initialize parallel simulator with multiple workers.

        :param model: model source or model
        :param actor_count: int,
        """
        if "actor_count" in kwargs:
            self.actor_count = kwargs.pop("actor_count")
        else:

            # FIXME: get virtual cores
            self.actor_count = max(cpu_count() - 1, 1)
        logger.info(f"Using '{self.actor_count}' cpu/core for parallel simulation.")

        # Create actors once
        logger.warning(f"Creating '{self.actor_count}' SimulationActors")
        self.simulators = [SimulatorActor.remote() for _ in range(self.actor_count)]

        super(SimulatorParallel, self).__init__(model=None, **kwargs)
        if model is not None:
            self.set_model(model=model)

    def set_model(self, model):
        """Set model."""
        super(SimulatorParallel, self).set_model(model)
        if model:
            if not self.model.state_path:
                raise ValueError("State path does not exist.")

            state_path = str(self.model.state_path)
            for simulator in self.simulators:
                simulator.set_model.remote(state_path)
            self.set_timecourse_selections(self.r.selections)

        # FIXME: set integrator settings

    def set_timecourse_selections(self, selections: Iterator[str]):
        """Set the timecourse selections."""
        for simulator in self.simulators:
            simulator.set_timecourse_selections.remote(selections)

    def _timecourses(self, simulations: List[TimecourseSim]) -> List[pd.DataFrame]:
        """Run all simulations with given model and collect the results.

        :param simulations: List[TimecourseSim]
        :return: Result
        """
        # Strip units for parallel simulations
        for sim in simulations:
            sim.strip_units()

        # Split simulations in chunks for actors
        # !simulation have to stay in same order to reconstruct dimensions!
        chunk_indices = np.array_split(np.arange(len(simulations)), self.actor_count)
        chunks = [[] for _ in range(self.actor_count)]
        for k, indices in enumerate(chunk_indices):
            for index in indices:
                chunks[k].append(simulations[index])

        tc_ids = []
        for k, simulator in enumerate(self.simulators):
            tcs_id = simulator.work.remote(chunks[k])
            tc_ids.append(tcs_id)

        results = ray.get(tc_ids)
        # flatten list of lists [[df, df], [df, df], ...]
        # indices = [k for sublist in chunks_indices for k in sublist]
        return [df for sublist in results for df in sublist]

    @staticmethod
    def _create_chunks(item, size: int):
        """Yield successive sized chunks from item."""
        for i in range(0, len(item), size):
            yield item[i : i + size]
