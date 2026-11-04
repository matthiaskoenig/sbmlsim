"""Simulation of SBML models with COPASI, the engine `copasi` of sbmlsim.

The engine is the optional extra `copasi` of sbmlsim, `pip install
"sbmlsim[copasi]"`; a condition is a timecourse whose changes are set before
the initialization of the model.
"""

from typing import override

import numpy as np
import pandas as pd

from examples.comparison.simulate import Condition, SimulateSBML
from sbmlsim.console import console
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator


class SimulateCopasiSBML(SimulateSBML):
    """Class for simulating an SBML model with the engine COPASI of sbmlsim."""

    def __init__(self, **kwargs):
        """Load the model into COPASI, with the selections of the comparison."""
        super().__init__(**kwargs)
        self.simulator = Simulator(
            n_workers=1,
            engine="copasi",
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance,
        )
        # the amounts of the species with only substance units, the
        # concentrations of the others, as the other simulators select them
        selections = ["time", *(self._target(s) for s in self.species)]
        selections += list(self.parameters)
        selections += list(self.compartments)
        self.model = self.simulator.load(
            AbstractModel(source=self.sbml_path, selections=selections)
        )

    def _target(self, sid: str) -> str:
        """Get the selection of an entity: `[S]` for a concentration species."""
        if sid in self.species and not self.has_only_substance[sid]:
            return f"[{sid}]"
        return sid

    @override
    def simulate_condition(
        self, condition: Condition, timepoints: np.ndarray
    ) -> pd.DataFrame:
        """Simulate the condition with COPASI."""
        console.print(f"simulate condition: {condition.sid}")
        preinit_changes: dict[str, float] = {}
        for change in condition.changes:
            if np.isnan(change.value):
                continue
            target = self._target(change.target_id)
            preinit_changes[target] = float(change.value)
            console.print(f"{target} = {change.value}", markup=False)

        simulation = Simulation(
            start=float(timepoints[0]),
            end=float(timepoints[-1]),
            times=[float(t) for t in timepoints],
            preinit_changes=preinit_changes,
        )
        res = self.simulator.run(self.model, simulation)
        df = res.ds.to_dataframe().reset_index()

        # the names of the concentration species without brackets
        return df.rename(columns=lambda c: str(c).removeprefix("[").removesuffix("]"))
