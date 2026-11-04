"""Simulate model with AMICI.

sudo apt-get install libatlas-base-dev swig libhdf5-serial-dev
pip install amici --upgrade
"""

import importlib
from typing import override

import numpy as np
import pandas as pd

from examples.comparison.simulate import Condition, SimulateSBML
from sbmlsim.console import console
from sbmlsim.engine import import_amici

amici = import_amici()
run_simulation = importlib.import_module("amici.sim.sundials").run_simulation


class SimulateAmiciSBML(SimulateSBML):
    """Class for simulating an SBML model with AMICI."""

    def __init__(self, **kwargs):
        """Import the model into AMICI and create the solver."""
        super().__init__(**kwargs)

        # custom model loading
        model_dir = self.results_dir / "amici" / self.mid
        sbml_importer = amici.SbmlImporter(self.sbml_path)
        sbml_importer.sbml2amici(self.mid, model_dir)

        # load the model module
        model_module = amici.import_model_module(self.mid, model_dir)

        # instantiate model
        self.model = model_module.get_model()
        self.state_ids = self.model.get_state_ids()

        # instantiate solver
        self.solver = self.model.create_solver()
        self.solver.set_absolute_tolerance(self.absolute_tolerance)
        self.solver.set_relative_tolerance(self.relative_tolerance)

    @override
    def simulate_condition(
        self, condition: Condition, timepoints: np.ndarray
    ) -> pd.DataFrame:
        """Simulate the condition with AMICI."""
        console.print(f"simulate condition: {condition.sid}")

        # changes
        x0 = np.asarray(self.model.get_initial_state())
        for change in condition.changes:
            tid = change.target_id
            value = change.value
            if np.isnan(value):
                continue

            if tid in self.state_ids:
                # AMICI state variables
                x0[self.state_ids.index(tid)] = value
                console.print(f"{tid} = {value} (state)")
            else:
                # AMICI parameters
                self.model.set_free_parameter_by_id(tid, value)
                console.print(f"{tid} = {value} (parameter)")

        self.model.set_initial_state(x0)

        # set timepoints
        t = np.asarray(timepoints)
        self.model.set_timepoints(t)

        # simulation
        rdata = run_simulation(self.model, self.solver)
        xids = self.model.get_state_ids()

        # create result dataframe
        df = pd.DataFrame(rdata.x, columns=xids)
        df.insert(loc=0, column="time", value=timepoints)

        return df
