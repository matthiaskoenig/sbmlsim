"""Simulate model with AMICI.

sudo apt-get install libatlas-base-dev swig libhdf5-serial-dev
pip install amici --upgrade
"""

from typing import List, Dict

import pandas as pd

from pathlib import Path
from typing import List

import amici
import numpy as np
import pandas as pd

from sbmlsim.comparison.simulate import SimulateSBML, Condition


class SimulateAmiciSBML(SimulateSBML):
    """Class for simulating an SBML model with AMICI."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # custom model loading
        model_dir = self.results_dir / "amici" / self.mid
        # sbml_importer = amici.SbmlImporter(self.sbml_path)
        # sbml_importer.sbml2amici(self.mid, model_dir)

        # load the model module
        model_module = amici.import_model_module(self.mid, model_dir)

        # instantiate model
        self.model = model_module.getModel()

        # instantiate solver
        self.solver = self.model.getSolver()
        self.solver.setAbsoluteTolerance(self.absolute_tolerance)
        self.solver.setRelativeTolerance(self.relative_tolerance)

    def simulate_condition(self, condition: Condition, timepoints: np.ndarray) -> pd.DataFrame:
        print(f"simulate condition: {condition.sid}")

        # changes
        # for change in condition.changes:
        #     tid = change.target_id
        #     value = change.value
        #     # is species
        #     if tid in self.parameters:
        #         set_parameters(tid, initial_value=value)
        #     elif tid in self.compartments:
        #         set_compartment(tid, initial_value=value)
        #     elif tid in self.species:
        #         if tid in self.has_only_substance:
        #             set_species(tid, initial_expression=f"{value}/compartment")
        #         else:
        #             # concentration
        #             set_species(tid, initial_concentration=value)

        # set timepoints
        t = np.asarray(timepoints)
        self.model.setTimepoints(t)

        # simulation
        rdata = amici.runAmiciSimulation(self.model, self.solver)
        print(t)
        print(rdata.x)  # state variables;

        # def _ids_and_names_to_rdata(

        # create result dataframe
        # print("Model parameters:", list(model.getParameterIds()), "\n")
        # print("Model const parameters:", list(model.getFixedParameterIds()), "\n")
        # print("Model outputs:", list(model.getObservableIds()), "\n")
        # print("Model states:", list(model.getStateIds()), "\n")
        df = None


        return df
