"""Simulate model with AMICI."""

from typing import List, Dict

import pandas as pd

from pathlib import Path
from typing import List

import amici
import numpy as np
import pandas as pd

from sbmlsim.comparison.simulate import SimulateSBML, Condition, Timepoints


class SimulateAmiciSBML(SimulateSBML):
    """Class for simulating an SBML model with AMICI."""

    def __init__(self, sbml_path, conditions: List[Condition], results_dir: Path):
        super().__init__(
            sbml_path=sbml_path,
            conditions=conditions,
            results_dir=results_dir
        )

        # custom model loading
        model_dir = self.results_dir / "amici" / self.mid
        # sbml_importer = amici.SbmlImporter(self.sbml_path)
        # sbml_importer.sbml2amici(self.mid, model_dir)

        # load the model module
        model_module = amici.import_model_module(self.mid, model_dir)

        # instantiate model
        self.model = model_module.getModel()


    def simulate_condition(self, condition: Condition, timepoints: Timepoints) -> pd.DataFrame:
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
        t = np.linspace(timepoints.start, timepoints.end, num=timepoints.steps + 1)
        self.model.setTimepoints(t)

        # instantiate solver
        solver = self.model.getSolver()
        # solver.setAbsoluteTolerance(1e-10)

        # simulation
        rdata = amici.runAmiciSimulation(self.model, solver)
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
