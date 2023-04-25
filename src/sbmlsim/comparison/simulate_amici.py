"""Simulate model with AMICI.

sudo apt-get install libatlas-base-dev swig libhdf5-serial-dev
pip install amici --upgrade
"""

from typing import List, Dict
from sbmlutils.console import console
import pandas as pd

from pathlib import Path
from typing import List

import amici
from amici.numpy import ReturnDataView
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
        for change in condition.changes:
            tid = change.target_id
            print(tid)
            value = change.value
            try:
                self.model.setParameterById(tid, value)
                console.print(f"{tid} = {value}")
            except RuntimeError:
                # FIXME: how to set the initial value of a state
                pass


        # set timepoints
        t = np.asarray(timepoints)
        self.model.setTimepoints(t)

        # simulation
        rdata = amici.runAmiciSimulation(self.model, self.solver)
        xids = self.model.getStateIds()

        # def _ids_and_names_to_rdata(

        # create result dataframe
        # print("Model parameters:", list(model.getParameterIds()), "\n")
        # print("Model const parameters:", list(model.getFixedParameterIds()), "\n")
        # print("Model outputs:", list(model.getObservableIds()), "\n")
        # print("Model states:", list(model.getStateIds()), "\n")
        df = pd.DataFrame(rdata.x, columns=xids)
        df.insert(loc=0, column="time", value=timepoints)

        return df
