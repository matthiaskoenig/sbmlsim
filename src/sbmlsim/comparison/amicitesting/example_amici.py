"""AMICI example simulation"""
from pathlib import Path
import amici
import numpy as np
import pandas as pd

mid = "icg_sd"
base_path: Path = Path(__file__).parent
model_dir = base_path / mid
model_path = base_path / f"{mid}.xml"

sbml_importer = amici.SbmlImporter(model_path)
sbml_importer.sbml2amici(mid, model_dir)

# load the model module
model_module = amici.import_model_module(mid, model_dir)

# instantiate model
model = model_module.getModel()

# instantiate solver
solver = model.getSolver()

# changes
changes = {
    "BW": 83.4,  # parameter initial condition
    "Ri_icg": 0.0,  # parameter initial condition
    "IVDOSE_icg": 41.7,  # species initial condition
}

state_ids = model.getStateIds()
x0 = np.asarray(model.getInitialStates())
for key, value in changes.items():
    if key in state_ids:
        # AMICI state variables
        x0[state_ids.index(key)] = value
        print(f"{key} = {value} (state)")
    else:
        # AMICI parameters
        model.setParameterById(key, value)
        print(f"{key} = {value} (parameter)")

# set timepoints
timepoints = np.linspace(0, 10, num=11)
model.setTimepoints(timepoints)

# simulation
rdata = amici.runAmiciSimulation(model, solver)
df = pd.DataFrame(rdata.x, columns=state_ids)
df.insert(loc=0, column="time", value=timepoints)
print(df)
