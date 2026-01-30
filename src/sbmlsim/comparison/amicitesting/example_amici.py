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
model = model_module.getModel()

# instantiate solver
solver = model.getSolver()

# update BW parameter which changes volumes via rules!
model.setParameterById("BW", 80)

# set timepoints
timepoints = np.linspace(0, 10, num=11)
model.setTimepoints(timepoints)

# simulation
rdata = amici.runAmiciSimulation(model, solver)
xids = model.getStateIds()
df = pd.DataFrame(rdata.x, columns=xids)
df.insert(loc=0, column="time", value=timepoints)
print(df)
