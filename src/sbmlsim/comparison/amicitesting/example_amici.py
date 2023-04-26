"""AMICI example simulation"""
from pathlib import Path
import amici
import numpy as np

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

for key, value in changes.items():

    try:
        # set parameters
        model.setParameterById(key, value)
        print(f"{key} = {value}")
    except RuntimeError as err:
        # FIXME: how to set the initial value of the state species IVDOSE_icg ??
        raise err

# set timepoints
t = np.linspace(0, 10, num=11)
model.setTimepoints(t)

# simulation
rdata = amici.runAmiciSimulation(model, solver)
print(rdata)
