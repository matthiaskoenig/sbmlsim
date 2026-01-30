"""Access variables in AMICI.

https://github.com/AMICI-dev/AMICI/blob/master/documentation/GettingStarted.ipynb
# MPIPoolEngine; https://github.com/ICB-DCM/pyPESTO/blob/86f4feaa8024c14bd0037cfc1522d9d5bb3fa3b6/doc/example/example_MPIPool.py
# https://github.com/ICB-DCM/pyPESTO/blob/main/pypesto/engine/mpi_pool.py

"""

import numpy as np
import amici

sbml_importer = amici.SbmlImporter("pravastatin_body_all_flat.xml")

model_name = "model_pravastatin"
model_dir = "model_pravastatin"
sbml_importer.sbml2amici(model_name, model_dir)

# load the model module
model_module = amici.import_model_module(model_name, model_dir)
# instantiate model
model = model_module.getModel()
# instantiate solver
solver = model.getSolver()

# set tolerances
solver.setAbsoluteTolerance(1e-10)

# set timepoints

timepoints = np.linspace(0, 24 * 60, 10)
model.setTimepoints(timepoints)
rdata = amici.runAmiciSimulation(model, solver)

# careful with constant compartments;
print(rdata.x)

print(rdata.by_id("IVDOSE_pra"))

# FIXME: How to make this fast and access parameters, compartments, variables, ...?
# https://amici.readthedocs.io/en/latest/ExampleSteadystate.html
# amici.pandas: rdata
