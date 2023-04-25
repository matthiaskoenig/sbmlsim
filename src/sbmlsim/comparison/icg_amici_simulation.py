
from pathlib import Path
from typing import List

import amici
import numpy as np
import pandas as pd


output_dir: Path = Path(__file__).parent / "results"
model_mids: List[str] = [
    "icg_body_flat",
    "icg_body_events_flat"
]

# Check if AMICI can simulate the model
for mid in model_mids:
    model_file: Path = output_dir / f"{mid}.xml"
    print("--- amici ---")
    sbml_importer = amici.SbmlImporter(model_file)
    model_dir = output_dir / "amici" / mid
    sbml_importer.sbml2amici(mid, model_dir)

    # load the model module
    model_module = amici.import_model_module(mid, model_dir)
    # instantiate model
    model = model_module.getModel()

    # model properties
    print("Model parameters:", list(model.getParameterIds()), "\n")
    print("Model const parameters:", list(model.getFixedParameterIds()), "\n")
    print("Model outputs:", list(model.getObservableIds()), "\n")
    print("Model states:", list(model.getStateIds()), "\n")

    # instantiate solver
    solver = model.getSolver()
    solver.setAbsoluteTolerance(1e-10)

    # run forward simulation
    t = np.linspace(0, 20, num=5)
    model.setTimepoints(t)
    rdata = amici.runAmiciSimulation(model, solver)
    print(t)
    print(rdata.x)

    # forward simulation with roadrunner
    print("--- roadrunner ---")
    import roadrunner

    r = roadrunner.RoadRunner(str(model_file))
    s = r.simulate(0, 20, steps=5)
    df_roadrunner = pd.DataFrame(s, columns=s.colnames)
    print(df_roadrunner.head())

