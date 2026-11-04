"""Simulation of the ICG model with AMICI."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from sbmlsim.engine import import_amici

amici = import_amici()
run_simulation = importlib.import_module("amici.sim.sundials").run_simulation

output_dir: Path = Path(__file__).parent / "results"
model_mids: list[str] = ["icg_body_flat", "icg_body_events_flat"]

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
    model = model_module.get_model()

    # model properties
    print("Model parameters:", list(model.get_free_parameter_ids()), "\n")
    print("Model const parameters:", list(model.get_fixed_parameter_ids()), "\n")
    print("Model outputs:", list(model.get_observable_ids()), "\n")
    print("Model states:", list(model.get_state_ids()), "\n")

    # instantiate solver
    solver = model.create_solver()
    solver.set_absolute_tolerance(1e-10)

    # run forward simulation
    t = np.linspace(0, 20, num=5)
    model.set_timepoints(t)
    rdata = run_simulation(model, solver)
    print(t)
    print(rdata.x)

    # forward simulation with roadrunner
    print("--- roadrunner ---")
    import roadrunner

    r = roadrunner.RoadRunner(str(model_file))
    s = r.simulate(0, 20, steps=5)
    df_roadrunner = pd.DataFrame(s, columns=s.colnames)
    print(df_roadrunner.head())
