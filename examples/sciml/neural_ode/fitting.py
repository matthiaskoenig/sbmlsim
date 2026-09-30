"""Fit of a neural ODE which is defined in python, in parallel.

    python -m examples.sciml.neural_ode.fitting --runs=2 --cores=2

The network sits in the right hand side: `compile_network` writes the model
with the network into `results/neural_ode` of the working directory, which is
the `base_path` of the problem, so the workers of the fit load the same file.
The elements are bounded, so every run starts from its own random values in
the bounds; an element without bounds starts from the value of the network in
every run. The fit uses a finite difference jacobian with a small step, which
the default step of `sbmlsim.fit.cli` (5%, made for parameters on a
logarithmic scale) is not. The results and the report are written into
`results/neural_ode/fit`.
"""

import argparse
import sys
from pathlib import Path

# run as a script (`python examples/sciml/neural_ode/fitting.py`, the "run
# file" of an IDE) the repository is not on `sys.path`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.sciml.neural_ode.experiment import (
    COMPILED_MODEL,
    EXAMPLE_PATH,
    MODEL_PATH,
    NeuralODE,
)
from examples.sciml.neural_ode.network import NETWORK_ID, build_network
from sbmlsim.fit import FitMappingCollection, FitSettings
from sbmlsim.fit.cli import FitDefinition, run_fit
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.sciml import (
    Hybridization,
    NetworkInput,
    NetworkPattern,
    compile_network,
    input_id,
    output_id,
)

#: where the fit writes, the model with the network included
RESULTS_PATH = Path("results") / "neural_ode"

NETWORK = build_network()

#: the network in the right hand side: the species are its inputs, the rates
#: its outputs. Every element is estimated within the bounds
HYBRIDIZATION = Hybridization(
    network=NETWORK,
    pattern=NetworkPattern.RHS,
    model="lv",
    inputs={
        input_id(NETWORK_ID, 0, (0,)): NetworkInput(formula="prey"),
        input_id(NETWORK_ID, 0, (1,)): NetworkInput(formula="predator"),
    },
    outputs={
        output_id(NETWORK_ID, 0, (0,)): "prey_param",
        output_id(NETWORK_ID, 0, (1,)): "predator_param",
    },
)
PARAMETERS, HYBRIDIZATION = HYBRIDIZATION.fit_parameters(
    estimate={NETWORK_ID: True}, bounds={NETWORK_ID: (-3.0, 3.0)}
)

#: the elements are searched on the linear scale, and a difference of the
#: cost needs a fixed grid
FIT_SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
)


def collections() -> dict[str, list[FitMappingCollection]]:
    """Get the fit mappings, both species of the one experiment."""
    return {
        "neural_ode": [FitMappingCollection(experiment=NeuralODE, sid="neural_ode")]
    }


FIT_DEFINITIONS: dict[str, FitDefinition] = {
    "NODE": FitDefinition(
        mapping_collections=collections,
        parameters=PARAMETERS,
        base_path=RESULTS_PATH,
        data_path=EXAMPLE_PATH,
        settings=FIT_SETTINGS,
        hybridizations=[HYBRIDIZATION],
    )
}


def compile_model() -> Path:
    """Write the model with the network into the results directory."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    return compile_network(MODEL_PATH, [HYBRIDIZATION], RESULTS_PATH / COMPILED_MODEL)


def main() -> None:
    """Compile the model, fit the network and report the fit."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--runs", type=int, default=2, help="optimization runs")
    parser.add_argument("--cores", type=int, default=2, help="workers of the fit")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the runs")
    parser.add_argument(
        "--max-nfev", type=int, default=100, help="evaluations of the cost per run"
    )
    options = parser.parse_args()

    compile_model()
    runs = run_fit(
        FIT_DEFINITIONS["NODE"],
        opid="neural_ode",
        size=options.runs,
        n_cores=options.cores,
        seed=options.seed,
        output_dir=RESULTS_PATH / "fit",
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=options.max_nfev,
    )
    for run in runs.values():
        run.report(output_dir=RESULTS_PATH / "fit", show_titles=False)


if __name__ == "__main__":
    main()
