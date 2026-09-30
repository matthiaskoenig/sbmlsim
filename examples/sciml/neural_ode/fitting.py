"""Fit of a neural ODE which is defined in python, in parallel.

    python -m examples.sciml.neural_ode.fitting --runs=2 --cores=2

The model has the species `prey` and `predator` whose rates a small network
(2-5-5-2, 57 elements) gives. The data are the first four seconds of a
Lotka-Volterra system with noise, the two seconds after them are validation
data: the report shows how the fit generalizes. The fit has more elements than
data points (32), so it can and does reach a cost below the one of the true
model: the network which describes the noise does not describe the system
after four seconds, which the validation data shows. The network is a
demonstration of the interface, not of a model.

The fit has two parts. The first run starts from the values of the network
(`SamplingType.START`, `build_network(seed)` decides where it starts), the
second part is a multistart: `--runs` runs which start from their own random
values in the bounds of the elements, in `--cores` workers. The library does
not mix the two in one fit, so they are two fits with a report each, and the
better of the two is written as PEtab SciML and read again.

The network sits in the right hand side: `compile_network` writes the model
with the network into `results/neural_ode` of the working directory, which is
the `base_path` of the problem, so the workers of the fit load the same file.
A fit uses a finite difference jacobian with a small step, which the default
step of `sbmlsim.fit.cli` (5%, made for parameters on a logarithmic scale) is
not. `--max-nfev`, the evaluations of the cost per run, is a demonstration
and stops early: the runs report that they did not converge. The results and
the reports are written into `results/neural_ode/fit`, the fitted problem into
`results/neural_ode/petab`.
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
    SPECIES,
    NeuralODE,
    mapping_id,
)
from examples.sciml.neural_ode.network import NETWORK_ID, build_network
from sbmlsim.fit import FitMappingCollection, FitSettings, MappingKind, display
from sbmlsim.fit.cli import FitDefinition, FitRun, run_fit
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2 import to_petab
from sbmlsim.fit.petab_v2.likelihood import log_likelihood, nominal_parameters
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.sampling import SamplingType
from sbmlsim.sciml import (
    Hybridization,
    NetworkInput,
    NetworkPattern,
    compile_network,
    input_id,
    output_id,
)

#: where the fit writes, the model with the network included; absolute, so the
#: base path of the problem and the paths of the report agree in every worker
RESULTS_PATH = (Path("results") / "neural_ode").resolve()

NETWORK = build_network()

#: the network in the right hand side: the species are its inputs, the rates
#: its outputs
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

#: every element is estimated within the bounds, the start value of an element
#: is its value in the network
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
    """Get the fit mappings: the first four seconds train, the rest validates."""
    return {
        "neural_ode": [
            FitMappingCollection(
                experiment=NeuralODE,
                mappings=[mapping_id(species) for species in SPECIES],
                sid="neural_ode",
            ),
            FitMappingCollection(
                experiment=NeuralODE,
                mappings=[mapping_id(species, validation=True) for species in SPECIES],
                sid="neural_ode_validation",
                kind=MappingKind.VALIDATION,
            ),
        ]
    }


FIT_DEFINITION = FitDefinition(
    mapping_collections=collections,
    parameters=PARAMETERS,
    base_path=RESULTS_PATH,
    data_path=EXAMPLE_PATH,
    settings=FIT_SETTINGS,
    hybridizations=[HYBRIDIZATION],
)


def compile_model() -> Path:
    """Write the model with the network into the results directory."""
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    return compile_network(MODEL_PATH, [HYBRIDIZATION], RESULTS_PATH / COMPILED_MODEL)


def fit(
    opid: str, size: int, n_cores: int, sampling: SamplingType, seed: int, max_nfev: int
) -> FitRun:
    """Fit the network and report the fit."""
    run = run_fit(
        FIT_DEFINITION,
        opid=opid,
        size=size,
        n_cores=n_cores,
        seed=seed,
        output_dir=RESULTS_PATH / "fit",
        sampling=sampling,
        # the step of the finite differences of the jacobian is absolute for
        # elements which are around zero
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=max_nfev,
    )[opid]
    run.report(output_dir=RESULTS_PATH / "fit", show_titles=False)
    return run


def main() -> None:
    """Compile the model, fit the network from its values and from random ones."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=2, help="runs of the multistart")
    parser.add_argument("--cores", type=int, default=2, help="workers of the fit")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the runs")
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=100,
        help="evaluations of the cost per run; the default is a demonstration "
        "which stops early, so the runs do not converge",
    )
    options = parser.parse_args()

    compile_model()

    # --- FIT FROM THE NETWORK ---
    start = fit(
        "neural_ode_start",
        size=1,
        n_cores=1,
        sampling=SamplingType.START,
        seed=options.seed,
        max_nfev=options.max_nfev,
    )
    problem = start.problem
    start_cost = problem.cost_least_square(
        problem.to_scale(nominal_parameters(problem).x(problem.pids))
    )
    # the fit shows its cost, the cost of the values it started from is not
    display.key_values({"start": f"cost {start_cost:.4g} of the network"})

    # --- MULTISTART, IN PARALLEL ---
    multistart = fit(
        "neural_ode_multistart",
        size=options.runs,
        n_cores=options.cores,
        sampling=SamplingType.UNIFORM,
        seed=options.seed,
        max_nfev=options.max_nfev,
    )

    # --- WRITE THE BETTER FIT AND READ IT AGAIN ---
    best = min((start, multistart), key=lambda run: run.result.parameter_set().cost)
    parameter_set = best.result.parameter_set()
    display.section("PEtab SciML", icon=":package:")
    yaml_file = to_petab(
        best.problem,
        RESULTS_PATH / "petab",
        settings=FIT_SETTINGS,
        parameter_set=parameter_set,
    )
    reader = PetabReader.from_yaml(yaml_file)
    reader.derived_dir = RESULTS_PATH / "petab" / "derived"
    restored = reader.to_optimization_problem(opid="restored")
    restored.initialize(FIT_SETTINGS)
    display.key_values(
        {
            "written": yaml_file.relative_to(Path.cwd()),
            "fit": best.problem.opid,
            "the fit": f"log-likelihood {log_likelihood(best.problem, parameter_set):.10g}",
            "read again": f"log-likelihood {log_likelihood(restored):.10g}",
        }
    )


if __name__ == "__main__":
    main()
