"""A PEtab SciML problem read, fitted, reported and written again.

    python -m examples.sciml.lotka_volterra_fit --runs=1

The problem is the case 001 of the PEtab SciML test suite,
`examples/sciml/lotka_volterra/`: the Lotka-Volterra model with a feed
forward network in the right hand side, which replaces the interaction term
of the predator. The example reads the problem, prints its networks, fits the
parameters of the model and the elements of the network together, reports the
fit, writes the problem as PEtab SciML again and reads it back: the
log-likelihoods of the two agree.

A problem which is read from PEtab builds its simulation experiment at
runtime, which the workers of a parallel fit cannot import, so the fit runs in
one process. The results are written into `results/lotka_volterra` of the
working directory, with the model which carries the network,
`lv_sciml.xml`.
"""

import argparse
import sys
from pathlib import Path

# run as a script (`python examples/sciml/lotka_volterra_fit.py`, the "run
# file" of an IDE) the repository is not on `sys.path`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sbmlsim.console import console
from sbmlsim.fit import display
from sbmlsim.fit.derived import hook_summaries
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table, to_petab
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization

#: the problem of the case 001 of the test suite
PROBLEM_PATH = Path(__file__).parent / "lotka_volterra" / "problem.yaml"

#: the elements of a network are searched on the linear scale, and a
#: difference of the cost needs a fixed grid
FIT_SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
)


def read(problem_path: Path, derived_dir: Path, opid: str) -> OptimizationProblem:
    """Read a problem, with the model which carries the network in `derived_dir`."""
    reader = PetabReader.from_yaml(problem_path)
    reader.derived_dir = derived_dir
    problem = reader.to_optimization_problem(opid=opid)
    problem.initialize(FIT_SETTINGS)
    return problem


def main() -> None:
    """Read, fit, report and write the problem."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--runs", type=int, default=1, help="optimization runs")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the runs")
    parser.add_argument(
        "--max-nfev", type=int, default=50, help="evaluations of the cost per run"
    )
    options = parser.parse_args()
    output_dir = Path("results") / "lotka_volterra"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- READ ---
    problem = read(PROBLEM_PATH, output_dir, opid="lotka_volterra")
    display.section("Problem", icon=display.ICON_FIT)
    display.key_values(
        {
            "problem": PROBLEM_PATH,
            "fit mappings": len(problem.mapping_keys),
            "parameters": len(problem.parameters),
            "log-likelihood": f"{log_likelihood(problem):.6g}",
        }
    )
    display.print_parameters(
        problem.parameters, hooks=hook_summaries(problem.hybridizations)
    )
    console.print(gaps_table(gaps_of_problem(problem), title="What PEtab v2 loses"))

    # --- FIT ---
    # the step of the finite differences of the jacobian is absolute for the
    # elements, which are around zero; the default of `sbmlsim.fit.cli` (5%)
    # is made for parameters on a logarithmic scale
    opt_result = run_optimization(
        problem=problem,
        settings=FIT_SETTINGS,
        size=options.runs,
        seed=options.seed,
        serial=True,
        show_progress=False,
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=options.max_nfev,
    )
    parameter_set = opt_result.parameter_set(0)

    # --- REPORT ---
    fisher = fisher_information(
        problem=problem, settings=FIT_SETTINGS, parameter_set=parameter_set
    )
    report = FitReport(
        problem=problem,
        settings=FIT_SETTINGS,
        parameter_sets=[parameter_set],
        opt_result=opt_result,
        fisher=fisher,
    )
    report.create(output_dir, name="report")

    # --- WRITE AND READ AGAIN ---
    display.section("PEtab SciML", icon=":package:")
    yaml_file = to_petab(problem, output_dir / "petab", settings=FIT_SETTINGS)
    restored = read(yaml_file, output_dir / "petab" / "derived", opid="restored")
    display.key_values(
        {
            "written": yaml_file,
            "log-likelihood": f"{log_likelihood(problem):.10g}",
            "read again": f"{log_likelihood(restored):.10g}",
        }
    )


if __name__ == "__main__":
    main()
