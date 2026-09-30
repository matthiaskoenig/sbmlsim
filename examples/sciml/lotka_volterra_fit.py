"""A PEtab SciML problem read, fitted, reported and written again.

    pip install sbmlsim[sciml]
    python -m examples.sciml.lotka_volterra_fit

The problem is the case 001 of the PEtab SciML test suite,
`examples/sciml/lotka_volterra/`: the Lotka-Volterra model with a feed
forward network in the right hand side, which replaces the interaction term
of the predator. The example reads the problem, prints its networks, fits the
parameters of the model and the elements of the network together, reports the
fit, writes the fitted problem as PEtab SciML again and reads it back: the
log-likelihood of the problem which is read agrees with the one of the fit.

The values of the problem are a solution already, its cost is small. The fit
starts from them (`SamplingType.START`, so every run would be the same one and
the example makes one run) and improves the cost, which `--max-nfev`, the
evaluations of the cost, limits: the default is a demonstration and stops early.
A problem which is read from PEtab builds its simulation experiment at
runtime, which the workers of a parallel fit cannot import, so the fit runs in
one process. The results are written into `results/lotka_volterra` of the
working directory, with the model which carries the network, `lv_sciml.xml`,
and the fitted problem in `petab/`.
"""

import argparse
import sys
from pathlib import Path

# run as a script (`python examples/sciml/lotka_volterra_fit.py`, the "run
# file" of an IDE) the repository is not on `sys.path`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sbmlsim.fit import display
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2 import gaps_of_problem, to_petab
from sbmlsim.fit.petab_v2.likelihood import log_likelihood, nominal_parameters
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization
from sbmlsim.fit.sampling import SamplingType

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
    parser.add_argument(
        "--max-nfev", type=int, default=50, help="evaluations of the cost of the fit"
    )
    options = parser.parse_args()
    output_dir = Path("results") / "lotka_volterra"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- READ ---
    problem = read(PROBLEM_PATH, output_dir, opid="lotka_volterra")
    start = nominal_parameters(problem)
    start_cost = problem.cost_least_square(problem.to_scale(start.x(problem.pids)))
    display.section("Problem", icon=display.ICON_FIT)
    display.key_values(
        {
            "problem": PROBLEM_PATH.relative_to(PROBLEM_PATH.parent.parent),
            "fit mappings": len(problem.mapping_keys),
            "parameters": len(problem.parameters),
            "start": f"cost {start_cost:.4g}, log-likelihood "
            f"{log_likelihood(problem, start):.6g}",
            "gaps": ", ".join(gap.id for gap in gaps_of_problem(problem)),
        }
    )

    # --- FIT ---
    # `run_optimization` shows the parameters and the networks it fits.
    # The fit starts from the values of the problem, which are the solution to
    # improve. The step of the finite differences of the jacobian is absolute
    # for the elements, which are around zero; the default of `sbmlsim.fit.cli`
    # (5%) is made for parameters on a logarithmic scale
    opt_result = run_optimization(
        problem=problem,
        settings=FIT_SETTINGS,
        size=1,
        sampling=SamplingType.START,
        serial=True,
        show_progress=False,
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=options.max_nfev,
    )
    parameter_set = opt_result.parameter_set(0)
    display.key_values(
        {
            "fit": f"cost {parameter_set.cost:.4g}, log-likelihood "
            f"{log_likelihood(problem, parameter_set):.6g}",
        }
    )

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

    # --- WRITE THE FIT AND READ IT AGAIN ---
    display.section("PEtab SciML", icon=":package:")
    yaml_file = to_petab(
        problem,
        output_dir / "petab",
        settings=FIT_SETTINGS,
        parameter_set=parameter_set,
    )
    restored = read(yaml_file, output_dir / "petab" / "derived", opid="restored")
    display.key_values(
        {
            "written": yaml_file,
            "the fit": f"log-likelihood {log_likelihood(problem, parameter_set):.10g}",
            "read again": f"log-likelihood {log_likelihood(restored):.10g}",
        }
    )


if __name__ == "__main__":
    main()
