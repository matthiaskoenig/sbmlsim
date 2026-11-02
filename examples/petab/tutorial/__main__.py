"""The problem of the PEtab v2 tutorial, defined in python, written and fitted.

`examples.petab.tutorial.experiment` defines the problem of the
[tutorial of PEtab v2](https://petab.readthedocs.io/en/latest/v2/tutorial/tutorial.html)
as a `SimulationExperiment`. This script writes it as a PEtab v2 problem,
compares the tables with the ones of the tutorial, reads the problem back and
fits it: the maximum likelihood estimate (`ObjectiveType.LIKELIHOOD`), which
estimates the standard deviations of the noise with the kinetic parameters.
The tutorial simplified its observables for illustration, they do not
describe the data as the published problem does, whose fit is
`python -m examples.petab.benchmark --problem=Boehm_JProteomeRes2014`.

    python -m examples.petab.tutorial
    python -m examples.petab.tutorial --runs=80 --cores=8

The PEtab problem and the report of the fit are written into
`results/petab_tutorial` of the working directory.
"""

import argparse
import sys
from pathlib import Path

# run as a script (the "run file" of an IDE) the repository is not on
# `sys.path`, so the `examples` package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import sympy

from examples.petab.tutorial.experiment import (
    EPO_CONCENTRATION,
    SETTINGS,
    TUTORIAL,
    tutorial_problem,
)
from sbmlsim.console import console
from sbmlsim.fit import display
from sbmlsim.fit.options import OptimizationAlgorithmType
from sbmlsim.fit.petab_v2 import from_petab, gaps_of_problem, gaps_table, to_petab
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization

ICON_PETAB = ":package:"


def compare_with_tutorial(petab_dir: Path) -> dict[str, bool]:
    """Compare the tables which were written with the tables of the tutorial.

    The ids of the experiment and of the condition are the ones the export
    gives them, the content is compared: the condition at the time 0 of the
    one experiment, the observables as math, their normal noise with the
    placeholder of the standard deviation, which every measurement gives the
    estimated parameter of its observable, and the estimated parameters with
    their bounds.

    Args:
        petab_dir: the directory of the problem.

    Returns:
        Every check and whether it holds.
    """

    def table(name: str) -> pd.DataFrame:
        return pd.read_csv(petab_dir / f"{name}.tsv", sep="\t")

    experiments, conditions = table("experiments"), table("conditions")
    observables, measurements = table("observables"), table("measurements")
    parameters = table("parameters").set_index("parameterId")
    ((_, time, target, value),) = TUTORIAL["experiments"]
    checks: dict[str, bool] = {}
    checks["one experiment with a condition at 0"] = (
        len(experiments) == 1 and float(experiments["time"].iloc[0]) == time
    )
    checks[f"the condition sets {target} = {value}"] = (
        len(conditions) == 1
        and conditions["targetId"].iloc[0] == target
        and float(conditions["targetValue"].iloc[0]) == EPO_CONCENTRATION
    )
    written = dict(
        zip(observables["observableId"], observables["observableFormula"], strict=True)
    )
    for sid, formula in TUTORIAL["observables"].items():
        checks[f"observable {sid}"] = (
            sid in written
            and sympy.simplify(sympy.sympify(written[sid]) - sympy.sympify(formula))
            == 0
        )
    checks["normal noise of a placeholder"] = bool(
        (observables["noiseDistribution"] == TUTORIAL["noise_distribution"]).all()
        and (observables["noiseFormula"] == observables["noisePlaceholders"]).all()
    )
    checks["the noise parameters of the measurements"] = len(measurements) == TUTORIAL[
        "measurements"
    ] and bool(
        (measurements["noiseParameters"] == "sd_" + measurements["observableId"]).all()
    )
    estimated = parameters[parameters["estimate"].astype(bool)]
    lower, upper = TUTORIAL["bounds"]
    checks["the estimated parameters and their bounds"] = (
        set(estimated.index) == set(TUTORIAL["estimated"])
        and bool((estimated["lowerBound"] == lower).all())
        and bool((estimated["upperBound"] == upper).all())
    )
    return checks


def main() -> None:
    """Write the problem, compare it with the tutorial, read it back and fit it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=40, help="optimization runs")
    parser.add_argument("--cores", type=int, default=1, help="workers of the fit")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the fit")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results") / "petab_tutorial"
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- WRITE ---
    problem = tutorial_problem(base_path=output_dir)
    problem.initialize(SETTINGS)
    yaml_file = to_petab(problem, output_dir / "petab", settings=SETTINGS)
    display.section("PEtab v2", icon=ICON_PETAB)
    display.link("problem", yaml_file)
    console.print(gaps_table(gaps_of_problem(problem), title="What PEtab v2 loses"))

    checks = compare_with_tutorial(yaml_file.parent)
    display.section("The tables of the tutorial", icon=ICON_PETAB)
    display.key_values(
        {check: "yes" if ok else "[red]no[/red]" for check, ok in checks.items()}
    )

    # --- READ ---
    read, settings = from_petab(yaml_file, opid="boehm_tutorial_read")
    read.initialize(settings)

    # --- FIT ---
    opt_result = run_optimization(
        problem=problem,
        settings=SETTINGS,
        size=args.runs,
        seed=args.seed,
        n_cores=args.cores,
        serial=args.cores == 1,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        show_progress=False,
    )
    parameter_set = opt_result.parameter_set(0)
    # the tutorial simplified the observables for illustration, they are not
    # the ones the data was measured as (the published problem weighs the
    # species with `specC17`, see `examples.petab.benchmark`), so the noise
    # which is estimated is large
    display.section("Maximum likelihood estimate", icon=display.ICON_PARAMETERS)
    display.key_values(
        {
            pid: f"{value:.6g}"
            for pid, value in zip(
                problem.pids, parameter_set.x(problem.pids), strict=True
            )
        }
    )
    llh = log_likelihood(problem, parameter_set)
    display.key_values(
        {
            "log-likelihood": f"{llh:.6f}",
            # the problem which was read back has the same likelihood
            "log-likelihood of the PEtab problem": (
                f"{log_likelihood(read, parameter_set):.6f}"
            ),
        }
    )

    report = FitReport(
        problem=problem,
        settings=SETTINGS,
        parameter_sets=[parameter_set],
        opt_result=opt_result,
    )
    report.create(output_dir, name="report")


if __name__ == "__main__":
    main()
