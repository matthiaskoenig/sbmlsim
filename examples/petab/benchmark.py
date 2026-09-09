"""Fit a problem of the PEtab benchmark collection with sbmlsim.

The problems are the ones of the
[benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab)
which are vendored next to this module, see their `README.md`:

- `Perelson_Science1996`, the viral dynamics of HIV-1 after the start of a
  protease inhibitor. One observable, which is a species of the model.
- `Boehm_JProteomeRes2014`, the dimerization of STAT5A and STAT5B. Three
  observables, each a formula over several species, which the fit gets as
  entities of the model, see `sbmlsim.fit.petab_v2.observables`.

The collection is PEtab 1.0 and `sbmlsim` reads PEtab 2.0, so the example
converts a problem with `petab.v2.petab1to2` first: the files of the collection
stay as they are and the conversion is part of the example. The problem is then
read into an `OptimizationProblem`, fitted, analysed with the profile
likelihood and reported.

    python -m examples.petab.benchmark
    python -m examples.petab.benchmark --problem=Boehm_JProteomeRes2014
    python -m examples.petab.benchmark --runs=8 --no-identifiability

The converted problem and the results are written into `results/<problem>` of
the working directory.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# run as a script (the "run file" of an IDE) the repository is not on
# `sys.path`, so the `examples` package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from petab.v2.petab1to2 import petab1to2

from sbmlsim.console import console
from sbmlsim.fit import display
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.identifiability import (
    IdentifiabilityResult,
    ProfileSettings,
    profile_likelihood,
)
from sbmlsim.fit.options import OptimizationAlgorithmType, ResidualType
from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table
from sbmlsim.fit.petab_v2.reader import from_petab
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization

#: the problems of the collection which are vendored here, PEtab 1.0
PROBLEMS: tuple[str, ...] = ("Perelson_Science1996", "Boehm_JProteomeRes2014")

ICON_PETAB = ":package:"


def main() -> None:
    """Convert, read, fit and analyse the problem of the collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        default=PROBLEMS[0],
        choices=PROBLEMS,
        help="problem of the collection to fit",
    )
    parser.add_argument(
        "--runs", type=int, default=4, help="number of optimization runs"
    )
    # the simulation experiment of a PEtab problem is created when the problem
    # is read, and a class which is created cannot be pickled, so the workers
    # of a parallel fit cannot be given the problem
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        choices=[1],
        help="workers of the fit, a PEtab problem is fitted in one process",
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed of the fit")
    parser.add_argument(
        "--no-identifiability",
        action="store_true",
        help="only fit, do not analyse the identifiability",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory of the converted problem and the results",
    )
    args = parser.parse_args()

    problem_dir = Path(__file__).parent / args.problem
    problem_yaml = f"{args.problem}.yaml"
    output_dir = Path(args.output_dir or Path("results") / args.problem)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- CONVERT ---
    display.section("PEtab 1.0 -> 2.0", icon=ICON_PETAB)
    petab2_dir = output_dir / "petab_v2"
    petab1to2(problem_dir / problem_yaml, output_dir=petab2_dir)
    yaml_file = petab2_dir / problem_yaml
    display.key_values({"collection": args.problem})
    display.link("problem", yaml_file)

    # --- READ ---
    problem, settings = from_petab(yaml_file, opid=args.problem)
    # the problem measures a viral load over orders of magnitude, which the
    # relative residuals describe and the absolute ones do not. PEtab says
    # log-normal noise here, which `sbmlsim` does not have, see the gaps
    settings = replace(settings, residual=ResidualType.NORMALIZED)
    problem.initialize(settings)

    display.section("Problem", icon=display.ICON_FIT)
    display.key_values(
        {
            "fit mappings": len(problem.mapping_keys),
            "data points": sum(len(y) for y in problem.y_references),
            "parameters": len(problem.parameters),
            "residual": settings.residual.name,
            "parameter scale": settings.parameter_scale.name,
        }
    )
    display.print_parameters(problem.parameters)

    console.print(gaps_table(gaps_of_problem(problem), title="What PEtab v2 loses"))

    # --- FIT ---
    opt_result = run_optimization(
        problem=problem,
        settings=settings,
        size=args.runs,
        seed=args.seed,
        n_cores=args.cores,
        serial=args.cores == 1,
        algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
        show_progress=False,
    )
    parameter_set = opt_result.parameter_set(0)

    display.section("Fitted", icon=display.ICON_PARAMETERS)
    display.key_values(
        {
            pid: f"{value:.6g}  (nominal {parameter.start_value:.6g})"
            for pid, value, parameter in zip(
                problem.pids,
                parameter_set.x(problem.pids),
                problem.parameters,
                strict=True,
            )
        }
    )

    # --- IDENTIFIABILITY ---
    identifiability: IdentifiabilityResult | None = None
    if not args.no_identifiability:
        # the profile likelihood reports the scans and the identifiability
        identifiability = profile_likelihood(
            problem=problem,
            settings=settings,
            parameter_set=parameter_set,
            profile_settings=ProfileSettings(),
            n_cores=args.cores,
            show_progress=False,
        )
        info = {
            "identifiable": (
                f"{identifiability.n_identifiable}/"
                f"{len(identifiability.pids)} parameters"
            )
        }
        if identifiability.better_optimum:
            # the scans found a lower cost than the parameter set, so the
            # confidence intervals are not intervals around an optimum
            info["converged"] = (
                "[orange3]no, the scans found a lower cost than the fit[/orange3]"
            )
        display.key_values(info)
        identifiability.to_json(output_dir / "identifiability.json")

    # --- FISHER INFORMATION ---
    # the local analysis, from one jacobian instead of a scan per parameter
    fisher = fisher_information(
        problem=problem, settings=settings, parameter_set=parameter_set
    )
    display.section("Fisher information", icon=display.ICON_IDENTIFIABILITY)
    display.key_values(
        {
            "rank": f"{fisher.rank} of {fisher.k}",
            "condition number": f"{fisher.condition_number:.4g}",
        }
    )

    # --- REPORT ---
    # `create` reports its own section with the link to the HTML report
    report = FitReport(
        problem=problem,
        settings=settings,
        parameter_sets=[parameter_set],
        opt_result=opt_result,
        identifiability=identifiability,
        fisher=fisher,
    )
    report.create(output_dir, name="report")


if __name__ == "__main__":
    main()
