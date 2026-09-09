"""Fit a problem of the PEtab benchmark collection with sbmlsim.

The problem is `Perelson_Science1996` of the
[benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab),
i.e. the viral dynamics of HIV-1 after the start of a protease inhibitor, see
`examples/petab/Perelson_Science1996/README.md`. The collection is PEtab 1.0
and `sbmlsim` reads PEtab 2.0, so the example converts the problem with
`petab.v2.petab1to2` first: the files of the collection stay as they are and
the conversion is part of the example.

The problem is then read into an `OptimizationProblem`, fitted, and the
identifiability of the fitted parameters is analysed with the profile
likelihood.

    python -m examples.petab.benchmark_perelson
    python -m examples.petab.benchmark_perelson --runs=8 --no-identifiability

The converted problem and the results are written into `results/perelson` of
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

#: the problem of the collection, PEtab 1.0
PROBLEM_DIR = Path(__file__).parent / "Perelson_Science1996"

#: name of its YAML file
PROBLEM_YAML = "Perelson_Science1996.yaml"

ICON_PETAB = ":package:"


def main() -> None:
    """Convert, read, fit and analyse the problem of the collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=4, help="number of optimization runs"
    )
    parser.add_argument(
        "--cores", type=int, default=1, help="number of workers of the fit"
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
        default=Path("results") / "perelson",
        help="directory of the converted problem and the results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- CONVERT ---
    display.section("PEtab 1.0 -> 2.0", icon=ICON_PETAB)
    petab2_dir = output_dir / "petab_v2"
    petab1to2(PROBLEM_DIR / PROBLEM_YAML, output_dir=petab2_dir)
    yaml_file = petab2_dir / PROBLEM_YAML
    display.key_values({"collection": PROBLEM_DIR.name})
    display.link("problem", yaml_file)

    # --- READ ---
    problem, settings = from_petab(yaml_file, opid="Perelson_Science1996")
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
        display.key_values(
            {
                "identifiable": (
                    f"{identifiability.n_identifiable}/"
                    f"{len(identifiability.pids)} parameters"
                )
            }
        )
        identifiability.to_json(output_dir / "identifiability.json")

    # --- REPORT ---
    # `create` reports its own section with the link to the HTML report
    report = FitReport(
        problem=problem,
        settings=settings,
        parameter_sets=[parameter_set, problem.parameter_set_model()],
        opt_result=opt_result,
        identifiability=identifiability,
    )
    report.create(output_dir, name="report")


if __name__ == "__main__":
    main()
