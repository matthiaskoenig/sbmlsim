"""Fit a problem of the PEtab benchmark collection with sbmlsim.

The problems are the ones of the
[benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab)
which are vendored next to this module, see their `README.md`:

- `Perelson_Science1996`, the viral dynamics of HIV-1 after the start of a
  protease inhibitor. One observable, which is a species of the model.
- `Boehm_JProteomeRes2014`, the dimerization of STAT5A and STAT5B. Three
  observables, each a formula over several species, which the fit evaluates
  on the simulation, see `sbmlsim.fit.objects.ObservableModel`.

Both are PEtab v2: `Boehm_JProteomeRes2014` is the encoding of the collection
in PEtab v2, `Perelson_Science1996` its problem of PEtab v1 converted with
`petab.v2.petab1to2`. The problem is read into an `OptimizationProblem`,
fitted, analysed with the profile likelihood and reported.

    python -m examples.petab.benchmark
    python -m examples.petab.benchmark --problem=Boehm_JProteomeRes2014
    python -m examples.petab.benchmark --runs=8 --no-identifiability

The results are written into `results/<problem>` of the working directory. The whole collection is read and simulated by
`scripts/petab_benchmark.py`, see `sbmlsim.fit.petab_v2.benchmark`.
"""

import argparse
import sys
from pathlib import Path

# run as a script (the "run file" of an IDE) the repository is not on
# `sys.path`, so the `examples` package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sbmlsim.console import console
from sbmlsim.fit import display
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.identifiability import (
    IdentifiabilityResult,
    ProfileSettings,
    profile_likelihood,
)
from sbmlsim.fit.options import OptimizationAlgorithmType
from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table
from sbmlsim.fit.petab_v2.reader import from_petab
from sbmlsim.fit.report import FitReport
from sbmlsim.fit.runner import run_optimization

#: the problems of the collection which are vendored here, in PEtab v2
PROBLEMS: tuple[str, ...] = ("Perelson_Science1996", "Boehm_JProteomeRes2014")

ICON_PETAB = ":package:"


def main() -> None:
    """Read, fit and analyse the problem of the collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        default=PROBLEMS[0],
        choices=PROBLEMS,
        help="problem of the collection to fit",
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="number of optimization runs"
    )
    parser.add_argument(
        "--cores", type=int, default=1, help="workers of the parallel fit"
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
        help="directory of the results",
    )
    args = parser.parse_args()

    yaml_file = Path(__file__).parent / args.problem / "problem.yaml"
    output_dir = Path(args.output_dir or Path("results") / args.problem)
    output_dir.mkdir(parents=True, exist_ok=True)

    display.section("PEtab v2", icon=ICON_PETAB)
    display.key_values({"collection": args.problem})
    display.link("problem", yaml_file)

    # --- READ ---
    # the settings of PEtab: the fit minimizes the negative log-likelihood of
    # the noise models of the problem, so the standard deviations of the noise
    # are estimated with the parameters of the model, e.g. the log-normal noise
    # of the viral load of Perelson, which spans orders of magnitude
    problem, settings = from_petab(yaml_file, opid=args.problem)
    problem.initialize(settings)

    display.section("Problem", icon=display.ICON_FIT)
    display.key_values(
        {
            "fit mappings": len(problem.mapping_keys),
            "data points": sum(len(y) for y in problem.y_references),
            "parameters": len(problem.parameters),
            "objective": settings.objective.value,
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
