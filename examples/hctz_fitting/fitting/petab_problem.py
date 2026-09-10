"""Write the HCTZ fit as a PEtab v2 problem and read it back.

The PEtab layer of `sbmlsim.fit.petab_v2` on the reference problem: the fit is
written as a PEtab v2 problem, validated with `petab` itself and read back into
an optimization problem, and what the tables of PEtab cannot express is
reported before anything is written.

    python -m examples.hctz_fitting.fitting.petab_problem
    python -m examples.hctz_fitting.fitting.petab_problem --subset=PK --portable

The problem is written into `results/petab/<subset>` of the working directory.

The module is not called `petab.py`: run as a script its directory is the first
entry of `sys.path`, so `import petab` would find this file instead of the
library and fail with "'petab' is not a package".
"""

import argparse
import sys
from pathlib import Path

# run as a script (`python examples/hctz_fitting/fitting/petab_problem.py`, the "run file" of an
# IDE) the repository is not on `sys.path`, so the `examples` package is not found
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import petab.v2 as petab_v2
from rich import box
from rich.table import Table

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.console import console
from sbmlsim.fit import display
from sbmlsim.fit.petab_v2 import gaps_of_problem, gaps_table, to_petab
from sbmlsim.fit.petab_v2.extension import extension_of
from sbmlsim.fit.petab_v2.reader import from_petab

ICON_PETAB = ":package:"


def main() -> None:
    """Write the HCTZ fit as PEtab v2 and read it back."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        default="PK",
        choices=sorted(FIT_DEFINITIONS),
        help="fit problem to write, the pharmacokinetics by default",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "petab",
        help="directory the problem is written to",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="write the `sbmlsim` extension as not required, i.e. a problem "
        "other tools fit with the objective of PEtab",
    )
    args = parser.parse_args()

    definition = FIT_DEFINITIONS[args.subset]
    problem = definition.problem(opid=f"hctz_{args.subset}")
    problem.initialize(definition.settings)

    display.section(f"PEtab v2: {args.subset}", icon=ICON_PETAB)
    display.key_values(
        {
            "collections": len(problem.mapping_collections),
            "fit mappings": len(problem.mapping_keys),
            "parameters": len(problem.parameters),
            "data points": sum(len(y) for y in problem.y_references),
        }
    )

    # what PEtab v2 cannot express about this fit, before it is written
    display.section("Gaps", icon=":triangular_flag:")
    console.print(gaps_table(gaps_of_problem(problem)))

    # --- WRITE ---
    output_dir = Path(args.output_dir) / args.subset
    yaml_file = to_petab(problem, output_dir, required_extension=not args.portable)

    display.section("Problem", icon=":open_file_folder:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            display.key_values({path.name: f"{path.stat().st_size / 1024:.1f} kB"})
    display.link("problem", yaml_file)

    # --- VALIDATE WITH PETAB ---
    petab_problem = petab_v2.Problem.from_yaml(yaml_file)
    extension = extension_of(petab_problem.config)
    issues = list(petab_problem.validate())
    display.section("PEtab", icon=":white_check_mark:")
    display.key_values(
        {
            "models": len(petab_problem.models),
            "experiments": len(petab_problem.experiments),
            "conditions": len(petab_problem.conditions),
            "observables": len(petab_problem.observables),
            "measurements": len(petab_problem.measurements),
            "parameters": len(petab_problem.parameters),
            "extension": f"required={extension.required}" if extension else "-",
            "issues": len(issues),
        }
    )
    for issue in issues:
        console.print(f"  {issue}")

    # every experiment of PEtab comes from a collection of fit mappings
    table = Table(box=box.SIMPLE_HEAD, header_style="bold", show_edge=False)
    for column in ["experiment", "collection", "periods", "observables"]:
        table.add_column(column)
    for experiment in petab_problem.experiments:
        info = extension.experiments.get(experiment.id, {}) if extension else {}
        observables = {
            m.observable_id
            for m in petab_problem.measurements
            if m.experiment_id == experiment.id
        }
        table.add_row(
            experiment.id,
            str(info.get("collection", "-")),
            str(len(experiment.periods)),
            str(len(observables)),
        )
    console.print(table)

    # --- READ BACK ---
    display.section("Read back", icon=":arrows_counterclockwise:")
    petab_problem_read, settings = from_petab(yaml_file, opid=f"{args.subset}_petab")
    petab_problem_read.initialize(settings)

    x = np.log10(np.asarray(problem.x0, dtype=float))
    cost = problem.cost_least_square(x)
    cost_petab = petab_problem_read.cost_least_square(x)
    display.key_values(
        {
            "collections": (
                f"{len(problem.mapping_collections)} -> "
                f"{len(petab_problem_read.mapping_collections)} "
                f"(one per PEtab experiment)"
            ),
            "fit mappings": (
                f"{len(problem.mapping_keys)} -> {len(petab_problem_read.mapping_keys)}"
            ),
            "settings": "kept" if settings == definition.settings else "LOST",
            "units": (
                "kept"
                if [p.unit for p in petab_problem_read.parameters]
                == [p.unit for p in problem.parameters]
                else "LOST"
            ),
            "cost": f"{cost:.6f} -> {cost_petab:.6f}",
            "difference": f"{abs(cost - cost_petab) / cost:.2e} (relative)",
        }
    )
    display.print_parameters(petab_problem_read.parameters)


if __name__ == "__main__":
    main()
