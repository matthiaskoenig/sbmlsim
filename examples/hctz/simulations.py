"""Run the HCTZ simulation experiments.

The experiments write their figures and reports into `results/` in the working
directory:

    python -m examples.hctz.simulations
"""

import shutil
from pathlib import Path

from examples.hctz.experiments.studies import Beermann1976, Patel1984
from examples.hctz.helpers import run_experiments
from sbmlsim.console import console
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.plot import Figure

EXPERIMENTS: dict[str, list[type[SimulationExperiment]]] = {
    "studies": [
        Beermann1976,
        Patel1984,
    ],
}
EXPERIMENTS["all"] = EXPERIMENTS["studies"]


def run_simulation_experiments(
    selected: str | None = None,
    experiment_classes: list[type[SimulationExperiment]] | None = None,
    output_dir: Path | str | None = None,
) -> None:
    """Run the HCTZ simulation experiments.

    Args:
        selected: group of experiments to run, one of `EXPERIMENTS`.
        experiment_classes: explicit list of experiments, overrides `selected`.
        output_dir: directory for the results, relative to the working directory.

    Raises:
        ValueError: if neither a group nor experiment classes are given, or the
            group does not exist.
    """
    Figure.fig_dpi = 300
    Figure.legend_fontsize = 10

    if experiment_classes is not None:
        experiments_to_run = experiment_classes
        output_dir = output_dir or "custom_selection"
    elif selected:
        if selected not in EXPERIMENTS:
            raise ValueError(
                f"Unknown group '{selected}', use one of: "
                f"'{sorted(EXPERIMENTS.keys())}'."
            )
        experiments_to_run = EXPERIMENTS[selected]
        output_dir = output_dir or selected
    else:
        raise ValueError(
            "No experiments specified, use 'selected' or 'experiment_classes'."
        )

    output_path = run_experiments(
        experiment_classes=experiments_to_run, output_dir=output_dir
    )

    # collect the figures in a single folder
    figures_dir = output_path / "_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for f in output_path.glob("**/*.png"):
        if f.parent == figures_dir:
            continue
        shutil.copy2(f, figures_dir / f.name)
    console.print(f"Figures copied to: {figures_dir.resolve().as_uri()}")


if __name__ == "__main__":
    run_simulation_experiments(selected="studies")
