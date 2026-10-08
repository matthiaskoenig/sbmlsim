"""
Example simulation experiment.
"""

from pathlib import Path
from typing import override

from examples.curve_types.model import create
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

#: selections of the timecourse, which the data uses
SELECTIONS = ["time", "S1", "S2", "[S1]", "[S2]"]


class CurveTypesExperiment(SimulationExperiment):
    """Simulation experiments for curve types."""

    #: path of the model, created by `run_curve_types_experiments`
    model_path: Path = Path.cwd() / "results" / "curve_types_model.xml"

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        """Define models."""
        return {"model": self.model_path}

    @override
    def simulations(self) -> dict[str, Simulation]:
        """Define simulations."""
        return {"tc": Simulation(end=10, steps=10)}

    @override
    def tasks(self) -> dict[str, Task]:
        """Define tasks."""
        return {"task_model_tc": Task(model="model", simulation="tc")}

    @override
    def data(self) -> dict[str, Data]:
        """Define the data of the experiment."""
        data = [Data(task="task_model_tc", index=selection) for selection in SELECTIONS]
        return {d.sid: d for d in data}

    @override
    def figures(self) -> dict[str, Figure]:
        """Define figure outputs (plots)."""
        fig = Figure(
            experiment=self,
            sid="figure0",
            name="Example curve type",
            num_cols=1,
            num_rows=1,
            width=5,
            height=5,
        )
        plots = fig.create_plots(
            xaxis=Axis("time", unit="min"), yaxis=Axis("data", unit="mM")
        )
        plots[0].set_title("Timecourse")
        plots[0].curve(
            x=Data("time", task="task_model_tc"),
            y=Data("[S1]", task="task_model_tc"),
            label="[S1]",
        )

        return {"fig1": fig}


def run_curve_types_experiments(output_path: Path) -> None:
    """Create the model and run the simulation experiments."""
    base_path = Path(__file__).parent

    CurveTypesExperiment.model_path = create(output_dir=output_path / "results")
    runner = ExperimentRunner(
        CurveTypesExperiment,
        simulator=Simulator(),
        data_path=base_path,
        base_path=base_path,
    )
    runner.run_experiments(output_path=output_path / "results", show_figures=False)


if __name__ == "__main__":
    run_curve_types_experiments(Path.cwd())
