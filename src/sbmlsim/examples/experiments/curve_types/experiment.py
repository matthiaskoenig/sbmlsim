"""
Example simulation experiment.
"""

from pathlib import Path
from typing import Union

from sbmlsim.combine.sedml.report import Report
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Figure, Plot
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.simulator.simulation_serial import SimulatorSerial
from sbmlsim.task import Task


class CurveTypesExperiment(SimulationExperiment):
    """Simulation experiments for curve types."""

    def models(self) -> dict[str, Union[Path, AbstractModel]]:
        """Define models."""
        return {
            "model": Path(__file__).parent / "results" / "curve_types_model.xml",
        }

    def simulations(self) -> dict[str, AbstractSim]:
        """Define simulations."""
        tc = TimecourseSim(
            timecourses=Timecourse(start=0, end=10, steps=10),
            time_offset=0,
        )
        return {"tc": tc}

    def tasks(self) -> dict[str, Task]:
        """Define tasks."""
        tasks = dict()
        for model in ["model"]:
            tasks[f"task_{model}_tc"] = Task(model=model, simulation="tc")
        return tasks

    def data(self) -> dict[str, Data]:
        """Define data generators."""
        # direct access via id
        data = []
        for model in ["model"]:
            for selection in ["time", "S1", "S2", "[S1]", "[S2]"]:
                data.append(Data(task=f"task_{model}_tc", index=selection))
        return {d.sid: d for d in data}

    def reports(self) -> dict[str, Report]:
        """Define reports."""
        report1 = Report(
            sid="report1",
            datasets={
                sid: f"task_model_tc__{sid}"
                for sid in ["time", "S1", "S2", "[S1]", "[S2]"]
            },
        )
        return {report1.sid: report1}

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

        # FIXME: add helper to easily create figure layouts with plots
        p0 = fig.add_subplot(Plot(sid="plot0", name="Timecourse"), row=1, col=1)
        p0.set_title("Timecourse")
        p0.set_xaxis("time", unit="min")
        p0.set_yaxis("data", unit="mM")

        p0.curve(
            x=Data("time", task="task_model_tc"),
            y=Data("[S1]", task="task_model_tc"),
            label="[S1]",
        )

        return {
            fig.sid: fig,
        }


def run_curve_types_experiments(output_path: Path) -> Path:
    """Run simulation experiments."""
    base_path = Path(__file__).parent
    data_path = base_path

    runner = ExperimentRunner(
        CurveTypesExperiment,
        simulator=SimulatorSerial(),
        data_path=data_path,
        base_path=base_path,
    )
    _results = runner.run_experiments(
        output_path=output_path / "results", show_figures=True
    )


if __name__ == "__main__":
    run_curve_types_experiments(Path(__file__).parent / "results")
