"""Scans and observables in a simulation experiment: midazolam at three doses.

The experiment declares the observables of the scan core next to its
simulations, the mass concentration of midazolam in plasma and its
non-compartmental analysis, and its data are labelled arrays which keep the
dimension of the doses: the cmax of every dose is read by its label. `Fig1` draws
the midazolam of every dose as one curve each, the cmax over the dose, and a band
of the midazolam over 40 Latin hypercube draws of two parameters.
"""

from pathlib import Path
from typing import override

from sbmlsim import Q
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.resources import MIDAZOLAM_SBML
from sbmlsim.simulation import (
    PK,
    Change,
    Dimension,
    Formula,
    Observable,
    Scan,
    Simulation,
    sampling,
)
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

#: the PK parameters the example prints
PARAMETERS = ("pk.cmax", "pk.tmax", "pk.auc_inf_obs")


class MidazolamDoses(SimulationExperiment):
    """Oral midazolam at three doses."""

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": MIDAZOLAM_SBML}

    @override
    def simulations(self) -> dict[str, Scan]:
        simulation = Simulation(
            time_unit="hr",
            end=24,
            steps=480,
            changes=[Change(0, {"PODOSE_mid": Q(7.5, "mg")})],
        )
        doses = Dimension(
            "dose",
            values={"PODOSE_mid": Q([5.0, 7.5, 15.0], "mg")},
            labels=["low", "standard", "high"],
        )
        draws = sampling.lhs(
            {
                "LI__MIDIM_Vmax": sampling.LogNormal(cv=0.3),
                "Ka_abs_mid": sampling.LogNormal(cv=0.3),
            },
            40,
            seed=1,
            model=self._models["model"],
            id="draw",
        )
        return {"doses": Scan(simulation, [doses]), "draws": Scan(simulation, [draws])}

    @override
    def observables(self) -> dict[str, Observable]:
        return {
            "mid": Formula("mid", "[Cve_mid] * Mr_mid", unit="ng/ml"),
            "pk": PK("pk", "mid", dose="PODOSE_mid", route="oral"),
        }

    @override
    def tasks(self) -> dict[str, Task]:
        return {
            "task_doses": Task(model="model", simulation="doses"),
            "task_draws": Task(model="model", simulation="draws"),
        }

    @override
    def data(self) -> dict[str, Data]:
        indices = ("time", "mid", *PARAMETERS)
        return {
            "data_" + i.replace(".", "_"): Data(i, task="task_doses") for i in indices
        }

    @override
    def figures(self) -> dict[str, Figure]:
        fig = Figure(experiment=self, sid="Fig1", num_cols=3, num_rows=1)
        plots = fig.create_plots(xaxis=Axis("time", unit="hr"), legend=True)
        plots[0].set_yaxis("midazolam", unit="ng/ml")
        plots[0].curve(
            Data("time", task="task_doses"),
            Data("mid", task="task_doses"),
            over="dose",
        )
        plots[1].legend = False
        plots[1].set_xaxis("dose", unit="mg")
        plots[1].set_yaxis("cmax", unit="ng/ml")
        plots[1].curve(
            Data("dose.PODOSE_mid", task="task_doses"),
            Data("pk.cmax", task="task_doses"),
            color="black",
            marker="o",
        )
        plots[2].set_yaxis("midazolam", unit="ng/ml")
        plots[2].band(
            Data("time", task="task_draws"),
            Data("mid", task="task_draws"),
            across="draw",
            name="mid",
        )
        return {"fig1": fig}


def run(output_path: Path) -> SimulationExperiment:
    """Run the experiment and print the PK parameters per dose."""
    base_path = Path(__file__).parent
    runner = ExperimentRunner(
        MidazolamDoses, simulator=Simulator(), base_path=base_path, data_path=base_path
    )
    experiment = runner.run_experiments(output_path=output_path, keep_results=True)[
        0
    ].experiment
    for index in PARAMETERS:
        values = Data(index, task="task_doses").get_data(experiment)
        per_dose = dict(
            zip(
                values["dose"].values.tolist(),
                values.values.round(3).tolist(),
                strict=True,
            )
        )
        print(index, per_dose, values.attrs["units"])
    high = Data("mid", task="task_doses", sel={"dose": "high"}).get_data(experiment)
    print(
        "mid of the high dose",
        high.dims,
        round(float(high.max()), 3),
        high.attrs["units"],
    )
    return experiment


if __name__ == "__main__":
    run(Path.cwd() / "results")
