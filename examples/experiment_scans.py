"""Scans and observables in a simulation experiment: midazolam at three doses.

The experiment declares the observables of the scan core next to its
simulations, the mass concentration of midazolam in plasma and its
non-compartmental analysis, and its data are labelled arrays which keep the
dimension of the doses: the cmax of every dose is read by its label. `Fig1` draws
the midazolam of every dose as one curve each, the cmax over the dose with the
data, and a band of the midazolam over 40 Latin hypercube draws of two
parameters. The cmax over the dose is also a fit mapping, values over a
dimension of a scan against a table: `run` fits the maximal velocity of the
metabolism of midazolam to the (synthetic) data.
"""

from pathlib import Path
from typing import override

import numpy as np
import pandas as pd

from sbmlsim import Q
from sbmlsim.data import Data, DataSet
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings
from sbmlsim.fit.sampling import SamplingType
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

#: the doses in mg
DOSES = [5.0, 7.5, 15.0]

#: the parameter the fit estimates and its value in the model
FIT_PARAMETER = "LI__MIDIM_Vmax"
REFERENCE_VALUE = 0.1


def _simulation() -> Simulation:
    return Simulation(
        time_unit="hr",
        end=24,
        steps=480,
        changes=[Change(0, {"PODOSE_mid": Q(7.5, "mg")})],
    )


def _dose_scan() -> Scan:
    dose = Dimension(
        "dose",
        values={"PODOSE_mid": Q(DOSES, "mg")},
        labels=["low", "standard", "high"],
    )
    return Scan(_simulation(), [dose])


def _observables() -> dict[str, Observable]:
    return {
        "mid": Formula("mid", "[Cve_mid] * Mr_mid", unit="ng/ml"),
        "pk": PK("pk", "mid", dose="PODOSE_mid", route="oral"),
    }


class MidazolamDoses(SimulationExperiment):
    """Oral midazolam at three doses."""

    @override
    def datasets(self) -> dict[str, DataSet]:
        # synthetic data: the cmax of the simulation of this example at the
        # reference parameters with 1 percent of noise (regenerated whenever the
        # experiment is initialized); the stated error is 5 percent
        result = Simulator().run(
            MIDAZOLAM_SBML,
            _dose_scan(),
            list(_observables().values()),
            keep=["pk.cmax"],
        )
        cmax = result.ds["pk.cmax"].values
        cmax = cmax * np.random.default_rng(1).normal(1.0, 0.01, len(cmax))
        df = pd.DataFrame({"dose": DOSES, "cmax": cmax, "cmax_sd": 0.05 * cmax})
        return {
            "cmax_doses": DataSet.from_df(
                df,
                udict={"dose": "mg", "cmax": "ng/ml", "cmax_sd": "ng/ml"},
                ureg=self.ureg,
            )
        }

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": MIDAZOLAM_SBML}

    @override
    def simulations(self) -> dict[str, Scan]:
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
        return {
            "doses": _dose_scan(),
            "draws": Scan(_simulation(), [draws]),
        }

    @override
    def observables(self) -> dict[str, Observable]:
        return _observables()

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
    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_cmax_dose": FitMapping(
                self,
                reference=FitData(
                    self,
                    dataset="cmax_doses",
                    xid="dose",
                    yid="cmax",
                    yid_sd="cmax_sd",
                ),
                observable=FitData(
                    self, task="task_doses", xid="dose.PODOSE_mid", yid="pk.cmax"
                ),
            )
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
        plots[1].set_xaxis("dose", unit="mg")
        plots[1].set_yaxis("cmax", unit="ng/ml")
        plots[1].curve(
            Data("dose.PODOSE_mid", task="task_doses"),
            Data("pk.cmax", task="task_doses"),
            color="black",
            marker="o",
            markersize=11,
            markerfacecolor="white",
            markeredgecolor="black",
            label="simulation",
        )
        plots[1].add_data(
            dataset="cmax_doses",
            xid="dose",
            yid="cmax",
            yid_sd="cmax_sd",
            label="data",
            color="tab:red",
            linestyle="None",
            markersize=5,
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
    fit_cmax(base_path)
    return experiment


def fit_cmax(base_path: Path) -> OptimizationProblem:
    """Fit the maximal velocity of the metabolism to the cmax over the doses.

    A short serial least squares fit which starts at twice the reference value
    (`SamplingType.START`, the default draws the start from the bounds), no
    report is written. The cmax depends only weakly on the parameter, so the
    fit moves from the start towards the reference and stops near it, not at it.
    """
    problem = OptimizationProblem(
        "midazolam_cmax",
        [FitMappingCollection(experiment=MidazolamDoses, mappings=["fm_cmax_dose"])],
        [
            FitParameter(
                pid=FIT_PARAMETER,
                lower_bound=REFERENCE_VALUE / 10,
                upper_bound=REFERENCE_VALUE * 10,
                start_value=2 * REFERENCE_VALUE,
                unit="mmol/min/l",
            )
        ],
        base_path=base_path,
        data_path=base_path,
    )
    problem.initialize(FitSettings())
    fits, _ = problem.optimize(size=1, seed=1, sampling=SamplingType.START, max_nfev=30)
    fitted = float(fits[0].x[0])
    print(
        f"fit of {FIT_PARAMETER}: start {2 * REFERENCE_VALUE:.4g}, "
        f"fitted {fitted:.4g}, reference {REFERENCE_VALUE:.4g}, "
        f"cost {fits[0].cost:.4g}"
    )
    return problem


if __name__ == "__main__":
    run(Path.cwd() / "results")
