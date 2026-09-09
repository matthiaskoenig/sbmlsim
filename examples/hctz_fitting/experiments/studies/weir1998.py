from typing import ClassVar

import pandas as pd

from examples.hctz_fitting.experiments.base_experiment import HCTZSimulationExperiment
from examples.hctz_fitting.experiments.metadata import (
    ApplicationForm,
    Coadministration,
    Dosing,
    Fasting,
    HCTZMappingMetaData,
    Health,
    Route,
    Tissue,
)
from examples.hctz_fitting.helpers import run_experiments
from sbmlsim.data import DataSet
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim


class Weir1998(HCTZSimulationExperiment):
    """Simulation experiment of Weir1998.

    Multiple dosing of hydrochlorothiazide 25 mg twice daily.
    """

    suffixes: ClassVar[list[str]] = ["", "_kombi"]
    colors: ClassVar[dict[str, str]] = {
        "": "black",
        "_kombi": "tab:blue",
    }

    def datasets(self) -> dict[str, DataSet]:
        dsets = {}
        for fig_id in ["Fig2", "Fig3", "Tab4"]:
            df: pd.DataFrame = self.load_dataframe(fig_id)
            for label, df_label in df.groupby("label"):
                label = str(label)
                dset = DataSet.from_df(df_label, self.ureg)
                if label.startswith(("hctz", "amount_", "excretion_")):
                    dset.unit_conversion("mean", 1 / self.Mr.hctz)
                dsets[f"{fig_id}_{label}"] = dset
        return dsets

    def simulations(self) -> dict[str, AbstractSim]:
        Q_ = self.Q_
        tcsims = {}

        # 11 doses, every 12 hours

        tc0 = Timecourse(
            start=0,
            end=12 * 60,  # [min]
            steps=500,
            changes={
                **self.default_changes(),
                "PODOSE_hctz": Q_(25, "mg"),
            },
        )
        tc1 = Timecourse(
            start=0,
            end=12 * 60,  # [min]
            steps=500,
            changes={
                "PODOSE_hctz": Q_(25, "mg"),
                "Aurine_hctz": Q_(0, "mmole"),  # reset urine collection
            },
        )
        tc2 = Timecourse(
            start=0,
            end=60 * 60,  # [min]
            steps=500,
            changes={
                "PODOSE_hctz": Q_(25, "mg"),
                "Aurine_hctz": Q_(0, "mmole"),  # reset urine collection
            },
        )
        tcsims["hctz25"] = TimecourseSim(
            [tc0] + [tc1 for _ in range(9)] + [tc2], time_offset=-10 * 12 * 60
        )

        return tcsims

    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = {}

        infos = [
            ("Fig2_hctz25", "[Cve_hctz]", Tissue.PLASMA),
            ("Fig3_amount_cumulative_hctz25", "Aurine_hctz", Tissue.URINE),
            ("Tab4_excretion_hctz25", "KI__HCTZEX", Tissue.URINE),
        ]

        for info in infos:
            (dset_id, yid, tissue) = info
            for suffix in self.suffixes:
                mappings[f"fm_{dset_id}{suffix}"] = FitMapping(
                    self,
                    reference=FitData(
                        self,
                        dataset=f"{dset_id}{suffix}",
                        xid="time",
                        yid="mean",
                        yid_sd="mean_sd",
                        count="count",
                    ),
                    observable=FitData(self, task="task_hctz25", xid="time", yid=yid),
                    metadata=HCTZMappingMetaData(
                        tissue=tissue,
                        application_form=ApplicationForm.TABLET,
                        route=Route.PO,
                        dosing=Dosing.MULTI,
                        health=Health.HEALTHY,
                        fasting=Fasting.NR,
                        coadministration=Coadministration.DILTIAZEM
                        if "kombi" in suffix
                        else Coadministration.NONE,
                    ),
                )

        return mappings

    def figures(self) -> dict[str, Figure]:
        return {
            **self.figure_Fig2_Fig3_Tab4(),
        }

    def figure_Fig2_Fig3_Tab4(self) -> dict[str, Figure]:
        name = "Fig2_Fig3_Tab4"
        fig = Figure(
            experiment=self,
            sid=name,
            num_rows=1,
            num_cols=3,
            name=f"{self.__class__.__name__} {name}",
        )

        plots = fig.create_plots(xaxis=Axis(self.label_time, unit="hr"), legend=True)
        plots[0].set_yaxis(self.label_hctz, unit=self.unit_hctz)
        plots[1].set_yaxis(
            label=self.label_hctz_excretion_urine, unit=self.unit_hctz_excretion_urine
        )
        plots[2].set_yaxis(label=self.label_hctz_urine, unit=self.unit_hctz_urine)

        # simulation
        for k, yid in enumerate(["[Cve_hctz]", "KI__HCTZEX", "Aurine_hctz"]):
            plots[k].add_data(
                task="task_hctz25",
                xid="time",
                yid=yid,
                label="Sim",
                color=self.color_hctz,
            )

        # data
        for k, dset_id in enumerate(
            ["Fig2_hctz25", "Tab4_excretion_hctz25", "Fig3_amount_cumulative_hctz25"]
        ):
            for suffix in self.suffixes:
                plots[k].add_data(
                    dataset=f"{dset_id}{suffix}",
                    xid="time",
                    yid="mean",
                    yid_sd="mean_sd",
                    count="count",
                    label="25 mg + DIL60" if "kombi" in suffix else "25 mg",
                    color=self.colors[suffix],
                )

        return {
            name: fig,
        }


if __name__ == "__main__":
    run_experiments(Weir1998, output_dir=Weir1998.__name__)
