"""
Conversion of radioactivity

1 Ci = 3.7×10^10 decays per second
C14: 62.4 mCi/mmol = 62.4E-3 *3.7*10^10 CPS/mmol = 62.4E-3/60 *3.7*10^10 CPM/mmole
=> 1 mmole = 62.4E-3/60 * 3.7*10^10 CPM
=> 1 mmole = 38480000 CPM
=> 1 CPM = 1/38480000 mmole
"""

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


class Beermann1976(HCTZSimulationExperiment):
    """Simulation experiment of Beermann1976.

    Oral dosing of 5, 50, 75 mg HCTZ and intravenous dosing of 1 and 35 mg HCTZ.
    """

    doses: ClassVar[list[int]] = [
        5,
        50,
        75,
        35,
        1,
    ]
    routes: ClassVar[list[str]] = [
        "po",
        "po",
        "po",
        "iv",
        "iv",
    ]
    colors: ClassVar[dict[int, str]] = {
        5: "tab:blue",
        50: "tab:orange",
        75: "tab:red",
        35: "tab:green",
        1: "tab:brown",
    }

    def datasets(self) -> dict[str, DataSet]:
        dsets = {}
        for fig_id in ["Tab1A", "Fig2", "Fig3"]:
            df: pd.DataFrame = self.load_dataframe(fig_id)
            for label_key, df_label in df.groupby("label"):
                label = str(label_key)
                dset = DataSet.from_df(df_label, self.ureg)
                if fig_id == "Tab1A":
                    dset.unit_conversion("value", 1 / self.Mr.hctz)
                if fig_id == "Fig3" and label.startswith("hctz"):
                    dset.unit_conversion("value", 1 / self.Mr.hctz)

                dsets[label] = dset

        # console.print(dsets.keys())
        # console.print(dsets)
        return dsets

    def simulations(self) -> dict[str, AbstractSim]:
        Q_ = self.Q_
        tcsims: dict[str, AbstractSim] = {}

        for kd, dose in enumerate(self.doses):
            route = self.routes[kd]

            tcsims[f"hctz_{route}{dose}"] = TimecourseSim(
                Timecourse(
                    start=0,
                    end=180 * 60,  # [min]
                    steps=1000,
                    changes={
                        **self.default_changes(),
                        f"{route.upper()}DOSE_hctz": Q_(dose, "mg"),
                    },
                )
            )

        return tcsims

    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = {}

        # urine and feces
        for dset_id in self._datasets:
            if not dset_id.startswith("amount"):
                continue

            tokens = dset_id.split("_")
            route = tokens[-2][-2:]
            dose = int(tokens[-2][4:-2])
            individual = tokens[-1]
            tissue = "urine" if "urine" in dset_id else "feces"
            if route == "iv":
                application_form = ApplicationForm.SOLUTION
            else:
                if dose == 5:
                    application_form = ApplicationForm.SUSPENSION
                elif dose == 50:
                    application_form = ApplicationForm.CAPSULE

            mappings[f"fm_hctz_{route}{dose}_{individual}_{tissue}"] = FitMapping(
                self,
                reference=FitData(
                    self,
                    dataset=dset_id,
                    xid="time",
                    yid="value",
                    count="count",
                ),
                observable=FitData(
                    self,
                    task=f"task_hctz_{route}{dose}",
                    xid="time",
                    yid=f"A{tissue}_hctz",
                ),
                metadata=HCTZMappingMetaData(
                    tissue=Tissue.URINE if tissue == "urine" else Tissue.FECES,
                    application_form=application_form,
                    route=Route.PO if route == "po" else Route.IV,
                    dosing=Dosing.SINGLE,
                    health=Health.HEALTHY,
                    fasting=Fasting.FASTED,
                    coadministration=Coadministration.NONE,
                ),
            )

        # Issues with data, outliers
        mappings["fm_hctz5po_4"] = FitMapping(
            self,
            reference=FitData(
                self,
                dataset="hctz5po_4",
                xid="time",
                yid="value",
                count="count",
            ),
            observable=FitData(
                self, task="task_hctz_po5", xid="time", yid="[Cve_hctz]"
            ),
            metadata=HCTZMappingMetaData(
                tissue=Tissue.PLASMA,
                application_form=ApplicationForm.SUSPENSION,
                route=Route.PO,
                dosing=Dosing.SINGLE,
                health=Health.HEALTHY,
                fasting=Fasting.FASTED,
                coadministration=Coadministration.NONE,
            ),
        )
        # Issues with data, outliers
        mappings["fm_excretion_hctz5po_4"] = FitMapping(
            self,
            reference=FitData(
                self,
                dataset="excretion_hctz5po_4",
                xid="time",
                yid="value",
                count="count",
            ),
            observable=FitData(
                self, task="task_hctz_po5", xid="time", yid="KI__HCTZEX"
            ),
            metadata=HCTZMappingMetaData(
                tissue=Tissue.URINE,
                application_form=ApplicationForm.SUSPENSION,
                route=Route.PO,
                dosing=Dosing.SINGLE,
                health=Health.HEALTHY,
                fasting=Fasting.FASTED,
                coadministration=Coadministration.NONE,
            ),
        )

        mappings["fm_hctz75po_6"] = FitMapping(
            self,
            reference=FitData(
                self,
                dataset="hctz75_po_6",
                xid="time",
                yid="value",
                count="count",
            ),
            observable=FitData(
                self, task="task_hctz_po75", xid="time", yid="[Cve_hctz]"
            ),
            metadata=HCTZMappingMetaData(
                tissue=Tissue.PLASMA,
                application_form=ApplicationForm.TABLET,
                route=Route.PO,
                dosing=Dosing.SINGLE,
                health=Health.HEALTHY,
                fasting=Fasting.FASTED,
                coadministration=Coadministration.NONE,
            ),
        )

        # console.print(mappings)
        return mappings

    def figures(self) -> dict[str, Figure]:
        return {
            **self.figure_Tab1A(),
            # **self.figure_Fig2(),
            **self.figure_Fig3(),
        }

    def figure_Tab1A(self) -> dict[str, Figure]:
        name = "Tab1A"
        fig = Figure(
            experiment=self,
            sid=name,
            num_rows=1,
            num_cols=4,
            name=f"{self.__class__.__name__} {name}",
        )

        plots = fig.create_plots(xaxis=Axis(self.label_time, unit="hr"), legend=True)
        plots[0].set_yaxis(label=self.label_hctz_urine, unit=self.unit_hctz_urine)
        plots[1].set_yaxis(label=self.label_hctz_feces, unit=self.unit_hctz_feces)
        plots[2].set_yaxis(label=self.label_hctz_urine, unit=self.unit_hctz_urine)
        plots[3].set_yaxis(
            label=self.label_hctz_feces, unit=self.unit_hctz_feces, min=-0.5, max=10.5
        )

        # simulation
        for kd, dose in enumerate(self.doses):
            route = self.routes[kd]
            kr = 0 if route == "po" else 1

            for ky, yid in enumerate(["Aurine_hctz", "Afeces_hctz"]):
                plots[kr * 2 + ky].add_data(
                    task=f"task_hctz_{route}{dose}",
                    xid="time",
                    yid=yid,
                    label=f"Sim {route}{dose} mg",
                    color=self.colors[dose],
                )

        # data urine
        for dset_id in self._datasets:
            if not dset_id.startswith("amount"):
                continue
            tokens = dset_id.split("_")
            route = tokens[-2][-2:]
            dose = int(tokens[-2][4:-2])
            individual = tokens[-1]
            kr = 0 if route == "po" else 1
            ky = 0 if "urine" in dset_id else 1

            plots[kr * 2 + ky].add_data(
                dataset=dset_id,
                xid="time",
                yid="value",
                count="count",
                label=f"{route}{dose} mg {individual}",
                color=self.colors[dose],
            )

        return {
            name: fig,
        }

    def figure_Fig2(self) -> dict[str, Figure]:
        # FIXME: conversion issues
        name = "Fig2"
        fig = Figure(
            experiment=self,
            sid=name,
            num_rows=1,
            num_cols=2,
            name=f"{self.__class__.__name__} {name}",
        )

        plots = fig.create_plots(
            xaxis=Axis(self.label_time, unit="hr", min=-0.1, max=25), legend=True
        )
        plots[0].set_yaxis(self.label_hctz, unit=self.unit_hctz)
        plots[1].set_yaxis(
            self.label_hctz_excretion_urine, unit=self.unit_hctz_excretion_urine
        )

        # simulation
        for kp, yid in enumerate(["[Cve_hctz]", "KI__HCTZEX"]):
            plots[kp].add_data(
                task="task_hctz_po5",
                xid="time",
                yid=yid,
                label="Sim po5 mg",
                color=self.colors[5],
            )

        # data
        plots[0].add_data(
            dataset="hctz5po_4",
            xid="time",
            yid="value",
            count="count",
            label="po5 mg HCTZ",
            color=self.colors[5],
        )

        plots[1].add_data(
            dataset="excretion_hctz5po_4",
            xid="time",
            yid="value",
            count="count",
            label="5 mg HCTZ",
            color=self.colors[5],
        )

        return {
            name: fig,
        }

    def figure_Fig3(self) -> dict[str, Figure]:
        name = "Fig3"
        fig = Figure(
            experiment=self,
            sid=name,
            name=f"{self.__class__.__name__} {name}",
        )

        plots = fig.create_plots(
            xaxis=Axis(self.label_time, unit="hr", min=-0.1, max=40), legend=True
        )
        plots[0].set_yaxis(self.label_hctz, unit=self.unit_hctz)

        # simulation
        plots[0].add_data(
            task="task_hctz_po75",
            xid="time",
            yid="[Cve_hctz]",
            label="Sim po75 mg",
            color=self.colors[75],
        )

        # data
        plots[0].add_data(
            dataset="hctz75_po_6",
            xid="time",
            yid="value",
            count="count",
            label="po75 mg HCTZ",
            color=self.colors[75],
        )

        return {
            name: fig,
        }


if __name__ == "__main__":
    run_experiments(Beermann1976, output_dir=Beermann1976.__name__)
