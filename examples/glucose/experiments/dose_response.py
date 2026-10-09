"""Dose response of the hormones glucagon, epinephrine and insulin on glucose."""

from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim import Q
from sbmlsim.data import Data, DataSet, load_pkdb_dataframe
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot.serialization_matplotlib import FigureMPL
from sbmlsim.result import ScanResult
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.task import Task

from .plotting import add_data

#: studies of the healthy controls per hormone, the data of the other studies is
#: not used
STUDIES: dict[str, list[str]] = {
    "Epinephrine": [
        "Degn2004",
        "Lerche2009",
        "Mitrakou1991",
        "Levy1998",
        "Israelian2006",
        "Jones1998",
        "Segel2002",
    ],
    "Glucagon": [
        "Butler1991",
        "Cobelli2010",
        "Fery1993Gerich1993",
        "Henkel2005",
        "Mitrakou1991Basu2009",
        "Mitrakou1992",
        "Degn2004",
        "Lerche2009",
        "Levy1998",
        "Israelian2006",
        "Segel2002",
    ],
    "Insulin": [
        "Ferrannini1988",
        "Fery1993",
        "Gerich1993",
        "Basu2009",
        "Lerche2009",
        "Henkel2005",
        "Butler1991",
        "Knop2007",
        "Cobelli2010",
        "Mitrakou1992",
    ],
}

#: glucagon studies with hyperinsulinemic clamps, the insulin suppresses the
#: glucagon by the factor `INSULIN_SUPPRESSION`, which the data is corrected for
GLUCAGON_CLAMP_STUDIES = [
    "Degn2004",
    "Lerche2009",
    "Levy1998",
    "Israelian2006",
    "Segel2002",
]
INSULIN_SUPPRESSION = 3.4

#: selections of the scan: the hormones, which are assignment rules of the
#: glucose; the glucose of the scan is a coordinate of the result
SELECTIONS = ["glu", "epi", "ins", "gamma"]


class DoseResponseExperiment(SimulationExperiment):
    """Hormone dose-response curves."""

    @override
    def datasets(self) -> dict[str, DataSet]:
        """Define the dose response data of the hormones."""
        if self.data_path is None:
            raise ValueError("data_path is required for the dose response data.")

        dsets = {}
        for hormone_key, studies in STUDIES.items():
            df = load_pkdb_dataframe(
                f"DoseResponse_Tab{hormone_key}", data_path=self.data_path
            )
            # only healthy controls
            df = df[(df.condition == "normal") & df.reference.isin(studies)].copy()
            if hormone_key == "Glucagon":
                clamp = df.reference.isin(GLUCAGON_CLAMP_STUDIES)
                df.loc[clamp, ["mean", "se"]] *= INSULIN_SUPPRESSION

            udict = {
                "glc": df["glc_unit"].unique()[0],
                "mean": df["unit"].unique()[0],
            }
            dsets[hormone_key.lower()] = DataSet.from_df(
                df, ureg=self.ureg, udict=udict
            )

        return dsets

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        """Define models."""
        return {"model1": Path(__file__).parent.parent / "model" / "liver_glucose.xml"}

    @override
    def simulations(self) -> dict[str, Scan]:
        """Scanning dose-response curves of hormones and gamma function.

        Vary external glucose concentrations (boundary condition).
        """
        glc_scan = Scan(
            simulation=Simulation(end=1, steps=1),
            dimensions=[
                Dimension(
                    "dim1",
                    values={"[glc_ext]": Q(np.linspace(2, 20, num=30), "mM")},
                ),
            ],
        )
        return {"glc_scan": glc_scan}

    @override
    def tasks(self) -> dict[str, Task]:
        """Define tasks."""
        return {"task_glc_scan": Task(model="model1", simulation="glc_scan")}

    @override
    def data(self) -> dict[str, Data]:
        """Define the data of the experiment."""
        self.add_selections_data(
            selections=["time", *SELECTIONS], task_ids=["task_glc_scan"]
        )
        return {}

    @override
    def figures_mpl(self) -> dict[str, FigureMPL]:
        """Define the matplotlib figure of the dose responses."""
        xunit = "mM"
        yunit_hormone = "pmol/l"
        yunit_gamma = "dimensionless"

        # the hormones are assignment rules of the glucose, the first time point
        # of every simulation of the scan is the dose response; the glucose of
        # the scan is the coordinate of the dimension
        res: ScanResult = self.results["task_glc_scan"]
        initial = res.ds.isel(time=0)
        columns = ["[glc_ext]", *SELECTIONS]
        dset = DataSet.from_df(
            pd.DataFrame({sid: np.asarray(initial[sid].values) for sid in columns}),
            udict={sid: res.units[sid] for sid in columns},
            ureg=self.ureg,
        )

        fig_mpl, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig_mpl.subplots_adjust(wspace=0.3, hspace=0.3)

        # selection, label (and dataset), unit, limits of the x and y axis
        panels = [
            ("glu", "glucagon", yunit_hormone, (2, 20), (0, 200)),
            ("epi", "epinephrine", yunit_hormone, (2, 8), (0, 7000)),
            ("ins", "insulin", yunit_hormone, (2, 20), (0, 800)),
            ("gamma", "gamma", yunit_gamma, (2, 20), (0, 1)),
        ]
        for ax, (sid, label, yunit, xlim, ylim) in zip(axes.flat, panels, strict=True):
            # simulation
            add_data(
                ax,
                dset,
                xid="[glc_ext]",
                yid=sid,
                xunit=xunit,
                yunit=yunit,
                linewidth=2,
                linestyle="-",
                marker="None",
                color="black",
            )
            # experimental data
            if label in self._datasets:
                add_data(
                    ax,
                    self._datasets[label],
                    xid="glc",
                    yid="mean",
                    yid_se="mean_se",
                    xunit=xunit,
                    yunit=yunit,
                    label=label.capitalize(),
                    color="black",
                    linestyle="None",
                    alpha=0.6,
                )
            ax.set_xlabel(f"glucose [{xunit}]")
            ax.set_ylabel(f"{label} [{yunit}]")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        return {"fig1": fig_mpl}
