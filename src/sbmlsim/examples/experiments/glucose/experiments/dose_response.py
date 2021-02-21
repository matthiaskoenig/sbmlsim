from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.pyplot import Figure

from sbmlsim.data import Data, DataSet, load_pkdb_dataframe
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot.plotting_matplotlib import add_data, plt
from sbmlsim.result import XResult
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.task import Task
from sbmlsim.utils import timeit


class DoseResponseExperiment(SimulationExperiment):
    """Hormone dose-response curves."""

    @timeit
    def models(self) -> Dict[str, Union[AbstractModel, Path]]:
        return {"model1": Path(__file__).parent.parent / "model" / "liver_glucose.xml"}

    @timeit
    def datasets(self) -> Dict[str, DataSet]:
        dsets = {}

        # dose-response data for hormones
        for hormone_key in ["Epinephrine", "Glucagon", "Insulin"]:
            df = load_pkdb_dataframe(
                f"DoseResponse_Tab{hormone_key}", data_path=self.data_path
            )
            df = df[df.condition == "normal"]  # only healthy controls
            epi_normal_studies = [
                "Degn2004",
                "Lerche2009",
                "Mitrakou1991",
                "Levy1998",
                "Israelian2006",
                "Jones1998",
                "Segel2002",
            ]
            glu_normal_studies = [
                "Butler1991",
                "Cobelli2010",
                "Fery1993" "Gerich1993",
                "Henkel2005",
                "Mitrakou1991" "Basu2009",
                "Mitrakou1992",
                "Degn2004",
                "Lerche2009",
                "Levy1998",
                "Israelian2006",
                "Segel2002",
            ]
            ins_normal_studies = [
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
            ]
            # filter studies
            if hormone_key == "Epinephrine":
                df = df[df.reference.isin(epi_normal_studies)]
            elif hormone_key == "Glucagon":
                df = df[df.reference.isin(glu_normal_studies)]
                # correct glucagon data for insulin suppression
                # (hyperinsulinemic clamps)
                insulin_supression = 3.4
                glu_clamp_studies = [
                    "Degn2004",
                    "Lerche2009",
                    "Levy1998",
                    "Israelian2006",
                    "Segel2002",
                ]
                df.loc[df.reference.isin(glu_clamp_studies), "mean"] = (
                    insulin_supression
                    * df[df.reference.isin(glu_clamp_studies)]["mean"]
                )
                df.loc[df.reference.isin(glu_clamp_studies), "se"] = (
                    insulin_supression * df[df.reference.isin(glu_clamp_studies)]["se"]
                )

            elif hormone_key == "Insulin":
                df = df[df.reference.isin(ins_normal_studies)]

            udict = {
                "glc": df["glc_unit"].unique()[0],
                "mean": df["unit"].unique()[0],
            }
            dsets[hormone_key.lower()] = DataSet.from_df(
                df, udict=udict, ureg=self.ureg
            )

        return dsets

    def datagenerators(self) -> None:
        for key in ["time", "glu", "ins", "epi", "gamma"]:
            Data(experiment=self, task="task_glc_scan", index=key)

    @timeit
    def tasks(self) -> Dict[str, Task]:
        """Tasks"""
        return {"task_glc_scan": Task(model="model1", simulation="glc_scan")}

    @timeit
    def simulations(self) -> Dict[str, ScanSim]:
        """Scanning dose-response curves of hormones and gamma function.

        Vary external glucose concentrations (boundary condition).
        """
        glc_scan = ScanSim(
            simulation=TimecourseSim([Timecourse(start=0, end=1, steps=1, changes={})]),
            dimensions=[
                Dimension(
                    "dim1",
                    changes={"[glc_ext]": self.Q_(np.linspace(2, 20, num=30), "mM")},
                ),
            ],
        )
        return {"glc_scan": glc_scan}

    @timeit
    def figures(self) -> Dict[str, Figure]:
        xunit = "mM"
        yunit_hormone = "pmol/l"
        yunit_gamma = "dimensionless"

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        f.subplots_adjust(wspace=0.3, hspace=0.3)
        axes = (ax1, ax2, ax3, ax4)

        # process scan results
        task = self._tasks["task_glc_scan"]
        model = self._models[task.model_id]
        tcscan = self._simulations[task.simulation_id]  # TimecourseScan Definition

        # FIXME: this must be simpler
        glc_vec = tcscan.dimensions[0].changes["[glc_ext]"]
        xres = self.results["task_glc_scan"]  # type: XResult

        # we already have all the data ordered, we only want the steady state value

        dose_response = {}
        for sid in ["glu", "epi", "ins", "gamma"]:
            da = xres[sid]  # type: xr.DataArray

            # get initial time
            head = da.head({"_time": 1}).to_series()
            dose_response[sid] = head.values

        dose_response["[glc_ext]"] = glc_vec
        df = pd.DataFrame(dose_response)
        dset = DataSet.from_df(df, udict=model.udict, ureg=self.ureg)
        print(dset)

        # plot scan results
        kwargs = {"linewidth": 2, "linestyle": "-", "marker": "None", "color": "black"}
        add_data(
            ax1,
            dset,
            xid="[glc_ext]",
            yid="glu",
            xunit=xunit,
            yunit=yunit_hormone,
            **kwargs,
        )
        add_data(
            ax2,
            dset,
            xid="[glc_ext]",
            yid="epi",
            xunit=xunit,
            yunit=yunit_hormone,
            **kwargs,
        )
        add_data(
            ax3,
            dset,
            xid="[glc_ext]",
            yid="ins",
            xunit=xunit,
            yunit=yunit_hormone,
            **kwargs,
        )
        add_data(
            ax4,
            dset,
            xid="[glc_ext]",
            yid="gamma",
            xunit=xunit,
            yunit=yunit_gamma,
            **kwargs,
        )

        # plot experimental data
        kwargs = {
            "color": "black",
            "linestyle": "None",
            "alpha": 0.6,
        }
        add_data(
            ax1,
            self._datasets["glucagon"],
            xid="glc",
            yid="mean",
            yid_se="mean_se",
            xunit=xunit,
            yunit=yunit_hormone,
            label="Glucagon",
            **kwargs,
        )
        add_data(
            ax2,
            self._datasets["epinephrine"],
            xid="glc",
            yid="mean",
            yid_se="mean_se",
            xunit=xunit,
            yunit=yunit_hormone,
            label="Epinephrine",
            **kwargs,
        )
        add_data(
            ax3,
            self._datasets["insulin"],
            xid="glc",
            yid="mean",
            yid_se="mean_se",
            xunit=xunit,
            yunit=yunit_hormone,
            label="Insulin",
            **kwargs,
        )

        ax1.set_ylabel(f"glucagon [{yunit_hormone}]")
        ax1.set_ylim(0, 200)
        ax2.set_ylabel(f"epinephrine [{yunit_hormone}]")
        ax2.set_ylim(0, 7000)
        ax3.set_ylabel(f"insulin [{yunit_hormone}]")
        ax3.set_ylim(0, 800)
        ax4.set_ylabel(f"gamma [{yunit_gamma}]")
        ax4.set_ylim(0, 1)

        for ax in axes:
            ax.set_xlabel(f"glucose [{xunit}]")
            ax.set_xlim(2, 20)

        ax2.set_xlim(2, 8)

        return {"fig1": f}
