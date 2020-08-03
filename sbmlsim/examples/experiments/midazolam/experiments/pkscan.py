
"""
Dose-dependency of pharmacokinetic parameters.
"""
from typing import Dict
import pandas as pd
import numpy as np

from sbmlsim.experiment import ExperimentDict
from sbmlsim.data import DataSet, Data
from sbmlsim.simulation import Timecourse, TimecourseSim, ScanSim, Dimension
from sbmlsim.plot.plotting_matplotlib import Figure, Axis

from . import MidazolamSimulationExperiment


class PKScanExperiment(MidazolamSimulationExperiment):
    def simulations(self) -> Dict[str, ScanSim]:
        """ Scanning dose-response of midazolam pharmacokinetics."""
        tend = 1000  # [min]
        steps = 2000
        Q_ = self.Q_
        scan_defs = {
            "po_scan": {'PODOSE_mid': Q_(np.logspace(-1, 1.61, num=20), 'mg')},
            "iv_scan": {'IVDOSE_mid': Q_(np.logspace(-1, 1.17, num=20), 'mg')},
        }

        tcscans = ExperimentDict()
        for key, changes in scan_defs.items():
            tcscans[key] = ScanSim(simulation=
                TimecourseSim([
                    Timecourse(start=0, end=tend, steps=steps,
                               changes={
                                   **self.default_changes(),
                               })
                ]),
                dimensions=[
                    Dimension("dim_dose", changes=changes)
                ]
            )

        return tcscans

    def datagenerators(self) -> None:
        for sim_key in self._simulations:
            for selection in ["time", "[Cve_mid]", "[Cve_mid1oh]", "IVDOSE_mid",
                              "PODOSE_mid"]:
                Data(self, index=selection, task=f"task_{sim_key}")

    def figures(self) -> Dict[str, Figure]:
        return {
            **self.fig1(),
            # **self.Fig2(),
            # **self.Fig3(),
        }

    def fig1(self) -> Dict[str, Figure]:
        unit_time = "min"
        unit_mid = "nmol/ml"
        unit_mid1oh = "nmol/ml"

        fig = Figure(self, sid="Fig1",
                     num_rows=2, num_cols=2, name=self.sid)
        plots = fig.create_plots(
            Axis("time", unit=unit_time),
            legend=True
        )

        # set titles and labs
        plots[0].set_title("scan po")
        plots[1].set_title("scan iv")
        for k in (0, 1):
            plots[k].set_yaxis("midazolam", unit_mid)
            plots[k].xaxis.label_visible = False
        for k in (2, 3):
            plots[k].set_yaxis("1-hydroxymidazolam", unit_mid1oh)

        # simulation
        for k, key in enumerate(["po_scan", "iv_scan"]):
            task_id = f"task_{key}"
            # plot midazolam
            plots[k].add_data(task=task_id, xid='time', yid='[Cve_mid]',
                              color="black")
            # plot 1-hydroxymidazolam
            plots[k+2].add_data(task=task_id, xid='time', yid='[Cve_mid1oh]',
                                color="black")

        return ExperimentDict({"fig1": fig})

    '''
    def Fig3(self) -> Dict[str, Figure]:
        Q_ = self.ureg.Quantity
        unit_dose = "mg"

        # calculate all pharmacokinetic parameters from scans
        dfs = {}
        for key in self._simulations.keys():
            xres = self.results[f"task_{key}"]
            if key.startswith("midpo"):
                dose_vec = Q_(xres["PODOSE_mid"].values[0], xres.udict["PODOSE_mid"])
            elif key.startswith("midiv"):
                dose_vec = Q_(xres["IVDOSE_mid"].values[0], xres.udict["IVDOSE_mid"])
            time_vec = xres.dim_mean('time')

            pk_dicts = []
            for k, dose in enumerate(dose_vec):
                dose = dose.to("mol", "chemistry", mw=self.Mr.mid)
                conc = Q_(xres["[Cve_mid]"].sel(dim_dose=k).values, xres.udict['[Cve_mid]'])
                tcpk = TimecoursePK(
                    time=time_vec,
                    concentration=conc,
                    substance="midazolam",
                    dose=dose,
                    ureg=self.ureg
                )
                pk_dicts.append(tcpk.pk.to_dict())

            df = pd.DataFrame(pk_dicts)
            # Fix for unsorted rows
            df = df.sort_values(by=["dose"])
            dfs[key] = df

        # TODO: calculate bioavailability from simulations (oral & iv simulation)

        # visualize pharmacokinetic parameters
        figures = {}
        for pkkey in ["auc", "aucinf", "tmax", "cmax", "kel", "thalf", "vd", "vdss", "cl"]:
            print("-" * 80)
            print(f"{pkkey}")
            print("-" * 80)

            f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            f.subplots_adjust(wspace=.3, hspace=.3)
            axes = (ax1, ax2)

            kwargs = {
                "linewidth": 2.0,
                "markersize": 8,
            }
            plot_kwargs = {
                "default": {**kwargs, "color": "black"},
                "inducer": {**kwargs, "color": "green"},
                "inhibitor": {**kwargs, "color": "magenta"}
            }

            # for all conditions
            for simkey, df in dfs.items():
                print(f"\t{simkey}")

                for ax in axes:
                    # simulation type
                    ax.set_title(pkkey)
                    sim_type = simkey.split("_")[1]

                    x = self.Q_(df['dose'].values, df['dose_unit'][0]) * self.Mr.mid
                    x = x.to(unit_dose)
                    y = df[pkkey]
                    print(y)

                    if simkey.startswith("midpo"):
                        marker = "o"
                        linestyle = "-"
                        alpha = 0.5
                        label = f"po, {sim_type}"
                    elif simkey.startswith("midiv"):
                        marker = "s"
                        linestyle = "--"
                        alpha = 1.0
                        label = f"iv, {sim_type}"

                    ax.plot(x, y, label=label, marker=marker, linestyle=linestyle, alpha=alpha,
                            **plot_kwargs[sim_type])
                    ax.set_xlabel(f"dose [{unit_dose}]")
                    yunit = df[f"{pkkey}_unit"][0]
                    ax.set_ylabel(f"{pkkey} [{yunit}]")
                    #ax.legend()

            ax2.set_yscale("log")
            figures[f'fig_{pkkey}'] = f

        return figures
    '''