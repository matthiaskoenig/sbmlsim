from typing import Dict, List

from sbmlsim.data import DataSet, load_pkdb_dataframes_by_substance
from sbmlsim.fit import FitMapping, FitData
from sbmlsim.plot import Figure, Axis
from sbmlsim.simulation import Timecourse, TimecourseSim

from . import MidazolamSimulationExperiment


class Mandema1992(MidazolamSimulationExperiment):
    def datasets(self) -> Dict[str, DataSet]:
        dsets = {}
        for fig_id in ["Fig1A", "Fig2A", "Fig3A"]:
            dframes = load_pkdb_dataframes_by_substance(
                f"{self.sid}_{fig_id}", data_path=self.data_path)

            for substance, df in dframes.items():
                dset = DataSet.from_df(df, self.ureg)
                if substance == "midazolam":
                    dset.unit_conversion("mean", 1 / self.Mr.mid)
                elif substance == "1-hydroxymidazolam":
                    dset.unit_conversion("mean", 1 / self.Mr.mid1oh)
                else:
                    raise ValueError

                dsets[f"{fig_id}_{substance}"] = dset
        return dsets

    def simulations(self) -> Dict[str, TimecourseSim]:
        return {
            **self.simulation_mid()
        }

    def simulation_mid(self) -> Dict[str, TimecourseSim]:
        """ Mandema1992

        - midazolam, iv, 0.1 [mg/kg] (infusion over 15 min)
        - 1-hydroxy midazolam, iv, 0.15 [mg/kg] (infusion over 15 min)
        - midazolam, po, 7.5 [mg]
        """
        Q_ = self.Q_

        bodyweight = Q_(69, 'kg') # avg. bodyweight of 8 individuals (sd=6kg)
        # mid_iv = Q_(0.1, 'mg/kg') * bodyweight
        mid_Ri = Q_(0.1, 'mg/kg') * bodyweight / Q_(15, 'min')

        # if injected in 1 min use the IVDOSE_mid1oh parameter
        # mid1oh_iv = Q_(0.15, 'mg/kg') * bodyweight
        # but infused in 15 [min]
        mid1oh_Ri = Q_(0.15, 'mg/kg') * bodyweight / Q_(15, 'min')

        mid_po = Q_(7.5, 'mg')

        tcsims = {}
        tcsims["mid_iv"] = TimecourseSim([
            Timecourse(start=0, end=15, steps=600, changes={
                **self.default_changes(),
                'Ri_mid': mid_Ri,
                "BW": bodyweight
            }),
            Timecourse(start=0, end=300, steps=600, changes={
                **self.default_changes(),
                'Ri_mid': Q_(0, "mg_per_min"),
                "BW": bodyweight
            }),
        ])
        tcsims["mid1oh_iv"] = TimecourseSim([
            Timecourse(start=0, end=15, steps=100, changes={
                **self.default_changes(),
                # 'IVDOSE_mid1oh': mid1oh_iv
                'Ri_mid1oh': mid1oh_Ri,
                "BW": bodyweight
            }),
            Timecourse(start=0, end=300, steps=600, changes={
                'Ri_mid1oh': Q_(0, 'mg_per_min'),
            }),
        ])
        tcsims["mid_po"] = TimecourseSim([
            Timecourse(start=0, end=315, steps=700, changes={
                **self.default_changes(),
                'PODOSE_mid': mid_po,
                "BW": bodyweight})
        ])

        return tcsims

    def fit_mappings(self) -> Dict[str, FitMapping]:
        # fit mapping: which data maps on which simulation
        fit_dict = {
            "fm1": {"ref": "Fig1A_midazolam", "obs": "task_mid_iv", "yid": "[Cve_mid]"},
            "fm2": {"ref": "Fig3A_midazolam", "obs": "task_mid_po", "yid": "[Cve_mid]"},
            "fm3": {"ref": "Fig1A_1-hydroxymidazolam", "obs": "task_mid_iv", "yid": "[Cve_mid1oh]"},
            "fm4": {"ref": "Fig2A_1-hydroxymidazolam", "obs": "task_mid1oh_iv", "yid": "[Cve_mid1oh]"},
            "fm5": {"ref": "Fig3A_1-hydroxymidazolam", "obs": "task_mid_po", "yid": "[Cve_mid1oh]"},
        }

        mappings = {}
        for key, values in fit_dict.items():
            mappings[key] = FitMapping(
                self,
                reference=FitData(self, dataset=values["ref"], xid="time", yid="mean", yid_sd="mean_sd"),
                observable=FitData(self, task=values["obs"], xid="time", yid=values["yid"])
            )
        return mappings

    def figures(self) -> Dict[str, Figure]:
        return {
            **self.figure_mid(),
            **self.figure_injection(),
        }

    def figure_mid(self):
        title = f"{self.sid}"
        unit_time = "min"
        unit_mid = "nmol/ml"
        unit_mid1oh = "nmol/ml"

        fig = Figure(self, sid="Fig1",
                     num_rows=2, num_cols=3,
                     height=15, width=20)
        plots = fig.create_plots(
            Axis("time", unit=unit_time),
            legend=True
        )

        # simulation
        plots[0].set_title(f"{title} (midazolam iv, 0.1 [mg/kg])")
        plots[1].set_title(f"{title} (1-hydroxymidazolam iv, 0.15 [mg/kg])")
        plots[2].set_title(f"{title} (midazolam po, 7.5 [mg])")
        for k in (0, 1, 2):
            plots[k].set_yaxis("midazolam", unit_mid)
        for k in (3, 4, 5):
            plots[k].set_yaxis("1-hydroxymidazolam", unit_mid1oh)

        plot_dict = {
            "mid_iv": (plots[0], plots[3]),
            "mid1oh_iv": (plots[1], plots[4]),
            "mid_po": (plots[2], plots[5])
        }

        for key, plot in plot_dict.items():
            # plot midazolam
            p = plot[0]
            p.add_data(task=f"task_{key}", xid="time", yid="[Cve_mid]",
                       label="mid (ve blood)")
            #plot 1-hydroxymidazolam
            p = plot[1]
            p.add_data(task=f"task_{key}", xid="time", yid="[Cve_mid1oh]",
                       label="mid1oh (ve blood)")

        # plot data
        data_def = {
            "Fig1A_midazolam": {'plot': plots[0], 'key': 'mid'},
            "Fig1A_1-hydroxymidazolam": {'plot': plots[3], 'key': 'mid1oh'},
            "Fig2A_1-hydroxymidazolam": {'plot': plots[4], 'key': 'mid1oh'},
            "Fig3A_midazolam": {'plot': plots[2], 'key': 'mid'},
            "Fig3A_1-hydroxymidazolam": {'plot': plots[5], 'key': 'mid1oh'},
        }
        for dset_key, dset_info in data_def.items():
            p = dset_info['plot']
            p.add_data(dataset=dset_key,
                       xid="time", yid="mean", yid_sd="mean_sd",
                       count=None, color="black", label=dset_info['key'])

        return {"fig1": fig}

    def figure_injection(self):
        title = f"{self.sid}"
        unit_time = "min"
        unit = "nmol/ml"
        unit_rate = "mmole_per_min"
        unit_amount = "mmole"

        fig = Figure(self, sid="Fig1",
                     num_rows=2, num_cols=2,
                     height=10, width=10)
        plots = fig.create_plots(
            Axis("time", unit=unit_time),
            legend=True
        )

        # simulation
        for k in (0, 1, 2, 3):
            plots[k].set_title(f"{title} (1-hydroxymidazolam iv)")
        plots[0].set_yaxis("IVDOSE_mid1oh", 'mg')
        plots[1].set_yaxis("rate", unit_rate)
        plots[2].set_yaxis("amount", unit_amount)

        plots[0].add_data(task='task_mid1oh_iv', xid='time', yid="IVDOSE_mid1oh",
                          label="IVDOSE_mid1oh", color="black")
        plots[1].add_data(task='task_mid1oh_iv', xid='time', yid="injection_mid1oh",
                          label="injection_mid1oh", color="blue")
        plots[1].add_data(task='task_mid1oh_iv', xid='time', yid="infusion_mid1oh",
                          label="infusion_mid1oh", color="darkorange")
        plots[2].add_data(task='task_mid1oh_iv', xid='time', yid="Aurine_mid1oh",
                          label="mid1oh (urine)", color="darkgrey")

        return {"fig2": fig}