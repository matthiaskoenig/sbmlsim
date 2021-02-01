from typing import Dict, List

from sbmlsim.data import DataSet, load_pkdb_dataframes_by_substance
from sbmlsim.fit import FitData, FitMapping
from sbmlsim.plot import Axis, Figure
from sbmlsim.simulation import Timecourse, TimecourseSim

from . import MidazolamSimulationExperiment


class Kupferschmidt1995(MidazolamSimulationExperiment):
    def datasets(self) -> Dict[str, DataSet]:
        dsets = {}
        for fig_id in ["Fig1", "Fig2"]:
            dframes = load_pkdb_dataframes_by_substance(
                f"{self.sid}_{fig_id}", data_path=self.data_path
            )
            for substance, df in dframes.items():
                dset = DataSet.from_df(df, self.ureg)
                if substance == "midazolam":
                    dset.unit_conversion("mean", 1 / self.Mr.mid)
                elif substance == "1-hydroxymidazolam":
                    dset.unit_conversion("mean", 1 / self.Mr.mid1oh)
                else:
                    raise ValueError

                for intervention in df.intervention.unique():
                    # tag route
                    if "po" in intervention:
                        route = "po"
                    elif "iv" in intervention:
                        route = "iv"
                    else:
                        raise ValueError
                    # tag intervention type
                    if "GRAP1" in intervention:
                        continue
                    else:
                        type = "control"

                    dsets[f"{fig_id}_{substance}_{route}_{type}"] = dset[
                        dset.intervention == intervention
                    ]
        return dsets

    def simulations(self) -> Dict[str, TimecourseSim]:
        return super(Kupferschmidt1995, self).simulations(
            simulations={**self.simulations_mid()}
        )

    def simulations_mid(self) -> Dict[str, TimecourseSim]:
        """Kupferschmidt1995

        - midazolam, iv, 5 [mg]
        - midazolam, po, 15 [mg]

        - grapefruit juice, po, 200 [mg], -60min and -15min

        """
        Q_ = self.Q_
        bodyweight = Q_(70, "kg")  # avg. bodyweight of 8 individuals. (se=1kg)
        mid_iv = Q_(5, "mg")
        mid_po = Q_(15, "mg")

        sim_def = {
            "mid_iv_c": {
                "end": 1500,
                "steps": 3000,
                "dose": {"IVDOSE_mid": mid_iv},
            },
            "mid_po_c": {
                "end": 1500,
                "steps": 3000,
                "dose": {"PODOSE_mid": mid_po},
            },
        }
        tcsims = {}
        for key, value in sim_def.items():
            tcsims[key] = TimecourseSim(
                [
                    Timecourse(
                        start=0,
                        end=value["end"],
                        steps=value["steps"],
                        changes={**value["dose"], "BW": bodyweight},
                    )
                ]
            )

        return tcsims

    def fit_mappings(self) -> Dict[str, FitMapping]:
        # fit mapping: which data maps on which simulation
        fit_dict = {
            "fm_mid_iv": {
                "ref": "Fig1_midazolam_iv_control",
                "obs": "task_mid_iv_c",
                "yid": "[Cve_mid]",
            },
            "fm_mid1oh_iv": {
                "ref": "Fig2_1-hydroxymidazolam_iv_control",
                "obs": "task_mid_iv_c",
                "yid": "[Cve_mid1oh]",
            },
            "fm_mid_po": {
                "ref": "Fig1_midazolam_po_control",
                "obs": "task_mid_po_c",
                "yid": "[Cve_mid]",
            },
            "fm_mid1oh_po": {
                "ref": "Fig2_1-hydroxymidazolam_po_control",
                "obs": "task_mid_po_c",
                "yid": "[Cve_mid1oh]",
            },
        }

        mappings = {}
        for key, values in fit_dict.items():
            mappings[key] = FitMapping(
                self,
                reference=FitData(
                    self,
                    dataset=values["ref"],
                    xid="time",
                    yid="mean",
                    yid_sd="mean_sd",
                ),
                observable=FitData(
                    self, task=values["obs"], xid="time", yid=values["yid"]
                ),
            )
        return mappings

    def figures(self) -> Dict[str, Figure]:
        return {**self.figure_mid()}

    def figure_mid(self):
        unit_time = "min"
        unit_mid = "nmol/ml"
        unit_mid1oh = "nmol/ml"

        fig = Figure(self, sid="Fig1", num_rows=2, num_cols=2, name=self.sid)
        plots = fig.create_plots(Axis("time", unit=unit_time), legend=True)

        # set titles and labs
        plots[0].set_title("midazolam iv, 5 [mg]")
        plots[1].set_title("midazolam iv, 5 [mg] + Grapefruit Juice")
        plots[2].set_title("midazolam po, 15 [mg]")
        plots[3].set_title("midazolam po, 15 [mg] + Grapefruit Juice")
        for k in (0, 1):
            plots[k].set_yaxis("midazolam", unit_mid)
            plots[k].xaxis.label_visible = False
        for k in (2, 3):
            plots[k].set_yaxis("1-hydroxymidazolam", unit_mid1oh)

        # simulation
        plot_dict = {
            "mid_iv_c": {
                "plot": (0, 1),
                "label": "mid (ve blood; control)",
                "color": "black",
            },
            "mid_po_c": {
                "plot": (2, 3),
                "label": "mid (ve blood; control)",
                "color": "black",
            },
        }

        for key, value in plot_dict.items():
            for suffix in ["_sensitivity", ""]:
                task_id = f"task_{key}{suffix}"
                # plot midazolam
                p = plots[value["plot"][0]]
                p.add_data(
                    task=task_id,
                    xid="time",
                    yid="[Cve_mid]",
                    label=value["label"],
                    color=value["color"],
                    linewidth=2,
                )
                # plot 1-hydroxymidazolam
                p = plots[value["plot"][1]]
                p.add_data(
                    task=task_id,
                    xid="time",
                    yid="[Cve_mid1oh]",
                    label=value["label"],
                    color=value["color"],
                    linewidth=2,
                )

        # plot data
        data_def = {
            "Fig1_midazolam_iv_control": {
                "plot": 0,
                "key": "control",
                "color": "black",
            },
            "Fig1_midazolam_po_control": {
                "plot": 1,
                "key": "control",
                "color": "black",
            },
            "Fig2_1-hydroxymidazolam_iv_control": {
                "plot": 2,
                "key": "control",
                "color": "black",
            },
            "Fig2_1-hydroxymidazolam_po_control": {
                "plot": 3,
                "key": "control",
                "color": "black",
            },
        }

        for dset_key, dset_info in data_def.items():
            p = plots[dset_info["plot"]]
            p.add_data(
                dataset=dset_key,
                xid="time",
                yid="mean",
                yid_sd="mean_sd",
                count=None,
                color=dset_info["color"],
                label=dset_info["key"],
            )

        return {"fig1": fig}
