"""Reusable functionality for multiple simulation experiments."""

from collections import namedtuple
from pathlib import Path
from typing import ClassVar

import pandas as pd

from examples.hctz_fitting import MODEL_PATH
from sbmlsim.data import load_pkdb_dataframe
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import AbstractSim
from sbmlsim.task import Task

MolecularWeights = namedtuple("MolecularWeights", "hctz ren ang1 ald")


class HCTZSimulationExperiment(SimulationExperiment):
    """Base class for all SimulationExperiments."""

    font: ClassVar[dict] = {"weight": "bold", "size": 22}
    scan_font: ClassVar[dict] = {"weight": "bold", "size": 15}
    tick_font_size = 15
    legend_font_size = 13
    suptitle_font_size = 40

    unit_time = "hr"
    unit_hctz = "µM"
    unit_hctz_urine = "µmol"
    unit_hctz_feces = "µmol"
    unit_hctz_excretion_urine = "nmol/min"

    unit_urine_volume = "ml"
    unit_na = "mM"
    unit_cl = "mM"
    unit_na_urine = "mmole"
    unit_cl_urine = "mmole"

    units: ClassVar[dict[str, str]] = {
        "time": "hr",
        "[Cve_hctz]": unit_hctz,
        "Aurine_hctz": unit_hctz_urine,
        "Afeces_hctz": unit_hctz_feces,
        "KI__HCTZEX": unit_hctz_excretion_urine,
        "[Cve_ang1]": "nM",
        "[Cve_ald]": "nM",
        "HR": "1/min",
        "bp_systolic": "mmHg",
        "bp_diastolic": "mmHg",
        "renin_activity": "pmole/min/l",
        "Vurine": unit_urine_volume,
        "NA_EXCRETION": "mmole/hr",
        "CL_EXCRETION": "mmole/hr",
        "vin_na": "mmole/hr",
        "vin_cl": "mmole/hr",
        "NA_UPTAKE": "mmole/hr",
        "CL_UPTAKE": "mmole/hr",
        "NA_FILTRATION": "mmole/hr",
        "CL_FILTRATION": "mmole/hr",
        "NA_REABSORPTION": "mmole/hr",
        "CL_REABSORPTION": "mmole/hr",
        "diuresis": "ml/hr",
        "vin_h2o": "ml/hr",
        "H2O_UPTAKE": "ml/hr",
        "h2o_reabsorption": "ml/hr",
        "ECF": "l",
        "[na]": unit_na,
        "[cl]": unit_cl,
        "na_urine": unit_na_urine,
        "cl_urine": unit_cl_urine,
    }

    label_time = "time"
    label_hctz = "HCTZ"
    label_hctz_urine = "HCTZ urine\n"
    label_hctz_feces = "HCTZ feces\n"
    label_hctz_excretion_urine = "HCTZ excretion\nurine"

    label_urine_volume = "Urine volume"
    label_na = "Sodium ECF"
    label_cl = "Chloride ECF"
    label_na_urine = "Sodium urine"
    label_cl_urine = "Chloride urine"

    labels: ClassVar[dict[str, str]] = {
        "time": label_time,
        "[Cve_hctz]": label_hctz,
        "Aurine_hctz": label_hctz_urine,
        "Afeces_hctz": label_hctz_feces,
        "KI__HCTZEX": label_hctz_excretion_urine,
        "HR": "heart rate",
        "bp_systolic": "blood pressure systolic\n",
        "bp_diastolic": "blood pressure diastolic\n",
        "renin_activity": "renin activity\n",
        "Vurine": label_urine_volume,
        "NA_EXCRETION": "Sodium excretion\n",
        "CL_EXCRETION": "Chloride excretion\n",
        "vin_na": "Na uptake baseline",
        "vin_cl": "Cl uptake baseline",
        "NA_UPTAKE": "Na uptake",
        "CL_UPTAKE": "Cl uptake",
        "NA_FILTRATION": "Na filtration",
        "CL_FILTRATION": "Cl filtration",
        "NA_REABSORPTION": "Na reabsorption",
        "CL_REABSORPTION": "Cl reabsorption",
        "vin_h2o": "H2O uptake baseline",
        "H2O_UPTAKE": "H2O uptake",
        "h2o_reabsorption": "H2O reabsorption",
        "diuresis": "diuresis",
        "ECF": "Extracellular fluid",
        "[na]": label_na,
        "[cl]": label_cl,
        "na_urine": label_na_urine,
        "cl_urine": label_cl_urine,
    }

    color_hctz = "black"
    color_hctz_urine = "black"

    def load_dataframe(self, fig_id: str) -> pd.DataFrame:
        """Load the data of a figure or table of the study."""
        if self.data_path is None:
            raise ValueError(
                f"'{self.sid}': no 'data_path' to load '{fig_id}' from, the "
                f"experiment must be created with the data path of the example."
            )
        return load_pkdb_dataframe(f"{self.sid}_{fig_id}", data_path=self.data_path)

    def models(self) -> dict[str, AbstractModel | Path]:
        return {
            "model": AbstractModel(
                source=MODEL_PATH,
                language_type=AbstractModel.LanguageType.SBML,
                changes={},
            )
        }

    @staticmethod
    def _default_changes(Q_):
        """Default changes to simulations.

        The fitted parameters of the last optimizations are kept as a reference,
        uncomment them to simulate with the fitted values.
        """
        return {
            # pharmacokinetics
            # 20260421_191243__251fc
            #     >>> !Optimal parameter 'Kp_hctz' within 5% of upper bound! <<<
            #     >>> !Optimal parameter 'GU__F_hctz_abs' within 5% of lower bound! <<<
            # 'ftissue_hctz': Q_(0.24614387687774153, 'l/min'),  # [0.01 - 10]
            # 'Kp_hctz': Q_(0.9997280036700814, 'dimensionless'),  # [0.25 - 1.0]
            # 'KI__HCTZEX_k': Q_(0.0037108904792554284, '1/ml'),  # [1e-10 - 1]
            # 'Ka_dis_hctz': Q_(0.35181331155360623, '1/hr'),  # [0.0001 - 10]
            # 'GU__F_hctz_abs': Q_(0.6121311521798801, 'dimensionless'),  # [0.6 - 0.8]
            # 'GU__HCTZABS_k': Q_(0.02041376871688115, '1/min'),  # [0.0001 - 10]
            # pharmacodynamics
            # 20260428_215542__acc84
            # 'gamma_hctz_nacl': Q_(3.139520154586461, 'dimensionless'),  # [1 - 10]
            # 'E50_hctz_nacl': Q_(0.00015768610209207848, 'mM'),  # [1e-06 - 0.01]
            # 'Emax_hctz_na': Q_(1.8546129025527704, 'dimensionless'),  # [1 - 20]
            # 'Emax_hctz_cl': Q_(1.0072984331883104, 'dimensionless'),  # [1 - 20]
            # 'k_na': Q_(0.0006459968399240859, 'l/min'),  # [1e-10 - 1000.0]
            # 'k_cl': Q_(0.003002476234054992, 'l/min'),  # [1e-10 - 1000.0]
        }

    def default_changes(self: SimulationExperiment) -> dict:
        """Default changes to simulations."""
        return HCTZSimulationExperiment._default_changes(Q_=self.Q_)

    def simulations(self) -> dict[str, AbstractSim]:
        return {}

    def tasks(self) -> dict[str, Task]:
        if self.simulations():
            return {
                f"task_{key}": Task(model="model", simulation=key)
                for key in self.simulations()
            }
        return {}

    def data(self) -> dict:
        self.add_selections_data(
            selections=[
                "time",
                # pharmacokinetics
                "IVDOSE_hctz",
                "PODOSE_hctz",
                "[Cve_hctz]",
                "Afeces_hctz",  # cumulative amount of hctz in feces
                "KI__HCTZEX",  # urinary excretion rate
                "Aurine_hctz",  # cumulative amount of hctz in urine
                "hctz_urine_excretion",  # amount of hctz in urine based on collected volume
                "HR",
                # Urine volume & ion balance
                "NA_EXCRETION",  # mmole/hr [mmole/min]
                "CL_EXCRETION",  # mmole/hr [mmole/min]
                "NA_FILTRATION",  # mmole/hr [mmole/min]
                "CL_FILTRATION",  # mmole/hr [mmole/min]
                "NA_REABSORPTION",  # mmole/hr [mmole/min]
                "CL_REABSORPTION",  # mmole/hr [mmole/min]
                "diuresis",  # ml/hr [l/min]
                "Vurine",  # urine volume
                "na_urine",  # sodium urine
                "cl_urine",  # chloride urine
                "[na]",  # sodium ECF
                "[cl]",  # chloride ECF
                "ECF",
                "vin_na",
                "vin_cl",
                "NA_UPTAKE",
                "CL_UPTAKE",
                "vin_h2o",
                "H2O_UPTAKE",
                "h2o_reabsorption",
                "bp_systolic",  # blood pressure systolic
                "bp_diastolic",  # blood pressure diastolic
                # parameter scans
                "PODOSE_hctz",
                "f_renal_function",
                "f_cirrhosis",
                "f_cardiac_function",
            ]
        )
        return {}

    @property
    def Mr(self):
        return MolecularWeights(
            hctz=self.Q_(297.7, "g/mole"),
            ren=self.Q_(45057, "g/mole"),
            ang1=self.Q_(1296.499, "g/mole"),
            ald=self.Q_(360.444, "g/mole"),
        )

    renal_colors: ClassVar[dict] = {
        "Control": "black",
        "Normal renal function": "black",
        "Mild renal impairment": "#66c2a4",
        "Moderate renal impairment": "#2ca25f",
        "Severe renal impairment": "#006d2c",
    }
    renal_map: ClassVar[dict] = {
        "Normal renal function": 101.0 / 101.0,  # 1.0,
        "Mild renal impairment": 69.5 / 101.0,  # 0.69
        "Moderate renal impairment": 32.5 / 101.0,  # 0.32
        "Severe renal impairment": 19.5 / 101.0,  # 0.19
    }

    cardiac_colors: ClassVar[dict] = {
        "Normal cardiac function": "black",
        "Mild cardiac impairment": "#FFB233",
        "Moderate cardiac impairment": "#FF8333",
        "Severe cardiac impairment": "tab:red",
        "Cardiac failure": "darkred",
    }

    # Normal CO: ~4.5 to 6 L/min
    # Mild impairment: Slight reduction or maintained near normal (e.g., 4-5.5 L/min)
    # Moderate impairment: Reduced CO possibly around 3-4 L/min
    # Severe impairment and failure: Marked reduction, often below 3 L/min, sometimes much lower depending on severity and heart failure stage.
    cardiac_map: ClassVar[dict] = {
        "Normal cardiac function": 1.0,
        "Mild cardiac impairment": 4.75 / 5.25,  # 0.90,
        "Moderate cardiac impairment": 3.5 / 5.25,  # 0.67,
        "Severe cardiac impairment": 3 / 5.25,  # 0.57,
        "Cardiac failure": 0.5,
    }

    # ----------- Cirrhosis map --------------
    cirrhosis_map: ClassVar[dict] = {
        "Control": 0,
        "Mild cirrhosis": 0.3994897959183674,  # CPT A
        "Moderate cirrhosis": 0.6979591836734694,  # CPT B
        "Severe cirrhosis": 0.8127551020408164,  # CPT C
    }
    cirrhosis_colors: ClassVar[dict] = {
        "Control": "black",
        "Mild cirrhosis": "#74a9cf",  # CPT A
        "Moderate cirrhosis": "#2b8cbe",  # CPT B
        "Severe cirrhosis": "#045a8d",  # CPT C
    }

    # --- Pharmacokinetic parameters ---
    pk_labels: ClassVar[dict] = {
        "auc": "AUCend",
        "aucinf": "AUC",
        "cl": "Total clearance",
        "cl_renal": "Renal clearance",
        "cl_hepatic": "Hepatic clearance",
        "cmax": "Cmax",
        "thalf": "Half-life",
        "kel": "kel",
        "vd": "vd",
    }

    pk_units: ClassVar[dict] = {
        "auc": "µmole/l*hr",
        "aucinf": "µmole/l*hr",
        "cl": "ml/min",
        "cl_renal": "ml/min",
        "cl_hepatic": "ml/min",
        "cmax": "µmole/l",
        "thalf": "hr",
        "kel": "1/hr",
        "vd": "l",
    }
