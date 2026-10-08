"""Reusable functionality for multiple simulation experiments."""

from pathlib import Path
from typing import ClassVar, NamedTuple, override

import pandas as pd

from examples.hctz_fitting import MODEL_PATH
from sbmlsim import Q
from sbmlsim.data import Data, load_pkdb_dataframe
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.task import Task
from sbmlsim.units import Quantity


class MolecularWeights(NamedTuple):
    """Molecular weights of HCTZ, renin, angiotensin I and aldosterone."""

    hctz: Quantity
    ren: Quantity
    ang1: Quantity
    ald: Quantity


class HCTZSimulationExperiment(SimulationExperiment):
    """Base class for all SimulationExperiments.

    A study defines its datasets, simulations, fit mappings and figures; the
    model, a task per simulation and the selections are shared.
    """

    unit_hctz = "µM"
    unit_hctz_urine = "µmol"
    unit_hctz_feces = "µmol"
    unit_hctz_excretion_urine = "nmol/min"

    label_time = "time"
    label_hctz = "HCTZ"
    label_hctz_urine = "HCTZ urine\n"
    label_hctz_feces = "HCTZ feces\n"
    label_hctz_excretion_urine = "HCTZ excretion\nurine"

    color_hctz = "black"

    Mr: ClassVar[MolecularWeights] = MolecularWeights(
        hctz=Q(297.7, "g/mole"),
        ren=Q(45057, "g/mole"),
        ang1=Q(1296.499, "g/mole"),
        ald=Q(360.444, "g/mole"),
    )

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        """Define the whole body model of HCTZ."""
        return {
            "model": AbstractModel(
                source=MODEL_PATH,
                language_type=AbstractModel.LanguageType.SBML,
            )
        }

    @override
    def tasks(self) -> dict[str, Task]:
        """Define a task per simulation of the study."""
        return {
            f"task_{key}": Task(model="model", simulation=key)
            for key in self._simulations
        }

    @override
    def data(self) -> dict[str, Data]:
        """Define the selections of every task."""
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
                # functions of the organs
                "f_renal_function",
                "f_cirrhosis",
                "f_cardiac_function",
            ]
        )
        return {}

    def load_dataframe(self, fig_id: str) -> pd.DataFrame:
        """Load the data of a figure or table of the study."""
        if self.data_path is None:
            raise ValueError(
                f"'{self.sid}': no 'data_path' to load '{fig_id}' from, the "
                f"experiment must be created with the data path of the example."
            )
        return load_pkdb_dataframe(f"{self.sid}_{fig_id}", data_path=self.data_path)

    def default_changes(self) -> dict[str, Quantity]:
        """Default changes to simulations.

        The fitted parameters of the last optimizations are kept as a reference,
        uncomment them to simulate with the fitted values.
        """
        return {
            # pharmacokinetics
            # 20260421_191243__251fc
            #     >>> !Optimal parameter 'Kp_hctz' within 5% of upper bound! <<<
            #     >>> !Optimal parameter 'GU__F_hctz_abs' within 5% of lower bound! <<<
            # 'ftissue_hctz': Q(0.24614387687774153, 'l/min'),  # [0.01 - 10]
            # 'Kp_hctz': Q(0.9997280036700814, 'dimensionless'),  # [0.25 - 1.0]
            # 'KI__HCTZEX_k': Q(0.0037108904792554284, '1/ml'),  # [1e-10 - 1]
            # 'Ka_dis_hctz': Q(0.35181331155360623, '1/hr'),  # [0.0001 - 10]
            # 'GU__F_hctz_abs': Q(0.6121311521798801, 'dimensionless'),  # [0.6 - 0.8]
            # 'GU__HCTZABS_k': Q(0.02041376871688115, '1/min'),  # [0.0001 - 10]
            # pharmacodynamics
            # 20260428_215542__acc84
            # 'gamma_hctz_nacl': Q(3.139520154586461, 'dimensionless'),  # [1 - 10]
            # 'E50_hctz_nacl': Q(0.00015768610209207848, 'mM'),  # [1e-06 - 0.01]
            # 'Emax_hctz_na': Q(1.8546129025527704, 'dimensionless'),  # [1 - 20]
            # 'Emax_hctz_cl': Q(1.0072984331883104, 'dimensionless'),  # [1 - 20]
            # 'k_na': Q(0.0006459968399240859, 'l/min'),  # [1e-10 - 1000.0]
            # 'k_cl': Q(0.003002476234054992, 'l/min'),  # [1e-10 - 1000.0]
        }
