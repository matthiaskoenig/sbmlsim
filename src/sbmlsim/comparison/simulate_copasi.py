from typing import List

import numpy as np
import pandas as pd
from basico import (
    load_model,
    run_time_course,
    set_parameters,
    set_species,
    set_compartment,
)


from sbmlsim.comparison.simulate import SimulateSBML, Condition
from pymetadata.console import console


class SimulateCopasiSBML(SimulateSBML):
    """Class for simulating an SBML model with COPASI via basico."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # custom model loading
        load_model(location=str(self.sbml_path))

    def simulate_condition(
        self, condition: Condition, timepoints: List[float]
    ) -> pd.DataFrame:
        print(f"simulate condition: {condition.sid}")

        # reset ? (reloading for resetting)
        # load_model(location=str(self.sbml_path))

        # changes
        for change in condition.changes:
            tid = change.target_id
            tname = self.sid2name[tid]
            value = change.value
            if np.isnan(value):
                continue

            # is species
            if tid in self.parameters:
                # necessary to set via name
                set_parameters(tname, initial_value=value, exact=True)
                console.print(f"{tid} = {value}")
            elif tid in self.compartments:
                set_compartment(tname, initial_value=value)
                console.print(f"{tid} = {value}")
            elif tid in self.species:
                if self.has_only_substance[tid] is True:
                    set_species(
                        tname,
                        initial_expression=f"{value}/{self.species_compartments_names[tid]}",
                    )
                    console.print(f"{tid} = {value}")
                else:
                    # concentration
                    set_species(tname, initial_concentration=value)
                    console.print(f"{tid} = {value}")

        df: pd.DataFrame = run_time_course(
            use_sbml_id=True,
            a_tol=self.absolute_tolerance,
            r_tol=self.relative_tolerance,
            use_concentrations=True,
            values=timepoints,
            update_model=False,
        )

        # cleanup amount columns
        df.columns = [
            c.replace("Values[amount(", "").replace(")]", "") for c in df.columns
        ]

        # add time column & remove index
        df.reset_index(inplace=True)
        df.rename(columns={"Time": "time"}, inplace=True)

        return df
