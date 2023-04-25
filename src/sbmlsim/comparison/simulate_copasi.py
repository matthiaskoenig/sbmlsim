from pathlib import Path
from typing import List, Dict

import pandas as pd
from basico import (
    load_model,
    run_time_course_with_output,
    run_time_course,
    set_parameters,
    set_species,
    set_compartment,
)


from sbmlsim.comparison.simulate import SimulateSBML, Condition, Timepoints


class SimulateCopasiSBML(SimulateSBML):
    """Class for simulating an SBML model with COPASI via basico."""

    def __init__(self, sbml_path, conditions: List[Condition], results_dir: Path):
        super().__init__(
            sbml_path=sbml_path,
            conditions=conditions,
            results_dir=results_dir
        )

        # custom model loading
        load_model(location=str(self.sbml_path))

    def simulate_condition(self, condition: Condition, timepoints: Timepoints) -> pd.DataFrame:
        print(f"simulate condition: {condition.sid}")

        # reset ? (reloading for resetting)
        load_model(location=str(self.sbml_path))

        # changes
        for change in condition.changes:
            tid = change.target_id
            value = change.value
            # is species
            if tid in self.parameters:
                set_parameters(tid, initial_value=value)
            elif tid in self.compartments:
                set_compartment(tid, initial_value=value)
            elif tid in self.species:
                if tid in self.has_only_substance:
                    set_species(tid, initial_expression=f"{value}/compartment")
                else:
                    # concentration
                    set_species(tid, initial_concentration=value)

        duration = timepoints.end - timepoints.start
        df: pd.DataFrame = run_time_course(
            timepoints.start,
            duration,
            timepoints.steps,
            use_sbml_id=True
        )

        # cleanup amount columns
        df.columns = [c.replace("Values[amount(", "").replace(")]", "") for c in df.columns]

        # add time column & remove index
        df.reset_index(inplace=True)
        df.rename(columns={"Time": "time"}, inplace=True)

        return df

