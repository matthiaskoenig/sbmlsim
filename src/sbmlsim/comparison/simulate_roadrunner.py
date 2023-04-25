from pathlib import Path
from typing import List, Dict

import pandas as pd
import roadrunner


from sbmlsim.comparison.simulate import SimulateSBML, Condition, Timepoints


class SimulateRoadrunnerSBML(SimulateSBML):
    """Class for simulating an SBML model."""

    def __init__(self, sbml_path, conditions: List[Condition], results_dir: Path):
        super().__init__(
            sbml_path=sbml_path,
            conditions=conditions,
            results_dir=results_dir
        )

        # custom model loading
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(self.sbml_path))

        # complete selections
        selections = ["time"]
        for s in self.species:
            if self.has_only_substance:
                selections.append(s)
            else:
                selections.append(f"[{s}]")
        selections += list(self.species)
        selections += list(self.parameters)

        self.r.selections = selections

    def simulate_condition(self, condition: Condition, timepoints: Timepoints) -> pd.DataFrame:
        print(f"simulate condition: {condition.sid}")

        # reset
        self.r.resetAll()

        # changes
        for change in condition.changes:
            tid = change.target_id
            value = change.value
            # is species
            if tid in self.species:

                if self.has_only_substance[tid]:
                    # amount
                    target = f"init({tid})"
                else:
                    # concentration
                    target = f"init([{tid}])"
                self.r.setValue(target, value)

        # simulate
        s = self.r.simulate(
            start=timepoints.start,
            end=timepoints.end,
            steps=timepoints.steps,
        )
        df = pd.DataFrame(s, columns=s.colnames)

        # cleanup column names for concentration species
        df.columns = [c.replace("[", "").replace("]", "") for c in df.columns]

        return df
