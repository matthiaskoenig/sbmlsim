import numpy as np
import pandas as pd
import roadrunner


from sbmlsim.comparison.simulate import SimulateSBML, Condition
from pymetadata.console import console


class SimulateRoadrunnerSBML(SimulateSBML):
    """Class for simulating an SBML model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # custom model loading
        self.r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(self.sbml_path))

        # complete selections
        selections = ["time"]
        for s in self.species:
            if self.has_only_substance[s] is True:
                selections.append(s)
            else:
                selections.append(f"[{s}]")

        selections += list(self.parameters)
        selections += list(self.compartments)
        self.r.selections = selections

        # tolerances
        integrator: roadrunner.Integrator = self.r.integrator
        integrator.setValue("absolute_tolerance", self.absolute_tolerance)
        integrator.setValue("relative_tolerance", self.relative_tolerance)

    def simulate_condition(
        self, condition: Condition, timepoints: list[float]
    ) -> pd.DataFrame:
        """Simulate condition"""
        # print(f"simulate condition: {condition.sid}")

        # reset
        self.r.resetAll()

        # changes
        for change in condition.changes:
            tid = change.target_id
            value = change.value
            if np.isnan(value):
                continue
            # is species
            # print(tid)
            if tid in self.species:
                if self.has_only_substance[tid] is True:
                    # amount
                    target = f"init({tid})"
                else:
                    # concentration
                    target = f"init([{tid}])"
                self.r.setValue(target, value)
                # print(f"{target} = {value}")
            else:
                self.r.setValue(tid, value)
                console.print(f"{tid} = {value}")

        # simulate
        s = self.r.simulate(times=timepoints)
        df = pd.DataFrame(s, columns=s.colnames)

        # cleanup column names for concentration species
        df.columns = [c.replace("[", "").replace("]", "") for c in df.columns]

        return df
