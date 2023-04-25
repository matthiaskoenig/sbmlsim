"""Example for model comparison."""
from pathlib import Path
from typing import List

from sbmlsim.comparison.simulate_copasi import SimulateCopasiSBML
from simulate_roadrunner import SimulateRoadrunnerSBML
from simulate import Condition, Timepoints


# FIXME: update selections to match
# FIXME: unify the results DataFrames

if __name__ == "__main__":

    base_path: Path = Path(__file__).parent
    model_path = base_path / "resources" / "icg_sd.xml"
    conditions_path = base_path / "resources" / "condition.tsv"
    timepoints = Timepoints(start=0, end=5, steps=10)

    conditions: List[Condition] = Condition.parse_conditions_from_file(conditions_path=conditions_path)

    # simulate roadrunner
    print("*** roadrunner ***")
    simulate_roadrunner = SimulateRoadrunnerSBML(
        sbml_path=model_path,
        conditions=conditions,
    )
    df = simulate_roadrunner.simulate_condition(
        condition=conditions[0],
        timepoints=timepoints,
    )
    print(df.columns)
    print(df)

    print("*** copasi ***")
    simulate_copasi = SimulateCopasiSBML(
        sbml_path=model_path,
        conditions=conditions,
    )
    df = simulate_copasi.simulate_condition(
        condition=conditions[0],
        timepoints=timepoints,
    )
    print(df.columns)
    print(df)

