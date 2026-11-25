"""Interpolations in a model definition of the factory."""

from pathlib import Path

import pandas as pd
import pytest
from sbmlutils.factory import AssignmentRule

from sbmlsim.interpolation import Interpolation
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from tests.interpolation.interpolation_models import uptake_model, write_model

DATA = pd.DataFrame({"time": [0.0, 2.0, 4.0], "f_data": [1.0, 1.0 / 3.0, 2.0]})


def test_assignment_rules_of_mapping() -> None:
    """One rule per driven element, the formula of its column."""
    interpolation = Interpolation(DATA, method="linear")
    rules = interpolation.assignment_rules({"f_data": "f"})
    assert [type(rule) for rule in rules] == [AssignmentRule]
    assert rules[0].variable == "f"
    assert rules[0].value == interpolation.interpolators[0].formula()
    assert repr(1.0 / 3.0) in str(rules[0].value)


def test_assignment_rules_in_model(tmp_path: Path) -> None:
    """A model definition takes the rules and simulates the data."""
    model = uptake_model()
    model.parameters[1].constant = False
    model.rules = Interpolation(DATA, method="linear").assignment_rules({"f_data": "f"})
    path = write_model(model, tmp_path / "uptake.xml")
    simulator = Simulator(n_workers=1)
    loaded = simulator.load(path)
    loaded.set_selections(["time", "f"])
    result = simulator.simulate(loaded, Simulation(end=4, times=[0, 2, 4]))
    assert result["f"] == pytest.approx([1.0, 1.0 / 3.0, 2.0])


def test_assignment_rules_column_not_sid_raises() -> None:
    """Without targets a column name has to be an SBML id."""
    data = DATA.rename(columns={"f_data": "f [1/min]"})
    with pytest.raises(ValueError, match="not an SBML id"):
        Interpolation(data).assignment_rules()
