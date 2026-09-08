"""Test the parameter sets which connect a fit to its report."""

from pathlib import Path

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, ParameterSet, ParameterSets

PARAMETERS = [
    FitParameter("p1", start_value=1.0, lower_bound=0.1, upper_bound=10.0, unit="mM"),
    FitParameter("p2", start_value=2.0, lower_bound=0.1, upper_bound=10.0, unit="1/hr"),
]


def test_from_fit_parameters() -> None:
    """A set is created from the parameters of a problem and a vector."""
    pset = ParameterSet.from_fit_parameters(
        parameters=PARAMETERS, x=[1.5, 2.5], sid="fit", cost=0.25
    )
    assert pset.sid == "fit"
    assert pset.values == {"p1": 1.5, "p2": 2.5}
    assert pset.units == {"p1": "mM", "p2": "1/hr"}
    assert pset.cost == 0.25
    assert len(pset) == 2


def test_from_fit_parameters_wrong_size() -> None:
    """A vector of the wrong length is reported."""
    with pytest.raises(ValueError, match="values for"):
        ParameterSet.from_fit_parameters(parameters=PARAMETERS, x=[1.0], sid="fit")


def test_x_orders_by_pids() -> None:
    """The vector follows the parameter order of the problem."""
    pset = ParameterSet(sid="s", values={"p2": 2.0, "p1": 1.0})
    assert np.allclose(pset.x(["p1", "p2"]), [1.0, 2.0])
    assert np.allclose(pset.x(["p2", "p1"]), [2.0, 1.0])


def test_x_missing_parameter() -> None:
    """A set without one of the parameters is reported."""
    pset = ParameterSet(sid="s", values={"p1": 1.0})
    with pytest.raises(KeyError, match="p2"):
        pset.x(["p1", "p2"])


def test_sets_json_round_trip(tmp_path: Path) -> None:
    """The sets are stored as JSON and read back."""
    psets = ParameterSets(
        [
            ParameterSet.from_fit_parameters(PARAMETERS, [1.0, 2.0], sid="model"),
            ParameterSet.from_fit_parameters(
                PARAMETERS, [1.5, 2.5], sid="fit", cost=0.1, provenance="run 0"
            ),
        ]
    )
    path = tmp_path / "parameters.json"
    psets.to_json(path=path)

    psets2 = ParameterSets.from_json(path)
    assert len(psets2) == 2
    assert [p.sid for p in psets2] == ["model", "fit"]
    assert psets2["fit"].values == {"p1": 1.5, "p2": 2.5}
    assert psets2["fit"].cost == 0.1
    assert psets2["fit"].provenance == "run 0"
    assert psets2[0].units == {"p1": "mM", "p2": "1/hr"}


def test_sets_unique_ids() -> None:
    """Sets need unique identifiers, they label the report."""
    pset = ParameterSet(sid="s", values={"p1": 1.0})
    with pytest.raises(ValueError, match="unique"):
        ParameterSets([pset, ParameterSet(sid="s", values={"p1": 2.0})])


def test_sets_of() -> None:
    """A single set, a list of sets and `ParameterSets` are all accepted."""
    pset = ParameterSet(sid="s", values={"p1": 1.0})
    assert len(ParameterSets.of(pset)) == 1
    assert len(ParameterSets.of([pset])) == 1
    assert len(ParameterSets.of(ParameterSets([pset]))) == 1
    with pytest.raises(ValueError, match="At least one"):
        ParameterSets.of([])


def test_sets_to_df() -> None:
    """The sets are a table with one column per set."""
    psets = ParameterSets(
        [
            ParameterSet.from_fit_parameters(PARAMETERS, [1.0, 2.0], sid="model"),
            ParameterSet.from_fit_parameters(PARAMETERS, [1.5, 2.5], sid="fit"),
        ]
    )
    df = psets.to_df()
    assert list(df.columns) == ["parameter", "unit", "model", "fit"]
    assert list(df.parameter) == ["p1", "p2"]
    assert list(df.fit) == [1.5, 2.5]


def test_missing_set() -> None:
    """An unknown identifier is reported."""
    psets = ParameterSets([ParameterSet(sid="s", values={"p1": 1.0})])
    with pytest.raises(KeyError, match="unknown"):
        _ = psets["unknown"]
