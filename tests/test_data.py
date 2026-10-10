"""Testing DataSet and Data functionality."""

from pathlib import Path

import pandas as pd
import pytest

from sbmlsim.data import Data, DataSet, load_pkdb_dataframe
from sbmlsim.units import UnitRegistry

data_dir = Path(__file__).parent / "data"


def test_dataset():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 3, 4], "col3": [4, 5, 6]})
    dset = DataSet.from_df(
        df, udict={"col1": "mM"}, ureg=UnitRegistry(on_redefinition="ignore")
    )
    assert "col1" in dset.uinfo
    assert dset.uinfo["col1"] == "mM"


def test_Faber1978_Fig1() -> None:
    """Test Faber1978 Fig1."""
    data_path = data_dir / "datasets"
    df = load_pkdb_dataframe(sid="Faber1978_Fig1", data_path=data_path)
    dset = DataSet.from_df(df, udict={}, ureg=UnitRegistry(on_redefinition="ignore"))
    assert "cpep" in dset.uinfo
    assert "time" in dset.uinfo
    assert dset.uinfo["time"] == "min"
    assert dset.uinfo["cpep"] == "pmol/ml"
    assert "time_unit" in dset.columns
    assert dset.time_unit.unique()[0] == "min"
    assert "cpep_unit" in dset.columns
    assert dset.cpep_unit.unique()[0] == "pmol/ml"


def test_Allonen1981_Fig3A() -> None:
    """Test Allonen1981 Fig3A."""
    data_path = data_dir / "datasets"
    df = load_pkdb_dataframe(sid="Allonen1981_Fig3A", data_path=data_path)
    for substance in df.substance.unique():
        dset = DataSet.from_df(
            df[df.substance == substance], ureg=UnitRegistry(on_redefinition="ignore")
        )

        assert "mean" in dset.uinfo
        assert "time" in dset.uinfo
        assert dset.uinfo["time"] == "hr"
        assert dset.uinfo["mean"] == "ng/ml"
        assert "time_unit" in dset.columns
        assert dset.time_unit.unique()[0] == "hr"
        assert "mean_unit" in dset.columns
        assert dset.mean_unit.unique()[0] == "ng/ml"
        assert "unit" not in dset.columns


def test_unit_conversion() -> None:
    """Test unit conversion."""
    data_path = data_dir / "datasets"
    df = load_pkdb_dataframe(sid="Allonen1981_Fig3A", data_path=data_path)

    ureg = UnitRegistry(on_redefinition="ignore")
    Q_ = ureg.Quantity
    Mr = Q_(300, "g/mole")
    for substance in df.substance.unique():
        d = DataSet.from_df(df[df.substance == substance], ureg=ureg)
        d.unit_conversion("mean", factor=1 / Mr)

        assert "mean" in d.uinfo
        assert "time" in d.uinfo
        assert d.uinfo["time"] == "hr"

        # check that units converted correctly
        mean_unit = ureg.Unit(d.uinfo["mean"])
        assert mean_unit.dimensionality == ureg.Unit("mole/meter**3").dimensionality

        # check that factor applied correctly
        assert d["mean"].values[0] < 0.00004
        assert d["mean"].values[0] > 0.00003


def test_dataset_external_units_are_columns() -> None:
    """A unit of the `udict` is a `*_unit` column, like the units of the table."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 3, 4]})
    dset = DataSet.from_df(
        df,
        udict={"col1": "mM", "absent": "mg"},
        ureg=UnitRegistry(on_redefinition="ignore"),
    )
    assert list(dset["col1_unit"].unique()) == ["mM"]
    # a unit of a column which the table does not have adds no column
    assert "absent_unit" not in dset.columns
    assert dset.uinfo["absent"] == "mg"


def test_load_pkdb_dataframe_missing(tmp_path: Path) -> None:
    """A dataset which is in none of the data paths names every path."""
    data_path = [tmp_path, data_dir / "datasets"]
    with pytest.raises(FileNotFoundError, match="Faber1978_Fig9") as err:
        load_pkdb_dataframe(sid="Faber1978_Fig9", data_path=data_path)
    assert all(str(p) in str(err.value) for p in data_path)


@pytest.mark.parametrize(
    ("index", "selection", "sid"),
    [
        ("[X]", "[X]", "task__conc__X"),
        ("X", "X", "task__X"),
        ("time", "time", "task__time"),
    ],
)
def test_the_selection_and_sid_of_data(index: str, selection: str, sid: str) -> None:
    """A Data selects what its index names, its sid marks a concentration."""
    data = Data(index, task="task")
    assert data.selection == selection
    assert data.sid == sid


@pytest.mark.parametrize(
    ("data", "selection", "sid", "name", "index"),
    [
        (Data("[X]", task="task"), "[X]", "task__conc__X", "X", "X"),
        (Data("X[1]", task="task"), "X[1]", "task__X[1]", "X[1]", "X[1]"),
        (Data("[X]", dataset="dset"), "[X]", "dset__conc__X", "X", "X"),
        (Data("mean", dataset="dset"), "mean", "dset__mean", "mean", "mean"),
        (Data("[X]", task="task", sid="given"), "[X]", "given", "X", "X"),
        # a function is named by its single variable, else by its own index
        (
            Data("[F]", function="Y/2", variables={"Y": Data("[Y]", task="task")}),
            "[F]",
            "conc__F",
            "Y",
            "F",
        ),
        (
            Data(
                "[F]",
                function="Y/Z",
                variables={"Y": Data("Y", task="task"), "Z": Data("Z", task="task")},
            ),
            "[F]",
            "conc__F",
            "F",
            "F",
        ),
    ],
)
def test_the_identifiers_of_data(
    data: Data, selection: str, sid: str, name: str, index: str
) -> None:
    """The brackets of a concentration are part of the selection, not of the names."""
    assert data.selection == selection
    assert data.sid == sid
    assert data.name == name
    assert data.index == index
    assert data.to_dict()["index"] == index


def test_from_df_ignores_unit_of_missing_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A row without a value has no unit, which is no second unit of the column (#275)."""
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "time_unit": ["hr", "hr", "hr"],
            "meanperiod": [float("nan"), 2.0, 3.0],
            "meanperiod_unit": [float("nan"), "hr", "hr"],
        }
    )
    with caplog.at_level("DEBUG", logger="sbmlsim.data"):
        dset = DataSet.from_df(df, ureg=UnitRegistry(on_redefinition="ignore"))
    assert [r for r in caplog.records if r.levelname == "ERROR"] == []
    assert dset.uinfo["meanperiod"] == "hr"


def test_from_df_missing_first_unit_is_not_nan() -> None:
    """The unit of a column is never nan, also when the first row has none (#275)."""
    df = pd.DataFrame(
        {
            "value": [float("nan"), 2.0],
            "value_unit": [float("nan"), "mM"],
        }
    )
    dset = DataSet.from_df(df, ureg=UnitRegistry(on_redefinition="ignore"))
    assert dset.uinfo["value"] == "mM"


def test_slices_own_their_units() -> None:
    """The unit conversion of a slice changes no other slice nor the parent (#267)."""
    ureg = UnitRegistry(on_redefinition="ignore")
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "value_unit": ["mM"] * 4,
        }
    )
    dset = DataSet.from_df(df, ureg=ureg)
    a = dset[dset.group == "a"]
    b = dset[dset.group == "b"]
    a.unit_conversion("value", 1000 * ureg.Quantity(1.0, "dimensionless"))
    assert dset.uinfo["value"] == "mM"
    assert b.uinfo["value"] == "mM"
    assert a.uinfo is not dset.uinfo
    assert a.uinfo.ureg is dset.uinfo.ureg
