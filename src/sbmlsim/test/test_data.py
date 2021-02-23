"""Testing DataSet and Data functionality."""
import pandas as pd

from sbmlsim.data import DataSet, load_pkdb_dataframe
from sbmlsim.test import DATA_DIR
from sbmlsim.units import UnitRegistry


def test_dataset():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 3, 4], "col3": [4, 5, 6]})
    dset = DataSet.from_df(df, udict={"col1": "mM"}, ureg=UnitRegistry())
    assert "col1" in dset.udict
    assert dset.udict["col1"] == "mM"


def test_Faber1978_Fig1():
    data_path = DATA_DIR / "datasets"
    df = load_pkdb_dataframe(sid="Faber1978_Fig1", data_path=data_path)
    dset = DataSet.from_df(df, ureg=UnitRegistry())
    assert "cpep" in dset.udict
    assert "time" in dset.udict
    assert dset.udict["time"] == "min"
    assert dset.udict["cpep"] == "pmol/ml"
    assert "time_unit" in dset.columns
    assert dset.time_unit.unique()[0] == "min"
    assert "cpep_unit" in dset.columns
    assert dset.cpep_unit.unique()[0] == "pmol/ml"


def test_Allonen1981_Fig3A():
    data_path = DATA_DIR / "datasets"
    df = load_pkdb_dataframe(sid="Allonen1981_Fig3A", data_path=data_path)
    for substance in df.substance.unique():
        dset = DataSet.from_df(df[df.substance == substance], ureg=UnitRegistry())

        print(dset)
        print(dset.udict)

        assert "mean" in dset.udict
        assert "time" in dset.udict
        assert dset.udict["time"] == "hr"
        assert dset.udict["mean"] == "ng/ml"
        assert "time_unit" in dset.columns
        assert dset.time_unit.unique()[0] == "hr"
        assert "mean_unit" in dset.columns
        assert dset.mean_unit.unique()[0] == "ng/ml"
        assert "unit" not in dset.columns


def test_unit_conversion1():
    data_path = DATA_DIR / "datasets"
    df = load_pkdb_dataframe(sid="Allonen1981_Fig3A", data_path=data_path)

    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    Mr = Q_(300, "g/mole")
    for substance in df.substance.unique():
        d = DataSet.from_df(df[df.substance == substance], ureg=ureg)
        d.unit_conversion("mean", factor=1 / Mr)

        assert "mean" in d.udict
        assert "time" in d.udict
        assert d.udict["time"] == "hr"

        # check that units converted correctly
        mean_unit = ureg.Unit(d.udict["mean"])
        assert mean_unit.dimensionality == ureg.Unit("mole/meter**3").dimensionality

        # check that factor applied correctly
        assert d["mean"].values[0] < 0.00004
        assert d["mean"].values[0] > 0.00003
