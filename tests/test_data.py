"""Testing DataSet and Data functionality."""
import pandas as pd
from conftest import data_dir

from sbmlsim.data import DataSet, load_pkdb_dataframe
from sbmlsim.units import UnitRegistry


from pathlib import Path

from sbmlsim import RESOURCES_DIR

TEST_PATH = Path(__file__).parents[0]  # directory of tests files
DATA_DIR = RESOURCES_DIR / "testdata"  # directory of data for tests

MODEL_DIR = DATA_DIR / "models"

MODEL_REPRESSILATOR = MODEL_DIR / "repressilator.xml"
MODEL_GLCWB = MODEL_DIR / "body21_livertoy_flat.xml"
MODEL_DEMO = MODEL_DIR / "Koenig_demo_14.xml"
MODEL_MIDAZOLAM = MODEL_DIR / "midazolam_body_flat.xml"


def test_dataset():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [2, 3, 4], "col3": [4, 5, 6]})
    dset = DataSet.from_df(df, udict={"col1": "mM"}, ureg=UnitRegistry(on_redefinition='ignore'))
    assert "col1" in dset.uinfo
    assert dset.uinfo["col1"] == "mM"


def test_Faber1978_Fig1() -> None:
    """Test Faber1978 Fig1."""
    data_path = data_dir / "datasets"
    df = load_pkdb_dataframe(sid="Faber1978_Fig1", data_path=data_path)
    dset = DataSet.from_df(df, udict={}, ureg=UnitRegistry(on_redefinition='ignore'))
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
        dset = DataSet.from_df(df[df.substance == substance], ureg=UnitRegistry(on_redefinition='ignore'))

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

    ureg = UnitRegistry(on_redefinition='ignore')
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
