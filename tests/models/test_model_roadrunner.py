"""Test the information tables of a roadrunner model."""

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML


def test_species_df() -> None:
    """Every column of the species table has values."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    assert model.r is not None
    df = RoadrunnerSBMLModel.species_df(model.r)
    assert list(df.columns) == [
        "sid",
        "concentration",
        "amount",
        "unit",
        "constant",
        "boundaryCondition",
        "name",
    ]
    assert not df.isna().all().any()


def test_parameter_df() -> None:
    """Every column of the parameter table has values."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    assert model.r is not None
    df = RoadrunnerSBMLModel.parameter_df(model.r)
    assert list(df.columns) == ["sid", "value", "unit", "constant", "name"]
    assert len(df) > 0
