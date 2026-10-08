"""Test the information tables of a roadrunner model."""

from pathlib import Path

import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.model_resources import Source
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


def test_a_url_is_no_source(tmp_path: Path) -> None:
    """A model is read from a file or the SBML itself, not downloaded."""
    with pytest.raises(OSError, match="does not exist"):
        Source.from_source(
            "https://www.ebi.ac.uk/biomodels/BIOMD0000000012", base_dir=tmp_path
        )
