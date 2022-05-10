"""Testing sbmlsim model handling."""

import pytest
import roadrunner

from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from tests import MODEL_REPRESSILATOR


def test_abstractmodel_creation() -> None:
    """Test creation of abstract model."""
    model = AbstractModel(
        sid="model1",
        source=MODEL_REPRESSILATOR,
        language_type=AbstractModel.LanguageType.SBML,
    )
    assert model
    assert model.sid == "model1"
    assert model.source.source == MODEL_REPRESSILATOR


def test_abstractmodel_creation_with_empty_changes() -> None:
    """Test creation of abstract model with empty changes."""
    model = AbstractModel(
        sid="model1",
        source=MODEL_REPRESSILATOR,
        language_type=AbstractModel.LanguageType.SBML,
        changes={},
    )
    assert model
    assert len(model.changes) == 0


def test_roadrunnermodel_creation() -> None:
    """Test RoadrunnerSBMLModel creation."""
    model = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)
    assert model
    assert model.sid is None
    assert model.source.source == MODEL_REPRESSILATOR
    assert model.language_type == AbstractModel.LanguageType.SBML


def test_load_roadrunner_model() -> None:
    """Test loading RoadRunner model."""
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    assert r
    assert isinstance(r, roadrunner.RoadRunner)


def test_parameter_df() -> None:
    """Test parameter DataFrame."""
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    df = RoadrunnerSBMLModel.parameter_df(r)

    assert df is not None
    assert "sid" in df


def test_species_df() -> None:
    """Test species DataFrame."""
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    df = RoadrunnerSBMLModel.species_df(r)
    assert df is not None
    assert "sid" in df


def test_copy_model() -> None:
    """Test copy model."""
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    r["X"] = 100.0
    r_copy = RoadrunnerSBMLModel.copy_roadrunner_model(r)
    assert r_copy
    assert isinstance(r_copy, roadrunner.RoadRunner)
    assert 100.0 == pytest.approx(r_copy["X"])
