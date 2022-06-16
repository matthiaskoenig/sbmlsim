"""Testing sbmlsim model handling."""
from pathlib import Path

import pytest
import roadrunner

from sbmlsim.model import AbstractModel
from sbmlsim.model.rr_model import RoadrunnerSBMLModel


def test_abstractmodel_creation(repressilator_path: Path) -> None:
    """Test creation of abstract model."""
    model = AbstractModel(
        sid="model1",
        source=repressilator_path,
        language_type=AbstractModel.LanguageType.SBML,
    )
    assert model
    assert model.sid == "model1"
    assert model.source.source == repressilator_path


def test_abstractmodel_creation_with_empty_changes(repressilator_path: Path) -> None:
    """Test creation of abstract model with empty changes."""
    model = AbstractModel(
        sid="model1",
        source=repressilator_path,
        language_type=AbstractModel.LanguageType.SBML,
        changes={},
    )
    assert model
    assert len(model.changes) == 0


def test_roadrunnermodel_creation(repressilator_path: Path) -> None:
    """Test RoadrunnerSBMLModel creation."""
    model = RoadrunnerSBMLModel(source=repressilator_path)
    assert model
    assert model.sid is None
    assert model.source.source == repressilator_path
    assert model.language_type == AbstractModel.LanguageType.SBML


def test_load_roadrunner_model(repressilator_path: Path) -> None:
    """Test loading RoadRunner model."""
    r = RoadrunnerSBMLModel.loda_model_from_source(repressilator_path)
    assert r
    assert isinstance(r, roadrunner.RoadRunner)


def test_parameter_df(repressilator_path: Path) -> None:
    """Test parameter DataFrame."""
    r = RoadrunnerSBMLModel.loda_model_from_source(repressilator_path)
    df = RoadrunnerSBMLModel.parameter_df(r)

    assert df is not None
    assert "sid" in df


def test_species_df(repressilator_path: Path) -> None:
    """Test species DataFrame."""
    r = RoadrunnerSBMLModel.loda_model_from_source(repressilator_path)
    df = RoadrunnerSBMLModel.species_df(r)
    assert df is not None
    assert "sid" in df


def test_copy_model(repressilator_path: Path) -> None:
    """Test copy model."""
    r = RoadrunnerSBMLModel.loda_model_from_source(repressilator_path)
    r["X"] = 100.0
    r_copy = RoadrunnerSBMLModel.copy_roadrunner_instance(r)
    assert r_copy
    assert isinstance(r_copy, roadrunner.RoadRunner)
    assert 100.0 == pytest.approx(r_copy["X"])
