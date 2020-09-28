import pytest
import roadrunner

from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.test import MODEL_REPRESSILATOR


def test_abstractmodel_creation():
    model = AbstractModel(
        sid="model1",
        source=MODEL_REPRESSILATOR,
        language_type=AbstractModel.LanguageType.SBML,
    )
    assert model
    assert model.sid == "model1"
    assert model.source.source == MODEL_REPRESSILATOR


def test_abstractmodel_creation_with_changes():
    model = AbstractModel(
        sid="model1",
        source=MODEL_REPRESSILATOR,
        language_type=AbstractModel.LanguageType.SBML,
        changes=[],
    )
    assert model
    assert len(model.changes) == 0


def test_roadrunnermodel_creation():
    model = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)
    assert model
    assert model.sid is None
    assert model.source.source == MODEL_REPRESSILATOR
    assert model.language_type == AbstractModel.LanguageType.SBML


def test_load_roadrunner_model():
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    assert r
    assert isinstance(r, roadrunner.RoadRunner)


def test_parameter_df():
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    df = RoadrunnerSBMLModel.parameter_df(r)

    assert df is not None
    assert "sid" in df


def test_species_df():
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    df = RoadrunnerSBMLModel.species_df(r)
    assert df is not None
    assert "sid" in df


def test_copy_model():
    r = RoadrunnerSBMLModel.load_roadrunner_model(MODEL_REPRESSILATOR)
    r["X"] = 100.0
    r_copy = RoadrunnerSBMLModel.copy_roadrunner_model(r)
    assert r_copy
    assert isinstance(r_copy, roadrunner.RoadRunner)
    assert pytest.approx(100.0, r_copy["X"])


"""
def test_clamp_sid():
    r = load_model(MODEL_REPRESSILATOR)

    # Perform clamping
    r_clamp = clamp_species(r, sids=["X"], boundary_condition=True)
    assert r_clamp
    assert isinstance(r_clamp, roadrunner.RoadRunner)
"""
