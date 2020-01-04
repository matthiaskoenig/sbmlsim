"""
Test model module.
"""
import roadrunner


from sbmlsim.model import load_model, copy_model, clamp_species
from sbmlsim.model import species_df, parameter_df

from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def test_load_model():
    r = load_model(MODEL_REPRESSILATOR)
    assert r
    assert isinstance(r, roadrunner.RoadRunner)


def test_copy_model():
    r = load_model(MODEL_REPRESSILATOR)
    r_copy = copy_model(r)
    assert r_copy
    assert isinstance(r_copy, roadrunner.RoadRunner)

    r_copy.timeCourseSelections = ['time', "X"]
    print(r.timeCourseSelections)
    print(r_copy.timeCourseSelections)


def test_clamp_sid():
    r = load_model(MODEL_REPRESSILATOR)

    # Perform clamping
    r_clamp = clamp_species(r, sids=["X"], boundary_condition=True)
    assert r_clamp
    assert isinstance(r_clamp, roadrunner.RoadRunner)


def test_parameter_df():
    r = load_model(MODEL_REPRESSILATOR)
    df = parameter_df(r)

    assert df is not None
    assert "sid" in df


def test_species_df():
    r = load_model(MODEL_REPRESSILATOR)
    df = species_df(r)
    assert df is not None
    assert "sid" in df
