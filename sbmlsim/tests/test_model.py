"""
Test model module.
"""
import roadrunner

import sbmlsim
from sbmlsim.model import clamp_species
from sbmlsim.model import species_df, parameter_df

from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def test_clamp_sid():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)

    # Perform clamping
    r_clamp = clamp_species(r, sids=["X"], boundary_condition=True)
    assert r_clamp
    assert isinstance(r_clamp, roadrunner.RoadRunner)


def test_parameter_df():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    df = parameter_df(r)

    assert df is not None
    assert "sid" in df


def test_species_df():
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    df = species_df(r)
    assert df is not None
    assert "sid" in df


