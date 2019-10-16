"""
Test model module.
"""
import roadrunner
from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim import plotting_matlab as plotting
from sbmlsim.simulation import TimecourseSimulation
from sbmlsim.model import clamp_species
from sbmlsim.parametrization import ChangeSet
from sbmlsim.model import species_df, parameter_df

from sbmlsim.tests.settings import MODEL_REPRESSILATOR


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


