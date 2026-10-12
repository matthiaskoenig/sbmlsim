"""Interpolation preserves exact values next to missing data."""

import numpy as np

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result.timecourse import GridWeights, apply_weights, interpolate
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml


def test_reinterpolating_a_grid_preserves_a_shorter_simulation_endpoint() -> None:
    model = RoadrunnerSBMLModel(
        source=sbml("model endpoint\n S=0; S'=1;\nend"), selections=["S"]
    )
    result = Simulator(n_workers=1).run(
        model, Simulation(end=1, steps=1), time=[0, 1, 2]
    )
    np.testing.assert_allclose(result["S"], [0, 1, np.nan])
    np.testing.assert_allclose(result.interpolate([1])["S"], [1])
    np.testing.assert_allclose(result.interpolate([0, 1, 2])["S"], [0, 1, np.nan])


def test_missing_neighbors_only_affect_intermediate_values() -> None:
    result = interpolate(
        np.array([0.0, 1.0, 2.0]),
        np.array([np.nan, 1.0, np.nan]),
        np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    )
    np.testing.assert_allclose(result, [np.nan, np.nan, 1, np.nan, np.nan])


def test_endpoint_weights_broadcast_over_multiple_timecourses() -> None:
    weights = GridWeights(
        lower=np.array([0, 0]),
        upper=np.array([1, 1]),
        weight=np.array([0.0, 1.0]),
        valid=np.array([True, True]),
    )
    values = np.array([[1.0, np.nan], [np.nan, 2.0], [np.inf, 3.0]])
    np.testing.assert_allclose(apply_weights(weights, values), values)
