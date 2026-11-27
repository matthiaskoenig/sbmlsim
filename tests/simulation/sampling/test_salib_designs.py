"""The designs of SALib, drawn by sbmlsim-core.

sbmlsim-core draws the points on the unit cube with a generator of its own, so
the points of a seed are not the ones SALib draws for it; it compares its
designs with SALib's where a seed does not decide them. These tests check that
a design is the one of sbmlsim-core mapped through the distributions.
"""

import numpy as np
import pytest
from sbmlsim_core import sensitivity as core

from sbmlsim.simulation.sampling import (
    Distribution,
    LogNormal,
    Normal,
    Uniform,
    fast,
    morris,
    sobol,
)
from sbmlsim.simulation.sampling.designs import unit_cube
from sbmlsim.simulation.scan import Dimension

UNIFORM: dict[str, Distribution] = {"a": Uniform(2.0, 5.0), "b": Uniform(0.0, 1.0)}


def _stack(dimension: Dimension) -> np.ndarray:
    assert dimension.values is not None
    return np.column_stack([np.asarray(dimension.values[t]) for t in ("a", "b")])


def _mapped(u: np.ndarray) -> np.ndarray:
    """The points of the unit cube mapped through the distributions of `UNIFORM`."""
    return np.column_stack([2.0 + 3.0 * u[:, 0], u[:, 1]])


def test_sobol_is_the_matrix_of_saltelli_of_sbmlsim_core() -> None:
    dimension = sobol(UNIFORM, 16, seed=3)
    np.testing.assert_allclose(
        _stack(dimension), _mapped(core.saltelli(2, 16, seed=3)), rtol=1e-15
    )
    assert len(dimension) == 16 * (2 + 2)
    assert len(sobol(UNIFORM, 16, seed=3, second_order=True)) == 16 * (2 * 2 + 2)


def test_sobol_needs_a_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        sobol(UNIFORM, 10, seed=1)


def test_fast_is_the_design_of_sbmlsim_core() -> None:
    dimension = fast(UNIFORM, 65, m=4, seed=2)
    np.testing.assert_allclose(
        _stack(dimension), _mapped(core.fast_sample(2, 65, m=4, seed=2)), rtol=1e-15
    )
    with pytest.raises(ValueError, match="4"):
        fast(UNIFORM, 64, m=4, seed=2)


def test_morris_maps_the_levels_to_the_centres_of_their_strata() -> None:
    dimension = morris(UNIFORM, 10, levels=4, seed=1)
    grid = core.morris_sample(2, 10, levels=4, seed=1)
    u = (grid * 3 + 0.5) / 4
    np.testing.assert_allclose(_stack(dimension), _mapped(u), rtol=1e-15)
    assert set(np.round(u.ravel() * 8).astype(int)) <= {1, 3, 5, 7}


def test_unbounded_marginals_stay_finite() -> None:
    distributions: dict[str, Distribution] = {
        "a": Normal(0.0, 1.0),
        "b": LogNormal(1.0, 0.5),
    }
    for dimension in (
        sobol(distributions, 8, seed=1),
        fast(distributions, 65, seed=1),
        morris(distributions, 4, seed=1),
    ):
        assert np.isfinite(_stack(dimension)).all()


def test_a_seed_is_below_2_to_the_64() -> None:
    sobol(UNIFORM, 8, seed=2**64 - 1)
    with pytest.raises(ValueError, match="below 2"):
        sobol(UNIFORM, 8, seed=2**64)


def test_a_correlation_is_refused() -> None:
    for design in (sobol, fast, morris):
        with pytest.raises(TypeError):
            design(UNIFORM, 8, seed=1, correlation=[[1.0, 0.5], [0.5, 1.0]])  # ty: ignore[unknown-argument]


def test_the_record_recreates_the_unit_cube() -> None:
    for dimension in (
        sobol(UNIFORM, 8, seed=4),
        fast(UNIFORM, 65, seed=4),
        morris(UNIFORM, 5, seed=4),
    ):
        assert dimension.design is not None and dimension.values is not None
        u = unit_cube(dimension.design, 2)
        np.testing.assert_allclose(
            2.0 + 3.0 * u[:, 0], np.asarray(dimension.values["a"])
        )
