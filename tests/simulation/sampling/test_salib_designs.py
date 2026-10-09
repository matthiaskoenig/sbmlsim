"""The designs of SALib."""

import numpy as np
import pytest
from SALib.sample import fast_sampler
from SALib.sample import morris as morris_sampler
from SALib.sample import sobol as sobol_sampler

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

PROBLEM = {"num_vars": 2, "names": ["x0", "x1"], "bounds": [[2.0, 5.0], [0.0, 1.0]]}
UNIFORM: dict[str, Distribution] = {"a": Uniform(2.0, 5.0), "b": Uniform(0.0, 1.0)}


def _stack(dimension: Dimension) -> np.ndarray:
    assert dimension.values is not None
    return np.column_stack([np.asarray(dimension.values[t]) for t in ("a", "b")])


def test_sobol_equals_salib() -> None:
    dimension = sobol(UNIFORM, 16, seed=3)
    expected = sobol_sampler.sample(
        PROBLEM, 16, calc_second_order=False, scramble=True, seed=3
    )
    np.testing.assert_allclose(_stack(dimension), expected)
    assert len(dimension) == 16 * (2 + 2)
    assert len(sobol(UNIFORM, 16, seed=3, second_order=True)) == 16 * (2 * 2 + 2)


def test_sobol_needs_a_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        sobol(UNIFORM, 10, seed=1)


def test_fast_equals_salib() -> None:
    dimension = fast(UNIFORM, 65, m=4, seed=2)
    np.testing.assert_allclose(
        _stack(dimension), fast_sampler.sample(PROBLEM, 65, M=4, seed=2)
    )
    with pytest.raises(ValueError, match="4"):
        fast(UNIFORM, 64, m=4, seed=2)


def test_morris_maps_the_levels_to_the_centres_of_their_strata() -> None:
    dimension = morris(UNIFORM, 10, levels=4, seed=1)
    grid = morris_sampler.sample(
        {"num_vars": 2, "names": ["x0", "x1"], "bounds": [[0.0, 1.0]] * 2},
        10,
        num_levels=4,
        seed=1,
    )
    u = (grid * 3 + 0.5) / 4
    np.testing.assert_allclose(
        _stack(dimension), np.column_stack([2.0 + 3.0 * u[:, 0], u[:, 1]])
    )


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
