"""The estimators of the global analyses against SALib, element by element.

The estimators are the ones of sbmlsim-core, whose bootstrap draws other
resamples than numpy does for a seed: the estimates are SALib's within the
rounding, the intervals SALib's in their spread (sbmlsim-core compares them
with SALib's for the resamples SALib draws).
"""

import warnings

import numpy as np
import pytest
from SALib.analyze import fast as salib_fast
from SALib.analyze import morris as salib_morris
from SALib.analyze import sobol as salib_sobol
from SALib.sample import fast_sampler
from SALib.sample import morris as morris_sampler
from SALib.sample import sobol as sobol_sampler

from sbmlsim.sensitivity.estimators import (
    fast_indices,
    morris_effects,
    sobol_indices,
)
from sbmlsim.simulation.sampling.designs import unit_problem

PI = np.pi
D = 3
PROBLEM = unit_problem(D)

#: the coefficients of the elements, the Ishigami function and two variations
COEFFICIENTS = [(7.0, 0.1), (1.0, 0.5), (0.0, 0.0)]


def _elements(cube: np.ndarray) -> np.ndarray:
    """Get a column per element: the Ishigami function with other coefficients."""
    x = cube * 2.0 * PI - PI
    return np.stack(
        [
            np.sin(x[:, 0])
            + a * np.sin(x[:, 1]) ** 2
            + b * x[:, 2] ** 4 * np.sin(x[:, 0])
            for a, b in COEFFICIENTS
        ],
        axis=1,
    )


def _same(mine: np.ndarray, theirs: np.ndarray, atol: float = 1e-14) -> None:
    """Compare an index of one element with the one of SALib, within the rounding."""
    np.testing.assert_allclose(
        mine, np.asarray(theirs, dtype=float), rtol=1e-10, atol=atol
    )


def _spread(mine: np.ndarray, theirs: np.ndarray) -> None:
    """Compare an interval with the one of SALib, another bootstrap of the same spread."""
    np.testing.assert_allclose(mine, np.asarray(theirs, dtype=float), rtol=0.15)


@pytest.mark.parametrize("second_order", [False, True])
def test_sobol_of_every_element_equals_salib(second_order: bool) -> None:
    cube = sobol_sampler.sample(
        PROBLEM, 64, calc_second_order=second_order, scramble=True, seed=2
    )
    y = _elements(cube)
    found = sobol_indices(
        y,
        d=D,
        second_order=second_order,
        conf_level=0.95,
        num_resamples=1000,
        seed=2,
    )
    for element in range(y.shape[1]):
        expected = salib_sobol.analyze(
            PROBLEM,
            y[:, element],
            calc_second_order=second_order,
            num_resamples=1000,
            conf_level=0.95,
            print_to_console=False,
            seed=np.random.default_rng(2),
        )
        for key in ("S1", "ST"):
            _same(found[key][:, element], expected[key])
        for key in ("S1_conf", "ST_conf"):
            _spread(found[key][:, element], expected[key])
        if second_order:
            upper = np.triu_indices(D, k=1)
            _same(found["S2"][..., element][upper], expected["S2"][upper])
            _spread(found["S2_conf"][..., element][upper], expected["S2_conf"][upper])
            for key in ("S2", "S2_conf"):
                assert np.isnan(found[key][..., element][np.tril_indices(D)]).all()
    assert "S2" in found if second_order else "S2" not in found


def test_fast_of_every_element_equals_salib() -> None:
    cube = fast_sampler.sample(PROBLEM, 257, M=4, seed=3)
    y = _elements(cube)
    found = fast_indices(y, d=D, m=4, conf_level=0.95, num_resamples=1000, seed=11)
    for element in range(y.shape[1]):
        np.random.seed(11)
        with warnings.catch_warnings():
            # SALib warns that its intervals of FAST are unreliable
            warnings.simplefilter("ignore", UserWarning)
            expected = salib_fast.analyze(
                PROBLEM,
                y[:, element],
                M=4,
                num_resamples=1000,
                conf_level=0.95,
                print_to_console=False,
                seed=11,
            )
        for key in ("S1", "ST"):
            _same(found[key][:, element], expected[key], atol=1e-12)
        for key in ("S1_conf", "ST_conf"):
            _spread(found[key][:, element], expected[key])


@pytest.mark.parametrize(("levels", "trajectories"), [(4, 20), (6, 13)])
def test_morris_of_every_element_equals_salib(levels: int, trajectories: int) -> None:
    grid = morris_sampler.sample(PROBLEM, trajectories, num_levels=levels, seed=4)
    # the centred cube of `designs.unit_cube`, whose steps have the signs of the grid
    cube = (grid * (levels - 1) + 0.5) / levels
    y = _elements(cube)
    found = morris_effects(
        cube, y, levels=levels, conf_level=0.95, num_resamples=1000, seed=4
    )
    for element in range(y.shape[1]):
        expected = salib_morris.analyze(
            PROBLEM,
            grid,
            y[:, element],
            num_levels=levels,
            scaled=False,
            num_resamples=1000,
            conf_level=0.95,
            print_to_console=False,
            seed=np.random.default_rng(4),
        )
        for key in ("mu", "mu_star", "sigma"):
            _same(found[key][:, element], expected[key])
        # a parameter whose effects do not vary in magnitude has no interval,
        # up to the rounding at the scale of the effects
        scale = float(np.max(np.abs(np.asarray(expected["mu_star"], dtype=float))))
        theirs = np.asarray(expected["mu_star_conf"], dtype=float)
        varying = theirs > 1e-10 * max(scale, 1.0)
        _spread(found["mu_star_conf"][varying, element], theirs[varying])
        assert (
            found["mu_star_conf"][~varying, element] < 1e-10 * max(scale, 1.0)
        ).all()


def test_an_element_without_variance_has_zero_indices() -> None:
    """A constant element is left out by the analyses, the estimator gives zeros."""
    cube = sobol_sampler.sample(
        PROBLEM, 16, calc_second_order=False, scramble=True, seed=1
    )
    y = np.zeros((cube.shape[0], 2))
    y[:, 1] = _elements(cube)[:, 0]
    found = sobol_indices(
        y, d=D, second_order=False, conf_level=0.95, num_resamples=10, seed=1
    )
    assert (found["S1"][:, 0] == 0.0).all()
    assert (found["ST"][:, 0] == 0.0).all()
    assert (found["S1_conf"][:, 0] == 0.0).all()
    assert np.any(found["ST"][:, 1] != 0.0)


def test_the_points_of_a_design_are_checked() -> None:
    """A design of three parameters has 5, 8 or 3 points per sample, block or trajectory."""
    with pytest.raises(ValueError, match="points per sample"):
        sobol_indices(
            np.zeros((9, 1)),
            d=D,
            second_order=False,
            conf_level=0.95,
            num_resamples=2,
            seed=1,
        )
    with pytest.raises(ValueError, match="points per sample"):
        sobol_indices(
            np.zeros((10, 1)),
            d=D,
            second_order=True,
            conf_level=0.95,
            num_resamples=2,
            seed=1,
        )
    with pytest.raises(ValueError, match="block of points per parameter"):
        fast_indices(
            np.zeros((10, 1)), d=D, m=4, conf_level=0.95, num_resamples=2, seed=1
        )
    with pytest.raises(ValueError, match="trajectories of 4 points"):
        morris_effects(
            np.zeros((10, D)),
            np.zeros((10, 1)),
            levels=4,
            conf_level=0.95,
            num_resamples=2,
            seed=1,
        )
