"""The estimators of the global sensitivity analyses, over every element at once.

An analysis of `sbmlsim.sensitivity.indices` computes the indices of every
element of its observables: every label of the other dimensions of the scan
(doses, conditions) and every time point of a timecourse on a grid. The
estimators of the three methods are means over the points of the design, so
an element is one column of an array and every element is estimated in the
same pass (`y` of the shape `(points, elements)`).

The estimators are the ones of SALib, from the papers they cite, in Rust in
sbmlsim-core (`sbmlsim_core.sensitivity`, the crate `sbmlsim-sensitivity`),
which spreads the elements over the CPUs of the process; the tests of
`tests/sensitivity/test_estimators.py` compare them with SALib on the same
values:

- `sobol_indices`: `S1` and `ST` of Saltelli et al. 2010 and `S2` of Saltelli
  2002 on the matrix of Saltelli, normalized by the variance of the sample;
- `fast_indices`: `S1` and `ST` of the extended FAST of Saltelli et al. 1999
  from the spectrum of the output over the frequency of every parameter;
- `morris_effects`: the elementary effects of Morris 1991 with the `mu_star`
  of Campolongo et al. 2007.

The intervals are the bootstrap of SALib: `z(conf_level)` times the standard
deviation of the estimator over resamples of the samples of Sobol, over
windows of half of the points of a block of FAST and over resamples of the
trajectories of Morris. The resamples are drawn from the seed by the
generator of sbmlsim-core, the same on every platform and the same for every
element, but not the ones numpy draws for the seed, so an interval of a seed
is another bootstrap of the same spread than the one of SALib.

The guard of SALib against a sample without variance is kept (such an index
is zero) and the one of its intervals against a resample without variance is
not: an element whose values are constant within the error of the integrator
is left out by the caller (`sbmlsim.sensitivity.indices.constant_tolerance`),
far above the machine epsilon of these guards. Groups of parameters are no
part of a design of sbmlsim, so the estimators have none.

References:
    Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., Tarantola,
    S. (2010). Variance based sensitivity analysis of model output. Design and
    estimator for the total sensitivity index. Computer Physics Communications
    181(2):259-270.

    Saltelli, A. (2002). Making best use of model evaluations to compute
    sensitivity indices. Computer Physics Communications 145(2):280-297.

    Saltelli, A., Tarantola, S., Chan, K. P.-S. (1999). A quantitative
    model-independent method for global sensitivity analysis of model output.
    Technometrics 41(1):39-56.

    Morris, M. D. (1991). Factorial sampling plans for preliminary
    computational experiments. Technometrics 33(2):161-174.

    Campolongo, F., Cariboni, J., Saltelli, A. (2007). An effective screening
    design for sensitivity analysis of large models. Environmental Modelling
    and Software 22(10):1509-1518.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np


def _core() -> ModuleType:
    """Get the estimators of sbmlsim-core, imported when an analysis runs.

    Returns:
        The module `sbmlsim_core.sensitivity`.

    Raises:
        ImportError: when the bindings of sbmlsim-core are not installed.
    """
    from sbmlsim.engine import import_core

    import_core()
    from sbmlsim_core import sensitivity

    return sensitivity


def sobol_indices(
    y: np.ndarray,
    *,
    d: int,
    second_order: bool,
    conf_level: float,
    num_resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Get the Sobol indices of the points of a design of Saltelli.

    The values are centred and scaled by their standard deviation per element,
    as SALib does, since the estimators are biased for an output which is not
    centred; an element of no standard deviation keeps its zeros and gets zero
    indices. `S1` is `mean(B (AB - A)) / var` and `ST` is
    `0.5 mean((A - AB)^2) / var` of Saltelli et al. 2010, `S2` is
    `mean(BA_j AB_k - A B) / var - S1_j - S1_k` of Saltelli 2002, each over the
    variance of the pooled sample of `A` and `B`. An interval is
    `z(conf_level)` times the standard deviation of the estimator over the
    resamples, each with the variance of its own resample.

    Args:
        y: the values, a row per point in the order of the design and a column
            per element.
        d: the number of parameters.
        second_order: whether the design carries the second order.
        conf_level: the level of the confidence intervals, in (0, 1).
        num_resamples: the number of bootstrap resamples.
        seed: the seed of the resamples.

    Returns:
        `S1`, `S1_conf`, `ST` and `ST_conf` of the shape `(d, elements)`, and
        with the second order `S2` and `S2_conf` of the shape
        `(d, d, elements)`, the upper triangle and `NaN` elsewhere.

    Raises:
        ValueError: if the number of rows is no multiple of the points of a
            sample of the design of `d` parameters.
    """
    return _core().sobol_indices(
        np.asarray(y, dtype=float),
        d,
        second_order=second_order,
        conf_level=conf_level,
        num_resamples=num_resamples,
        seed=seed,
    )


def fast_indices(
    y: np.ndarray,
    *,
    d: int,
    m: int,
    conf_level: float,
    num_resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Get the indices of the extended FAST of the points of a FAST design.

    The points of a parameter are the block of the design which walks the
    search curve at its frequency `omega = floor((n - 1) / (2 m))`: `S1` is the
    power of the output at the harmonics `1..m` of `omega` and `ST` what is
    left of the power above the interferences below `omega / 2`, each over the
    power of the whole spectrum. An interval is the standard deviation of the
    orders over `num_resamples` windows of half of the points of the block,
    from starts drawn from the seed: a window wraps around the end of the block
    and its frequency is the one of its own length. SALib documents these
    intervals as unreliable, so they are indicative only.

    Args:
        y: the values, a row per point in the order of the design and a column
            per element.
        d: the number of parameters.
        m: the interference factor of the design.
        conf_level: the level of the confidence intervals, in (0, 1).
        num_resamples: the number of bootstrap resamples.
        seed: the seed of the windows.

    Returns:
        `S1`, `S1_conf`, `ST` and `ST_conf` of the shape `(d, elements)`.

    Raises:
        ValueError: if the number of rows is no multiple of `d`, or a block has
            not more than `4 m^2` points.
    """
    return _core().fast_indices(
        np.asarray(y, dtype=float),
        d,
        m=m,
        conf_level=conf_level,
        num_resamples=num_resamples,
        seed=seed,
    )


def morris_effects(
    cube: np.ndarray,
    y: np.ndarray,
    *,
    levels: int,
    conf_level: float,
    num_resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Get the elementary effects of the trajectories of a Morris design.

    One step of a trajectory moves one parameter, so the effect of that
    parameter is the change of the output over the step divided by
    `delta = levels / (2 (levels - 1))`, the step of the grid of SALib on
    `[0, 1]`. Which parameter a step moves and in which direction follows from
    the points alone and is the same for every element. `mu` and `mu_star` are
    the mean of the effects and of their magnitudes over the trajectories
    (Campolongo et al. 2007), `sigma` is their standard deviation and
    `mu_star_conf` the interval of `mu_star` over resamples of the
    trajectories.

    Args:
        cube: the points of the design on the unit cube, a row per point; a
            centred grid gives the same effects as the grid of SALib, since
            only the sign of a step is read from it.
        y: the values, a row per point in the order of the design and a column
            per element.
        levels: the levels of the grid of the design.
        conf_level: the level of the confidence interval, in (0, 1).
        num_resamples: the number of bootstrap resamples.
        seed: the seed of the resamples.

    Returns:
        `mu`, `mu_star`, `sigma` and `mu_star_conf` of the shape
        `(d, elements)`.

    Raises:
        ValueError: if the points are no trajectories of `d + 1` points of the
            `d` parameters of the cube, every step moving one parameter.
    """
    return _core().morris_effects(
        np.asarray(cube, dtype=float),
        np.asarray(y, dtype=float),
        levels=levels,
        conf_level=conf_level,
        num_resamples=num_resamples,
        seed=seed,
    )
