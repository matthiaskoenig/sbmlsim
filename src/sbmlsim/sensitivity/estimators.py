"""The estimators of the global sensitivity analyses, over every element at once.

An analysis of `sbmlsim.sensitivity.indices` computes the indices of every
element of its observables: every label of the other dimensions of the scan
(doses, conditions) and every time point of a timecourse. The estimators of
the three methods are means over the points of the design, so an element is
one column of an array and every element is estimated in the same pass
(`y` of the shape `(points, elements)`); the analyses called SALib once per
element before, about 10 ms, which a timecourse of a thousand time points
under a few conditions paid for every observable.

The estimators are the ones of SALib, which the tests of
`tests/sensitivity/test_global.py` compare them with on the same values, from
the papers they cite:

- `sobol_indices`: `S1` and `ST` of Saltelli et al. 2010 and `S2` of Saltelli
  2002 on the matrix of Saltelli, normalized by the variance of the sample;
- `fast_indices`: `S1` and `ST` of the extended FAST of Saltelli et al. 1999
  from the spectrum of the output over the frequency of every parameter;
- `morris_effects`: the elementary effects of Morris 1991 with the `mu_star`
  of Campolongo et al. 2007.

The intervals are the bootstrap of SALib with the resamples of its seed,
which it draws once per analysis, so every element shared them before and
shares them now. A mean over a resample is the mean of the values of the
sample weighted by how often the resample draws each of them (`_counts`), so
the intervals of Sobol and Morris are products with that matrix of counts
rather than means over an array of `points x resamples x elements` values,
which is the same mean and holds no array of that size. FAST resamples a
window of the output and not its points, so its intervals are one transform
per parameter and resample, which is the cost of a FAST analysis of many
elements.

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

import math

import numpy as np
from scipy.stats import norm

#: the smallest variance an estimator divides by, the guard of SALib against a
#: sample without variance; a smaller one gives a zero index
EPSILON = float(np.finfo(float).eps)


def _z(conf_level: float) -> float:
    """Get the factor of the normal of a confidence interval.

    Args:
        conf_level: the level of the interval, in (0, 1).

    Returns:
        The quantile `0.5 + conf_level / 2` of the standard normal.
    """
    return float(norm.ppf(0.5 + conf_level / 2.0))


def _counts(index: np.ndarray, n: int) -> np.ndarray:
    """Get how often every row of a sample is drawn by every resample.

    A mean over a resample of `n` draws is the mean of the values of the
    sample weighted by these counts, so such a mean is a product with this
    matrix.

    Args:
        index: the drawn rows, a row per resample.
        n: the number of rows of the sample.

    Returns:
        The counts as floats, a row per resample.
    """
    return np.stack([np.bincount(row, minlength=n) for row in index]).astype(float)


def _quotient(
    numerator: np.ndarray, variance: np.ndarray, usable: np.ndarray
) -> np.ndarray:
    """Divide by the variance of the sample where it has one, else give zero.

    Args:
        numerator: the numerator of the estimator.
        variance: the variance of the sample of every element.
        usable: whether the sample of an element has a variance to share out.

    Returns:
        The quotient, zero where the sample has no variance.
    """
    shape = np.broadcast_shapes(numerator.shape, variance.shape)
    out = np.zeros(shape, dtype=float)
    return np.divide(numerator, variance, out=out, where=usable & (variance > EPSILON))


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
    step = 2 * d + 2 if second_order else d + 2
    rows = y.shape[0]
    if rows == 0 or rows % step:
        raise ValueError(
            f"The design of {d} parameters has {step} points per sample, which "
            f"{rows} rows are no multiple of; an analysis of the second order "
            f"needs a design of the second order."
        )
    n = rows // step
    elements = y.shape[1]
    scale = y.std(axis=0)
    y = np.divide(
        y - y.mean(axis=0), scale, out=np.zeros_like(y, dtype=float), where=scale > 0.0
    )
    a = y[0:rows:step]
    b = y[step - 1 : rows : step]
    pooled = np.concatenate([a, b])
    variance = np.var(pooled, axis=0)
    usable = np.ptp(pooled, axis=0) > EPSILON
    weights = _counts(
        np.random.default_rng(seed).integers(n, size=(n, num_resamples)).T, n
    )
    # the variance of every resample of the pooled sample, from its sums: the
    # values are centred and scaled, so the mean is small against the spread
    # and the difference of the two moments does not cancel
    sums = weights @ (a + b)
    squares = weights @ (a * a + b * b)
    variance_boot = squares / (2 * n) - (sums / (2 * n)) ** 2
    usable_boot = np.broadcast_to(usable, variance_boot.shape)
    z = _z(conf_level)
    found: dict[str, np.ndarray] = {
        key: np.zeros((d, elements)) for key in ("S1", "S1_conf", "ST", "ST_conf")
    }
    # the first order of every resample, which the second order subtracts
    first_boot: list[np.ndarray] = []
    for j in range(d):
        ab = y[j + 1 : rows : step]
        first = b * (ab - a)
        total = 0.5 * (a - ab) ** 2
        found["S1"][j] = _quotient(np.mean(first, axis=0), variance, usable)
        found["ST"][j] = _quotient(np.mean(total, axis=0), variance, usable)
        s1_boot = _quotient(weights @ first / n, variance_boot, usable_boot)
        st_boot = _quotient(weights @ total / n, variance_boot, usable_boot)
        found["S1_conf"][j] = z * s1_boot.std(axis=0, ddof=1)
        found["ST_conf"][j] = z * st_boot.std(axis=0, ddof=1)
        if second_order:
            first_boot.append(s1_boot)
    if not second_order:
        return found
    found["S2"] = np.full((d, d, elements), np.nan)
    found["S2_conf"] = np.full((d, d, elements), np.nan)
    for j in range(d):
        ba = y[j + 1 + d : rows : step]
        for k in range(j + 1, d):
            ab = y[k + 1 : rows : step]
            closed = ba * ab - a * b
            pair = _quotient(np.mean(closed, axis=0), variance, usable)
            found["S2"][j, k] = pair - found["S1"][j] - found["S1"][k]
            pair_boot = _quotient(weights @ closed / n, variance_boot, usable_boot)
            found["S2_conf"][j, k] = z * (
                pair_boot - first_boot[j] - first_boot[k]
            ).std(axis=0, ddof=1)
    return found


def _orders(block: np.ndarray, m: int, omega: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the first and the total order of a block of a FAST design.

    `S1` is the power of the output at the harmonics `1..m` of the frequency of
    the parameter and `ST` what is left of the power above the interferences
    below `omega / 2`, each over the power of the whole spectrum. The transform
    is the one of a real signal, whose coefficients are the ones of the full
    transform up to the Nyquist frequency.

    Args:
        block: the values of the points of the parameter, a column per element.
        m: the interference factor.
        omega: the frequency of the parameter.

    Returns:
        The first and the total order of every element, zero for a block whose
        spectrum carries no power.
    """
    n = block.shape[0]
    spectrum = np.fft.rfft(block, axis=0)
    power = np.square(np.abs(spectrum[1 : math.ceil(n / 2)]) / n)
    whole = 2.0 * np.sum(power, axis=0)
    first = 2.0 * np.sum(power[np.arange(1, m + 1) * omega - 1], axis=0)
    interference = 2.0 * np.sum(power[: math.floor(omega / 2.0)], axis=0)
    usable = whole > EPSILON
    return (
        _quotient(first, whole, usable),
        np.where(usable, 1.0 - _quotient(interference, whole, usable), 0.0),
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
    search curve at its frequency `omega = floor((n - 1) / (2 m))`, see
    `_orders`. An interval is the standard deviation of the orders over
    `num_resamples` windows of half of the points of the block, drawn from the
    legacy random state of SALib: a window wraps around the end of the block
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
        ValueError: if the number of rows is no multiple of `d`.
    """
    rows = y.shape[0]
    if rows == 0 or rows % d:
        raise ValueError(
            f"The design of {d} parameters has a block of points per parameter, "
            f"which {rows} rows are no multiple of."
        )
    n = rows // d
    elements = y.shape[1]
    omega = math.floor((n - 1) / (2 * m))
    window = math.ceil(n * 0.5)
    inside_omega = math.floor((window - 1) / (2 * m))
    z = _z(conf_level)
    found: dict[str, np.ndarray] = {
        key: np.zeros((d, elements)) for key in ("S1", "S1_conf", "ST", "ST_conf")
    }
    # the legacy generator of SALib, which it seeds once and draws the starts
    # of the windows of one parameter after the other from
    rng = np.random.RandomState(seed)
    for i in range(d):
        block = y[i * n : (i + 1) * n]
        found["S1"][i], found["ST"][i] = _orders(block, m, omega)
        resampled = np.zeros((2, num_resamples, elements))
        for r in range(num_resamples):
            start = int(rng.randint(0, n))
            inside = block[np.arange(start, start + window) % n]
            resampled[0, r], resampled[1, r] = _orders(inside, m, inside_omega)
        found["S1_conf"][i] = z * resampled[0].std(axis=0, ddof=1)
        found["ST_conf"][i] = z * resampled[1].std(axis=0, ddof=1)
    return found


def _gathered(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pick the point of every trajectory which a mask marks for a parameter.

    Args:
        values: the values, `(trajectories, points, elements)`.
        mask: the marked point of every trajectory and parameter,
            `(trajectories, points, parameters)`, one per trajectory and
            parameter.

    Returns:
        The values of the marked points, `(trajectories, parameters, elements)`.
    """
    return np.einsum("tpe,tpj->tje", values, mask.astype(float))


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
    the points alone and is the same for every element, so the rows of the
    output are gathered once and the effects of every element are the
    differences of those rows. `mu` and `mu_star` are the mean of the effects
    and of their magnitudes over the trajectories (Campolongo et al. 2007),
    `sigma` is their standard deviation and `mu_star_conf` the interval of
    `mu_star` over resamples of the trajectories.

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
            `d` parameters of the cube.
    """
    rows, d = cube.shape
    size = d + 1
    if rows == 0 or rows % size or y.shape[0] != rows:
        raise ValueError(
            f"The design of {d} parameters has trajectories of {size} points, "
            f"which the {rows} points of the cube and the {y.shape[0]} rows of "
            f"the values must be a multiple of."
        )
    trajectories = rows // size
    elements = y.shape[1]
    steps = np.diff(cube.reshape(trajectories, size, d), axis=1)
    values = y.reshape(trajectories, size, elements)
    # the point of a trajectory at the larger and at the smaller value of
    # every parameter, the one after and the one before the step which moves it
    up, down = steps > 0, steps < 0
    pad = ((0, 0), (1, 0), (0, 0)), ((0, 0), (0, 1), (0, 0))
    larger = _gathered(values, np.pad(up, pad[0]) + np.pad(down, pad[1]))
    smaller = _gathered(values, np.pad(up, pad[1]) + np.pad(down, pad[0]))
    delta = levels / (2.0 * (levels - 1))
    effects = np.transpose(larger - smaller, (1, 0, 2)) / delta
    magnitude = np.abs(effects)
    # the generator of SALib, which spawns a child of the seed and draws the
    # resamples of one parameter after the other from it
    rng = np.random.default_rng(seed).spawn(1)[0]
    z = _z(conf_level)
    conf = np.zeros((d, elements))
    for j in range(d):
        weights = _counts(
            rng.integers(trajectories, size=(num_resamples, trajectories)), trajectories
        )
        resampled = weights @ magnitude[j] / trajectories
        conf[j] = z * resampled.std(axis=0, ddof=1)
    return {
        "mu": np.mean(effects, axis=1),
        "mu_star": np.mean(magnitude, axis=1),
        "sigma": np.std(effects, axis=1, ddof=1),
        "mu_star_conf": conf,
    }
