"""Correlated draws of the variables of an uncertainty analysis.

The points of the unit cube are quasi-random (scrambled Sobol points, the
default, Owen 1998; Joe & Kuo 2008), a Latin hypercube or random. Their normal
scores `z = Φ⁻¹(u)` are correlated with the Cholesky factor of the
correlation of the normal scores, `z L^T`, mapped back to the unit interval,
`Φ(z)`, and through the inverse distribution function of every marginal (a
Gaussian copula, Nelsen 2006). For normal marginals this is the multivariate
normal of the covariance, for log-normal marginals the multivariate normal of
their logarithms; for other marginals the correlation is the one of the
normal scores, about the same rank correlation.

The correlation of the normal scores comes from the covariances of the
analysis (`normal_score_correlation`):

- a correlation: the coefficients between the normal scores (for log-normal
  marginals between the logarithms)
- a covariance of normal marginals: `Σ_ij / (sd_i sd_j)`
- a covariance on the log scale of log-normal marginals: `Σ_ij / (σ_i σ_j)`
  with the standard deviations `σ` of the logarithms
- a covariance on the linear scale of log-normal marginals: the exact
  conversion `ln(1 + Σ_ij / (μ_i μ_j)) / (σ_i σ_j)` with the means `μ`

The variances of a covariance must be the variances of the marginals (a
relative tolerance of 1e-6), which the marginals define.

Owen AB (1998). Scrambling Sobol' and Niederreiter-Xing points. J Complexity
14:466-489. Joe S, Kuo FY (2008). Constructing Sobol sequences with better
two-dimensional projections. SIAM J Sci Comput 30:2635-2654. Nelsen RB
(2006). An Introduction to Copulas, 2nd ed. Springer.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy import stats
from scipy.stats import qmc

from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.sampling.distributions import EPS, LogNormal, Normal
from sbmlsim.simulation.sampling.references import references
from sbmlsim.uncertainty.covariance import Covariance
from sbmlsim.uncertainty.sources import Variable
from sbmlsim.units import Quantity

#: the relative tolerance of the variances of a covariance against the marginals
VARIANCE_RTOL: float = 1e-6


def _magnitude(value: Any, unit: str | None) -> float:
    """Get a number of a value, a quantity in a unit if one is given."""
    if isinstance(value, Quantity):
        return float(value.to(unit).magnitude if unit else value.magnitude)
    return float(value)


def _moments(variable: Variable, unit: str | None) -> tuple[str, float, float]:
    """Get the family, the mean and the standard deviation of a marginal.

    The family is `normal` or `log-normal`; for a log-normal the mean is
    the arithmetic mean and the standard deviation the one of the logarithm.

    Raises:
        ValueError: for another marginal or a relative one.
    """
    d = variable.distribution
    if d.is_relative:
        raise ValueError(
            f"The variable '{variable.key}' is relative to the reference of its "
            f"target; a covariance needs marginals with a location."
        )
    if isinstance(d, Normal):
        mean = _magnitude(d.mean, unit)
        sd = (
            _magnitude(d.sd, unit) if d.sd is not None else float(d.cv or 0) * abs(mean)
        )
        return "normal", mean, sd
    if isinstance(d, LogNormal):
        sigma = math.sqrt(math.log1p(float(d.cv or 0.0) ** 2))
        return (
            "log-normal",
            _magnitude(d.median, unit) * math.exp(sigma**2 / 2.0),
            sigma,
        )
    raise ValueError(
        f"The covariance of '{variable.key}' needs a normal or log-normal "
        f"marginal, not {d!r}; give a correlation for other marginals."
    )


def normal_score_correlation(
    variables: Sequence[Variable], covariance: Sequence[Covariance]
) -> np.ndarray:
    """Get the correlation of the normal scores of the variables, see the module.

    Args:
        variables: the variables, in the order of the columns.
        covariance: the covariances and correlations, by the keys.

    Returns:
        The correlation matrix of the normal scores.

    Raises:
        ValueError: for a target of a covariance which is no variable, a
            variable in two covariances, variances which disagree with the
            marginals, or a correlation which is not positive semi-definite.
    """
    index = {variable.key: k for k, variable in enumerate(variables)}
    r = np.eye(len(variables))
    covered: set[str] = set()
    for cov in covariance:
        unknown = [t for t in cov.targets if t not in index]
        if unknown:
            raise ValueError(
                f"The covariance of {list(cov.targets)} names {unknown}, which "
                f"are no variables of the analysis: {list(index)}."
            )
        twice = sorted(covered & set(cov.targets))
        if twice:
            raise ValueError(f"The variables {twice} are in two covariances.")
        covered |= set(cov.targets)
        rows = [index[t] for t in cov.targets]
        if cov.kind == "correlation":
            r[np.ix_(rows, rows)] = cov.matrix
            continue
        varying = [t for t in cov.targets if variables[index[t]].by_scenario]
        if varying:
            raise ValueError(
                f"The variables {varying} of the covariance of {list(cov.targets)} "
                f"have the marginals of its scenarios, whose variances differ; "
                f"give a correlation instead."
            )
        units = cov.units or (None,) * len(cov.targets)
        moments = [
            _moments(variables[k], unit) for k, unit in zip(rows, units, strict=True)
        ]
        families = {family for family, _, _ in moments}
        if len(families) > 1:
            raise ValueError(
                f"The covariance of {list(cov.targets)} mixes normal and log-normal "
                f"marginals; give a correlation instead."
            )
        family = families.pop()
        m = cov.matrix
        for a, (_, mean, sd) in enumerate(moments):
            if family == "normal" or cov.scale == "log":
                expected = sd**2
            else:
                expected = (mean * math.sqrt(math.expm1(sd**2))) ** 2
            if cov.scale == "log" and family == "normal":
                raise ValueError(
                    f"The covariance of {list(cov.targets)} is on the log scale, "
                    f"its marginals are normal."
                )
            if not math.isclose(m[a, a], expected, rel_tol=VARIANCE_RTOL):
                raise ValueError(
                    f"The variance {m[a, a]:.6g} of '{cov.targets[a]}' in the "
                    f"covariance differs from the variance {expected:.6g} of its "
                    f"marginal (the standard deviation of the distribution); the "
                    f"marginals define the variances."
                )
        for a, (_, mean_a, sd_a) in enumerate(moments):
            for b, (_, mean_b, sd_b) in enumerate(moments):
                if a == b:
                    continue
                if family == "normal" or cov.scale == "log":
                    rho = m[a, b] / (sd_a * sd_b)
                else:
                    rho = math.log1p(m[a, b] / (mean_a * mean_b)) / (sd_a * sd_b)
                r[rows[a], rows[b]] = rho
    # validated as a correlation, which names the conflicting pair
    Covariance([v.key for v in variables], (r + r.T) / 2.0, kind="correlation")
    return (r + r.T) / 2.0


def _factor(r: np.ndarray) -> np.ndarray:
    """Get a factor `L` of a positive semi-definite correlation, `L L^T = r`."""
    try:
        return np.linalg.cholesky(r)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(r)
        return vectors * np.sqrt(np.clip(values, 0.0, None))


def unit_points(
    d: int,
    n: int,
    *,
    design: Literal["sobol", "lhs", "random"] = "sobol",
    seed: int,
    start: int = 0,
) -> np.ndarray:
    """Get `n` points of the unit cube of `d` dimensions.

    Sobol points and random points continue their sequence from `start`, so
    the batches of a propagation are the first points of one sequence; a
    Latin hypercube has no continuation, a hypercube of more points is
    another one.

    Args:
        d: the number of dimensions.
        n: the number of points.
        design: `sobol`, `lhs` or `random`.
        seed: the seed.
        start: the index of the first point.

    Returns:
        The points, `n` rows of `d` columns.

    Raises:
        ValueError: for another design, or a Latin hypercube after `start`.
    """
    if design == "sobol":
        sampler = qmc.Sobol(d, scramble=True, seed=seed)
        with warnings.catch_warnings():
            # the balance of the points is the propagation's, which draws
            # powers of two
            warnings.filterwarnings("ignore", message=".*balance properties.*")
            if start:
                sampler.fast_forward(start)
            return sampler.random(n)
    if design == "lhs":
        if start:
            raise ValueError(
                "A Latin hypercube has no continuation, its points of more draws "
                "are another hypercube: give the number of draws, or use 'sobol' "
                "or 'random' for more draws."
            )
        return qmc.LatinHypercube(d, seed=np.random.default_rng(seed)).random(n)
    if design == "random":
        # a double of the generator is one step of its bit generator
        bits = np.random.PCG64(seed).advance(start * d)
        return np.random.Generator(bits).random((n, d))
    raise ValueError(f"The design {design!r} is none of 'sobol', 'lhs' and 'random'.")


def draw(
    variables: Sequence[Variable],
    covariance: Sequence[Covariance],
    n: int,
    *,
    design: Literal["sobol", "lhs", "random"] = "sobol",
    seed: int,
    start: int = 0,
    model: Any = None,
    simulation: Simulation | None = None,
    scenario: str | None = None,
) -> dict[str, np.ndarray | Quantity]:
    """Get `n` correlated draws of the variables, see the module.

    Args:
        variables: the variables.
        covariance: their covariances and correlations.
        n: the number of draws.
        design: `sobol`, `lhs` or `random`, see `unit_points`.
        seed: the seed.
        start: the index of the first draw of the sequence.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the
            references of relative distributions.
        scenario: the scenario of the marginals of the variables, see
            `Variable.by_scenario`; the same points give the same quantiles in
            every scenario.

    Returns:
        Key -> the values of the variable, a quantity where it has a unit.

    Raises:
        ValueError: see `normal_score_correlation`, or a relative distribution
            without a model.
    """
    if not variables:
        return {}
    r = normal_score_correlation(variables, covariance)
    u = unit_points(len(variables), n, design=design, seed=seed, start=start)
    if not np.allclose(r, np.eye(len(variables))):
        z = stats.norm.ppf(np.clip(u, EPS, 1.0 - EPS)) @ _factor(r).T
        u = stats.norm.cdf(z)
    marginals = [variable.marginal(scenario) for variable in variables]
    relative = [v for v, d in zip(variables, marginals, strict=True) if d.is_relative]
    refs: dict[str, Any] = {}
    if relative:
        if model is None:
            raise ValueError(
                f"The variables {[v.key for v in relative]} are relative to the "
                f"references of their targets: give the model."
            )
        found = references(model, [v.targets[0] for v in relative], simulation)
        refs = {v.key: found[v.targets[0]] for v in relative}
    values: dict[str, np.ndarray | Quantity] = {}
    for k, (variable, marginal) in enumerate(zip(variables, marginals, strict=True)):
        try:
            values[variable.key] = marginal.ppf(u[:, k], refs.get(variable.key))
        except ValueError as err:
            raise ValueError(f"'{variable.key}': {err}") from err
    return values
