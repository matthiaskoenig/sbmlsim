"""The joint prior of the parameters of a fit and the priors as distributions.

A parameter of a fit has the prior of `FitParameter.prior`, truncated at its
bounds, or else the uniform prior over its bounds. The priors are independent
unless the problem has a `prior_correlation`, a `Covariance.correlation` of
parameters which have a normal or a log-normal prior: the priors of these
parameters are one Gaussian copula with this correlation of their normal
scores (Nelsen 2006), i.e. the multivariate normal of their values (normal)
and of the logarithms of their values (log-normal), restricted to their
bounds. It is the dependence of the uncertainty analyses as well, see
`sbmlsim.uncertainty.Covariance`.

A fit of `ObjectiveType.POSTERIOR` minimizes the negative log posterior as a
sum of squares, `JointPrior.residuals`:

- an independent prior has the residual `sqrt(c - 2 log p)` with the offset
  `c` of `_independent_term`, so half its square minus `c / 2` is the negative
  log density of the prior truncated at the bounds;
- the correlated priors are whitened (Tarantola 2005): with the normal scores
  `z = (v - mu) / sigma` of their values `v` (the logarithm of a log-normal
  one) and the Cholesky factor `L` of their correlation `R = L L'`, the
  residuals `w = L⁻¹ z` have the half sum of squares `z' R⁻¹ z / 2`, the
  quadratic form of the multivariate normal; a log-normal prior has the
  residual `sqrt(c + 2 ln(x))` of the density of the logarithm in the value,
  `1 / x`. The density of correlated priors is normalized as the
  multivariate normal without the bounds, so the cost is the negative log
  posterior up to a constant of the problem, which neither the estimate, its
  profiles nor its Laplace approximation depend on.

`prior_distributions` and `prior_model` give the priors of a problem as the
distributions of the designs of `sbmlsim.simulation.sampling` and as the
`UncertaintyModel` of a propagation: a prior truncated at the bounds of its
parameter, the uniform distribution over the bounds for a parameter without
one, and the correlation of the problem. The marginals of correlated priors
are truncated each at its bounds, which equals the joint prior of the fit
where the bounds leave the priors untouched.

Nelsen RB (2006). An Introduction to Copulas, 2nd ed. Springer. Tarantola A
(2005). Inverse Problem Theory and Methods for Model Parameter Estimation.
SIAM.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sbmlsim.fit.objects import FitParameter, PriorDistribution

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.simulation.sampling.distributions import Distribution
    from sbmlsim.uncertainty.covariance import Covariance
    from sbmlsim.uncertainty.sources import UncertaintyModel

#: the priors which a correlation of priors takes: their normal scores are
#: the standardized values or logarithms
CORRELATED_PRIORS: tuple[PriorDistribution, ...] = (
    PriorDistribution.NORMAL,
    PriorDistribution.LOG_NORMAL,
)


def _independent_term(
    parameter: FitParameter,
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Get the log density of the prior of a parameter and the offset of its residual.

    Args:
        parameter: a parameter with a prior.

    Returns:
        The function of the log density of the prior truncated at the bounds
        and the offset of the residual: twice the largest log density on a
        grid of the bounds and at the parameters of the prior, plus one, so
        `c - 2 log p` is positive within the bounds.
    """
    if parameter.prior is None:
        raise ValueError(f"The parameter '{parameter.pid}' has no prior.")
    lower, upper = parameter.lower_bound, parameter.upper_bound
    log_density = parameter.prior.log_density_function(lower, upper)
    candidates = [np.asarray(parameter.prior.parameters, dtype=float)]
    if np.isfinite(lower) and np.isfinite(upper):
        candidates.append(np.linspace(lower, upper, 2001))
        if lower > 0.0:
            candidates.append(np.geomspace(lower, upper, 2001))
    grid = np.concatenate(candidates)
    grid = grid[(grid >= lower) & (grid <= upper)]
    values = log_density(grid) if grid.size else np.array([0.0])
    finite = values[np.isfinite(values)]
    largest = float(np.max(finite)) if finite.size else 0.0
    return log_density, 2.0 * max(largest, 0.0) + 1.0


@dataclass(frozen=True)
class _Independent:
    """The prior of one parameter which is not correlated.

    Attributes:
        index: the position of the parameter in the problem.
        log_density: the log density of the prior truncated at the bounds.
        offset: the offset `c` of the residual `sqrt(c - 2 log p)`.
    """

    index: int
    log_density: Callable[[np.ndarray], np.ndarray]
    offset: float


@dataclass(frozen=True)
class _Correlated:
    """The correlated priors of a problem, a multivariate normal of their scores.

    Attributes:
        indices: the positions of the parameters in the problem.
        log: whether a prior is log-normal, i.e. its score is of the logarithm.
        mu: the mean of the value or of its logarithm.
        sigma: the standard deviation of the value or of its logarithm.
        factor: the lower Cholesky factor `L` of the correlation.
        offsets: the offset of the residual `sqrt(c + 2 ln(x))` of every
            log-normal prior, in the order of the log-normal priors.
        lower: the lower bounds.
        upper: the upper bounds.
        constant: the logarithm of the normalization of the multivariate
            normal, `sum(ln(sigma)) + ln(det(L)) + k ln(2 pi) / 2`.
    """

    indices: tuple[int, ...]
    log: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    factor: np.ndarray
    offsets: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    constant: float

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Get the normal scores of the values, `NaN` for a log of a value which is not positive."""
        values = np.asarray(x, dtype=float)[list(self.indices)]
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(
                self.log, np.log(np.where(values > 0.0, values, np.nan)), values
            )
        return (values - self.mu) / self.sigma

    def whitened(self, x: np.ndarray) -> np.ndarray:
        """Get the whitened scores `L⁻¹ z`."""
        from scipy.linalg import solve_triangular

        z = self.scores(x)
        if not np.all(np.isfinite(z)):
            return np.full(z.size, np.nan)
        return np.asarray(solve_triangular(self.factor, z, lower=True), dtype=float)


def _correlated_term(
    parameters: Sequence[FitParameter], correlation: Covariance
) -> _Correlated:
    """Get the correlated priors of a correlation of the parameters of a problem.

    Args:
        parameters: the parameters of the problem.
        correlation: the correlation of the normal scores of their priors.

    Returns:
        The correlated priors in the order of the parameters of the problem.

    Raises:
        ValueError: for a correlation which is a covariance, a target which
            is no parameter, a parameter without a normal or log-normal prior
            or a log-normal prior of a parameter whose lower bound is not
            positive.
    """
    if correlation.kind != "correlation":
        raise ValueError(
            "The correlation of the priors is a correlation of their normal "
            "scores, Covariance.correlation(...): the variances are the ones of "
            "the priors of the parameters."
        )
    positions = {p.pid: j for j, p in enumerate(parameters)}
    unknown = [t for t in correlation.targets if t not in positions]
    if unknown:
        raise ValueError(
            f"The correlated priors {unknown} are no parameters of the fit, the "
            f"parameters are {list(positions)}."
        )
    order = sorted(
        range(len(correlation.targets)), key=lambda i: positions[correlation.targets[i]]
    )
    matrix = np.asarray(correlation.matrix, dtype=float)[np.ix_(order, order)]
    indices = tuple(positions[correlation.targets[i]] for i in order)
    log, mu, sigma, offsets, lower, upper = [], [], [], [], [], []
    for j in indices:
        parameter = parameters[j]
        prior = parameter.prior
        if prior is None or prior.distribution not in CORRELATED_PRIORS:
            raise ValueError(
                f"The prior of '{parameter.pid}' is "
                f"'{None if prior is None else prior.distribution.value}'; a "
                f"correlation of priors takes the priors "
                f"{[d.value for d in CORRELATED_PRIORS]}, whose normal scores "
                f"are the standardized values or logarithms."
            )
        is_log = prior.distribution is PriorDistribution.LOG_NORMAL
        if is_log:
            if not parameter.lower_bound > 0.0:
                raise ValueError(
                    f"The log-normal prior of '{parameter.pid}' is correlated, its "
                    f"lower bound must be positive, not {parameter.lower_bound}."
                )
            offsets.append(max(-2.0 * math.log(parameter.lower_bound), 0.0) + 1.0)
        log.append(is_log)
        mu.append(prior.parameters[0])
        sigma.append(prior.parameters[1])
        lower.append(parameter.lower_bound)
        upper.append(parameter.upper_bound)
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"The correlation of the priors of {[parameters[j].pid for j in indices]} "
            f"is singular; a correlation of 1 between two priors is one prior."
        ) from err
    sigma_array = np.asarray(sigma, dtype=float)
    constant = float(
        np.sum(np.log(sigma_array))
        + np.sum(np.log(np.diag(factor)))
        + 0.5 * len(indices) * math.log(2.0 * math.pi)
    )
    return _Correlated(
        indices=indices,
        log=np.asarray(log, dtype=bool),
        mu=np.asarray(mu, dtype=float),
        sigma=sigma_array,
        factor=factor,
        offsets=np.asarray(offsets, dtype=float),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        constant=constant,
    )


class JointPrior:
    """The prior of the parameters of a problem, see the module."""

    def __init__(
        self,
        parameters: Sequence[FitParameter],
        correlation: Covariance | None = None,
    ) -> None:
        """Create the prior of the parameters.

        Args:
            parameters: the parameters of the problem.
            correlation: the correlation of the normal scores of some of their
                priors, see the module.

        Raises:
            ValueError: for a correlation the priors do not take, see
                `_correlated_term`.
        """
        self.correlated: _Correlated | None = (
            None if correlation is None else _correlated_term(parameters, correlation)
        )
        in_correlation = (
            set() if self.correlated is None else set(self.correlated.indices)
        )
        self.independent: list[_Independent] = []
        for j, parameter in enumerate(parameters):
            if parameter.prior is None or j in in_correlation:
                continue
            log_density, offset = _independent_term(parameter)
            self.independent.append(_Independent(j, log_density, offset))

    def __len__(self) -> int:
        """Get the number of the residuals of the prior."""
        size = len(self.independent)
        if self.correlated is not None:
            size += len(self.correlated.indices) + len(self.correlated.offsets)
        return size

    def residuals(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Get the residuals of the prior, see the module.

        Args:
            x: values of the parameters on the linear scale.

        Returns:
            The residuals, `NaN` where the prior is not defined, e.g. outside
            the bounds of a truncated prior, and the offset, so that half the
            sum of squares minus the offset is the negative log density.
        """
        residuals: list[float] = []
        offset = 0.0
        for term in self.independent:
            log_p = float(term.log_density(np.asarray(x[term.index], dtype=float)))
            residuals.append(
                math.sqrt(max(term.offset - 2.0 * log_p, 0.0))
                if math.isfinite(log_p)
                else math.nan
            )
            offset += 0.5 * term.offset
        if self.correlated is not None:
            c = self.correlated
            residuals.extend(c.whitened(x).tolist())
            values = np.asarray(x, dtype=float)[list(c.indices)][c.log]
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = c.offsets + 2.0 * np.log(values)
            residuals.extend(
                np.where(np.isfinite(terms), np.sqrt(np.clip(terms, 0.0, None)), np.nan)
            )
            offset += 0.5 * float(np.sum(c.offsets)) - c.constant
        return np.asarray(residuals, dtype=float), offset

    def log_density(self, x: np.ndarray) -> float:
        """Get the log density of the prior.

        Args:
            x: values of the parameters on the linear scale.

        Returns:
            The log density, `-inf` outside the bounds of a prior; the
            correlated priors are normalized without their bounds, see the
            module.
        """
        total = 0.0
        for term in self.independent:
            total += float(term.log_density(np.asarray(x[term.index], dtype=float)))
        if self.correlated is not None:
            c = self.correlated
            values = np.asarray(x, dtype=float)[list(c.indices)]
            if np.any(values < c.lower) or np.any(values > c.upper):
                return -math.inf
            w = c.whitened(x)
            if not np.all(np.isfinite(w)):
                return -math.inf
            total += (
                -0.5 * float(np.sum(np.square(w)))
                - float(np.sum(np.log(values[c.log])))
                - c.constant
            )
        return total


def check_prior_correlation(
    parameters: Sequence[FitParameter], correlation: Covariance | None
) -> None:
    """Check that the priors of the parameters take a correlation.

    Args:
        parameters: the parameters of a problem.
        correlation: the correlation of their priors, nothing to check without.

    Raises:
        ValueError: see `_correlated_term`.
    """
    if correlation is not None:
        _correlated_term(parameters, correlation)


def _value(magnitude: float, unit: str | None) -> Any:
    """Get a number in a unit, a plain float without one."""
    from sbmlsim.units import ureg

    return ureg.Quantity(magnitude, unit) if unit else magnitude


def prior_distributions(
    problem: OptimizationProblem, targets: Mapping[str, str] | None = None
) -> dict[str, Distribution]:
    """Get the priors of the parameters of a problem as distributions of their targets.

    Args:
        problem: the problem, initialized, so the units of its parameters are
            the units of pint of its models, see
            `OptimizationProblem.parameter_units`.
        targets: pid -> the target of the model, overrides
            `FitParameter.target_id`.

    Returns:
        target -> its prior truncated at the bounds of its parameter, or the
        uniform distribution over the bounds for a parameter without a prior,
        in the unit of the parameter.

    Raises:
        ValueError: for a parameter without a prior whose bounds are not
            finite, a prior without a distribution of the sampler, or two
            parameters of one target (the versions of a parameter).
    """
    from sbmlsim.simulation.sampling.distributions import Truncated, Uniform
    from sbmlsim.simulation.sampling.fit import _targets

    units = problem.parameter_units
    names = _targets(problem.pids, [p.target_id for p in problem.parameters], targets)
    distributions: dict[str, Distribution] = {}
    for name, parameter in zip(names, problem.parameters, strict=True):
        unit = units[parameter.pid]
        lower, upper = parameter.lower_bound, parameter.upper_bound
        if parameter.prior is None:
            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError(
                    f"The parameter '{parameter.pid}' has no prior and the bounds "
                    f"[{lower}, {upper}]; its uniform prior needs finite bounds."
                )
            distributions[name] = Uniform(
                lower=_value(lower, unit), upper=_value(upper, unit)
            )
            continue
        distributions[name] = Truncated(
            parameter.prior.to_distribution(unit),
            lower=_value(lower, unit) if math.isfinite(lower) else None,
            upper=_value(upper, unit) if math.isfinite(upper) else None,
        )
    return distributions


def prior_model(
    problem: OptimizationProblem,
    targets: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> UncertaintyModel:
    """Get the priors of the parameters of a problem as the uncertainty of a propagation.

    Args:
        problem: the problem, initialized, see `prior_distributions`.
        targets: pid -> the target of the model, overrides
            `FitParameter.target_id`.
        kwargs: the other fields of the `UncertaintyModel`, e.g. `inputs`;
            a `covariance` is added to the correlation of the priors.

    Returns:
        The model with the priors of `prior_distributions` and the correlation
        of the priors of the problem, by the targets.
    """
    from sbmlsim.uncertainty.covariance import Covariance
    from sbmlsim.uncertainty.sources import UncertaintyModel

    distributions = prior_distributions(problem, targets)
    covariance = list(kwargs.pop("covariance", ()))
    correlation = problem.prior_correlation
    if correlation is not None:
        names = dict(zip(problem.pids, distributions, strict=True))
        covariance.insert(
            0,
            Covariance(
                [names[t] for t in correlation.targets],
                correlation.matrix,
                kind="correlation",
            ),
        )
    return UncertaintyModel(parameters=distributions, covariance=covariance, **kwargs)
