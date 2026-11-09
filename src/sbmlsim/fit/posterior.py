"""The posterior of the parameters of a fit: the Laplace approximation.

The Laplace approximation of the posterior is the multivariate normal around
the estimate of a fit whose precision is the curvature of the negative log
posterior at the estimate, in the space the optimizer searches, the
`parameter_scale` of the fit (Tierney & Kadane 1986; Gelman et al. 2013, ch.
4.1):

    precision = J' J / σ² + H_prior,   covariance = precision⁻¹

with the jacobian `J` of the residuals of the data at the estimate, the
Gauss-Newton approximation of the curvature of the negative log-likelihood
(exact for normal noise of a known scale and a model linear in the
parameters; an approximation for a scale of the noise which is estimated and
for Laplace noise), see `sbmlsim.fit.fisher`, and the Hessian of the negative
log density of the priors `H_prior`, by central differences in the space of
the optimizer. What the curvature contains is what the fit minimized, its
`ObjectiveType`:

- `posterior`: the likelihood and the priors, around the maximum a
  posteriori estimate;
- `likelihood`: the likelihood with the uniform priors of the bounds, which
  have no curvature; priors of the parameters are no part of the objective
  and are left out;
- `least_squares`: the weighted least squares as the likelihood of normal
  noise of the scale `σ² = 2 cost / (n - k)`, the posterior of the uniform
  priors, whose covariance is the covariance of the Fisher information.

On a logarithmic scale of the optimizer the normal of the logarithms is a
log-normal of the values. The priors are the densities of the values, as the
fit evaluated them, without the jacobian of the scale.

A direction which neither the data nor the priors constrain has no curvature,
the posterior is the uniform prior of the bounds there, which a normal does
not describe: `posterior_laplace` raises and names the parameters of such a
direction. It warns where the approximation is questionable: a parameter at
a bound, where the posterior is cut off, and a posterior whose interval
reaches past a bound. These are the problems of a sampler of the posterior
(#374).

A `ParameterPosterior` is stored as JSON (`posterior.json` next to the records
of a fit, see `sbmlsim.run`), so an analysis of the posterior does not run the
fit, and is the source of the parameters of a propagation
(`ParameterPosterior.uncertainty_model`), of the designs of
`sbmlsim.simulation.sampling` (`posterior_parameters`, or `distributions` and
`rank_correlation` for `lhs` and `random`) and of a sensitivity analysis.

Tierney L, Kadane JB (1986). Accurate approximations for posterior moments
and marginal densities. J Am Stat Assoc 81:82-86. Gelman A, Carlin JB, Stern
HS, Dunson DB, Vehtari A, Rubin DB (2013). Bayesian Data Analysis, 3rd ed.
CRC Press.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from sbmlsim.atomic import write_text_atomic
from sbmlsim.fit.fisher import (
    DEFAULT_RANK_TOLERANCE,
    FisherInformation,
    fisher_information,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ObjectiveType, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet

if TYPE_CHECKING:
    from sbmlsim.simulation.sampling.distributions import Distribution
    from sbmlsim.uncertainty.covariance import Covariance
    from sbmlsim.uncertainty.sources import UncertaintyModel

logger = logging.getLogger(__name__)

#: the file of a posterior next to the records of a fit
POSTERIOR_FILE = "posterior.json"

#: the step of the central differences of the log density of the priors,
#: relative to the parameters in the space of the optimizer; the priors are
#: no simulation and have no noise
PRIOR_STEP = 1e-4

#: a parameter within this fraction of the width of its bounds, in the space
#: of the optimizer, from a bound is at the bound
BOUND_TOLERANCE = 1e-3


def _log_sd(scale: ParameterScaleType, sd: float) -> float:
    """Get the standard deviation of the natural logarithm of an SD on a log scale."""
    return sd * math.log(10.0) if scale is ParameterScaleType.LOG10 else sd


@dataclass
class ParameterPosterior:
    """The posterior of the parameters of a fit, a normal in the space of its optimizer.

    Attributes:
        opid: id of the optimization problem.
        sid: id of the parameter set it is centered on.
        pids: the parameters, in the order of the matrix.
        values: the values of the parameters at the center, in the units of
            the model.
        scales: the space of every parameter, in which the posterior is
            normal; a logarithmic scale is a log-normal of the values.
        covariance: the covariance in the space of the parameters.
        units: the unit of every parameter, a unit of pint with the unit ids
            of the model resolved, `None` for the unit of the model.
        targets: the target of the model of every parameter
            (`FitParameter.target_id`).
        lower_bounds: the lower bounds of the fit.
        upper_bounds: the upper bounds of the fit.
        objective: what the fit minimized, see the module.
        method: how the posterior was computed, `laplace`.
        n: the number of data points.
        sigma2: the variance of the residuals the curvature of the data was
            divided by, 1 for a likelihood.
        step: the relative step of the jacobian of the data.
        problems: where the approximation is questionable, see the module.
    """

    opid: str
    sid: str
    pids: list[str]
    values: np.ndarray
    scales: list[ParameterScaleType]
    covariance: np.ndarray
    units: list[str | None]
    targets: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    objective: ObjectiveType
    method: str = "laplace"
    n: int = 0
    sigma2: float = 1.0
    step: float = 0.0
    problems: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Coerce the arrays and check their sizes.

        Raises:
            ValueError: if a field has not one entry per parameter or the
                covariance is not a symmetric matrix of them.
        """
        self.values = np.asarray(self.values, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.lower_bounds = np.asarray(self.lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(self.upper_bounds, dtype=float)
        self.objective = ObjectiveType(self.objective)
        k = len(self.pids)
        sizes = {
            "values": self.values.size,
            "scales": len(self.scales),
            "units": len(self.units),
            "targets": len(self.targets),
            "lower_bounds": self.lower_bounds.size,
            "upper_bounds": self.upper_bounds.size,
        }
        wrong = {name: size for name, size in sizes.items() if size != k}
        if wrong:
            raise ValueError(
                f"'{self.opid}': the posterior of the {k} parameters {self.pids} "
                f"has the sizes {wrong}."
            )
        if self.covariance.shape != (k, k) or not np.allclose(
            self.covariance, self.covariance.T, rtol=1e-10, atol=0.0
        ):
            raise ValueError(
                f"'{self.opid}': the covariance of the posterior must be a "
                f"symmetric {k}x{k} matrix, not of the shape "
                f"{self.covariance.shape}."
            )

    @property
    def k(self) -> int:
        """Get the number of parameters."""
        return len(self.pids)

    @property
    def mean(self) -> np.ndarray:
        """Get the mean in the space of the parameters, the values on their scales."""
        return np.array(
            [s.to_scale(v) for s, v in zip(self.scales, self.values, strict=True)],
            dtype=float,
        )

    @property
    def standard_deviations(self) -> np.ndarray:
        """Get the standard deviation of every parameter in its space."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Get the correlation of the parameters, of their normal scores."""
        sd = self.standard_deviations
        return self.covariance / np.outer(sd, sd)

    @property
    def correlation(self) -> pd.DataFrame:
        """Get the correlation of the parameters as a table."""
        return pd.DataFrame(self.correlation_matrix, index=self.pids, columns=self.pids)

    def intervals(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Get the central credible interval of every parameter.

        Args:
            level: the probability of the interval.

        Returns:
            The lower and the upper ends in the units of the model, the
            quantiles of the normal in the space of every parameter.
        """
        z = float(norm.ppf(0.5 + level / 2.0))
        mean, delta = self.mean, z * self.standard_deviations
        with np.errstate(over="ignore"):
            lower = [
                s.from_scale(m - d)
                for s, m, d in zip(self.scales, mean, delta, strict=True)
            ]
            upper = [
                s.from_scale(m + d)
                for s, m, d in zip(self.scales, mean, delta, strict=True)
            ]
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    @property
    def summary_df(self) -> pd.DataFrame:
        """Get the parameters with their standard deviations and 95 % intervals."""
        lower, upper = self.intervals(0.95)
        return pd.DataFrame(
            {
                "parameter": self.pids,
                "target": self.targets,
                "value": self.values,
                "scale": [s.name for s in self.scales],
                "sd": self.standard_deviations,
                "q0.025": lower,
                "q0.975": upper,
                "unit": self.units,
            }
        )

    def _names(self, targets: Mapping[str, str] | None) -> list[str]:
        """Get the target of every parameter, see `posterior_parameters`."""
        from sbmlsim.simulation.sampling.fit import _targets

        return _targets(self.pids, self.targets, targets)

    def distributions(
        self, targets: Mapping[str, str] | None = None
    ) -> dict[str, Distribution]:
        """Get the marginals of the posterior as distributions of their targets.

        Args:
            targets: pid -> the target of the model, overrides `targets`.

        Returns:
            target -> its marginal: a normal of the value on the linear scale,
            a log-normal of the value on a logarithmic one, in the unit of the
            parameter; with `correlation` the joint posterior (a Gaussian
            copula of these marginals is the normal of the space of the
            parameters).

        Raises:
            ValueError: for two parameters of one target.
        """
        from sbmlsim.simulation.sampling.distributions import LogNormal, Normal
        from sbmlsim.units import ureg

        distributions: dict[str, Distribution] = {}
        for name, scale, value, sd, unit in zip(
            self._names(targets),
            self.scales,
            self.values,
            self.standard_deviations,
            self.units,
            strict=True,
        ):

            def quantity(x: float, unit: str | None = unit) -> Any:
                return ureg.Quantity(x, unit) if unit else x

            if scale is ParameterScaleType.LINEAR:
                distributions[name] = Normal(mean=quantity(value), sd=quantity(sd))
            else:
                sigma = _log_sd(scale, float(sd))
                distributions[name] = LogNormal(
                    median=quantity(value), cv=math.sqrt(math.expm1(sigma**2))
                )
        return distributions

    def normal_correlation(
        self, targets: Mapping[str, str] | None = None
    ) -> Covariance:
        """Get the correlation of the normal scores of the posterior, by the targets.

        Args:
            targets: pid -> the target of the model, overrides `targets`.

        Returns:
            The correlation, which with the marginals of `distributions` is
            the posterior, see `sbmlsim.uncertainty.Covariance`.
        """
        from sbmlsim.uncertainty.covariance import Covariance

        matrix = self.correlation_matrix
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 1.0)
        return Covariance(self._names(targets), matrix, kind="correlation")

    def rank_correlation(self) -> np.ndarray:
        """Get the Spearman rank correlation of the parameters.

        The rank correlation of the normal of the space of the parameters,
        `6 / π asin(r / 2)` (Pearson 1907), which the designs `lhs` and
        `random` of `sbmlsim.simulation.sampling` take as `correlation` and
        convert back into this correlation of the normal scores.

        Returns:
            The matrix in the order of the parameters.
        """
        rank = (
            6.0 / np.pi * np.arcsin(np.clip(self.correlation_matrix, -1.0, 1.0) / 2.0)
        )
        np.fill_diagonal(rank, 1.0)
        return (rank + rank.T) / 2.0

    def uncertainty_model(
        self, targets: Mapping[str, str] | None = None, **kwargs: Any
    ) -> UncertaintyModel:
        """Get the posterior as the uncertainty of the parameters of a propagation.

        Args:
            targets: pid -> the target of the model, overrides `targets`.
            kwargs: the other fields of the `UncertaintyModel`, e.g. `inputs`;
                a `covariance` is added to the correlation of the posterior.

        Returns:
            The model with the marginals of `distributions` and the correlation
            of `normal_correlation`, so the draws are the normal of the space
            of the parameters.
        """
        from sbmlsim.uncertainty.sources import UncertaintyModel

        covariance = [self.normal_correlation(targets), *kwargs.pop("covariance", ())]
        return UncertaintyModel(
            parameters=self.distributions(targets), covariance=covariance, **kwargs
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "opid": self.opid,
            "sid": self.sid,
            "method": self.method,
            "objective": self.objective.value,
            "pids": list(self.pids),
            "targets": list(self.targets),
            "values": [float(v) for v in self.values],
            "units": list(self.units),
            "scales": [s.name for s in self.scales],
            "covariance": [[float(v) for v in row] for row in self.covariance],
            "lower_bounds": [float(v) for v in self.lower_bounds],
            "upper_bounds": [float(v) for v in self.upper_bounds],
            "n": self.n,
            "sigma2": self.sigma2,
            "step": self.step,
            "problems": list(self.problems),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ParameterPosterior:
        """Create from a dictionary, the inverse of `to_dict`."""
        return cls(
            opid=d["opid"],
            sid=d["sid"],
            method=d.get("method", "laplace"),
            objective=ObjectiveType(d["objective"]),
            pids=list(d["pids"]),
            targets=list(d["targets"]),
            values=np.asarray(d["values"], dtype=float),
            units=list(d["units"]),
            scales=[ParameterScaleType[name] for name in d["scales"]],
            covariance=np.asarray(d["covariance"], dtype=float),
            lower_bounds=np.asarray(d["lower_bounds"], dtype=float),
            upper_bounds=np.asarray(d["upper_bounds"], dtype=float),
            n=int(d.get("n", 0)),
            sigma2=float(d.get("sigma2", 1.0)),
            step=float(d.get("step", 0.0)),
            problems=list(d.get("problems", [])),
        )

    def write(self, path: Path) -> Path:
        """Write the posterior as JSON, atomically.

        Args:
            path: the file, or a directory for its `POSTERIOR_FILE`.

        Returns:
            The file.
        """
        path = Path(path)
        if path.is_dir():
            path = path / POSTERIOR_FILE
        write_text_atomic(path, json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def read(cls, path: Path) -> ParameterPosterior:
        """Read a posterior written by `write`.

        Args:
            path: the file, or a directory with its `POSTERIOR_FILE`, e.g. the
                record of a fit.

        Returns:
            The posterior.
        """
        path = Path(path)
        if path.is_dir():
            path = path / POSTERIOR_FILE
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def prior_hessian(
    log_density: Callable[[np.ndarray], float],
    z: np.ndarray,
    to_linear: Callable[[np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    step: float = PRIOR_STEP,
) -> np.ndarray:
    """Get the Hessian of the negative log density of a prior in the space of the optimizer.

    Central differences of `-log p(x(z))` with the step `step * max(|z|, 1)`;
    a stencil which would leave the bounds is moved inside them, so the
    Hessian of a parameter at its bound is the one next to it.

    Args:
        log_density: the log density of the prior, of the values on the
            linear scale.
        z: the parameters in the space of the optimizer.
        to_linear: the transformation of the space of the optimizer to the
            linear scale.
        lower: the lower bounds in the space of the optimizer.
        upper: the upper bounds in the space of the optimizer.
        step: the relative step.

    Returns:
        The symmetric Hessian.

    Raises:
        ValueError: if the log density is not finite at the stencil.
    """
    k = z.size
    h = step * np.maximum(np.abs(z), 1.0)
    center = np.array(z, dtype=float)
    for i in range(k):
        center[i] = min(max(center[i], lower[i] + h[i]), upper[i] - h[i])

    def f(point: np.ndarray) -> float:
        value = -float(log_density(to_linear(point)))
        if not math.isfinite(value):
            raise ValueError(
                f"The prior is not defined at {to_linear(point).tolist()}, its "
                f"Hessian cannot be computed."
            )
        return value

    hessian = np.zeros((k, k))
    f0 = f(center)
    for i in range(k):
        e_i = np.zeros(k)
        e_i[i] = h[i]
        hessian[i, i] = (f(center + e_i) - 2.0 * f0 + f(center - e_i)) / h[i] ** 2
        for j in range(i):
            e_j = np.zeros(k)
            e_j[j] = h[j]
            value = (
                f(center + e_i + e_j)
                - f(center + e_i - e_j)
                - f(center - e_i + e_j)
                + f(center - e_i - e_j)
            ) / (4.0 * h[i] * h[j])
            hessian[i, j] = hessian[j, i] = value
    return hessian


def posterior_laplace(
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    *,
    fisher: FisherInformation | None = None,
    step: float | None = None,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> ParameterPosterior:
    """Get the Laplace approximation of the posterior of the parameters of a fit.

    Args:
        problem: optimization problem, initialized with the settings.
        settings: settings the parameters were fitted with, whose objective
            decides what the posterior contains, see the module.
        parameter_set: the estimate of the fit, the center of the posterior.
        fisher: the Fisher information of the estimate on the problem, which
            is computed without; its curvature of the data is reused.
        step: relative step of the jacobian of the data, see
            `fisher_information`.
        rank_tolerance: a precision whose smallest eigenvalue is below this
            fraction of its largest one constrains a direction too little for a
            normal, see the module.

    Returns:
        The posterior, a normal in the space of the optimizer.

    Raises:
        ValueError: if the Fisher information is not the one of the parameter
            set, a fit of least squares has no degrees of freedom, or the
            data and the priors leave a direction unconstrained.
    """
    problem.initialize(settings)
    values = np.asarray(
        problem.bind_parameter_set(parameter_set).x(problem.pids), dtype=float
    )
    if fisher is None:
        fisher = fisher_information(problem, settings, parameter_set, step=step)
    elif (
        fisher.pids != list(problem.pids)
        or fisher.objective is not problem.objective
        or not np.allclose(fisher.values, values, rtol=1e-12, atol=0.0)
    ):
        raise ValueError(
            f"'{problem.opid}': the Fisher information of '{fisher.sid}' (objective "
            f"'{fisher.objective.value}') is not the one of the parameter set "
            f"'{parameter_set.sid}' (objective '{problem.objective.value}')."
        )
    objective = problem.objective
    sigma2 = fisher.sigma2
    if not math.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError(
            f"'{problem.opid}': the variance of the residuals of the fit of least "
            f"squares is {sigma2} (n={fisher.n} data points, k={fisher.k} "
            f"parameters); the posterior is not defined."
        )
    precision = np.asarray(fisher.matrix, dtype=float) / sigma2
    with_priors = [p.pid for p in problem.parameters if p.prior is not None]
    scales = list(problem.scales_initialized)
    z = problem.to_scale(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = problem.to_scale(problem.bounds[0])
        upper = problem.to_scale(problem.bounds[1])
    if objective is ObjectiveType.POSTERIOR:
        if with_priors:
            precision = precision + prior_hessian(
                problem.log_prior_density, z, problem.from_scale, lower, upper
            )
    elif with_priors:
        logger.info(
            "'%s': the priors of %s are no part of the objective '%s', the "
            "posterior is the one of the uniform priors of the bounds; fit with "
            "the objective 'posterior' for a posterior with the priors.",
            problem.opid,
            with_priors,
            objective.value,
        )
    precision = (precision + precision.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    largest = float(eigenvalues.max())
    weak = (
        eigenvalues <= largest * rank_tolerance
        if largest > 0.0
        else np.ones(eigenvalues.size, dtype=bool)
    )
    if weak.any():
        directions = [
            [
                problem.pids[i]
                for i in np.argsort(-np.abs(eigenvectors[:, d]))
                if abs(eigenvectors[i, d]) > 0.1
            ]
            for d in np.flatnonzero(weak)
        ]
        raise ValueError(
            f"'{problem.opid}': the data and the priors do not constrain "
            f"{len(directions)} direction(s) of the parameters, of {directions}; "
            f"the posterior is the uniform prior of the bounds there, which the "
            f"Laplace approximation cannot describe. Give these parameters "
            f"priors and fit the objective 'posterior', or see the profile "
            f"likelihood."
        )
    covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
    covariance = (covariance + covariance.T) / 2.0

    problems: list[str] = []
    sd = np.sqrt(np.diag(covariance))
    width = upper - lower
    for i, pid in enumerate(problem.pids):
        if not np.isfinite(width[i]):
            continue
        if min(z[i] - lower[i], upper[i] - z[i]) <= BOUND_TOLERANCE * width[i]:
            problems.append(
                f"'{pid}' is at a bound, the posterior is cut off there and is no "
                f"normal"
            )
        elif z[i] - 1.96 * sd[i] < lower[i] or z[i] + 1.96 * sd[i] > upper[i]:
            problems.append(
                f"the 95 % interval of '{pid}' reaches past a bound, which cuts off "
                f"the posterior"
            )
    for problem_text in problems:
        logger.warning("'%s': %s.", problem.opid, problem_text)

    return ParameterPosterior(
        opid=problem.opid,
        sid=parameter_set.sid,
        pids=list(problem.pids),
        values=values,
        scales=scales,
        covariance=covariance,
        units=list(problem.parameter_units.values()),
        targets=[p.target_id for p in problem.parameters],
        lower_bounds=np.asarray(problem.bounds[0], dtype=float),
        upper_bounds=np.asarray(problem.bounds[1], dtype=float),
        objective=objective,
        n=fisher.n,
        sigma2=float(sigma2),
        step=fisher.step,
        problems=problems,
    )
