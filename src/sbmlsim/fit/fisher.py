"""Fisher information of a parameter fit.

The profile likelihood of `sbmlsim.fit.identifiability` scans a parameter and
re-optimizes the others, which is exact and costs a simulation per point. The
Fisher information is the local alternative: the curvature of the cost at the
optimum, from which the standard errors, the correlations of the parameters and
the directions the data does not constrain follow, at the cost of one jacobian.

For a fit which minimizes `cost = 0.5 * Σ r²` with the weighted residuals `r`,
the Gauss-Newton approximation of the Hessian is `J' J` with the jacobian
`J = ∂r/∂θ`, which is the Fisher information matrix of the estimate. Its
inverse, scaled by the variance of the residuals, is the covariance of the
parameters:

    FIM = J' J,   cov = σ² (J' J)⁻¹,   σ² = 2 cost / (n - k)

The two analyses answer different questions and disagree where the cost is not
a quadratic: the Fisher information is a local statement about the optimum, the
profile likelihood follows the cost until it rises by the threshold. A
parameter which the Fisher information calls determined and the profile
likelihood calls non-identifiable is a parameter whose cost is flat away from
the optimum, which is what the profile is for.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet

logger = logging.getLogger(__name__)

#: relative step of the finite differences of the jacobian, in the space the
#: optimizer searches. The square root of the machine epsilon is the step which
#: balances the truncation and the rounding error of a central difference
DEFAULT_STEP = 1e-6

#: eigenvalues below this fraction of the largest one are a direction the data
#: does not constrain, i.e. the matrix is numerically rank deficient
DEFAULT_RANK_TOLERANCE = 1e-8


@dataclass
class FisherInformation:
    """The Fisher information of a parameter set on a problem.

    Attributes:
        opid: id of the optimization problem.
        sid: id of the parameter set.
        pids: parameters, in the order of the matrix.
        values: values of the parameters, in the units of the model.
        scale: space the matrix is expressed in, i.e. the space the optimizer
            searches, see `ParameterScaleType`.
        matrix: the Fisher information `J' J`.
        cost: cost of the parameter set.
        n: number of data points of the fit.
        alpha: confidence level of the intervals.
    """

    opid: str
    sid: str
    pids: list[str]
    values: np.ndarray
    scale: ParameterScaleType
    matrix: np.ndarray
    cost: float
    n: int
    alpha: float = 0.95
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE
    units: list[str | None] = field(default_factory=list)

    #: the covariance is read by the errors, the correlations and the table, so
    #: the warning of a rank deficient information is logged for the first of
    #: them and not once per reader
    _warned: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def k(self) -> int:
        """Get the number of parameters."""
        return len(self.pids)

    @property
    def sigma2(self) -> float:
        """Get the variance of the residuals, `2 cost / (n - k)`.

        A fit with as many parameters as data points has no degrees of freedom
        left and no variance to estimate, which is `nan`.
        """
        dof = self.n - self.k
        if dof <= 0:
            return float("nan")
        return float(2.0 * self.cost / dof)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Get the eigenvalues of the Fisher information, largest first.

        The matrix is symmetric and positive semi-definite, so the eigenvalues
        are real and not negative up to rounding. A small eigenvalue is a
        direction in parameter space the data does not constrain.
        """
        values = np.linalg.eigvalsh(self.matrix)
        return np.asarray(sorted(values, reverse=True), dtype=float)

    @property
    def condition_number(self) -> float:
        """Get the ratio of the largest to the smallest eigenvalue.

        A large condition number is a problem which is sloppy: the data
        determines some combinations of the parameters much better than others.
        """
        values = self.eigenvalues
        if values[-1] <= 0.0:
            return float("inf")
        return float(values[0] / values[-1])

    @property
    def rank(self) -> int:
        """Get the number of directions the data constrains."""
        values = self.eigenvalues
        if values[0] <= 0.0:
            return 0
        return int(np.sum(values > values[0] * self.rank_tolerance))

    @property
    def is_identifiable(self) -> bool:
        """Check whether the data constrains every direction locally."""
        return self.rank == self.k

    @property
    def covariance(self) -> np.ndarray:
        """Get the covariance of the parameters in the scaled space.

        `cov = σ² (J' J)⁻¹`, with the pseudo-inverse for a matrix which is rank
        deficient, i.e. for a problem which is locally not identifiable. The
        covariance of such a problem is not a covariance, its entries of the
        unconstrained directions are arbitrary; `is_identifiable` says whether
        it can be read. The warning about it is logged once per information.
        """
        if not self.is_identifiable and not self._warned:
            self._warned = True
            logger.warning(
                "'%s': the Fisher information has rank %s of %s, the covariance "
                "of the unconstrained directions is not meaningful.",
                self.opid,
                self.rank,
                self.k,
            )
        return np.asarray(self.sigma2 * np.linalg.pinv(self.matrix), dtype=float)

    @property
    def standard_errors(self) -> np.ndarray:
        """Get the standard error of every parameter in the scaled space."""
        variances = np.diag(self.covariance)
        return np.sqrt(np.where(variances > 0.0, variances, np.nan))

    @property
    def correlation(self) -> pd.DataFrame:
        """Get the correlation of the parameters.

        Two parameters which are correlated to `±1` are the pair the data only
        determines together, i.e. the fit can trade one against the other.
        """
        cov = self.covariance
        sd = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(sd, sd)
        return pd.DataFrame(corr, index=self.pids, columns=self.pids)

    def confidence_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the confidence interval of every parameter.

        The interval is `θ ± t · SE` in the space the optimizer searches, with
        the quantile of the t distribution of `n - k` degrees of freedom, and
        is transformed back into the units of the model. On a logarithmic scale
        the interval is therefore not symmetric around the value.

        Returns:
            The lower and the upper bound in the units of the model.
        """
        dof = self.n - self.k
        if dof <= 0:
            nan = np.full(self.k, np.nan)
            return nan, nan
        quantile = float(student_t.ppf(0.5 + self.alpha / 2.0, dof))
        scaled = self.scale.to_scale(self.values)
        delta = quantile * self.standard_errors
        return (
            np.asarray(self.scale.from_scale(scaled - delta), dtype=float),
            np.asarray(self.scale.from_scale(scaled + delta), dtype=float),
        )

    @property
    def summary_df(self) -> pd.DataFrame:
        """Get the parameters with their errors and intervals as a table."""
        lower, upper = self.confidence_intervals()
        errors = self.standard_errors
        with np.errstate(divide="ignore", invalid="ignore"):
            # the error relative to the value, in the scaled space
            cv = np.abs(errors / self.scale.to_scale(self.values)) * 100.0
        units = self.units or [None] * self.k
        return pd.DataFrame(
            {
                "parameter": self.pids,
                "value": self.values,
                "se": errors,
                "cv": cv,
                "ci_lower": lower,
                "ci_upper": upper,
                "unit": units,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "opid": self.opid,
            "sid": self.sid,
            "pids": list(self.pids),
            "values": [float(v) for v in self.values],
            "scale": self.scale.name,
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "cost": self.cost,
            "n": self.n,
            "alpha": self.alpha,
            "units": list(self.units),
        }


def jacobian(
    problem: OptimizationProblem, x: np.ndarray, step: float = DEFAULT_STEP
) -> np.ndarray:
    """Get the jacobian of the weighted residuals by central differences.

    Args:
        problem: initialized optimization problem.
        x: parameters in the space the optimizer searches.
        step: relative step of the differences.

    Returns:
        The jacobian with one row per residual and one column per parameter.
    """
    columns: list[np.ndarray] = []
    for k in range(len(x)):
        h = step * max(abs(float(x[k])), 1.0)
        x_plus, x_minus = np.array(x, dtype=float), np.array(x, dtype=float)
        x_plus[k] += h
        x_minus[k] -= h
        r_plus = np.asarray(problem.residuals(x_plus), dtype=float)
        r_minus = np.asarray(problem.residuals(x_minus), dtype=float)
        columns.append((r_plus - r_minus) / (2.0 * h))
    return np.column_stack(columns)


def fisher_information(
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    alpha: float = 0.95,
    step: float = DEFAULT_STEP,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> FisherInformation:
    """Calculate the Fisher information of a parameter set on a problem.

    Args:
        problem: optimization problem, initialized with the settings.
        settings: settings the parameters were fitted with.
        parameter_set: parameters to evaluate the information at, i.e. the
            result of a fit.
        alpha: confidence level of the intervals.
        step: relative step of the finite differences of the jacobian.
        rank_tolerance: eigenvalues below this fraction of the largest one are
            a direction the data does not constrain.

    Returns:
        The Fisher information with the errors, the correlations and the
        directions the data does not constrain.
    """
    problem.initialize(settings)
    values = np.asarray(parameter_set.x(problem.pids), dtype=float)
    x = problem.to_scale(values)

    residuals = np.asarray(problem.residuals(x), dtype=float)
    jac = jacobian(problem, x, step=step)
    matrix = jac.T @ jac

    return FisherInformation(
        opid=problem.opid,
        sid=parameter_set.sid,
        pids=list(problem.pids),
        values=values,
        scale=problem.parameter_scale,
        matrix=matrix,
        cost=float(0.5 * np.sum(np.square(residuals))),
        n=int(residuals.size),
        alpha=alpha,
        rank_tolerance=rank_tolerance,
        units=[p.unit for p in problem.parameters],
    )
