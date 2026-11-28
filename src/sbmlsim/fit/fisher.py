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

A fit of the likelihood (`ObjectiveType.LIKELIHOOD` or `POSTERIOR`) has the
scale of the noise in its residuals, so its covariance is `(J' J)⁻¹` without
`σ²`, and its intervals are the ones of the normal distribution. The
information is the one of the data: the residuals of the priors of a fit of
the posterior are no part of it, the Laplace approximation of the posterior,
`sbmlsim.fit.posterior`, adds the curvature of the priors.

The jacobian is the forward sensitivities of sbmlsim-core where the problem
is one the engine fits (`sbmlsim.fit.fit_core`, #556) and its conditions carry
them, and the central differences of the residuals in python otherwise, see
`_jacobian_of`; `FisherInformation.jacobian_of` says which of the two a matrix
came from.

The two analyses answer different questions and disagree where the cost is not
a quadratic: the Fisher information is a local statement about the optimum, the
profile likelihood follows the cost until it rises by the threshold. A
parameter which the Fisher information calls determined and the profile
likelihood calls non-identifiable is a parameter whose cost is flat away from
the optimum, which is what the profile is for.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from sbmlsim.engine import Engine, import_core
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ObjectiveType, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.progress import activity as progress_activity

logger = logging.getLogger(__name__)

#: what evaluates the points of the central differences of a jacobian, in the
#: order of the points, see `evaluator`
Evaluate = Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]

#: relative step of the finite differences of a record which does not store
#: its step, the step of the information before it followed the tolerance of
#: the integrator, see `default_step`
DEFAULT_STEP = 1e-6

#: where the jacobian of an information comes from: the central differences of
#: the residuals in python, which is what every engine can do, or the forward
#: sensitivities of sbmlsim-core, which are the derivatives themselves
#: (`sbmlsim.fit.fit_core.SENSITIVITIES`, #556)
CENTRAL_DIFFERENCES = "central_differences"
SENSITIVITIES = "sensitivities"


def default_step(relative_tolerance: float) -> float:
    """Get the relative step of the central differences of the jacobian.

    A simulation is exact only to the relative tolerance of its integrator, so
    the residuals have a noise of about that size. The error of a central
    difference is the noise divided by the step plus the truncation error,
    which grows with the square of the step; the cube root of the noise
    balances them. A step at the machine epsilon divides the noise of the
    integrator by a tiny step and the jacobian is noise.

    Args:
        relative_tolerance: relative tolerance of the integrator of the fit.

    Returns:
        The step, relative to the parameters in the space the optimizer
        searches, `1e-2` for the default tolerance `1e-6`.
    """
    return float(np.cbrt(relative_tolerance))


#: eigenvalues below this fraction of the largest one are a direction the data
#: does not constrain, i.e. the matrix is numerically rank deficient,
#: `DEFAULT_RANK_TOLERANCE` of `sbmlsim_core.stats`
DEFAULT_RANK_TOLERANCE = 1e-8


class FisherStatistics(TypedDict):
    """The statistics of a Fisher information, see `FisherInformation.statistics`.

    Attributes:
        sigma2: the variance of the residuals, `2 cost / (n - k)`, `nan`
            without degrees of freedom, 1 for a likelihood.
        spectrum: the eigenvalues, largest first.
        rank: the number of eigenvalues above the rank tolerance times the
            largest one.
        condition: the ratio of the largest to the smallest eigenvalue, `inf`
            of a smallest one which is not positive.
        covariance: `sigma2` times the pseudo-inverse of the information.
        standard_errors: the standard error of every parameter, `nan` of a
            variance which is not positive (zero allowed of an identifiable
            information).
        correlation: the correlation of the parameters.
    """

    sigma2: float
    spectrum: np.ndarray
    rank: int
    condition: float
    covariance: np.ndarray
    standard_errors: np.ndarray
    correlation: np.ndarray


def decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the eigenvalues and the pseudo-inverse of a Fisher information.

    One eigendecomposition of `sbmlsim_core.stats` (the cyclic Jacobi method)
    gives the eigenvalues and the pseudo-inverse of numpy (the eigenvalues cut
    at `1e-15` of the largest one), with the same bits on every machine. They
    are the decomposition a record stores, and every statistic of
    `FisherInformation` follows from them, the ones of a record which stores
    them as well, see `FisherInformation.statistics`. The crate gives them
    only with its statistics (`fisher_statistics`, of a likelihood, whose
    covariance is the pseudo-inverse); the others are of the order of the
    matrix and are dropped, the eigendecomposition is the cost.

    Args:
        matrix: the Fisher information `J' J`, square, finite and symmetric.

    Returns:
        The eigenvalues, largest first, and the pseudo-inverse.

    Raises:
        ValueError: for a matrix which is not a square, finite and symmetric
            matrix of at least one parameter.
    """
    import_core()
    from sbmlsim_core import stats

    # sigma2 of a likelihood is 1 whatever the cost and the number of points
    d = stats.fisher_statistics(matrix, 0.0, 0, likelihood=True)
    return (
        np.asarray(d["spectrum"], dtype=float),
        np.asarray(d["covariance"], dtype=float),
    )


@dataclass
class FisherInformation:
    """The Fisher information of a parameter set on a problem.

    Its statistics, see `statistics`, follow from the decomposition of the
    matrix, its eigenvalues and its pseudo-inverse, which a record stores and
    sbmlsim-core computes for a matrix without one (`decomposition`).

    Attributes:
        opid: id of the optimization problem.
        sid: id of the parameter set.
        pids: parameters, in the order of the matrix.
        values: values of the parameters, in the units of the model.
        scale: space the matrix is expressed in, i.e. the space the optimizer
            searches, see `ParameterScaleType`. It is the scale of the
            settings of the fit, i.e. of the parameters without a scale of
            their own.
        scales: the space of every parameter, in the order of the matrix.
            Without them every parameter has the scale `scale`.
        matrix: the Fisher information `J' J` of the residuals of the data.
        cost: cost of the data at the parameter set, the weighted least
            squares or the negative log-likelihood, without the priors.
        n: number of data points of the fit.
        alpha: confidence level of the intervals.
        rank_tolerance: eigenvalues below this fraction of the largest one are
            a direction the data does not constrain.
        step: relative step of the finite differences of the jacobian the
            matrix was computed with, see `jacobian`; `0` for a jacobian which
            took no difference, i.e. the sensitivities of the engine.
        jacobian_of: where the jacobian of the matrix comes from,
            `CENTRAL_DIFFERENCES` or `SENSITIVITIES`.
        units: the unit of every parameter in the order of the matrix, a unit
            of pint with the unit ids of the model resolved, `None` for the
            unit of the model; empty for an information which does not know
            them.
        targets: the target of the model every parameter is written to, in the
            order of the matrix (`FitParameter.target_id`); empty for an
            information which does not know them, then the pids are the targets.
        objective: what the fit minimized; a fit of the likelihood has the
            noise in its residuals, see `scaled`.
        inverse: the pseudo-inverse of `matrix`, `None` to compute it. It is
            stored with the information and read back, so a record gives the
            standard errors it was written with: a record of a release before
            sbmlsim-core computed them took the SVD of LAPACK, which depends
            on the kernels of the CPU, so the same matrix gave standard errors
            which differ in their last digits on another machine and a model
            whose factory writes them into its SBML did not rebuild byte for
            byte (#523). sbmlsim-core computes them with the same bits on
            every machine, see `decomposition`. A record written before
            they were stored has none and computes them, and a matrix which is
            changed after the information was created drops them.
        spectrum: the eigenvalues of `matrix`, largest first, `None` to
            compute them; stored with the information for the same reason, so
            the rank of a record is the rank on every machine.
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
    step: float = DEFAULT_STEP
    jacobian_of: str = CENTRAL_DIFFERENCES
    units: list[str | None] = field(default_factory=list)
    scales: list[ParameterScaleType] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    objective: ObjectiveType = ObjectiveType.LEAST_SQUARES
    inverse: np.ndarray | None = None
    spectrum: np.ndarray | None = None

    #: the covariance is read by the errors, the correlations and the table, so
    #: the warning of a rank deficient information is logged for the first of
    #: them and not once per reader
    _warned: bool = field(default=False, init=False, repr=False, compare=False)
    #: the rank tolerance the warning was decided at, a change asks again
    _warned_tolerance: float | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _matrix_fingerprint: tuple[tuple[int, ...], str, bytes] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cached_decomposition: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _fingerprint(self) -> tuple[tuple[int, ...], str, bytes]:
        """Get the fingerprint of the matrix, which detects a change of it."""
        # Hashing avoids retaining a second full matrix just to detect public
        # in-place edits. Non-contiguous inputs need only a temporary copy.
        return (
            self.matrix.shape,
            self.matrix.dtype.str,
            hashlib.sha256(memoryview(np.ascontiguousarray(self.matrix))).digest(),
        )

    def _refresh_matrix_cache(self) -> None:
        """Reuse the decomposition while retaining the mutable matrix API.

        The decomposition is the one of the matrix, whatever the rank
        tolerance, so only a change of the matrix computes it again or drops
        the stored one; a change of the tolerance asks again for the warning
        of a rank deficient information.
        """
        fingerprint = self._fingerprint()
        if fingerprint != self._matrix_fingerprint:
            # the stored decomposition belongs to the matrix the information
            # was created with, so a matrix which is changed afterwards drops it
            self.inverse = None
            self.spectrum = None
            self._matrix_fingerprint = fingerprint
            self._cached_decomposition = None
            self._warned = False
        if self.rank_tolerance != self._warned_tolerance:
            self._warned_tolerance = self.rank_tolerance
            self._warned = False

    def _decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the eigenvalues and the pseudo-inverse of the matrix, computed once.

        Returns:
            The eigenvalues, largest first, and the pseudo-inverse, see
            `decomposition`.
        """
        if self._cached_decomposition is None:
            self._cached_decomposition = decomposition(self.matrix)
        return self._cached_decomposition

    def _inverse(self) -> np.ndarray:
        """Get the pseudo-inverse of the matrix, the stored one when there is one.

        Returns:
            The pseudo-inverse, not to be changed by the caller.
        """
        self._refresh_matrix_cache()
        if self.inverse is not None:
            return np.asarray(self.inverse, dtype=float)
        return self._decomposition()[1]

    def _spectrum(self) -> np.ndarray:
        """Get the eigenvalues of the matrix, largest first, the stored ones when there are.

        Returns:
            The eigenvalues, not to be changed by the caller.
        """
        self._refresh_matrix_cache()
        if self.spectrum is not None:
            return np.asarray(self.spectrum, dtype=float)
        return self._decomposition()[0]

    def __post_init__(self) -> None:
        """Check the scales of the parameters and the stored decomposition.

        Raises:
            ValueError: if the scales are given and not one per parameter, or
                if the stored decomposition does not have the shape of the
                matrix.
        """
        if self.targets and len(self.targets) != len(self.pids):
            raise ValueError(
                f"'{self.opid}': the Fisher information requires one target per "
                f"parameter, but '{len(self.targets)}' targets are given for the "
                f"'{len(self.pids)}' parameters '{self.pids}'."
            )
        if self.scales and len(self.scales) != len(self.pids):
            raise ValueError(
                f"'{self.opid}': the Fisher information requires one scale per "
                f"parameter, but '{len(self.scales)}' scales are given for the "
                f"'{len(self.pids)}' parameters '{self.pids}'."
            )
        k = len(self.pids)
        if self.inverse is not None and np.shape(self.inverse) != (k, k):
            raise ValueError(
                f"'{self.opid}': the stored inverse of the Fisher information "
                f"requires the shape of its matrix, '{(k, k)}', but has the "
                f"shape '{np.shape(self.inverse)}'."
            )
        if self.spectrum is not None and np.shape(self.spectrum) != (k,):
            raise ValueError(
                f"'{self.opid}': the stored eigenvalues of the Fisher "
                f"information require one per parameter, '{(k,)}', but have "
                f"the shape '{np.shape(self.spectrum)}'."
            )
        # the matrix the stored decomposition belongs to, so that a change of
        # it before the first read drops the decomposition as well
        self._matrix_fingerprint = self._fingerprint()

    @property
    def k(self) -> int:
        """Get the number of parameters."""
        return len(self.pids)

    @property
    def parameter_scales(self) -> list[ParameterScaleType]:
        """Get the space of every parameter, in the order of the matrix."""
        return list(self.scales) if self.scales else [self.scale] * self.k

    def to_scale(self, values: Any) -> np.ndarray:
        """Transform one value per parameter into the space of the optimizer."""
        return np.array(
            [
                scale.to_scale(value)
                for scale, value in zip(self.parameter_scales, values, strict=True)
            ],
            dtype=float,
        )

    def from_scale(self, values: Any) -> np.ndarray:
        """Transform one value per parameter into the units of the model."""
        return np.array(
            [
                scale.from_scale(value)
                for scale, value in zip(self.parameter_scales, values, strict=True)
            ],
            dtype=float,
        )

    @property
    def scaled(self) -> bool:
        """Check whether the covariance is scaled by the variance of the residuals.

        The residuals of a fit of least squares are weighted by weights whose
        scale is arbitrary, so the variance of the residuals is estimated from
        them; the residuals of a likelihood are in units of the noise.
        """
        return self.objective is ObjectiveType.LEAST_SQUARES

    @property
    def sigma2(self) -> float:
        """Get the variance of the residuals, `2 cost / (n - k)`.

        A fit with as many parameters as data points has no degrees of freedom
        left and no variance to estimate, which is `nan`. A fit of the
        likelihood has residuals in units of the noise, the variance 1, see
        `scaled`.
        """
        if not self.scaled:
            return 1.0
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
        return self._spectrum().copy()

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
        it can be read. The warning about it is logged once per information and
        rank tolerance.
        """
        self._refresh_matrix_cache()
        if not self.is_identifiable and not self._warned:
            self._warned = True
            logger.warning(
                "'%s': the Fisher information has rank %s of %s, the covariance "
                "of the unconstrained directions is not meaningful.",
                self.opid,
                self.rank,
                self.k,
            )
        return np.asarray(self.sigma2 * self._inverse(), dtype=float)

    @property
    def standard_errors(self) -> np.ndarray:
        """Get the standard error of every parameter in the scaled space."""
        variances = np.diag(self.covariance)
        # A perfect identifiable fit has zero estimated variance. A zero on
        # the diagonal of a deficient pseudo-inverse may be unconstrained.
        valid = variances >= 0.0 if self.is_identifiable else variances > 0.0
        return np.sqrt(np.where(valid, variances, np.nan))

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

    @property
    def statistics(self) -> FisherStatistics:
        """Get the statistics of the information.

        They follow from its decomposition, the stored one of a record or the
        one of sbmlsim-core, see `decomposition`, so a record gives the numbers
        it was written with.
        """
        return FisherStatistics(
            sigma2=self.sigma2,
            spectrum=self.eigenvalues,
            rank=self.rank,
            condition=self.condition_number,
            covariance=self.covariance,
            standard_errors=self.standard_errors,
            correlation=self.correlation.to_numpy(),
        )

    def confidence_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the confidence interval of every parameter.

        The interval is `θ ± t · SE` in the space the optimizer searches, with
        the quantile of the t distribution of `n - k` degrees of freedom (of
        the normal distribution for a fit of the likelihood, see `scaled`),
        and is transformed back into the units of the model, every parameter with
        its own scale. On a logarithmic scale the interval is therefore not
        symmetric around the value.

        Returns:
            The lower and the upper bound in the units of the model, `nan`
            for a fit of least squares without degrees of freedom; the
            interval of a direction the data does not constrain is unbounded
            on a logarithmic scale.
        """
        if self.scaled and self.n - self.k <= 0:
            nan = np.full(self.k, np.nan)
            return nan, nan
        import_core()
        from sbmlsim_core import stats

        return stats.fisher_intervals(
            self.values,
            self.standard_errors,
            self.parameter_scales,
            level=self.alpha,
            dof=self.n - self.k if self.scaled else None,
        )

    @property
    def has_mixed_scales(self) -> bool:
        """Check whether the parameters are searched in more than one space."""
        return len(set(self.parameter_scales)) > 1

    @property
    def relative_errors(self) -> np.ndarray:
        """Get the standard error of every parameter relative to its value.

        The coefficient of variation of the parameter in the units of the
        model, to first order: `se / |value|` for a parameter searched on the
        linear scale, the standard error of `ln(p)` for one searched as a
        natural logarithm and `ln(10)` times the standard error of `log10(p)`
        for a decadic logarithm. It does not depend on the unit of the
        parameter (the error of a logarithm divided by the logarithm of the
        value did, and was near infinite for a value near 1).
        """
        import_core()
        from sbmlsim_core import stats

        return stats.fisher_relative_errors(
            self.values, self.standard_errors, self.parameter_scales
        )

    @property
    def summary_df(self) -> pd.DataFrame:
        """Get the parameters with their errors and intervals as a table.

        The table has the column `scale` with the space of every parameter
        when the parameters are searched in different spaces, see
        `has_mixed_scales`.
        """
        lower, upper = self.confidence_intervals()
        errors = self.standard_errors
        cv = self.relative_errors * 100.0
        units = self.units or [None] * self.k
        columns: dict[str, Any] = {"parameter": self.pids, "value": self.values}
        if self.has_mixed_scales:
            columns["scale"] = [scale.name for scale in self.parameter_scales]
        columns.update(
            {
                "se": errors,
                "cv": cv,
                "ci_lower": lower,
                "ci_upper": upper,
                "unit": units,
            }
        )
        return pd.DataFrame(columns)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FisherInformation:
        """Create from a dictionary, the inverse of `to_dict`.

        Args:
            d: dictionary of `to_dict`, e.g. read from the `fisher.json` of a
                fit record.

        Returns:
            The Fisher information with the arrays restored.
        """
        return cls(
            opid=d["opid"],
            sid=d["sid"],
            pids=list(d["pids"]),
            values=np.asarray(d["values"], dtype=float),
            scale=ParameterScaleType[d["scale"]],
            matrix=np.asarray(d["matrix"], dtype=float),
            cost=d["cost"],
            n=d["n"],
            alpha=d.get("alpha", 0.95),
            rank_tolerance=d.get("rank_tolerance", DEFAULT_RANK_TOLERANCE),
            # a record before the step was stored has the default step
            step=d.get("step", DEFAULT_STEP),
            # a record before the jacobian was stored is one of the differences
            jacobian_of=str(d.get("jacobian_of", CENTRAL_DIFFERENCES)),
            units=list(d.get("units", [])),
            scales=[ParameterScaleType[name] for name in d.get("scales", [])],
            targets=list(d.get("targets", [])),
            # a record before the objective was stored is of least squares
            objective=ObjectiveType(d.get("objective", "least_squares")),
            # a record before the decomposition was stored computes it (#523)
            inverse=(
                None
                if d.get("inverse") is None
                else np.asarray(d["inverse"], dtype=float)
            ),
            spectrum=(
                None
                if d.get("spectrum") is None
                else np.asarray(d["spectrum"], dtype=float)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "opid": self.opid,
            "sid": self.sid,
            "pids": list(self.pids),
            "values": [float(v) for v in self.values],
            "scale": self.scale.name,
            "scales": [scale.name for scale in self.parameter_scales],
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "cost": self.cost,
            "n": self.n,
            "alpha": self.alpha,
            "rank_tolerance": self.rank_tolerance,
            "step": self.step,
            "jacobian_of": self.jacobian_of,
            "units": list(self.units),
            "targets": list(self.targets),
            "objective": self.objective.value,
            # the decomposition of the matrix, so that a record gives the same
            # standard errors, correlations and rank on every machine (#523)
            "inverse": [[float(v) for v in row] for row in self._inverse()],
            "spectrum": [float(v) for v in self._spectrum()],
        }


def _points(x: np.ndarray, step: float) -> tuple[list[np.ndarray], list[float]]:
    """Get the points of the central differences of a jacobian and their steps.

    The points are the pairs of one parameter after another, `x + h` before
    `x - h`, which is the order the columns are built from.

    Args:
        x: parameters in the space the optimizer searches.
        step: relative step of the differences.

    Returns:
        The points and the step of every parameter.
    """
    points: list[np.ndarray] = []
    steps: list[float] = []
    for k in range(len(x)):
        h = step * max(abs(float(x[k])), 1.0)
        x_plus, x_minus = np.array(x, dtype=float), np.array(x, dtype=float)
        x_plus[k] += h
        x_minus[k] -= h
        points += [x_plus, x_minus]
        steps.append(h)
    return points, steps


def _residuals_serial(
    problem: OptimizationProblem, points: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Evaluate the residuals of the points in this process, with its progress."""
    rows: list[np.ndarray] = []
    with progress_activity(
        "jacobian", "Fisher information", total=len(points)
    ) as evaluated:
        for point in points:
            rows.append(np.asarray(problem.residuals(point), dtype=float))
            if evaluated is not None:
                evaluated.advance()
    return rows


@contextlib.contextmanager
def evaluator(
    problem: OptimizationProblem, settings: FitSettings, n_cores: int = 1
) -> Generator[Evaluate | None]:
    """Get what evaluates the residuals of the points of a jacobian.

    The points of the central differences are independent of each other.
    sbmlsim-core evaluates them as one batch on threads of its own in this
    process (`OptimizationProblem.evaluate_residuals_many`); any other engine
    on a pool of workers, the pool of a fit with the problem initialized in
    every worker (`sbmlsim.fit.runner.JacobianPool`, what the jacobian of
    least squares uses). The values are the values of this process bitwise,
    see `JacobianPool`.

    Args:
        problem: initialized optimization problem.
        settings: the settings the problem was initialized with, which the
            workers initialize it with.
        n_cores: the number of workers, one for the evaluation in this
            process.

    Yields:
        What evaluates the points, `None` for the evaluation in this process.
    """
    # imported here, so that the fisher information of a module which does not
    # fit (a report, a record) does not pull in the runner of the fits
    from sbmlsim.fit.runner import JacobianPool, resolve_n_cores, worker_pool

    cores = resolve_n_cores(n_cores)
    if cores <= 1:
        yield None
        return
    if problem.engine is Engine.CORE:
        # sbmlsim-core simulates the points of a jacobian at once on threads
        # of its own: one batch in this process, which is the batch a worker
        # of a pool would evaluate (`runner._worker_residuals_many`). A pool
        # started a process and initialized the problem in it again for this
        # one batch, ten seconds of the Fisher information of the HCTZ PK fit
        # for half a second of simulations (sbmlsim#664)

        def batch(points: Sequence[np.ndarray]) -> list[np.ndarray]:
            """Evaluate the points as one batch of the engine, in their order."""
            with progress_activity(
                "jacobian", "Fisher information", total=len(points)
            ) as evaluated:
                rows = [
                    np.asarray(residuals, dtype=float)
                    for residuals, _ in problem.evaluate_residuals_many(
                        points, threads=cores
                    )
                ]
                if evaluated is not None:
                    evaluated.advance(len(points))
            return rows

        yield batch
        return
    with worker_pool(problem, settings, cores, threads=1) as pool:
        points_pool = JacobianPool(pool)
        points_pool.start()

        def evaluate(points: Sequence[np.ndarray]) -> list[np.ndarray]:
            """Evaluate the points on the workers, in the order of the points."""
            rows: list[np.ndarray | None] = [None] * len(points)
            for index, (residuals, _) in points_pool(points, None):
                rows[index] = np.asarray(residuals, dtype=float)
            missing = [k for k, row in enumerate(rows) if row is None]
            if missing:
                raise RuntimeError(
                    f"the points {missing} of the jacobian were not evaluated."
                )
            return [row for row in rows if row is not None]

        yield evaluate


def jacobian(
    problem: OptimizationProblem,
    x: np.ndarray,
    step: float = DEFAULT_STEP,
    evaluate: Evaluate | None = None,
) -> np.ndarray:
    """Get the jacobian of the weighted residuals by central differences.

    Args:
        problem: initialized optimization problem.
        x: parameters in the space the optimizer searches.
        step: relative step of the differences.
        evaluate: what evaluates the points of the differences, see
            `evaluator`; the evaluation in this process by default.

    Returns:
        The jacobian with one row per residual and one column per parameter.
    """
    points, steps = _points(x, step)
    rows = _residuals_serial(problem, points) if evaluate is None else evaluate(points)
    columns = [
        (rows[2 * k] - rows[2 * k + 1]) / (2.0 * steps[k]) for k in range(len(steps))
    ]
    return np.column_stack(columns)


def _jacobian_of(
    problem: OptimizationProblem,
    settings: FitSettings,
    x: np.ndarray,
    rows: int,
    *,
    step: float | None,
    engine: bool | None,
    n_cores: int,
) -> tuple[np.ndarray, str, float]:
    """Get the jacobian of the residuals of the data, and where it comes from.

    The forward sensitivities of sbmlsim-core are the derivatives of the
    residuals themselves and a simulation per condition, where a difference
    quotient is two simulations per parameter and condition whose accuracy is
    the noise of the integrator over its step: at the default tolerance of
    `1e-6` the central differences of the iv fit of the HCTZ example are 1.5 %
    off the sensitivities over the step which balances that noise
    (`default_step`), and the sensitivities of that tolerance are the ones of
    a tolerance of `1e-10` to `1e-5`, which the differences reach only there.
    So the information takes them where the engine has them.

    The finite differences of the engine are not taken: their step is a fixed
    `1e-3` of the parameter, the `diff_step` of an optimizer, and does not
    follow the tolerance of the integrator, so they are a worse estimate than
    the central differences of `default_step` in python.

    Args:
        problem: the initialized problem.
        settings: the settings the problem was initialized with.
        x: the point, in the space the problem searches.
        rows: the number of residuals of the data, which the jacobian of the
            differences is cut to: the residuals of the priors of a fit of the
            posterior are no part of the information.
        step: the relative step of the differences, `None` for the step of the
            tolerance of the integrator; a step which is given asks for the
            differences.
        engine: whether the jacobian is the one of the engine, see
            `fisher_information`.
        n_cores: the workers of the differences, the threads of the engine.

    Returns:
        The jacobian, where it comes from (`SENSITIVITIES` or
        `CENTRAL_DIFFERENCES`) and the step it took, `0` for the
        sensitivities.

    Raises:
        ValueError: if the jacobian of the engine is required and the engine
            cannot compute it.
    """
    from sbmlsim.fit import fit_core

    asked = engine or (engine is None and step is None)
    if asked:
        gaps = fit_core.gaps_of_a_jacobian(problem)
        reason = "; ".join(f"{gap.code}: {gap.message}" for gap in gaps)
        if engine and gaps:
            raise ValueError(
                f"'{problem.opid}': the jacobian of the Fisher information cannot "
                f"come from the engine: {reason}."
            )
        if not gaps:
            from sbmlsim.fit.runner import resolve_n_cores

            try:
                jac = fit_core.residual_jacobian(
                    problem, x, threads=resolve_n_cores(n_cores)
                )
            except ValueError as err:
                # a condition which carries no forward sensitivities (an event
                # of the model, a delay; the changes and the steady state of a
                # request since sbmlsim-core 0.29.0) is refused by the engine,
                # and the differences answer instead
                if engine:
                    raise
                reason = f"sensitivities: {err}"
            else:
                if jac.shape[0] == rows:
                    return jac, SENSITIVITIES, 0.0
                reason = (
                    f"residuals: the engine computes {jac.shape[0]} residuals of "
                    f"the data, the problem {rows}"
                )
                if engine:
                    raise ValueError(
                        f"'{problem.opid}': the jacobian of the Fisher information "
                        f"cannot come from the engine: {reason}."
                    )
        logger.info(
            "'%s': the jacobian of the Fisher information is the central "
            "differences of python: %s.",
            problem.opid,
            reason,
        )
    if step is None:
        step = default_step(settings.relative_tolerance)
    with evaluator(problem, settings, n_cores) as evaluate:
        jac = jacobian(problem, x, step=step, evaluate=evaluate)[:rows]
    return jac, CENTRAL_DIFFERENCES, step


def fisher_information(
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    alpha: float = 0.95,
    step: float | None = None,
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE,
    n_cores: int = 1,
    engine_jacobian: bool | None = None,
) -> FisherInformation:
    """Calculate the Fisher information of a parameter set on a problem.

    The jacobian is the forward sensitivities of sbmlsim-core where the
    problem and its conditions carry them, and the central differences of the
    residuals in python otherwise, see `_jacobian_of`.

    Args:
        problem: optimization problem, initialized with the settings.
        settings: settings the parameters were fitted with.
        parameter_set: parameters to evaluate the information at, i.e. the
            result of a fit.
        alpha: confidence level of the intervals.
        step: relative step of the finite differences of the jacobian,
            `default_step` of the relative tolerance of the settings by
            default; a step which is given asks for the differences, so the
            jacobian of such a call comes from the engine only with
            `engine_jacobian=True`.
        rank_tolerance: eigenvalues below this fraction of the largest one are
            a direction the data does not constrain.
        n_cores: the workers which evaluate the points of the jacobian, the
            two points of every parameter, see `evaluator`; one evaluates them
            in this process, which is `2 k` simulations of the experiments one
            after another. It is the threads of the batch of a jacobian of the
            engine.
        engine_jacobian: whether the jacobian is the forward sensitivities of
            the engine: `None` takes them where the engine has them, `True`
            requires them and names what keeps the problem from them, `False`
            takes the central differences in python.

    Returns:
        The Fisher information with the errors, the correlations and the
        directions the data does not constrain.

    Raises:
        ValueError: if `engine_jacobian` is required and the engine cannot
            compute the jacobian of the problem.
    """
    problem.initialize(settings)
    values = problem.bind_parameter_set(parameter_set).x(problem.pids)
    x = problem.to_scale(values)

    residuals, cost, n = data_residuals(problem, x)
    jac, jacobian_of, step = _jacobian_of(
        problem,
        settings,
        x,
        residuals.size,
        step=step,
        engine=engine_jacobian,
        n_cores=n_cores,
    )
    import_core()
    from sbmlsim_core import stats

    matrix = stats.gram(jac)

    return FisherInformation(
        opid=problem.opid,
        sid=parameter_set.sid,
        pids=list(problem.pids),
        values=values,
        scale=problem.parameter_scale,
        matrix=matrix,
        cost=cost,
        n=n,
        alpha=alpha,
        rank_tolerance=rank_tolerance,
        step=step,
        jacobian_of=jacobian_of,
        # units of pint, a unit id of a model resolved
        units=list(problem.parameter_units.values()),
        scales=list(problem.scales_initialized),
        targets=[p.target_id for p in problem.parameters],
        objective=problem.objective,
    )


def data_residuals(
    problem: OptimizationProblem, x: np.ndarray
) -> tuple[np.ndarray, float, int]:
    """Get the residuals of the data at parameters, without the ones of the priors.

    Args:
        problem: initialized optimization problem.
        x: parameters in the space the optimizer searches.

    Returns:
        The residuals of the data, the first ones of `problem.residuals`, the
        cost of the data (half their sum of squares minus the offset of a
        likelihood, i.e. the negative log-likelihood) and the number of data
        points: a residual per point for least squares, two (of the data and
        of the scale of its noise) for a likelihood.
    """
    residuals = np.asarray(problem.residuals(x), dtype=float)
    offset = problem._cost_offset
    n_prior = problem.prior_residual_count
    if n_prior:
        offset -= problem.prior.residuals(problem.from_scale(x))[1]
        residuals = residuals[:-n_prior]
    cost = float(0.5 * np.sum(np.square(residuals))) - offset
    n = residuals.size
    if problem.objective is not ObjectiveType.LEAST_SQUARES:
        n //= 2
    return residuals, cost, int(n)
