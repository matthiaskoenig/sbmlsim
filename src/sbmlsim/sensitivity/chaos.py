"""Polynomial chaos expansion of the output of a scan.

A polynomial chaos expansion writes an output of the model as a series of
polynomials of its inputs which are orthonormal under the distributions of
the inputs,

    y(u) = sum over alpha of c_alpha * Psi_alpha(u),

so the mean of the output is `c_0`, its variance is the sum of the squares of
the other coefficients, and the variance of every group of inputs is the sum
of the squares of the coefficients whose multi-index is supported exactly on
that group (Sudret 2008). The Sobol indices follow from the coefficients
without a single further simulation: the first order index of an input is the
share of the variance of the coefficients of that input alone, its total
index the share of every coefficient which reads it, and the interaction of
two inputs the share of the coefficients supported on the pair. The
expansion is also a surrogate: `PolynomialChaos.predict` evaluates it at any
point of the unit cube at the cost of a polynomial.

The points of a design of `sbmlsim.simulation.sampling` are the image of its
unit cube under the marginals of its targets (`Distribution.ppf`), and the
value of a target is a monotone function of the coordinate of its own
dimension alone, so an expansion in the coordinates of the unit cube has the
indices of the inputs, whatever their distributions. The basis is therefore
the one of the uniform distribution of the unit cube, the shifted Legendre
polynomials normalized to one, and the design of the analysis needs neither
a transform of the inputs nor a basis per marginal. The inputs must be
independent, which is what the share of a variance means, so a design with a
correlation is refused.

The basis of a degree `p` over `d` inputs has `(p + d)! / (p! d!)` terms,
which a regression needs more points than, so the basis is truncated: the
hyperbolic norm `q` drops the terms of many inputs at a high degree (Blatman
and Sudret 2010, `q = 1` is the full basis of the degree and `q = 0.5` keeps
about the terms of one and two inputs) and `max_interaction` caps the number
of inputs of a term. The degree is not given but chosen per element of the
observable from the corrected leave-one-out error (Chapelle, Vapnik and
Bengio 2002; Blatman and Sudret 2010), which needs no simulation beyond the
ones of the fit and which the result reports as the variable `<observable>.loo`
of every element, so a fit which the points do not support is visible rather
than silent. The expansions of the candidate degrees share the design matrix
and every element of an observable is fitted at once, one decomposition per
degree.

References:
    Sudret B. Global sensitivity analysis using polynomial chaos expansions.
    Reliability Engineering & System Safety. 2008;93(7):964-979.

    Blatman G, Sudret B. An adaptive algorithm to build up sparse polynomial
    chaos expansions for stochastic finite element analysis. Probabilistic
    Engineering Mechanics. 2010;25(2):183-197.

    Chapelle O, Vapnik V, Bengio Y. Model selection for small sample
    regression. Machine Learning. 2002;48(1):231-241.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from sbmlsim.result.scan import ScanResult
from sbmlsim.sensitivity.indices import (
    _assemble,
    _elements,
    _moved,
    _observables,
    _unit_points,
    _warn_failed,
    constant_tolerance,
    design_of,
)
from sbmlsim.sensitivity.result import PARAMETER, PARAMETER_2, SensitivityResult

logger = logging.getLogger(__name__)

#: the methods of a design a chaos expansion is fitted on: the samples of the
#: distributions of the targets, not the structured designs of `fast`,
#: `morris` or `local`, whose points lie on a curve, on trajectories or on a
#: star and carry no information about the rest of the input space
METHODS: frozenset[str] = frozenset({"random", "lhs", "sobol"})

#: the largest basis a fit builds, a guard against a degree of a high
#: dimension which no number of points supports
MAX_TERMS: int = 20000

#: the smallest number of points per term of the basis of a candidate degree;
#: the leave-one-out error of a fit with fewer points is not informative
POINTS_PER_TERM: float = 2.0

#: the relative tolerance of the rank of a design matrix, below which a
#: candidate degree is not fitted
RANK_TOLERANCE: float = 1e-10

#: the truncations `choose_truncation` tries, in order: the exponent of the
#: hyperbolic truncation and the cap on the number of inputs of a term. The
#: exponent falls before the cap, so a basis keeps the interactions of a low
#: degree as long as the points support them; the last entry is the additive
#: basis, which `d * degree` terms large and which carries no interaction at
#: all, so the total indices it gives are its first order ones
TRUNCATIONS: tuple[tuple[float, int | None], ...] = (
    (1.0, None),
    (0.75, None),
    (0.5, None),
    (0.5, 2),
    (1.0, 1),
)


def expansion(
    d: int, degree: int, *, q: float = 1.0, max_interaction: int | None = None
) -> np.ndarray:
    """Get the multi-indices of a truncated basis.

    Args:
        d: the number of inputs.
        degree: the largest total degree, not negative.
        q: the exponent of the hyperbolic truncation, in `(0, 1]`; a term is
            kept when `(sum alpha_i**q)**(1/q) <= degree`, so `1` keeps every
            term of the degree.
        max_interaction: the largest number of inputs of a term, every number
            by default.

    Returns:
        The multi-indices, a row per term, ordered by their total degree and
        then by the inputs they read; the first row is the constant term.

    Raises:
        ValueError: if the arguments are out of range or the basis is larger
            than `MAX_TERMS`.
    """
    if d < 1:
        raise ValueError(f"The basis needs at least one input, not {d}.")
    if degree < 0:
        raise ValueError(f"The degree must not be negative, not {degree}.")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"The hyperbolic truncation q must be in (0, 1], not {q}.")
    if max_interaction is not None and max_interaction < 1:
        raise ValueError(
            f"The interaction order must be at least 1, not {max_interaction}."
        )
    rows: list[np.ndarray] = []
    for total in range(degree + 1):
        for combination in itertools.combinations_with_replacement(range(d), total):
            alpha = np.bincount(combination, minlength=d)
            inputs = int(np.count_nonzero(alpha))
            if max_interaction is not None and inputs > max_interaction:
                continue
            if q < 1.0 and total > 1:
                norm = float(np.sum(alpha.astype(float) ** q) ** (1.0 / q))
                if norm > degree * (1.0 + 1e-12):
                    continue
            rows.append(alpha)
            if len(rows) > MAX_TERMS:
                raise ValueError(
                    f"The basis of the degree {degree} over {d} inputs has more than "
                    f"{MAX_TERMS} terms; lower the degree, lower q or cap "
                    f"max_interaction."
                )
    return np.asarray(rows, dtype=int)


def choose_truncation(d: int, n: int, degree: int) -> tuple[float, int | None]:
    """Choose a truncation of the basis of a degree which `n` points support.

    The full basis of a degree over many inputs is larger than any sample: the
    one of the degree 5 over 22 inputs has 80730 terms, more than `MAX_TERMS`,
    and even the `n (d + 2)` points of a Saltelli design of the same inputs
    would not support it. The truncations of `TRUNCATIONS` are tried in order
    and the first whose basis has at most `n / POINTS_PER_TERM` terms is
    taken, so a sample which supports the interactions keeps them and one
    which does not falls back to the additive basis rather than failing.

    Args:
        d: the number of inputs, at least 1.
        n: the number of points of the sample, at least 1.
        degree: the largest total degree of the basis, not negative.

    Returns:
        The exponent of the hyperbolic truncation and the cap on the
        interaction, for `expansion`, `fit_chaos` or `pce`.

    Raises:
        ValueError: if the arguments are out of range, or no truncation is
            supported, i.e. the additive basis of the degree over the inputs
            needs more points than the sample has.
    """
    if d < 1:
        raise ValueError(f"The basis needs at least one input, not {d}.")
    if n < 1:
        raise ValueError(f"The sample needs at least one point, not {n}.")
    if degree < 0:
        raise ValueError(f"The degree must not be negative, not {degree}.")
    budget = int(n // POINTS_PER_TERM)
    for q, max_interaction in TRUNCATIONS:
        try:
            terms = len(expansion(d, degree, q=q, max_interaction=max_interaction))
        except ValueError:
            # the arguments are checked above, so the basis is larger than
            # `MAX_TERMS`: it is larger than every budget as well
            continue
        if terms <= budget:
            return q, max_interaction
    smallest = len(expansion(d, degree, q=TRUNCATIONS[-1][0], max_interaction=1))
    raise ValueError(
        f"No truncation of the basis of the degree {degree} over {d} inputs is "
        f"supported by {n} points: the smallest, the additive basis, has "
        f"{smallest} terms and needs {math.ceil(smallest * POINTS_PER_TERM)} "
        f"points. Lower the degree or sample more points."
    )


def legendre_values(u: np.ndarray, max_degree: int) -> np.ndarray:
    """Evaluate the orthonormal Legendre polynomials of the unit interval.

    The polynomials are `sqrt(2k + 1) P_k(2u - 1)` of the Legendre
    polynomials `P_k`, which are orthonormal on `[0, 1]` under the uniform
    distribution, computed with their recurrence.

    Args:
        u: the points, a row per point and a column per input.
        max_degree: the largest degree, not negative.

    Returns:
        The values over `(point, input, degree)`.
    """
    points = np.asarray(u, dtype=float)
    t = 2.0 * points - 1.0
    out = np.empty((*points.shape, max_degree + 1))
    out[..., 0] = 1.0
    if max_degree >= 1:
        out[..., 1] = t
    for k in range(1, max_degree):
        out[..., k + 1] = ((2 * k + 1) * t * out[..., k] - k * out[..., k - 1]) / (
            k + 1
        )
    return out * np.sqrt(2.0 * np.arange(max_degree + 1) + 1.0)


def design_matrix(u: np.ndarray, multi_indices: np.ndarray) -> np.ndarray:
    """Evaluate the basis at the points.

    Args:
        u: the points, a row per point and a column per input.
        multi_indices: the multi-indices of the basis, a row per term.

    Returns:
        The values of the terms, a row per point and a column per term.

    Raises:
        ValueError: if the points and the multi-indices have different numbers
            of inputs.
    """
    points = np.asarray(u, dtype=float)
    indices = np.asarray(multi_indices, dtype=int)
    if points.ndim != 2 or indices.ndim != 2:
        raise ValueError("The points and the multi-indices are two-dimensional.")
    if points.shape[1] != indices.shape[1]:
        raise ValueError(
            f"The points have {points.shape[1]} inputs, the multi-indices "
            f"{indices.shape[1]}."
        )
    values = legendre_values(points, int(indices.max(initial=0)))
    out = np.ones((points.shape[0], indices.shape[0]))
    for i in range(points.shape[1]):
        out *= values[:, i, :][:, indices[:, i]]
    return out


def _fit_least_squares(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit the terms of a design matrix to every element by least squares.

    Args:
        x: the design matrix, a row per point and a column per term.
        y: the values, a row per point and a column per element.

    Returns:
        The coefficients over `(term, element)` and the corrected
        leave-one-out error of every element, or `None` when the points do not
        support the terms (too few points or a design matrix without full
        rank).
    """
    n, p = x.shape
    if n < POINTS_PER_TERM * p or n <= p + 1:
        return None
    q, r = np.linalg.qr(x)
    diagonal = np.abs(np.diag(r))
    if diagonal.min(initial=np.inf) <= RANK_TOLERANCE * diagonal.max(initial=0.0):
        return None
    coefficients = solve_triangular(r, q.T @ y)
    # the leverages of the points, the diagonal of the hat matrix `Q Q'`
    leverage = np.einsum("ij,ij->i", q, q)
    residuals = y - x @ coefficients
    held = residuals / np.maximum(1.0 - leverage, np.finfo(float).eps)[:, None]
    error = np.mean(held**2, axis=0)
    variance = y.var(axis=0)
    # the correction of the error of a fit of many terms, whose trace is about
    # `p / n` for a basis which is orthonormal under the distribution
    inverse = solve_triangular(r, np.eye(p))
    factor = (n / (n - p)) * (1.0 + float(np.sum(inverse**2)))
    return coefficients, np.divide(
        factor * error,
        variance,
        out=np.full(error.shape, np.inf),
        where=variance > 0.0,
    )


@dataclass(frozen=True)
class PolynomialChaos:
    """A polynomial chaos expansion of the elements of an output.

    The expansion of every element shares the basis `multi_indices`, in which
    the coefficients of the terms above the degree of an element are zero, so
    `coefficients` is one array whatever the degrees are.

    Attributes:
        parameters: the inputs, in the order of the columns of the
            multi-indices.
        multi_indices: the multi-indices of the basis, a row per term.
        coefficients: the coefficients over `(term, element)`.
        loo: the corrected leave-one-out error of every element, the share of
            the variance of its values the expansion does not predict.
        degree: the total degree of the basis of every element.
    """

    parameters: tuple[str, ...]
    multi_indices: np.ndarray
    coefficients: np.ndarray
    loo: np.ndarray
    degree: np.ndarray

    @property
    def d(self) -> int:
        """Get the number of inputs."""
        return len(self.parameters)

    @property
    def mean(self) -> np.ndarray:
        """Get the mean of every element, the coefficient of the constant term."""
        return np.asarray(self.coefficients[0], dtype=float)

    @property
    def variance(self) -> np.ndarray:
        """Get the variance of every element, the squares of the other coefficients."""
        return np.sum(self.coefficients[1:] ** 2, axis=0)

    def _shares(self, mask: np.ndarray) -> np.ndarray:
        """Get the share of the variance of the terms of a mask, 0 without variance.

        Args:
            mask: a row per input and a column per term.

        Returns:
            The shares over `(input, element)`.
        """
        squares = self.coefficients**2
        squares[0] = 0.0
        numerator = mask.astype(float) @ squares
        variance = self.variance
        return np.divide(
            numerator,
            variance,
            out=np.zeros_like(numerator),
            where=variance > 0.0,
        )

    def first_order(self) -> np.ndarray:
        """Get the first order index of every input, over `(input, element)`."""
        reads = self.multi_indices > 0
        alone = reads & (np.count_nonzero(reads, axis=1) == 1)[:, None]
        return self._shares(alone.T)

    def total_order(self) -> np.ndarray:
        """Get the total index of every input, over `(input, element)`."""
        return self._shares((self.multi_indices > 0).T)

    def second_order(self) -> np.ndarray:
        """Get the interaction of every pair of inputs.

        Returns:
            The indices over `(input, input, element)`, the pairs `i < j` and
            `NaN` on and below the diagonal, as the second order indices of a
            Sobol analysis.
        """
        reads = self.multi_indices > 0
        pairs = np.flatnonzero(np.count_nonzero(reads, axis=1) == 2)
        out = np.full((self.d, self.d, self.coefficients.shape[1]), np.nan)
        rows, columns = np.triu_indices(self.d, k=1)
        out[rows, columns] = 0.0
        squares = self.coefficients**2
        variance = self.variance
        for term in pairs:
            i, j = np.flatnonzero(reads[term])
            out[i, j] += squares[term]
        out[rows, columns] = np.divide(
            out[rows, columns],
            variance,
            out=np.zeros_like(out[rows, columns]),
            where=variance > 0.0,
        )
        return out

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Evaluate the expansion of every element at points of the unit cube.

        Args:
            u: the points, a row per point and a column per input.

        Returns:
            The values, a row per point and a column per element.
        """
        return design_matrix(u, self.multi_indices) @ self.coefficients


def fit_chaos(
    u: np.ndarray,
    y: np.ndarray,
    *,
    degree: int = 5,
    q: float = 1.0,
    max_interaction: int | None = None,
    parameters: Sequence[str] | None = None,
) -> PolynomialChaos:
    """Fit a chaos expansion of every element of an output on the unit cube.

    The candidate degrees are `1..degree`; every element keeps the one of the
    smallest corrected leave-one-out error, the bases of the candidates being
    the prefixes of the basis of `degree`, which are ordered by their total
    degree.

    Args:
        u: the points of the unit cube, a row per point and a column per input.
        y: the values, a row per point and a column per element.
        degree: the largest total degree of the basis.
        q: the exponent of the hyperbolic truncation, see `expansion`.
        max_interaction: the largest number of inputs of a term.
        parameters: the names of the inputs, `x0..xd-1` by default.

    Returns:
        The expansion.

    Raises:
        ValueError: if the points and the values do not match, or no candidate
            degree is supported by the points (`POINTS_PER_TERM` points per
            term of its basis).
    """
    points = np.asarray(u, dtype=float)
    values = np.asarray(y, dtype=float)
    if points.ndim != 2 or values.ndim != 2:
        raise ValueError("The points and the values are two-dimensional.")
    if points.shape[0] != values.shape[0]:
        raise ValueError(
            f"The points have {points.shape[0]} rows, the values {values.shape[0]}."
        )
    if degree < 1:
        raise ValueError(f"The degree of an expansion is at least 1, not {degree}.")
    d = points.shape[1]
    names = (
        tuple(parameters)
        if parameters is not None
        else tuple(f"x{k}" for k in range(d))
    )
    if len(names) != d:
        raise ValueError(
            f"The points have {d} inputs, the parameters are {list(names)}."
        )
    multi_indices = expansion(d, degree, q=q, max_interaction=max_interaction)
    totals = multi_indices.sum(axis=1)
    x = design_matrix(points, multi_indices)
    elements = values.shape[1]
    coefficients = np.zeros((multi_indices.shape[0], elements))
    loo = np.full(elements, np.inf)
    chosen = np.zeros(elements, dtype=int)
    for candidate in range(1, degree + 1):
        terms = int(np.count_nonzero(totals <= candidate))
        fit = _fit_least_squares(x[:, :terms], values)
        if fit is None:
            break
        found, error = fit
        better = error < loo
        if not better.any():
            continue
        coefficients[:, better] = 0.0
        coefficients[:terms, better] = found[:, better]
        loo[better] = error[better]
        chosen[better] = candidate
    if not np.isfinite(loo).any():
        raise ValueError(
            f"No degree of the basis is supported by {points.shape[0]} points: a "
            f"basis needs {POINTS_PER_TERM} points per term. Add points, lower the "
            f"degree, lower q or cap max_interaction."
        )
    return PolynomialChaos(
        parameters=names,
        multi_indices=multi_indices,
        coefficients=coefficients,
        loo=loo,
        degree=chosen,
    )


def pce(
    result: ScanResult,
    *,
    observables: Sequence[str] | None = None,
    degree: int = 5,
    q: float = 1.0,
    max_interaction: int | None = None,
    second_order: bool = False,
    dim: str | None = None,
    tolerance: float | None = None,
) -> SensitivityResult:
    """Get the Sobol indices of a scan from a polynomial chaos expansion.

    The expansion of every element of every observable is fitted on the points
    of the design of the scan, see the module; the indices are the shares of
    the variance of its coefficients and need no simulation of their own, so
    one sample of a few hundred points gives the first order and total indices
    which a Sobol analysis needs `n * (d + 2)` simulations for.

    The variables of the result are `S1`, `ST` and, with `second_order`, `S2`,
    the quality of the fit of every element (`loo`, the corrected
    leave-one-out error, and `degree`, the total degree of its basis) and the
    mean and standard deviation of the expansion (`mean`, `sd`, in the unit of
    the observable). An element whose values are constant has zero indices and
    no fit, an element of a failed point has none (`NaN`).

    Args:
        result: the result of a scan over a design of `METHODS` without a
            correlation.
        observables: the observables, every one of the result by default.
        degree: the largest total degree of the basis.
        q: the exponent of the hyperbolic truncation, see `expansion`.
        max_interaction: the largest number of inputs of a term.
        second_order: whether to compute the interaction of every pair.
        dim: the dimension of the design, the only one of a design by default.
        tolerance: the tolerance of a constant element, see
            `constant_tolerance`.

    Returns:
        The sensitivity result.

    Raises:
        ValueError: if the scan has no such design, the design has a
            correlation, or no degree is supported by the points.
    """
    tolerance = constant_tolerance(result, tolerance)
    dim, design = design_of(result, set(METHODS), dim)
    if design.options.get("correlation") is not None:
        raise ValueError(
            f"The design of the dimension '{dim}' has a correlation; the share of a "
            f"variance is the one of independent inputs, so an expansion needs a "
            f"design without one."
        )
    parameters, cube = _unit_points(result, dim, design)
    d = len(parameters)
    pairs = ["S2"] if second_order else []
    indices: dict[str, tuple[list[str], np.ndarray]] = {}
    units: dict[str, str] = {}
    failed = 0
    order = np.argsort(np.asarray(result.ds[dim].values))
    for name in _observables(result, observables):
        values, others = _moved(result, name, dim)
        flat = values[order].reshape(values.shape[0], -1)
        out: dict[str, np.ndarray] = {
            key: np.full((d, flat.shape[1]), np.nan) for key in ("S1", "ST")
        }
        out.update({key: np.full((d, d, flat.shape[1]), np.nan) for key in pairs})
        quality: dict[str, np.ndarray] = {
            key: np.full(flat.shape[1], np.nan)
            for key in ("loo", "degree", "mean", "sd")
        }
        usable, varying = _elements(flat, tolerance)
        failed += int(flat.shape[1] - usable.size)
        constant = usable[~varying]
        for key in out:
            out[key][..., constant] = 0.0
        quality["loo"][constant] = 0.0
        quality["degree"][constant] = 0
        quality["sd"][constant] = 0.0
        quality["mean"][constant] = flat[0, constant]
        estimated = usable[varying]
        if estimated.size:
            chaos = fit_chaos(
                cube,
                flat[:, estimated],
                degree=degree,
                q=q,
                max_interaction=max_interaction,
                parameters=parameters,
            )
            out["S1"][..., estimated] = chaos.first_order()
            out["ST"][..., estimated] = chaos.total_order()
            if pairs:
                out["S2"][..., estimated] = chaos.second_order()
            quality["loo"][estimated] = chaos.loo
            quality["degree"][estimated] = chaos.degree
            quality["mean"][estimated] = chaos.mean
            quality["sd"][estimated] = np.sqrt(chaos.variance)
        unit = result.units.get(name, "")
        for key, array in out.items():
            lead = [PARAMETER, PARAMETER_2] if key in pairs else [PARAMETER]
            shape = array.shape[: len(lead)] + values.shape[1:]
            indices[f"{name}.{key}"] = ([*lead, *others], array.reshape(shape))
            units[f"{name}.{key}"] = "dimensionless"
        for key, array in quality.items():
            indices[f"{name}.{key}"] = (others, array.reshape(values.shape[1:]))
            units[f"{name}.{key}"] = unit if key in ("mean", "sd") else "dimensionless"
    _warn_failed(failed)
    options = {
        **design.options,
        "degree": degree,
        "q": q,
        "max_interaction": max_interaction,
        "second_order": second_order,
        "tolerance": tolerance,
    }
    return _assemble(result, dim, "pce", options, parameters, indices, units)
