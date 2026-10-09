"""The designs of a scan: dimensions whose values follow a design.

Every design returns a `Dimension` of values with a `Design` record, which a
scan writes into the provenance of its result:

- `local`: the reference point and every target alone at `1 + delta` and
  `1 - delta` times its reference;
- `random`: independent draws, with a correlation a Gaussian copula;
- `lhs`: a Latin hypercube, with a correlation the rank reordering of Iman
  and Conover, which keeps one point per stratum;
- `sobol`, `fast`, `morris`: the designs of SALib for the sensitivity
  analyses.

A design draws points of the unit cube and maps them through the inverse CDF
of the distribution of every target, see
`sbmlsim.simulation.sampling.distributions`. A relative distribution reads the
reference of its target from the model, see `references`. Every random design
takes `seed`; `None` draws a seed, which the record keeps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.stats import qmc

from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.sampling.distributions import Distribution, Number
from sbmlsim.simulation.sampling.references import references
from sbmlsim.simulation.scan import Design, Dimension
from sbmlsim.simulator.simulator import ModelLike
from sbmlsim.units import Quantity, ureg


def _seed(seed: int | None) -> int:
    """Get the seed of a design; `None` draws one, which the record keeps.

    Raises:
        TypeError: if the seed is no integer.
    """
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"The seed of a design is an integer, not {seed!r}.")
    return int(seed)


def _count(n: int, name: str = "n") -> int:
    """Check a number of points.

    Raises:
        ValueError: if it is no positive integer.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        raise ValueError(f"'{name}' of a design is a positive integer, not {n!r}.")
    return int(n)


def _check(distributions: Mapping[str, Distribution]) -> None:
    """Check the distributions of a design.

    Raises:
        TypeError: if they are no mapping of targets to distributions.
        ValueError: if there is none.
    """
    if not isinstance(distributions, Mapping):
        raise TypeError(
            f"The distributions of a design map targets to distributions, not "
            f"{distributions!r}."
        )
    if not distributions:
        raise ValueError("A design needs at least one distribution.")
    for target, distribution in distributions.items():
        if not isinstance(distribution, Distribution):
            raise TypeError(f"'{target}': {distribution!r} is no distribution.")


def _resolve(
    distributions: Mapping[str, Distribution],
    model: ModelLike | None,
    simulation: Simulation | None,
) -> dict[str, Number]:
    """Read the references of the relative distributions.

    Raises:
        ValueError: if a distribution is relative and there is no model.
    """
    _check(distributions)
    relative = [t for t, d in distributions.items() if d.is_relative]
    if not relative:
        return {}
    if model is None:
        raise ValueError(
            f"The distributions of {relative} are relative to the references of "
            f"their targets: give the design the model (model=)."
        )
    return references(model, relative, simulation)


def _values(
    distributions: Mapping[str, Distribution],
    refs: Mapping[str, Number],
    u: np.ndarray,
) -> dict[str, Any]:
    """Map the points of the unit cube through the distributions, a column per target.

    Raises:
        ValueError: if a distribution does not fit its reference.
    """
    values: dict[str, Any] = {}
    for k, (target, distribution) in enumerate(distributions.items()):
        try:
            values[target] = distribution.ppf(u[:, k], refs.get(target))
        except ValueError as err:
            raise ValueError(f"'{target}': {err}") from err
    return values


def _encode_reference(value: Number) -> dict[str, Any]:
    """Encode a reference for the record."""
    if isinstance(value, Quantity):
        return {"value": float(value.magnitude), "unit": str(value.units)}
    return {"value": float(value), "unit": ""}


def _record(
    method: str,
    distributions: Mapping[str, Distribution],
    options: Mapping[str, Any],
    refs: Mapping[str, Number],
) -> Design:
    """Create the record of a design."""
    return Design(
        method=method,
        distributions={t: d.to_dict() for t, d in distributions.items()},
        options=dict(options),
        references={t: _encode_reference(v) for t, v in refs.items()},
    )


def _correlation(correlation: ArrayLike | None, d: int) -> np.ndarray | None:
    """Get the Cholesky factor of a correlation matrix, `None` without one.

    Raises:
        ValueError: if it is no symmetric, positive definite `d x d` matrix with
            ones on the diagonal.
    """
    if correlation is None:
        return None
    matrix = np.asarray(correlation, dtype=float)
    if (
        matrix.shape != (d, d)
        or not np.allclose(matrix, matrix.T)
        or not np.allclose(np.diag(matrix), 1.0)
    ):
        raise ValueError(
            f"The correlation of a design is a symmetric {d} x {d} matrix with ones "
            f"on the diagonal, not {matrix.tolist()}."
        )
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"The correlation {matrix.tolist()} of a design is not positive definite."
        ) from err


def _iman_conover(
    u: np.ndarray, factor: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Reorder the columns of a sample to the ranks of correlated scores.

    The scores are the van der Waerden scores `Φ⁻¹(i / (n + 1))` in random
    orders, freed of their sample correlation and given the target one; every
    column of the sample is sorted into their ranks, so its values, and the
    strata of a Latin hypercube, stay.

    Raises:
        ValueError: if the sample has not more points than columns.
    """
    n, d = u.shape
    if d == 1:
        return u
    if n <= d:
        raise ValueError(
            f"A correlated Latin hypercube of {d} targets needs more than {d} "
            f"points, it has {n}."
        )
    scores = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    s = np.column_stack([rng.permutation(scores) for _ in range(d)])
    q = np.linalg.cholesky(np.corrcoef(s, rowvar=False))
    target = s @ np.linalg.inv(q).T @ factor.T
    out = np.empty_like(u)
    for k in range(d):
        ranks = np.argsort(np.argsort(target[:, k]))
        out[:, k] = np.sort(u[:, k])[ranks]
    return out


def local(
    targets: Sequence[str],
    delta: float = 0.1,
    *,
    model: ModelLike,
    simulation: Simulation | None = None,
    id: str = "local",
) -> Dimension:
    """Get the local design: the reference and every target alone at `1 ± delta` times it.

    Args:
        targets: the targets, e.g. the parameters of `parameters_of`.
        delta: the relative change, `0 < delta < 1`.
        model: the model, whose references the design varies.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension of `2 k + 1` points, labelled `reference`, `<target>+`,
        `<target>-`.

    Raises:
        ValueError: if `delta` is not in `(0, 1)`, there is no target or a target
            is no target of the model.
    """
    if isinstance(targets, str):
        raise TypeError(
            f"The targets of a local design are a sequence, not {targets!r}."
        )
    names = list(dict.fromkeys(targets))
    if not names:
        raise ValueError("A local design needs at least one target.")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"The delta of a local design is in (0, 1), not {delta}.")
    refs = references(model, names, simulation)
    n = 2 * len(names) + 1
    values: dict[str, Any] = {}
    for j, target in enumerate(names):
        reference = refs[target]
        magnitude = float(getattr(reference, "magnitude", reference))
        column = np.full(n, magnitude)
        column[1 + 2 * j] = magnitude * (1.0 + delta)
        column[2 + 2 * j] = magnitude * (1.0 - delta)
        unit = str(reference.units) if isinstance(reference, Quantity) else None
        values[target] = ureg.Quantity(column, unit) if unit else column
    labels = ["reference", *(f"{t}{sign}" for t in names for sign in "+-")]
    record = _record("local", {}, {"delta": delta, "targets": names}, refs)
    return Dimension(id, values=values, labels=labels, design=record)


def random(
    distributions: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    correlation: ArrayLike | None = None,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "random",
) -> Dimension:
    """Get `n` random draws of the distributions.

    Without a correlation the points of the unit cube are `rng.random((n, d))`;
    with one they are standard normals with the Cholesky factor of the
    correlation, mapped through the normal CDF (a Gaussian copula), so the
    rank correlation of the values is the asked one for any marginals.

    Args:
        distributions: target -> its distribution.
        n: the number of points.
        seed: the seed; `None` draws one, which the record keeps.
        correlation: the correlation of the targets, in the order of
            `distributions`.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: if `n` is no positive integer, the correlation is not
            valid, or a relative distribution has no model.
    """
    n = _count(n)
    refs = _resolve(distributions, model, simulation)
    factor = _correlation(correlation, len(distributions))
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    if factor is None:
        u = rng.random((n, len(distributions)))
    else:
        u = stats.norm.cdf(rng.standard_normal((n, len(distributions))) @ factor.T)
    options = {
        "n": n,
        "seed": seed,
        "correlation": None
        if correlation is None
        else np.asarray(correlation, dtype=float).tolist(),
    }
    return Dimension(
        id,
        values=_values(distributions, refs, u),
        design=_record("random", distributions, options, refs),
    )


def lhs(
    distributions: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    correlation: ArrayLike | None = None,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "lhs",
) -> Dimension:
    """Get a Latin hypercube of `n` points of the distributions.

    The points of the unit cube are `qmc.LatinHypercube(d, rng=rng).random(n)`,
    one point per stratum of every target; with a correlation the columns are
    reordered by the method of Iman and Conover, which keeps the strata.

    Args:
        distributions: target -> its distribution.
        n: the number of points, more than the number of targets with a
            correlation.
        seed: the seed; `None` draws one, which the record keeps.
        correlation: the correlation of the targets, in the order of
            `distributions`.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: see `random`, and a correlated hypercube with too few points.
    """
    n = _count(n)
    refs = _resolve(distributions, model, simulation)
    factor = _correlation(correlation, len(distributions))
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    u = qmc.LatinHypercube(d=len(distributions), rng=rng).random(n=n)
    if factor is not None:
        u = _iman_conover(u, factor, rng)
    options = {
        "n": n,
        "seed": seed,
        "correlation": None
        if correlation is None
        else np.asarray(correlation, dtype=float).tolist(),
    }
    return Dimension(
        id,
        values=_values(distributions, refs, u),
        design=_record("lhs", distributions, options, refs),
    )
