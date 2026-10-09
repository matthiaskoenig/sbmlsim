"""The designs of a scan: dimensions whose values follow a design.

Every design returns a `Dimension` of values with a `Design` record, which a
scan writes into the provenance of its result:

- `local`: the reference point and every target alone at `1 + delta` and
  `1 - delta` times its reference;
- `random`: independent draws, with a correlation a Gaussian copula;
  `correlation` is the Spearman rank correlation of the values
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

from collections.abc import Callable, Mapping, Sequence
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

MAX_REDRAWS = 100
"""How often the orders of the scores of a correlated hypercube are drawn again."""


def _seed(seed: int | None) -> int:
    """Get the seed of a design; `None` draws one, which the record keeps.

    Raises:
        TypeError: if the seed is no integer.
        ValueError: if the seed is negative.
    """
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1)[0])
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"The seed of a design is an integer, not {seed!r}.")
    if seed < 0:
        raise ValueError(f"The seed of a design is not negative, not {seed}.")
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
    """Get the Cholesky factor of the normal-score correlation of a rank correlation.

    The asked matrix is the Spearman rank correlation of the values. The
    rank correlation of normal scores with the correlation `r` is
    `(6 / pi) arcsin(r / 2)`, so the scores get `2 sin(pi rho / 6)`.

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
    converted = 2.0 * np.sin(np.pi * matrix / 6.0)
    np.fill_diagonal(converted, 1.0)
    try:
        return np.linalg.cholesky(converted)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"The correlation {matrix.tolist()} of a design is not positive "
            f"definite once converted to the correlation of normal scores."
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
        ValueError: if the sample has not more points than columns, or the
            scores are collinear in every draw.
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
    for _ in range(MAX_REDRAWS):
        s = np.column_stack([rng.permutation(scores) for _ in range(d)])
        try:
            q = np.linalg.cholesky(np.corrcoef(s, rowvar=False))
            break
        except np.linalg.LinAlgError:
            continue
    else:
        raise ValueError(
            f"{n} points are too few for a correlated Latin hypercube of {d} "
            f"targets: the scores stay collinear after {MAX_REDRAWS} draws."
        )
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


def _drawn(
    method: str,
    distributions: Mapping[str, Distribution],
    n: int,
    seed: int | None,
    correlation: ArrayLike | None,
    model: ModelLike | None,
    simulation: Simulation | None,
    id: str,
    draw: Callable[[np.random.Generator, int, int, np.ndarray | None], np.ndarray],
) -> Dimension:
    """Get a random design: check the arguments, draw the unit cube, map it.

    The cheap arguments are checked before the references are read, which may
    load the model.

    Args:
        method: the name of the design.
        distributions: target -> its distribution.
        n: the number of points.
        seed: the seed, `None` draws one.
        correlation: the asked rank correlation.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.
        draw: draws the points of the unit cube from the generator, given the
            number of points, the number of targets and the Cholesky factor.

    Returns:
        The dimension.
    """
    _check(distributions)
    n = _count(n)
    d = len(distributions)
    factor = _correlation(correlation, d)
    seed = _seed(seed)
    refs = _resolve(distributions, model, simulation)
    u = draw(np.random.default_rng(seed), n, d, factor)
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
        design=_record(method, distributions, options, refs),
    )


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
    correlation of normal scores which has the asked rank correlation, mapped
    through the normal CDF (a Gaussian copula), so the Spearman rank
    correlation of the values is the asked one for any marginals.

    Args:
        distributions: target -> its distribution.
        n: the number of points.
        seed: the seed, not negative; `None` draws one, which the record keeps.
        correlation: the Spearman rank correlation of the values of the
            targets, in the order of `distributions`.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: if `n` is no positive integer, the seed is negative, the
            correlation is not valid, or a relative distribution has no model.
    """

    def draw(
        rng: np.random.Generator, n: int, d: int, factor: np.ndarray | None
    ) -> np.ndarray:
        if factor is None:
            return rng.random((n, d))
        return stats.norm.cdf(rng.standard_normal((n, d)) @ factor.T)

    return _drawn(
        "random", distributions, n, seed, correlation, model, simulation, id, draw
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
    reordered by the method of Iman and Conover, which keeps the strata, so
    the Spearman rank correlation of the values is the asked one.

    Args:
        distributions: target -> its distribution.
        n: the number of points, more than the number of targets with a
            correlation.
        seed: the seed, not negative; `None` draws one, which the record keeps.
        correlation: the Spearman rank correlation of the values of the
            targets, in the order of `distributions`.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: see `random`, and a correlated hypercube with too few points.
    """

    def draw(
        rng: np.random.Generator, n: int, d: int, factor: np.ndarray | None
    ) -> np.ndarray:
        u = qmc.LatinHypercube(d=d, rng=rng).random(n=n)
        return u if factor is None else _iman_conover(u, factor, rng)

    return _drawn(
        "lhs", distributions, n, seed, correlation, model, simulation, id, draw
    )


def _problem(d: int) -> dict[str, Any]:
    """Get the problem of SALib on the unit cube of `d` dimensions."""
    return {
        "num_vars": d,
        "names": [f"x{k}" for k in range(d)],
        "bounds": [[0.0, 1.0]] * d,
    }


def unit_cube(design: Design, d: int) -> np.ndarray:
    """Get the points of the unit cube of a recorded design of SALib.

    The points follow from the method, the options and the seed of the
    record, so a result of the design needs not store them; phase 2 creates
    them again for the analysis.

    Args:
        design: the record of a `sobol`, `fast` or `morris` design.
        d: the number of targets.

    Returns:
        The points, a row per point of the dimension.

    Raises:
        ValueError: if the record is of another method.
    """
    options = design.options
    if design.method == "sobol":
        from SALib.sample import sobol as sobol_sampler

        return sobol_sampler.sample(
            _problem(d),
            options["n"],
            calc_second_order=options["second_order"],
            scramble=True,
            seed=options["seed"],
        )
    if design.method == "fast":
        from SALib.sample import fast_sampler

        return fast_sampler.sample(
            _problem(d), options["n"], M=options["m"], seed=options["seed"]
        )
    if design.method == "morris":
        from SALib.sample import morris as morris_sampler

        levels = options["levels"]
        grid = morris_sampler.sample(
            _problem(d),
            options["trajectories"],
            num_levels=levels,
            seed=options["seed"],
        )
        # the centre of the stratum of every level, so an unbounded marginal is finite
        return (grid * (levels - 1) + 0.5) / levels
    raise ValueError(f"The design '{design.method}' is no design of SALib.")


def _salib(
    method: str,
    distributions: Mapping[str, Distribution],
    options: dict[str, Any],
    model: ModelLike | None,
    simulation: Simulation | None,
    id: str,
) -> Dimension:
    """Create a design of SALib: its unit cube mapped through the distributions."""
    refs = _resolve(distributions, model, simulation)
    record = _record(method, distributions, options, refs)
    u = unit_cube(record, len(distributions))
    return Dimension(id, values=_values(distributions, refs, u), design=record)


def sobol(
    distributions: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    second_order: bool = False,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "sobol",
) -> Dimension:
    """Get the design of Saltelli for the Sobol indices.

    `SALib.sample.sobol.sample` on the unit cube, scrambled, mapped through the
    distributions: `n (d + 2)` points, `n (2 d + 2)` with second order. The
    targets are independent, a correlation is not taken.

    Args:
        distributions: target -> its distribution.
        n: the base number of points, a power of two.
        seed: the seed, not negative; `None` draws one, which the record keeps.
        second_order: include the points of the second order indices.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: if `n` is no power of two, or see `random`.
    """
    _check(distributions)
    n = _count(n)
    if n < 2 or n & (n - 1):
        raise ValueError(f"The n of a Sobol design is a power of two, not {n}.")
    options = {"n": n, "seed": _seed(seed), "second_order": bool(second_order)}
    return _salib("sobol", distributions, options, model, simulation, id)


def fast(
    distributions: Mapping[str, Distribution],
    n: int,
    *,
    m: int = 4,
    seed: int | None = None,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "fast",
) -> Dimension:
    """Get the design of the extended FAST.

    `SALib.sample.fast_sampler.sample` on the unit cube, mapped through the
    distributions: `n d` points. The targets are independent, a correlation is
    not taken.

    Args:
        distributions: target -> its distribution.
        n: the number of points per target, more than `4 m^2`.
        m: the interference factor of SALib.
        seed: the seed, not negative; `None` draws one, which the record keeps.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: if `n` is not more than `4 m^2`, or see `random`.
    """
    _check(distributions)
    n = _count(n)
    m = _count(m, "m")
    if n <= 4 * m * m:
        raise ValueError(
            f"The n of a FAST design is more than 4 m^2 = {4 * m * m}, not {n}."
        )
    options = {"n": n, "seed": _seed(seed), "m": m}
    return _salib("fast", distributions, options, model, simulation, id)


def morris(
    distributions: Mapping[str, Distribution],
    trajectories: int,
    *,
    levels: int = 4,
    seed: int | None = None,
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "morris",
) -> Dimension:
    """Get the design of Morris for the elementary effects.

    `SALib.sample.morris.sample` on the unit cube: `trajectories (d + 1)`
    points. The level `k` of the grid of SALib (`0` to `1` in `levels - 1`
    steps) is mapped to `(k + 0.5) / levels`, the centre of the stratum of
    that level, so an unbounded marginal stays finite. The targets are
    independent, a correlation is not taken.

    Args:
        distributions: target -> its distribution.
        trajectories: the number of trajectories.
        levels: the number of levels of the grid, at least two.
        seed: the seed, not negative; `None` draws one, which the record keeps.
        model: the model, which a relative distribution needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        TypeError: if the distributions or the seed have the wrong type.
        ValueError: if `levels` is below two, or see `random`.
    """
    _check(distributions)
    trajectories = _count(trajectories, "trajectories")
    levels = _count(levels, "levels")
    if levels < 2:
        raise ValueError(f"The levels of a Morris design are at least 2, not {levels}.")
    options = {"trajectories": trajectories, "seed": _seed(seed), "levels": levels}
    return _salib("morris", distributions, options, model, simulation, id)
