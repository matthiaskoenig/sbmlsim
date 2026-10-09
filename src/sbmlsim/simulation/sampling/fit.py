"""The designs of a fit: draws of what a fit knows about its parameters.

- `fit_parameters`: the multivariate normal of the Fisher covariance around
  the fitted values, in the space of the `parameter_scale` (correlated, a
  local approximation);
- `profile_parameters`: every parameter from its profile likelihood (follows
  asymmetric and open profiles, independent across parameters, since the
  profiles carry no joint information);
- `fit_repeats`: the best parameter sets of the repeats of a fit.

A fitted parameter is written to its `pid` unless `targets` maps it to another
target; two parameters of one target (the versions of a parameter) raise,
since a scan sets a target once per point.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import IdentifiabilityResult, ParameterProfile
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.simulation.sampling.designs import _count, _seed
from sbmlsim.simulation.scan import Design, Dimension
from sbmlsim.units import ureg

logger = logging.getLogger(__name__)


class _Repeats(Protocol):
    """What `fit_repeats` reads of a result of a fit."""

    def parameter_sets(self, size: int = 1) -> Iterable[ParameterSet]:
        """Get the best parameter sets."""
        ...


def _targets(pids: Sequence[str], targets: Mapping[str, str] | None) -> list[str]:
    """Get the target of every parameter.

    Raises:
        ValueError: if two parameters have one target.
    """
    names = [(targets or {}).get(pid, pid) for pid in pids]
    twice = sorted({t for t in names if names.count(t) > 1})
    if twice:
        raise ValueError(
            f"The fitted parameters set the targets {twice} more than once (the "
            f"versions of a parameter); a scan sets a target once per point, "
            f"map them to different targets with targets=."
        )
    return names


def _column(values: np.ndarray, unit: str | None) -> Any:
    """Give a column its unit, plain floats without one."""
    return ureg.Quantity(values, unit) if unit else values


def fit_parameters(
    fisher: FisherInformation,
    n: int,
    *,
    seed: int | None = None,
    targets: Mapping[str, str] | None = None,
    id: str = "fit",
) -> Dimension:
    """Draw the fitted parameters from the normal of their Fisher covariance.

    The draws are normal around the fitted values in the space of the
    `parameter_scale` with the covariance of the Fisher information, drawn
    with its eigendecomposition (the eigenvalues of a rank deficient
    covariance clipped at zero), and transformed back into the units of the
    model. A covariance which is rank deficient warns once, see
    `FisherInformation.covariance`.

    Args:
        fisher: the Fisher information of the fitted parameters.
        n: the number of points.
        seed: the seed; `None` draws one, which the record keeps.
        targets: pid -> the target of the model, the pid by default.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        ValueError: if `n` is no positive integer or two parameters have one
            target.
    """
    n = _count(n)
    names = _targets(fisher.pids, targets)
    seed = _seed(seed)
    mean = fisher.to_scale(fisher.values)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher.covariance)
    scale = np.sqrt(np.clip(eigenvalues, 0.0, None))
    rng = np.random.default_rng(seed)
    draws = mean + (rng.standard_normal((n, fisher.k)) * scale) @ eigenvectors.T
    values = np.array([fisher.from_scale(row) for row in draws])
    units = list(fisher.units) or [None] * fisher.k
    record = Design(
        method="fit_parameters",
        options={
            "n": n,
            "seed": seed,
            "opid": fisher.opid,
            "sid": fisher.sid,
            "alpha": fisher.alpha,
            "pids": list(fisher.pids),
            "scales": [str(s) for s in fisher.parameter_scales],
        },
        references={
            target: {"value": float(value), "unit": unit or ""}
            for target, value, unit in zip(names, fisher.values, units, strict=True)
        },
    )
    return Dimension(
        id,
        values={
            target: _column(values[:, k], units[k]) for k, target in enumerate(names)
        },
        design=record,
    )


def _density(
    profile: ParameterProfile,
    parameter: FitParameter | None,
    scale: ParameterScaleType,
    cost_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the points and the density of a profile in the space of its scale.

    The points which did not converge are dropped, and a side whose confidence
    bound is open is extended to the (finite) bound of the parameter with the
    density of its outermost point.

    Raises:
        ValueError: if the optimum did not converge or fewer than two points
            are left.
    """
    if not profile.converged[profile.index_optimum]:
        raise ValueError(
            f"The profile of '{profile.pid}' has no converged optimum, the "
            f"likelihood of the parameter cannot be drawn from it."
        )
    keep = profile.converged & np.isfinite(profile.costs)
    values = np.asarray(profile.values, dtype=float)[keep]
    density = np.exp(-(np.asarray(profile.costs, dtype=float)[keep] - cost_min))
    if len(values) < 2:
        raise ValueError(
            f"The profile of '{profile.pid}' has fewer than two converged points."
        )
    x = np.asarray(scale.to_scale(values), dtype=float)
    if parameter is not None:
        for side, bound, outer in (
            ("lower", parameter.lower_bound, values[0]),
            ("upper", parameter.upper_bound, values[-1]),
        ):
            open_side = (
                profile.ci_lower if side == "lower" else profile.ci_upper
            ) is None
            beyond = bound < outer if side == "lower" else bound > outer
            if not (open_side and np.isfinite(bound) and beyond):
                continue
            xb = float(scale.to_scale(np.array([bound]))[0])
            if not np.isfinite(xb):
                continue
            if side == "lower":
                x, density = np.r_[xb, x], np.r_[density[0], density]
            else:
                x, density = np.r_[x, xb], np.r_[density, density[-1]]
    return x, density


def profile_parameters(
    identifiability: IdentifiabilityResult,
    n: int,
    *,
    seed: int | None = None,
    targets: Mapping[str, str] | None = None,
    id: str = "profile",
) -> Dimension:
    """Draw every parameter from the likelihood of its profile.

    The density of a parameter is `exp(-(cost - cost_min))` along its profile,
    interpolated in the space of the `parameter_scale`, so asymmetric and open
    profiles are followed; a side which stays below the threshold reaches the
    bound of the parameter. The parameters are independent, since the profiles
    carry no joint information.

    Args:
        identifiability: the profile likelihood of a fit.
        n: the number of points.
        seed: the seed; `None` draws one, which the record keeps.
        targets: pid -> the target of the model, the pid by default.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        ValueError: if `n` is no positive integer, two parameters have one
            target or a profile has no converged optimum.
    """
    n = _count(n)
    pids = list(identifiability.profiles)
    names = _targets(pids, targets)
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    parameters = {p.pid: p for p in identifiability.parameters}
    columns: dict[str, Any] = {}
    for pid, target in zip(pids, names, strict=True):
        parameter = parameters.get(pid)
        scale = (
            parameter.scale
            if parameter is not None and parameter.scale is not None
            else identifiability.fit_settings.parameter_scale
        )
        x, density = _density(
            identifiability.profiles[pid], parameter, scale, identifiability.cost_min
        )
        steps = np.diff(x) * 0.5 * (density[1:] + density[:-1])
        cdf = np.r_[0.0, np.cumsum(steps)]
        cdf /= cdf[-1]
        draws = np.interp(rng.random(n), cdf, x)
        columns[target] = _column(
            np.asarray(scale.from_scale(draws), dtype=float),
            parameter.unit if parameter is not None else None,
        )
    values = identifiability.parameter_set.values
    record = Design(
        method="profile_parameters",
        options={
            "n": n,
            "seed": seed,
            "opid": identifiability.opid,
            "alpha": identifiability.settings.alpha,
            "pids": pids,
        },
        references={
            target: {
                "value": float(values[pid]),
                "unit": (parameters[pid].unit if pid in parameters else None) or "",
            }
            for pid, target in zip(pids, names, strict=True)
            if pid in values
        },
    )
    return Dimension(id, values=columns, design=record)


def fit_repeats(
    result: _Repeats,
    size: int,
    *,
    targets: Mapping[str, str] | None = None,
    id: str = "repeats",
) -> Dimension:
    """Take the best parameter sets of the repeats of a fit.

    Args:
        result: the result of a fit.
        size: the number of parameter sets.
        targets: pid -> the target of the model, the pid by default.
        id: the id of the dimension.

    Returns:
        The dimension, its points labelled with the ids of the sets.

    Raises:
        ValueError: if the fit has no parameter set or two parameters have one
            target.
    """
    sets = list(result.parameter_sets(size=size))
    if not sets:
        raise ValueError("The result of the fit has no parameter set.")
    pids = list(sets[0].values)
    names = _targets(pids, targets)
    columns = {
        target: _column(
            np.array([s.values[pid] for s in sets], dtype=float), sets[0].units.get(pid)
        )
        for pid, target in zip(pids, names, strict=True)
    }
    record = Design(
        method="fit_repeats",
        options={
            "size": size,
            "costs": [s.cost for s in sets],
            "sids": [s.sid for s in sets],
        },
    )
    return Dimension(id, values=columns, labels=[s.sid for s in sets], design=record)
