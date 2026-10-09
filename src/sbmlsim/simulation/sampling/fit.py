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

    parameters: list[FitParameter]

    def parameter_sets(self, size: int = 1) -> Iterable[ParameterSet]:
        """Get the best parameter sets."""
        ...


def _targets(
    pids: Sequence[str],
    defaults: Sequence[str] | None,
    targets: Mapping[str, str] | None,
) -> list[str]:
    """Get the target of every parameter.

    Args:
        pids: the parameters.
        defaults: the target of every parameter (`FitParameter.target_id`), the
            pids if `None`.
        targets: pid -> target, overrides the defaults.

    Raises:
        ValueError: if a key of `targets` is no parameter or two parameters
            have one target.
    """
    unknown = sorted(set(targets or {}) - set(pids))
    if unknown:
        raise ValueError(
            f"The targets {unknown} are no fitted parameters, the parameters are "
            f"{list(pids)}."
        )
    names = [
        (targets or {}).get(pid, default)
        for pid, default in zip(pids, defaults or pids, strict=True)
    ]
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
    `parameter_scale` with the covariance of the Fisher information, drawn in
    the directions the information constrains only (the eigenvalues of
    `J'J` above the rank tolerance, with the variance `sigma2 / eigenvalue`),
    so a direction the data does not constrain has no spread, and transformed
    back into the units of the model. An information which is rank deficient
    warns once, see `FisherInformation.covariance`. The target of a parameter
    is the one of the information (`FitParameter.target_id`) unless `targets`
    maps it.

    Args:
        fisher: the Fisher information of the fitted parameters.
        n: the number of points.
        seed: the seed; `None` draws one, which the record keeps.
        targets: pid -> the target of the model, overrides `FitParameter.target_id`.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        ValueError: if `n` is no positive integer, the information has no
            degrees of freedom or constrains no direction, the draws are not
            finite in the units of the model, a key of `targets` is no
            parameter or two parameters have one target.
    """
    n = _count(n)
    if fisher.n <= fisher.k:
        raise ValueError(
            f"The Fisher information of '{fisher.opid}' has no degrees of freedom "
            f"(n={fisher.n} data points, k={fisher.k} parameters), its covariance "
            f"is not defined."
        )
    names = _targets(fisher.pids, fisher.targets or None, targets)
    seed = _seed(seed)
    if not fisher.is_identifiable:
        # logs the warning about the unconstrained directions once
        _ = fisher.covariance
    eigenvalues, eigenvectors = np.linalg.eigh(fisher.matrix)
    constrained = eigenvalues > eigenvalues.max() * fisher.rank_tolerance
    if not constrained.any():
        raise ValueError(
            f"The Fisher information of '{fisher.opid}' constrains no direction "
            f"of the parameters {list(fisher.pids)}."
        )
    scale = np.sqrt(fisher.sigma2 / eigenvalues[constrained])
    basis = eigenvectors[:, constrained]
    mean = fisher.to_scale(fisher.values)
    rng = np.random.default_rng(seed)
    draws = mean + (rng.standard_normal((n, len(scale))) * scale) @ basis.T
    values = np.column_stack(
        [
            np.asarray(s.from_scale(draws[:, j]), dtype=float)
            for j, s in enumerate(fisher.parameter_scales)
        ]
    )
    bad = [
        pid for j, pid in enumerate(fisher.pids) if not np.isfinite(values[:, j]).all()
    ]
    if bad:
        raise ValueError(
            f"The draws of the parameters {bad} of '{fisher.opid}' are not finite "
            f"in the units of the model; the Fisher information does not describe "
            f"them."
        )
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
            "scales": [s.name for s in fisher.parameter_scales],
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


TAIL_RISE = 8.0
TAIL_POINTS = 20


def _density(
    profile: ParameterProfile,
    parameter: FitParameter | None,
    scale: ParameterScaleType,
    cost_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the points and the density of a profile in the space of its scale.

    The points which did not converge are dropped. A profile stops at the
    first point at or above the threshold, so a closed side (one with a
    confidence bound, whose outermost point is not at the bound of the
    parameter) is extended by its quadratic tail: the cost beyond the outermost
    point follows `cost_min + b (x - x_opt)^2` through the optimum and that
    point, out to a rise of `TAIL_RISE` or to the bound of the parameter (a side
    which reaches the bound gets no tail). An
    open side is extended to the (finite) bound with the density of its
    outermost point.

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
    i_opt = int(np.cumsum(keep)[profile.index_optimum]) - 1
    values = np.asarray(profile.values, dtype=float)[keep]
    costs = np.asarray(profile.costs, dtype=float)[keep]
    if len(values) < 2:
        raise ValueError(
            f"The profile of '{profile.pid}' has fewer than two converged points."
        )
    x = np.asarray(scale.to_scale(values), dtype=float)
    density = np.exp(-(costs - cost_min))
    x_opt, c_opt = x[i_opt], costs[i_opt]
    for lower in (True, False):
        bound = np.nan
        if parameter is not None:
            bound = parameter.lower_bound if lower else parameter.upper_bound
        outer = values[0] if lower else values[-1]
        beyond = bool(bound < outer) if lower else bool(bound > outer)
        if np.isfinite(bound) and not beyond:
            # the profile reaches the bound: nothing to extend past it
            continue
        xb = np.nan
        if np.isfinite(bound) and beyond:
            xb = float(scale.to_scale(np.array([bound]))[0])
        ci = profile.ci_lower if lower else profile.ci_upper
        if ci is None:
            if not np.isfinite(xb):
                continue
            new_x = np.array([xb])
            new_density = np.array([density[0] if lower else density[-1]])
        else:
            x_o, c_o = (x[0], costs[0]) if lower else (x[-1], costs[-1])
            b = (c_o - c_opt) / (x_o - x_opt) ** 2 if x_o != x_opt else 0.0
            if b <= 0.0 or c_o - c_opt >= TAIL_RISE:
                continue
            reach = float(np.sqrt(TAIL_RISE / b))
            x_end = x_opt - reach if lower else x_opt + reach
            if np.isfinite(xb):
                x_end = max(x_end, xb) if lower else min(x_end, xb)
            if (x_end >= x_o) if lower else (x_end <= x_o):
                continue
            new_x = np.linspace(x_o, x_end, TAIL_POINTS + 1)[1:]
            new_density = np.exp(-(c_opt + b * (new_x - x_opt) ** 2 - cost_min))
        if lower:
            x, density = np.r_[new_x[::-1], x], np.r_[new_density[::-1], density]
        else:
            x, density = np.r_[x, new_x], np.r_[density, new_density]
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
    profiles are followed. A side which stays below the threshold reaches the
    bound of the parameter, a closed side (the profile stops at the threshold)
    is continued by its quadratic tail, see `TAIL_RISE`. The parameters are independent, since the profiles
    carry no joint information.

    Args:
        identifiability: the profile likelihood of a fit.
        n: the number of points.
        seed: the seed; `None` draws one, which the record keeps.
        targets: pid -> the target of the model, overrides `FitParameter.target_id`.
        id: the id of the dimension.

    Returns:
        The dimension.

    Raises:
        ValueError: if `n` is no positive integer, two parameters have one
            target or a profile has no converged optimum.
    """
    n = _count(n)
    pids = list(identifiability.profiles)
    parameters = {p.pid: p for p in identifiability.parameters}
    defaults = [parameters[p].target_id if p in parameters else p for p in pids]
    names = _targets(pids, defaults, targets)
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
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
        targets: pid -> the target of the model, overrides `FitParameter.target_id`.
        id: the id of the dimension.

    Returns:
        The dimension, its points labelled with the ids of the sets.

    Raises:
        ValueError: if `size` is no positive integer, the fit has no parameter
            set, a key of `targets` is no parameter or two parameters have one
            target.
    """
    size = _count(size, "size")
    sets = list(result.parameter_sets(size=size))
    if not sets:
        raise ValueError("The result of the fit has no parameter set.")
    pids = list(sets[0].values)
    fitted = {p.pid: p.target_id for p in result.parameters}
    names = _targets(pids, [fitted.get(pid, pid) for pid in pids], targets)
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
