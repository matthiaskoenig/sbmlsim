"""Parameter identifiability by profile likelihood.

A fit gives the parameters which describe the data best. Identifiability asks
how well the data determines every one of them: a parameter is identifiable if
the data constrains it to a finite interval, and non-identifiable if it can be
changed without the model describing the data worse.

The profile likelihood answers this by scanning every parameter around the
optimum. The scanned parameter is fixed at a value and all other parameters are
optimized again, so the profile is the best cost the model reaches with the
parameter at that value (Raue et al. 2009):

    PL(θi) = min over θj≠i of cost(θ)

The cost of `sbmlsim` is the cost of `scipy.optimize.least_squares`, i.e., half
the sum of the squared weighted residuals, `cost = 0.5 * Σ r²`. With residuals
which are standardized by the errors of the data, `2 * cost` is the negative
log-likelihood up to a constant, and the profile is compared with the threshold
of the likelihood ratio test (Raue et al. 2009, Kreutz et al. 2013):

    cost_threshold = cost_min + chi2.ppf(alpha, df) / 2

with `df = 1` for the pointwise confidence intervals of single parameters, i.e.,
a threshold of `1.92` above the minimal cost at 95% confidence, and
`df = #parameters` for simultaneous intervals. The values of a parameter at
which the profile crosses the threshold are the bounds of its confidence
interval. The intervals are invariant under a transformation of the parameters
and may be asymmetric, which is where the intervals of the Fisher information
matrix fail for the non-linear models of systems biology (Wieland et al. 2021).

The shape of the profile classifies the parameter (Raue et al. 2009):

- identifiable: the profile crosses the threshold on both sides of the optimum,
  the confidence interval is finite,
- practically non-identifiable: the profile has a minimum but stays below the
  threshold on one or both sides, i.e., up to the bound of the parameter. The
  data does not determine the parameter towards small and/or large values,
- structurally non-identifiable: the profile is flat over the whole scanned
  range, the parameter is compensated by the other parameters and the data
  carries no information about it.

The scans run in logarithmic parameter space, as the optimization does. Every
scan starts at the scale of its parameter, `sqrt(2 max_rise)` times the
standard error of the Fisher information of the optimum (`first_steps`), so
the first step is the one the profile allows and not one step for every
parameter. The step along a profile is adaptive (Schälte et al. 2023): a step
which raises the cost by more than a fraction of the threshold is reduced and
repeated, a step which raises it by little is enlarged by `step_factor`, so
the profile is resolved where it changes. A step is reduced by the curvature
its own rise gives, `step * sqrt(max_rise / (2 rise))`, which is the step of
half the allowed rise where the profile is quadratic: the correction reaches
the scale of the profile at once, where a fixed factor needs several
corrections and discards the re-optimization which measured each of them
(#647). The other parameters start from the previous point of the profile
(Simpson & Maclaren 2023) and their paths are stored, so a parameter which is
coupled to the scanned one is seen in its path (Maiwald et al. 2016). A scan
stops when the profile crosses the threshold, at the bound of the parameter or
after `max_points`; a bound which is reached below the threshold is an open
interval (Borisov & Metelkin 2020). A scan which finds a lower cost than the
optimum reports it: the fit did not converge, and the threshold is taken
relative to the lowest cost of all profiles.

The scans are independent, so they run in the worker pool of the fit runner,
two scans per parameter. An analysis of fewer scans than one re-optimization
spreads over, e.g. a profile of a single parameter, runs its scans one after
another instead and evaluates the points of the jacobian of every
re-optimization on all workers, see `sbmlsim.fit.runner.split_cores`. A
parallel analysis needs the same `if __name__ == "__main__":` guard as a
parallel fit.

References:
    Raue A, Kreutz C, Maiwald T, Bachmann J, Schilling M, Klingmüller U,
    Timmer J. Structural and practical identifiability analysis of partially
    observed dynamical models by exploiting the profile likelihood.
    Bioinformatics. 2009;25(15):1923-1929. doi:10.1093/bioinformatics/btp358

    Kreutz C, Raue A, Kaschek D, Timmer J. Profile likelihood in systems
    biology. FEBS J. 2013;280(11):2564-2571. doi:10.1111/febs.12276

    Maiwald T, Hass H, Steiert B, Vanlier J, Engesser R, Raue A, Kipkeew F,
    Bock HH, Kaschek D, Kreutz C, Timmer J. Driving the model to its limit:
    profile likelihood based model reduction. PLoS ONE. 2016;11(9):e0162366.
    doi:10.1371/journal.pone.0162366

    Wieland FG, Hauber AL, Rosenblatt M, Tönsing C, Timmer J. On structural
    and practical identifiability. Curr Opin Syst Biol. 2021;25:60-69.
    doi:10.1016/j.coisb.2021.03.005

    Borisov I, Metelkin E. Confidence intervals by constrained optimization,
    an algorithm and software package for practical identifiability analysis
    in systems biology. PLoS Comput Biol. 2020;16(12):e1008495.
    doi:10.1371/journal.pcbi.1008495

    Simpson MJ, Maclaren OJ. Profile-wise analysis: a profile likelihood-based
    workflow for identifiability analysis, estimation, and prediction with
    mechanistic mathematical models. PLoS Comput Biol. 2023;19(9):e1011515.
    doi:10.1371/journal.pcbi.1011515

    Schälte Y, Fröhlich F, Jost PJ, Vanhoefer J, Pathirana D, Stapor P,
    Lakrisenko P, Wang D, Raimúndez E, Merkt S, Schmiester L, Städter P,
    Grein S, Dudkin E, Doresic D, Weindl D, Hasenauer J. pyPESTO: a modular
    and scalable tool for parameter estimation for dynamic models.
    Bioinformatics. 2023;39(11):btad711. doi:10.1093/bioinformatics/btad711
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import chi2

from sbmlsim.engine import Engine
from sbmlsim.events import emit
from sbmlsim.fit import display, fit_core, runner
from sbmlsim.fit.derived import group_parameters, hook_summaries
from sbmlsim.fit.fisher import fisher_information
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.optimization import (
    FitTimeout,
    OptimizationProblem,
    PoolError,
    ResidualsEvaluator,
)
from sbmlsim.fit.options import (
    RUN_IN_PYTHON,
    RUN_IN_THE_ENGINE,
    FitSettings,
    OptimizationAlgorithmType,
    ParameterScaleType,
    run_place,
)
from sbmlsim.fit.parameters import ParameterSet, dump_json, load_json
from sbmlsim.progress import activity as progress_activity

if TYPE_CHECKING:
    # the chart is imported when a profile is drawn, not with the analysis
    from sbmlsim.plot.chart import Chart, ChartAxis, ChartPlot

logger = logging.getLogger(__name__)

#: numerical tolerance in logarithmic parameter space, a parameter closer to
#: its bound than this is at the bound
BOUND_TOLERANCE = 1e-9


class Identifiability(StrEnum):
    """Identifiability of a parameter, read from the shape of its profile.

    The classification follows Raue et al. 2009: a parameter is identifiable
    if its profile crosses the threshold on both sides of the optimum,
    practically non-identifiable if the profile stays below the threshold up to
    a bound of the parameter, on one or both sides, and structurally
    non-identifiable if the profile is flat on both sides.
    """

    IDENTIFIABLE = "identifiable"
    NON_IDENTIFIABLE_LOWER = "non_identifiable_lower"
    NON_IDENTIFIABLE_UPPER = "non_identifiable_upper"
    NON_IDENTIFIABLE = "non_identifiable"
    STRUCTURAL = "structural"

    @property
    def label(self) -> str:
        """Get the description of the identifiability."""
        return LABELS[self]

    @property
    def is_identifiable(self) -> bool:
        """Check whether the parameter is identifiable."""
        return self is Identifiability.IDENTIFIABLE


#: description of every identifiability
LABELS: dict[Identifiability, str] = {
    Identifiability.IDENTIFIABLE: "identifiable",
    Identifiability.NON_IDENTIFIABLE_LOWER: "non-identifiable (lower)",
    Identifiability.NON_IDENTIFIABLE_UPPER: "non-identifiable (upper)",
    Identifiability.NON_IDENTIFIABLE: "non-identifiable",
    Identifiability.STRUCTURAL: "structurally non-identifiable",
}


@dataclass(frozen=True)
class ProfileSettings:
    """Settings of a profile likelihood analysis.

    The steps are in logarithmic parameter space, i.e., in decades of the
    parameter, because the scans run in the space the optimization runs in.

    Attributes:
        alpha: confidence level of the intervals.
        degrees_of_freedom: degrees of freedom of the chi-square threshold,
            `1` for pointwise confidence intervals of single parameters, the
            number of parameters for simultaneous intervals.
        initial_step: first step of a scan in decades of the parameter, where
            the analysis has no scale of it: a scan starts at the scale of its
            parameter where the Fisher information of the optimum gives one,
            see `first_steps`.
        min_step: smallest step in decades, a step is not reduced below it.
        max_step: largest step in decades.
        step_factor: factor a step which rises slowly is enlarged by; a step
            which rises too fast is reduced by the curvature of the profile,
            see `_scan_direction`.
        max_cost_fraction: largest increase of the cost a single step may
            cause, as a fraction of the distance between the minimal cost and
            the threshold. A step with a larger increase is reduced and
            repeated, so a profile has at least `1 / max_cost_fraction` points
            between the optimum and the threshold.
        max_points: largest number of points of a scan in one direction.
        flatness: a profile which rises by less than this fraction of the
            distance between the minimal cost and the threshold, on both sides,
            is flat, i.e., the parameter is structurally non-identifiable.
        reoptimize: optimize the other parameters at every point of a scan,
            which is the profile likelihood. Without it the other parameters
            stay at the optimum, which is a plain scan of the cost and a lower
            bound of the profile: it is faster and finds the non-identifiable
            parameters, but its intervals are too narrow for coupled
            parameters.
        optimizer_kwargs: arguments of `scipy.optimize.least_squares` for the
            optimization of the other parameters, without `loss` and
            `f_scale`: the loss is the one of the fit,
            `FitSettings.loss_function`. The step of the finite differences
            is the one of a fit, `OptimizationProblem.optimizer_arguments`,
            unless `diff_step` is given.
        scan_timeout: seconds without any scan finishing after which the scans
            which run or wait are given up, `None` for no limit. It bounds a
            scan which does not return, which stopped a parallel analysis with
            no message and no result at all (#522); the parameters of the
            scans which were given up have no profile and are named in
            `incomplete`, and the analysis writes the profiles it has.
        timeout: seconds after which the whole analysis is given up, `None`
            for no limit, so that an overnight run is bounded and writes what
            it has. It is the budget of the scans and, in a serial analysis,
            is checked before every scan.
    """

    alpha: float = 0.95
    degrees_of_freedom: int = 1
    initial_step: float = 0.1
    min_step: float = 0.01
    max_step: float = 1.0
    step_factor: float = 2.0
    max_cost_fraction: float = 0.2
    max_points: int = 50
    flatness: float = 0.05
    reoptimize: bool = True
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    scan_timeout: float | None = None
    timeout: float | None = None

    def __post_init__(self) -> None:
        """Check the settings.

        Raises:
            ValueError: if a setting is out of its range.
        """
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"'alpha' must be in (0, 1), got '{self.alpha}'.")
        if self.degrees_of_freedom < 1:
            raise ValueError(
                f"'degrees_of_freedom' must be at least 1, got "
                f"'{self.degrees_of_freedom}'."
            )
        if not 0.0 < self.min_step <= self.initial_step <= self.max_step:
            raise ValueError(
                f"the steps must satisfy 0 < min_step <= initial_step <= max_step, "
                f"got '{self.min_step}', '{self.initial_step}', '{self.max_step}'."
            )
        if self.step_factor <= 1.0:
            raise ValueError(
                f"'step_factor' must be larger than 1, got '{self.step_factor}'."
            )
        if not 0.0 < self.max_cost_fraction <= 1.0:
            raise ValueError(
                f"'max_cost_fraction' must be in (0, 1], got "
                f"'{self.max_cost_fraction}'."
            )
        for name in ("scan_timeout", "timeout"):
            seconds = getattr(self, name)
            if seconds is not None and seconds <= 0.0:
                raise ValueError(f"'{name}' must be positive or None, got '{seconds}'.")
        if self.max_points < 1:
            raise ValueError(
                f"'max_points' must be at least 1, got '{self.max_points}'."
            )
        if not 0.0 <= self.flatness < 1.0:
            raise ValueError(f"'flatness' must be in [0, 1), got '{self.flatness}'.")
        object.__setattr__(self, "optimizer_kwargs", dict(self.optimizer_kwargs))

    @property
    def delta(self) -> float:
        """Get the chi-square quantile of the confidence level.

        This is the threshold on `-2 log L`, i.e., on twice the cost.
        """
        return float(chi2.ppf(self.alpha, df=self.degrees_of_freedom))

    def threshold(self, cost: float) -> float:
        """Get the threshold on the cost for a minimal cost.

        Args:
            cost: minimal cost.

        Returns:
            The cost at which the profile crosses the confidence level,
            `cost + delta / 2` for the cost convention `0.5 * Σ r²`.
        """
        return cost + self.delta / 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "alpha": self.alpha,
            "degrees_of_freedom": self.degrees_of_freedom,
            "initial_step": self.initial_step,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "step_factor": self.step_factor,
            "max_cost_fraction": self.max_cost_fraction,
            "max_points": self.max_points,
            "flatness": self.flatness,
            "reoptimize": self.reoptimize,
            "optimizer_kwargs": dict(self.optimizer_kwargs),
            "scan_timeout": self.scan_timeout,
            "timeout": self.timeout,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> ProfileSettings:
        """Create settings from a dictionary, i.e., from the stored JSON.

        A record written before the limits of #522 has neither, which is no
        limit, the behaviour it ran with.
        """
        return ProfileSettings(**d)


def cost_threshold(cost: float, alpha: float = 0.95, df: int = 1) -> float:
    """Get the threshold on the cost of a likelihood ratio test.

    The cost is `0.5 * Σ r²`, so the threshold is half the chi-square quantile
    above the minimal cost; at 95% and one degree of freedom it is `1.92`.

    Args:
        cost: minimal cost.
        alpha: confidence level.
        df: degrees of freedom.

    Returns:
        The threshold on the cost.
    """
    return cost + float(chi2.ppf(alpha, df=df)) / 2.0


@dataclass
class ParameterProfile:
    """Profile likelihood of a single parameter.

    The points are sorted by the value of the parameter, the optimum is one of
    them. Every point carries the full parameter vector the cost was reached
    with, so the paths of the other parameters along the profile are known.

    Attributes:
        pid: id of the profiled parameter.
        values: values of the parameter at the points, ascending.
        costs: cost at every point.
        paths: values of all parameters at every point, one row per point in
            the order of the parameters of the problem.
        converged: whether the optimization of the other parameters converged
            at every point.
        index_optimum: index of the optimum in the points.
        ci_lower: lower bound of the confidence interval, `None` if the
            profile stays below the threshold down to the bound.
        ci_upper: upper bound of the confidence interval, `None` if the
            profile stays below the threshold up to the bound.
        identifiability: classification of the parameter, set by `evaluate`,
            `None` if a point failed or did not converge.
    """

    pid: str
    values: np.ndarray
    costs: np.ndarray
    paths: np.ndarray
    converged: np.ndarray
    index_optimum: int
    ci_lower: float | None = None
    ci_upper: float | None = None
    identifiability: Identifiability | None = None

    def __post_init__(self) -> None:
        """Normalize the arrays.

        Raises:
            ValueError: if the points are inconsistent.
        """
        self.values = np.asarray(self.values, dtype=float)
        self.costs = np.asarray(self.costs, dtype=float)
        self.paths = np.asarray(self.paths, dtype=float)
        self.converged = np.asarray(self.converged, dtype=bool)
        n = len(self.values)
        if n == 0:
            raise ValueError(f"'{self.pid}': a profile needs at least one point.")
        if len(self.costs) != n or len(self.converged) != n or len(self.paths) != n:
            raise ValueError(
                f"'{self.pid}': the points of the profile are inconsistent, "
                f"'{n}' values, '{len(self.costs)}' costs, '{len(self.paths)}' "
                f"paths and '{len(self.converged)}' flags."
            )
        if not 0 <= self.index_optimum < n:
            raise ValueError(
                f"'{self.pid}': the optimum '{self.index_optimum}' is not one of "
                f"the '{n}' points."
            )
        if np.any(np.diff(self.values) < 0):
            raise ValueError(f"'{self.pid}': the values must be ascending.")

    def __len__(self) -> int:
        """Get the number of points."""
        return len(self.values)

    @property
    def value_optimum(self) -> float:
        """Get the value of the parameter at the optimum."""
        return float(self.values[self.index_optimum])

    @property
    def cost_optimum(self) -> float:
        """Get the cost at the optimum."""
        return float(self.costs[self.index_optimum])

    @property
    def cost_min(self) -> float:
        """Get the lowest cost of the profile, of the converged points."""
        costs = self.costs[self.converged & np.isfinite(self.costs)]
        return float(np.min(costs)) if len(costs) else float("inf")

    def side(self, direction: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the points on one side of the optimum, from the optimum outward.

        Args:
            direction: `-1` for the points below the optimum, `+1` above.

        Returns:
            The values and the costs, the optimum first.
        """
        k = self.index_optimum
        if direction < 0:
            return self.values[: k + 1][::-1], self.costs[: k + 1][::-1]
        return self.values[k:], self.costs[k:]

    def crossing(
        self,
        threshold: float,
        direction: int,
        scale: ParameterScaleType,
    ) -> float | None:
        """Get the value at which the profile crosses the threshold.

        The crossing is interpolated between the last point below and the
        first point above the threshold, in the scale of the parameter, i.e.
        the scale the scan ran on.

        Args:
            threshold: threshold on the cost.
            direction: `-1` for the crossing below the optimum, `+1` above.
            scale: the scale of the parameter.

        Returns:
            The value of the parameter at the crossing, `None` if the profile
            does not reach the threshold on that side.
        """
        values, costs = self.side(direction)
        valid = self.converged & np.isfinite(self.costs)
        k_opt = self.index_optimum
        valid = valid[: k_opt + 1][::-1] if direction < 0 else valid[k_opt:]
        failed = np.flatnonzero(~valid)
        if failed.size:
            # A failed point is not a crossing and cannot be bridged by later
            # points, even when they happened to converge.
            values, costs = values[: failed[0]], costs[: failed[0]]
        above = np.nonzero(costs >= threshold)[0]
        if len(above) == 0:
            return None
        k = int(above[0])
        if k == 0:
            # the optimum itself is at the threshold
            return float(values[0])
        value_a, value_b = float(values[k - 1]), float(values[k])
        cost_a, cost_b = costs[k - 1], costs[k]
        if cost_b == cost_a:
            return float(values[k])
        fraction = (threshold - cost_a) / (cost_b - cost_a)
        if scale.is_log:
            log_a, log_b = np.log10(value_a), np.log10(value_b)
            return float(10 ** (log_a + fraction * (log_b - log_a)))
        return float(value_a + fraction * (value_b - value_a))

    def rise(self, direction: int) -> float:
        """Get how far the profile rises above its minimum on one side.

        Args:
            direction: `-1` for the side below the optimum, `+1` above.

        Returns:
            The largest cost on the side minus the lowest cost of the profile.
        """
        _, costs = self.side(direction)
        return float(np.max(costs) - self.cost_min)

    def evaluate(
        self,
        threshold: float,
        flatness_cost: float,
        scale: ParameterScaleType,
    ) -> None:
        """Set the confidence interval and the classification.

        Args:
            threshold: threshold on the cost of the confidence intervals.
            flatness_cost: rise of the cost below which a side is flat.
            scale: the scale of the parameter, see `crossing`.
        """
        self.ci_lower = self.crossing(threshold, direction=-1, scale=scale)
        self.ci_upper = self.crossing(threshold, direction=+1, scale=scale)

        if not np.all(self.converged & np.isfinite(self.costs)):
            # Missing scan results give no evidence of identifiability or of a
            # flat profile. Keep any proven interval endpoint, but no class.
            self.identifiability = None
            return

        flat_lower = self.rise(-1) <= flatness_cost
        flat_upper = self.rise(+1) <= flatness_cost
        if flat_lower and flat_upper:
            self.identifiability = Identifiability.STRUCTURAL
        elif self.ci_lower is not None and self.ci_upper is not None:
            self.identifiability = Identifiability.IDENTIFIABLE
        elif self.ci_lower is None and self.ci_upper is None:
            self.identifiability = Identifiability.NON_IDENTIFIABLE
        elif self.ci_lower is None:
            self.identifiability = Identifiability.NON_IDENTIFIABLE_LOWER
        else:
            self.identifiability = Identifiability.NON_IDENTIFIABLE_UPPER

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "pid": self.pid,
            "values": self.values.tolist(),
            "costs": self.costs.tolist(),
            "paths": self.paths.tolist(),
            "converged": self.converged.tolist(),
            "index_optimum": self.index_optimum,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "identifiability": (
                self.identifiability.value if self.identifiability else None
            ),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> ParameterProfile:
        """Create a profile from a dictionary, i.e., from the stored JSON."""
        identifiability = d.get("identifiability")
        return ParameterProfile(
            pid=d["pid"],
            values=np.asarray(d["values"], dtype=float),
            costs=np.asarray(d["costs"], dtype=float),
            paths=np.asarray(d["paths"], dtype=float),
            converged=np.asarray(d["converged"], dtype=bool),
            index_optimum=int(d["index_optimum"]),
            ci_lower=d.get("ci_lower"),
            ci_upper=d.get("ci_upper"),
            identifiability=(
                Identifiability(identifiability) if identifiability else None
            ),
        )


@dataclass
class IdentifiabilityResult:
    """Result of a profile likelihood analysis.

    Attributes:
        opid: id of the optimization problem.
        parameter_set: parameters the profiles were computed around, in the
            units of pint of the problem, see
            `OptimizationProblem.parameter_units`.
        parameters: fit parameters of the problem, with their bounds and units.
        settings: settings of the analysis.
        fit_settings: settings of the fit, which define the cost.
        cost: cost of the parameter set.
        profiles: profile of every analysed parameter by parameter id.
        duration: duration of the analysis in seconds.
        incomplete: why a parameter which was analysed has no profile, by
            parameter id, e.g. the scan which the `scan_timeout` or the
            `timeout` of the settings gave up (#522). Empty for an analysis
            whose every parameter has a profile.
        profiles_of: where the profiles were walked, `RUN_IN_THE_ENGINE` or
            `RUN_IN_PYTHON` (`RUN_MIXED` for a result whose parts were not all
            walked in one place, see `combine`); `None` for a result which did
            not record it, i.e. one of sbmlsim 0.27 or earlier.
    """

    opid: str
    parameter_set: ParameterSet
    parameters: list[FitParameter]
    settings: ProfileSettings
    fit_settings: FitSettings
    cost: float
    profiles: dict[str, ParameterProfile]
    duration: float = 0.0
    incomplete: dict[str, str] = field(default_factory=dict)
    profiles_of: str | None = None

    def __post_init__(self) -> None:
        """Evaluate the profiles."""
        self.parameters = list(self.parameters)
        self.evaluate()

    def __str__(self) -> str:
        """Get string representation."""
        return (
            f"<IdentifiabilityResult: {self.opid}, {len(self.profiles)} profiles, "
            f"{self.n_identifiable} identifiable>"
        )

    @property
    def pids(self) -> list[str]:
        """Get the ids of the parameters of the problem."""
        return [p.pid for p in self.parameters]

    @property
    def cost_min(self) -> float:
        """Get the lowest cost, of the parameter set or of any profile point."""
        costs = [self.cost, *(profile.cost_min for profile in self.profiles.values())]
        return float(np.min(costs))

    @property
    def threshold(self) -> float:
        """Get the threshold on the cost of the confidence intervals."""
        return self.settings.threshold(self.cost_min)

    @property
    def flatness_cost(self) -> float:
        """Get the rise of the cost below which a profile is flat."""
        return self.settings.flatness * self.settings.delta / 2.0

    @property
    def n_identifiable(self) -> int:
        """Get the number of identifiable parameters."""
        return sum(
            1
            for profile in self.profiles.values()
            if profile.identifiability is not None
            and profile.identifiability.is_identifiable
        )

    @property
    def better_optimum(self) -> bool:
        """Check whether a profile found a lower cost than the parameter set.

        The threshold is relative to the lowest cost, so the intervals are
        still valid, but the fit did not converge to the optimum.
        """
        return self.cost_min < self.cost - self.flatness_cost

    def evaluate(self) -> None:
        """Set the confidence intervals and the classifications."""
        threshold, flatness_cost = self.threshold, self.flatness_cost
        for pid, profile in self.profiles.items():
            profile.evaluate(
                threshold=threshold,
                flatness_cost=flatness_cost,
                scale=self.scale(pid),
            )

    def _definition(self) -> str:
        """Get what the profiles of a result are computed around, as JSON.

        It is `to_dict` without the profiles, the duration and the parameters
        which have none, which the parts of an analysis share, see `combine`;
        JSON compares the values bitwise and a `NaN` equal to itself. A part
        which gave a scan up is a part of the analysis like any other, so
        `incomplete` is no part of the definition, and neither is
        `profiles_of`: the walk of the engine is the walk of python, so a part
        of either is a part of the same analysis.

        Raises:
            TypeError: for a value which is not serializable as JSON.
        """
        d = self.to_dict()
        del d["profiles"], d["duration"], d["incomplete"], d["profiles_of"]
        return json.dumps(d)

    @staticmethod
    def combine(parts: Sequence[IdentifiabilityResult]) -> IdentifiabilityResult:
        """Combine the results of parts of an analysis into one result.

        An analysis split into parts, e.g. one per parameter, or the results
        `profile_likelihood` reports with `on_profile`, combine to the result
        of the whole analysis: the profiles of the parts in the order of the
        parts, evaluated against the lowest cost of all of them, since the
        threshold of the confidence intervals depends on it.

        Args:
            parts: the results of the parts, around the same parameter set
                with the same settings.

        Returns:
            The result with the profiles of all parts, its duration the sum of
            the durations of the parts, the parameters without a profile of
            all of them (`incomplete`) and, as `profiles_of`, the one place
            the parts were walked in, `RUN_MIXED` if they were not all walked
            in one place. The parts are not changed.

        Raises:
            ValueError: if there is no part, if a parameter is profiled in two
                parts, or if the parts differ in the problem, the parameter
                set, the parameters, the settings, the settings of the fit or
                the cost.
        """
        if not parts:
            raise ValueError("There is no part of an identifiability analysis.")
        first = parts[0]
        definition = first._definition()
        profiles: dict[str, ParameterProfile] = {}
        incomplete: dict[str, str] = {}
        for k, part in enumerate(parts):
            if part._definition() != definition:
                raise ValueError(
                    f"'{first.opid}': the part {k} ('{part.opid}') is not a part of "
                    f"the analysis of the first part, they differ in the problem, "
                    f"the parameter set, the parameters, the settings or the cost."
                )
            for pid, profile in part.profiles.items():
                if pid in profiles:
                    raise ValueError(
                        f"'{first.opid}': the parameter '{pid}' is profiled twice, "
                        f"again in the part {k}."
                    )
                profiles[pid] = deepcopy(profile)
            incomplete.update(part.incomplete)
        # a parameter a part gave up and another profiled has its profile
        incomplete = {
            pid: reason for pid, reason in incomplete.items() if pid not in profiles
        }
        return IdentifiabilityResult(
            opid=first.opid,
            parameter_set=deepcopy(first.parameter_set),
            parameters=deepcopy(first.parameters),
            settings=first.settings,
            fit_settings=first.fit_settings,
            cost=first.cost,
            profiles=profiles,
            duration=float(sum(part.duration for part in parts)),
            incomplete=incomplete,
            profiles_of=run_place(part.profiles_of for part in parts),
        )

    def scale(self, pid: str) -> ParameterScaleType:
        """Get the scale a parameter was profiled on.

        It is the `FitParameter.scale` of the parameter and the
        `parameter_scale` of the settings of the fit for a parameter without
        one, i.e. `OptimizationProblem.scales_initialized`.

        Raises:
            KeyError: if the problem has no parameter with the id.
        """
        scale = self.parameter(pid).scale
        return self.fit_settings.parameter_scale if scale is None else scale

    def parameter(self, pid: str) -> FitParameter:
        """Get the fit parameter of an id.

        Raises:
            KeyError: if the problem has no parameter with the id.
        """
        for p in self.parameters:
            if p.pid == pid:
                return p
        raise KeyError(f"'{self.opid}': no parameter '{pid}' in '{self.pids}'.")

    def summary_df(self) -> pd.DataFrame:
        """Get the table of the analysis, one row per profiled parameter.

        Returns:
            Table with the parameter, its value, unit (as the user defined
            it, `model` for the unit of the model) and bounds, the confidence
            interval (`NaN` for an open side), the classification,
            the number of points of the profile and its lowest cost.
        """
        rows = []
        for pid, profile in self.profiles.items():
            p = self.parameter(pid)
            rows.append(
                {
                    "parameter": pid,
                    "value": profile.value_optimum,
                    # the unit as the user defined it, as every table of a
                    # report shows it
                    "unit": p.unit or "model",
                    "lower_bound": p.lower_bound,
                    "upper_bound": p.upper_bound,
                    "ci_lower": (
                        np.nan if profile.ci_lower is None else profile.ci_lower
                    ),
                    "ci_upper": (
                        np.nan if profile.ci_upper is None else profile.ci_upper
                    ),
                    "identifiability": (
                        profile.identifiability.value
                        if profile.identifiability
                        else None
                    ),
                    "n_points": len(profile),
                    "cost_min": profile.cost_min,
                    "converged": bool(
                        np.all(profile.converged & np.isfinite(profile.costs))
                    ),
                }
            )
        return pd.DataFrame(rows)

    def report(self, path: Path | None = None, print_output: bool = False) -> str:
        """Get the text report of the analysis.

        Args:
            path: file to write the report to.
            print_output: print the report.

        Returns:
            The report.
        """
        info = [
            "-" * 80,
            f"Identifiability '{self.opid}'",
            "-" * 80,
            f"parameter set: {self.parameter_set.sid}",
            f"cost: {self.cost:.6g}",
            f"cost_min: {self.cost_min:.6g}",
            f"alpha: {self.settings.alpha}",
            f"degrees of freedom: {self.settings.degrees_of_freedom}",
            f"threshold: {self.threshold:.6g}",
            f"reoptimize: {self.settings.reoptimize}",
            f"duration: {self.duration:.1f} s",
            "",
            self.summary_df().to_string(index=False),
            "-" * 80,
        ]
        if self.better_optimum:
            info.insert(
                -2,
                f"!A profile found a cost of {self.cost_min:.6g} below the cost "
                f"{self.cost:.6g} of the parameter set, the fit did not converge!",
            )
        text = "\n".join(info)
        if print_output:
            print(text)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f_report:
                f_report.write(text)
        return text

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "opid": self.opid,
            "parameter_set": self.parameter_set.to_dict(),
            "parameters": [p.to_dict() for p in self.parameters],
            "settings": self.settings.to_dict(),
            "fit_settings": self.fit_settings.to_dict(),
            "cost": self.cost,
            "profiles": {
                pid: profile.to_dict() for pid, profile in self.profiles.items()
            },
            "duration": self.duration,
            "incomplete": dict(self.incomplete),
            "profiles_of": self.profiles_of,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> IdentifiabilityResult:
        """Create a result from a dictionary, i.e., from the stored JSON."""
        return IdentifiabilityResult(
            opid=d["opid"],
            parameter_set=ParameterSet.from_dict(d["parameter_set"]),
            parameters=[FitParameter(**p) for p in d["parameters"]],
            settings=ProfileSettings.from_dict(d["settings"]),
            fit_settings=FitSettings.from_dict(d["fit_settings"]),
            cost=float(d["cost"]),
            profiles={
                pid: ParameterProfile.from_dict(profile)
                for pid, profile in d["profiles"].items()
            },
            duration=float(d.get("duration", 0.0)),
            # a record written before #522 has every parameter it analysed
            incomplete=dict(d.get("incomplete", {})),
            # a record of sbmlsim 0.27 or earlier does not say where it walked
            profiles_of=d.get("profiles_of"),
        )

    def to_json(self, path: Path | None = None) -> str | Path:
        """Store the result as JSON.

        Args:
            path: file to write, the JSON string is returned if it is `None`.

        Returns:
            The path or the JSON string.
        """
        return dump_json(self.to_dict(), path)

    @staticmethod
    def from_json(json_info: str | Path) -> IdentifiabilityResult:
        """Load a result from a JSON file or string.

        Args:
            json_info: path of the file or the JSON string.

        Returns:
            The result.
        """
        return IdentifiabilityResult.from_dict(load_json(json_info))


# ----------------------------------------------------------------------------
# the scans
# ----------------------------------------------------------------------------
#: a point of a scan: the full parameter vector in logarithmic space, the cost
#: and whether the optimization of the other parameters converged
ProfilePoint = tuple[np.ndarray, float, bool]


def _scaled_bounds(problem: OptimizationProblem) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounds of the parameters in the space of the optimizer.

    The scans run in the space the fit searches, i.e. every parameter in its
    own scale, which is the `parameter_scale` of the settings for a parameter
    without one, so that a step of the scan is a step of the optimizer.
    """
    return (
        problem.to_scale([p.lower_bound for p in problem.parameters]),
        problem.to_scale([p.upper_bound for p in problem.parameters]),
    )


def _evaluate_point(
    problem: OptimizationProblem,
    theta: np.ndarray,
    index: int,
    settings: ProfileSettings,
    jacobian: ResidualsEvaluator | None = None,
) -> ProfilePoint:
    """Evaluate a point of a scan.

    The parameter `index` of `theta` is fixed, the other parameters are
    optimized from their values in `theta` if the settings say so.

    Args:
        problem: initialized optimization problem.
        theta: parameter vector in logarithmic space, the start of the
            optimization of the other parameters.
        index: index of the fixed parameter.
        settings: settings of the analysis.
        jacobian: evaluates the points of the jacobian of the re-optimization
            of the other parameters on the workers of the fit, see
            `OptimizationProblem.jacobian_map`; the jacobian is the serial one
            bitwise. The scans of a parallel analysis are the tasks of the
            pool and get none.

    Returns:
        The point: the parameter vector the cost was reached with, the cost
        and whether the evaluation converged.
    """
    theta = np.array(theta, dtype=float)
    free = np.array([k != index for k in range(len(theta))])

    try:
        if not settings.reoptimize or not np.any(free):
            cost = problem.cost_least_square(theta)
            return theta, cost, bool(np.isfinite(cost))

        lower, upper = _scaled_bounds(problem)

        def embed(theta_free: np.ndarray) -> np.ndarray:
            """Put the free parameters into the vector of all parameters."""
            theta_full = theta.copy()
            theta_full[free] = theta_free
            return theta_full

        def residuals(theta_free: np.ndarray) -> np.ndarray:
            """Residuals as a function of the free parameters."""
            return problem.residuals(embed(theta_free))  # ty: ignore[invalid-return-type]

        arguments = problem.optimizer_arguments(
            OptimizationAlgorithmType.LEAST_SQUARE, settings.optimizer_kwargs
        )
        if jacobian is not None:
            arguments["workers"] = problem.jacobian_map(jacobian, embed=embed)
        # the start must be within the bounds, a warm start from the previous
        # point is, the optimum of the fit is by the checks of the problem
        x0 = np.clip(theta[free], lower[free], upper[free])
        opt_result = scipy.optimize.least_squares(
            fun=residuals,
            x0=x0,
            bounds=(lower[free], upper[free]),
            **arguments,
        )
        theta_opt = theta.copy()
        theta_opt[free] = opt_result.x
        cost = problem.cost_least_square(theta_opt)
        return theta_opt, cost, bool(opt_result.success and np.isfinite(cost))
    except (RuntimeError, FitTimeout, ValueError) as err:
        logger.error(
            "'%s': the scan of '%s' failed at '%s': %s",
            problem.opid,
            problem.pids[index],
            problem.from_scale(theta)[index],
            err,
        )
        return theta, float("inf"), False


def _scan_direction(
    problem: OptimizationProblem,
    theta_optimum: np.ndarray,
    cost_optimum: float,
    index: int,
    direction: int,
    settings: ProfileSettings,
    first_step: float | None = None,
    jacobian: ResidualsEvaluator | None = None,
) -> list[ProfilePoint]:
    """Scan a parameter from the optimum in one direction.

    The step is adaptive: a step which raises the cost by more than
    `max_cost_fraction` of the distance to the threshold is reduced by the
    curvature its own rise gives and repeated, a step which raises it by
    little is enlarged by `step_factor`. The other parameters start from the
    previous point. The scan ends when the cost crosses the threshold, at the
    bound of the parameter, after `max_points` or when an evaluation fails.

    Args:
        problem: initialized optimization problem.
        theta_optimum: optimum in logarithmic space.
        cost_optimum: cost at the optimum.
        index: index of the scanned parameter.
        direction: `-1` towards the lower bound, `+1` towards the upper bound.
        settings: settings of the analysis.
        first_step: the step the scan starts at, the scale of the parameter
            where the analysis knows it (`first_steps`); `initial_step` of the
            settings without.
        jacobian: evaluates the jacobian of every point on the workers of the
            fit, see `_evaluate_point`.

    Returns:
        The points of the scan, from the optimum outward, without the optimum.
    """
    # the points of a scan which runs in this process, see `sbmlsim.progress`;
    # a scan which runs in a worker of the fit reports nothing, its task is a
    # step of the activity of the analysis
    with progress_activity(
        "scan",
        f"{problem.pids[index]} {'up' if direction > 0 else 'down'}",
        total=settings.max_points,
    ) as scanned:
        lower, upper = _scaled_bounds(problem)
        bound = upper[index] if direction > 0 else lower[index]
        threshold = settings.threshold(cost_optimum)
        max_rise = settings.max_cost_fraction * settings.delta / 2.0

        points: list[ProfilePoint] = []
        theta_prev, cost_prev = np.array(theta_optimum, dtype=float), cost_optimum
        step = settings.initial_step if first_step is None else first_step

        while len(points) < settings.max_points:
            if abs(theta_prev[index] - bound) <= BOUND_TOLERANCE:
                # the parameter is at its bound, the interval is open
                break

            while True:
                value = theta_prev[index] + direction * step
                if (direction > 0 and value >= bound) or (
                    direction < 0 and value <= bound
                ):
                    value = bound
                theta_start = theta_prev.copy()
                theta_start[index] = value
                theta, cost, converged = _evaluate_point(
                    problem, theta_start, index, settings, jacobian=jacobian
                )
                rise = cost - cost_prev
                if converged and rise > max_rise and step > settings.min_step:
                    # the rise of a step which is walked again is the curvature
                    # of the profile there, which is locally quadratic in the
                    # step: the step whose rise is half of what a step may
                    # raise the cost by is `step * sqrt(max_rise / (2 rise))`,
                    # which reaches the scale of the profile in one correction
                    # where a fixed factor needs several and discards the
                    # re-optimization which measured each of them (#647).
                    # `rise > max_rise` keeps the factor below `1 / sqrt(2)`,
                    # so the step always shrinks, and a rise which is not
                    # finite takes it to `min_step` at once
                    step = max(
                        step * math.sqrt(max_rise / (2.0 * rise)), settings.min_step
                    )
                    continue
                break

            points.append((theta, cost, converged))
            scanned.advance()
            if not converged or cost >= threshold:
                break
            if rise < max_rise / settings.step_factor:
                # the profile changes slowly, larger steps
                step = min(step * settings.step_factor, settings.max_step)
            theta_prev, cost_prev = theta, cost

        return points


def _scan_task(
    problem: OptimizationProblem,
    task: Mapping[str, Any],
    jacobian: ResidualsEvaluator | None = None,
) -> list[ProfilePoint]:
    """Run the scan of a task, see `_scan_direction`."""
    return _scan_direction(
        problem=problem,
        theta_optimum=np.asarray(task["theta_optimum"], dtype=float),
        cost_optimum=float(task["cost_optimum"]),
        index=int(task["index"]),
        direction=int(task["direction"]),
        settings=task["settings"],
        first_step=task.get("first_step"),
        jacobian=jacobian,
    )


def _worker_scan(
    token: str,
    problem: OptimizationProblem,
    settings: FitSettings,
    task: dict[str, Any],
) -> tuple[int, int, list[ProfilePoint]]:
    """Run a scan in a worker process of the pool of the fit.

    Returns:
        The index of the parameter, the direction and the points of the scan.
    """
    initialized = runner.worker_problem(token, problem, settings)
    with runner.worker_progress():
        points = _scan_task(initialized, task)
    return int(task["index"]), int(task["direction"]), points


def _assemble_profile(
    problem: OptimizationProblem,
    pid: str,
    theta_optimum: np.ndarray,
    cost_optimum: float,
    index: int,
    scans: Mapping[int, list[ProfilePoint]],
) -> ParameterProfile:
    """Combine the scans of both directions into the profile of a parameter.

    The scans run in the space the optimizer searches every parameter in, the
    profile has the values of the parameters, i.e. every parameter is
    converted back with its own scale.
    """
    lower = list(reversed(scans.get(-1, [])))
    upper = scans.get(+1, [])
    points = [*lower, (np.asarray(theta_optimum), cost_optimum, True), *upper]
    paths = np.array([problem.from_scale(theta) for theta, _, _ in points], dtype=float)
    return ParameterProfile(
        pid=pid,
        values=paths[:, index],
        costs=np.array([cost for _, cost, _ in points], dtype=float),
        paths=paths,
        converged=np.array([converged for _, _, converged in points], dtype=bool),
        index_optimum=len(lower),
    )


def first_steps(
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    profile_settings: ProfileSettings,
    pids: Sequence[str],
    n_cores: int = 1,
) -> list[float]:
    """Get the first step of the scan of every parameter from its scale.

    A profile is quadratic around the optimum, the cost rising by
    `d² / (2 s²)` over a distance `d` with `s` the standard error of the
    parameter in the space the scans run in, so the step which raises the cost
    by the most a step may raise it, `max_cost_fraction` of the distance to
    the threshold, is `sqrt(2 max_rise) s`. A scan which starts there walks
    the profile at its scale from its first point, where one which starts at
    `initial_step` corrects that step down to the scale and discards the
    re-optimization which measured every correction (#647).

    The standard errors are the ones of the Fisher information of the optimum
    (`sbmlsim.fit.fisher.fisher_information`), whose jacobian is the forward
    sensitivities of the engine where the problem has them and the central
    differences of python otherwise, so the step of a parameter costs a
    fraction of the point it saves. A direction the data does not constrain
    has no standard error and keeps `initial_step`, and an information which
    cannot be computed leaves every parameter with it: a first step is where a
    scan starts and not a result of the analysis, so a failure of it is
    reported and the analysis runs.

    Args:
        problem: optimization problem, initialized with the settings.
        settings: settings of the fit, which define the cost.
        parameter_set: the optimum the profiles are walked around.
        profile_settings: settings of the analysis.
        pids: the parameters to profile, in the order of the profiles.
        n_cores: the workers of the jacobian of the information, which is the
            threads of its batch in the engine.

    Returns:
        The first step of every parameter of `pids`, in its order, in the
        space the scans run in and within `[min_step, max_step]`.
    """
    max_rise = profile_settings.max_cost_fraction * profile_settings.delta / 2.0
    steps = [profile_settings.initial_step] * len(pids)
    try:
        information = fisher_information(
            problem, settings, parameter_set, n_cores=n_cores
        )
    except Exception as err:
        logger.warning(
            "'%s': the scans start at the step '%s' of the settings, the Fisher "
            "information of the optimum, which gives the scale of a parameter, "
            "could not be computed (%s).",
            problem.opid,
            profile_settings.initial_step,
            err,
        )
        return steps
    errors = dict(zip(information.pids, information.standard_errors, strict=True))
    for i, pid in enumerate(pids):
        step = math.sqrt(2.0 * max_rise) * float(errors.get(pid, float("nan")))
        if math.isfinite(step) and step > 0.0:
            steps[i] = min(
                max(step, profile_settings.min_step), profile_settings.max_step
            )
    return steps


def profile_likelihood(
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    profile_settings: ProfileSettings | None = None,
    pids: Sequence[str] | None = None,
    n_cores: int = 1,
    serial: bool = False,
    show_progress: bool = True,
    on_profile: Callable[[IdentifiabilityResult], None] | None = None,
    engine_profiles: bool | None = None,
) -> IdentifiabilityResult:
    """Analyse the identifiability of the parameters of a fit.

    Computes the profile likelihood of every parameter around the parameter set
    and classifies the parameters, see the module documentation.

    Args:
        problem: optimization problem, initialized with the settings.
        settings: settings of the fit, which define the cost.
        parameter_set: optimal parameters, e.g., the best run of a fit.
        profile_settings: settings of the analysis, the defaults if `None`.
        pids: parameters to profile, all parameters of the problem which are no
            elements of a network by default. The elements of a network are
            profiled when they are named: one scan per element is impractical
            for hundreds of elements.
        n_cores: number of worker processes, the scans of the parameters run
            in parallel, or, for fewer scans than one re-optimization spreads
            over, the scans run one after another and the points of the
            jacobian of every re-optimization run on all workers, see
            `sbmlsim.fit.runner.split_cores`. A parallel analysis needs the
            `if __name__ == "__main__":` guard like a parallel fit.
        serial: run the scans in this process, whatever `n_cores` says.
        show_progress: show the progress of the scans on the console.
        on_profile: called in the calling process with the result of one
            profile as soon as both scans of its parameter finished, e.g. to
            store the profiles while the analysis runs; its duration is the
            time since the profile before it finished. The results of all
            profiles combine to the result of the analysis, see
            `IdentifiabilityResult.combine`. An analysis which walks in the
            engine answers with all of its profiles at once, so it calls the
            callback for every one of them when the answer is there.
        engine_profiles: whether the whole analysis runs in the engine, i.e.
            the walks of the profiles, the optimizations of their points and
            the evaluations of those, see `sbmlsim.fit.fit_core`: `None` walks
            there if the engine can walk it
            (`fit_core.gaps_of_the_profiles`, which a problem of the engine
            core of a profile of the other parameters passes), `True` requires
            it and `False` scans in python, which evaluates the points through
            the engine.

    Returns:
        The profiles with the confidence intervals and the classification.

    Raises:
        KeyError: if a parameter id is not a parameter of the problem.
        ValueError: if the parameter set is outside of the bounds of the problem,
            if there is no parameter to profile, if the arguments of the
            optimizer change the loss, which is `FitSettings.loss_function`,
            or if `engine_profiles` is required and the engine cannot walk the
            profiles.
        TypeError: if an argument of the optimizer is not an argument of
            `scipy.optimize.least_squares`.
    """
    profile_settings = profile_settings or ProfileSettings()
    # the other parameters are optimized as a fit optimizes them
    problem.check_optimizer_arguments(
        OptimizationAlgorithmType.LEAST_SQUARE, profile_settings.optimizer_kwargs
    )
    problem.initialize(settings)

    if pids is None:
        single, _ = group_parameters(
            problem.parameters, hook_summaries(problem.hybridizations)
        )
        if len(single) < len(problem.parameters):
            logger.info(
                "'%s': the %d elements of the networks are not profiled, name "
                "an element to profile it.",
                problem.opid,
                len(problem.parameters) - len(single),
            )
        if not single:
            raise ValueError(
                f"'{problem.opid}': there is no parameter to profile, the "
                f"parameters of the problem are all elements of networks, name "
                f"the ones to profile."
            )
        pids = [p.pid for p in single]
    pids = list(pids)
    if not pids:
        raise ValueError(
            f"'{problem.opid}': there is no parameter to profile, `pids` names none."
        )
    unknown = [pid for pid in pids if pid not in problem.pids]
    if unknown:
        raise KeyError(
            f"'{problem.opid}': the parameters '{unknown}' are not parameters of "
            f"the problem '{problem.pids}'."
        )

    parameter_set = problem.bind_parameter_set(parameter_set)
    x = parameter_set.x(problem.pids)
    lower, upper = _scaled_bounds(problem)
    negative = {
        pid: (float(value), scale.name)
        for pid, value, scale in zip(
            problem.pids, x, problem.scales_initialized, strict=True
        )
        if scale.is_log and value <= 0.0
    }
    if negative:
        raise ValueError(
            f"'{problem.opid}': the parameters must be positive, the scans run in "
            f"the logarithmic space of a parameter, got the values and scales "
            f"'{negative}'."
        )
    theta_optimum = problem.to_scale(x)
    outside = [
        pid
        for pid, value, lb, ub in zip(
            problem.pids, theta_optimum, lower, upper, strict=True
        )
        if value < lb - BOUND_TOLERANCE or value > ub + BOUND_TOLERANCE
    ]
    if outside:
        raise ValueError(
            f"'{problem.opid}': the parameters '{outside}' of '{parameter_set.sid}' "
            f"are outside of the bounds of the problem."
        )
    # the optimum is within the bounds up to rounding, a parameter at its bound
    # after a round trip through JSON must not start a scan outside
    theta_optimum = np.clip(theta_optimum, lower, upper)

    ts = time.time()
    cost_optimum = problem.cost_least_square(theta_optimum)
    threads = 1 if serial else runner.resolve_n_cores(n_cores)
    # every scan starts at the scale of its parameter, the same step in python
    # and in the engine, so no scan corrects `initial_step` down to that scale
    # and discards a re-optimization at every correction (#647)
    steps = first_steps(
        problem, settings, parameter_set, profile_settings, pids, n_cores=threads
    )
    tasks: list[dict[str, Any]] = [
        {
            "index": problem.pids.index(pid),
            "direction": direction,
            "theta_optimum": theta_optimum,
            "cost_optimum": cost_optimum,
            "settings": profile_settings,
            "first_step": step,
        }
        for pid, step in zip(pids, steps, strict=True)
        for direction in (-1, +1)
    ]
    # a worker cannot start workers of its own, so the workers go either to the
    # scans or to the jacobian of the re-optimization of one point: one
    # parameter is fixed, the others are optimized, see `split_cores`
    scan_workers, jacobian_workers = runner.split_cores(
        OptimizationAlgorithmType.LEAST_SQUARE,
        size=len(tasks),
        n_cores=1 if serial else runner.resolve_n_cores(n_cores),
        kwargs=profile_settings.optimizer_kwargs,
        n_parameters=(
            max(len(problem.parameters) - 1, 1) if profile_settings.reoptimize else 1
        ),
    )
    parallel = scan_workers > 1
    parallel_jacobian = jacobian_workers > 1

    # the engine walks the sides of every profile next to each other on the
    # threads of one batch, where the scans in python are a process each
    in_the_engine = False
    if engine_profiles is not False:
        gaps = fit_core.gaps_of_the_profiles(problem, profile_settings)
        if engine_profiles and gaps:
            raise ValueError(
                f"'{problem.opid}': the profiles cannot be walked in the engine: "
                + "; ".join(f"{gap.code}: {gap.message}" for gap in gaps)
                + "."
            )
        in_the_engine = not gaps
        if gaps:
            logger.info(
                "%s: the profiles are not walked in the engine: %s",
                problem.opid,
                "; ".join(f"{gap.code}: {gap.message}" for gap in gaps),
            )

    display.section("Identifiability", icon=display.ICON_IDENTIFIABILITY)
    display.key_values(
        {
            "problem": problem.opid,
            "parameter set": parameter_set.sid,
            "cost": f"{cost_optimum:.6g}",
            "threshold": (
                f"{profile_settings.threshold(cost_optimum):.6g} "
                f"(alpha={profile_settings.alpha}, "
                f"df={profile_settings.degrees_of_freedom}, "
                f"delta/2={profile_settings.delta / 2:.4g})"
            ),
            "profiles": ", ".join(pids),
            "reoptimize": profile_settings.reoptimize,
            "scans": (
                f"{len(tasks)} on {threads} thread(s) of the engine"
                if in_the_engine
                else (
                    f"{len(tasks)} one after another, the jacobian of a point on "
                    f"{jacobian_workers} core(s)"
                    if parallel_jacobian
                    else f"{len(tasks)} on {scan_workers} core(s)"
                )
            ),
        }
    )

    # the units of pint, with the unit ids of the models resolved, which a
    # design of the profiles reads, see `profile_parameters`
    result_parameter_set = replace(
        parameter_set, units={**parameter_set.units, **problem.parameter_units}
    )
    scans: dict[int, dict[int, list[ProfilePoint]]] = {
        task["index"]: {} for task in tasks
    }
    assembled: dict[str, ParameterProfile] = {}
    incomplete: dict[str, str] = {}
    last_finished = ts
    deadline = (
        None
        if profile_settings.timeout is None
        else time.monotonic() + profile_settings.timeout
    )

    def _give_up(task: Mapping[str, Any], err: BaseException) -> None:
        """Record a scan which was given up, so its parameter has no profile.

        A parameter needs both of its scans for a profile, so one which was
        given up leaves it without one; the reason is kept per parameter and
        the first one is the one reported (#522).
        """
        pid = problem.pids[task["index"]]
        side = "below" if task["direction"] < 0 else "above"
        logger.error(
            "'%s': the scan of '%s' %s the optimum was given up (%s), so the "
            "parameter has no profile.",
            problem.opid,
            pid,
            side,
            err,
        )
        incomplete.setdefault(pid, f"the scan {side} the optimum: {err}")

    def collect(index: int, direction: int, points: list[ProfilePoint]) -> None:
        """Keep the scan, the profile once both of its scans finished."""
        nonlocal last_finished
        scans[index][direction] = points
        if len(scans[index]) < 2:
            return
        pid = problem.pids[index]
        assembled[pid] = _assemble_profile(
            problem=problem,
            pid=pid,
            theta_optimum=theta_optimum,
            cost_optimum=cost_optimum,
            index=index,
            scans=scans[index],
        )
        now = time.time()
        emit("identifiability.profile", opid=problem.opid, pid=pid)
        if on_profile is not None:
            on_profile(
                IdentifiabilityResult(
                    opid=problem.opid,
                    # copies: a part is the caller's, it shares nothing with
                    # the result of the analysis, which evaluates its profiles
                    parameter_set=deepcopy(result_parameter_set),
                    parameters=deepcopy(problem.parameters),
                    settings=deepcopy(profile_settings),
                    fit_settings=deepcopy(settings),
                    cost=cost_optimum,
                    profiles={pid: deepcopy(assembled[pid])},
                    duration=now - last_finished,
                    profiles_of=(RUN_IN_THE_ENGINE if in_the_engine else RUN_IN_PYTHON),
                )
            )
        last_finished = now

    with runner.optimization_progress(
        "profiling",
        len(tasks),
        show_progress,
        unit="scans",
        workers=scan_workers,
        kind="identifiability",
    ) as progress:
        if in_the_engine:
            walked = fit_core.profile_points(
                problem,
                profile_settings,
                theta_optimum,
                pids,
                steps,
                threads=threads,
            )
            for pid, (below, above) in walked.items():
                index = problem.pids.index(pid)
                for direction, points in ((-1, below), (+1, above)):
                    runner._advance(progress)
                    collect(index, direction, points)
        elif parallel:
            with runner.worker_pool(problem, settings, scan_workers) as pool:
                for key, outcome in pool.run(
                    _worker_scan,
                    dict(enumerate(tasks)),
                    timeout=profile_settings.scan_timeout,
                    deadline=deadline,
                ):
                    if isinstance(outcome, TimeoutError | PoolError):
                        _give_up(tasks[key], outcome)
                        runner._advance(progress)
                        continue
                    if isinstance(outcome, Exception):
                        raise outcome
                    index, direction, points = outcome
                    runner._advance(progress)
                    collect(index, direction, points)
        else:
            with ExitStack() as stack:
                jacobian: runner.JacobianPool | None = None
                if parallel_jacobian:
                    # the scans run here, their re-optimizations on the
                    # workers; sbmlsim-core simulates the points of a
                    # jacobian at once on the threads of one worker, see
                    # `runner.FitPool.threads`
                    batched = problem.engine is Engine.CORE
                    jacobian = runner.JacobianPool(
                        stack.enter_context(
                            runner.worker_pool(
                                problem,
                                settings,
                                1 if batched else jacobian_workers,
                                threads=jacobian_workers if batched else 1,
                            )
                        )
                    )
                    jacobian.start()
                for task in tasks:
                    if deadline is not None and time.monotonic() >= deadline:
                        _give_up(
                            task,
                            TimeoutError(
                                f"the analysis reached its timeout of "
                                f"{profile_settings.timeout} s before this scan "
                                f"started"
                            ),
                        )
                        runner._advance(progress)
                        continue
                    points = _scan_task(problem, task, jacobian=jacobian)
                    runner._advance(progress)
                    collect(task["index"], task["direction"], points)

    result = IdentifiabilityResult(
        opid=problem.opid,
        parameter_set=result_parameter_set,
        parameters=problem.parameters,
        settings=profile_settings,
        fit_settings=settings,
        cost=cost_optimum,
        # a parameter whose scan was given up has no profile, see `_give_up`
        profiles={pid: assembled[pid] for pid in pids if pid in assembled},
        duration=time.time() - ts,
        incomplete=incomplete,
        profiles_of=RUN_IN_THE_ENGINE if in_the_engine else RUN_IN_PYTHON,
    )
    if incomplete:
        logger.warning(
            "'%s': %s of the %s analysed parameters have no profile, %s; the "
            "analysis wrote the profiles it has, see the 'scan_timeout' and "
            "the 'timeout' of its settings.",
            problem.opid,
            len(incomplete),
            len(pids),
            sorted(incomplete),
        )
    if result.better_optimum:
        logger.warning(
            "'%s': a profile found a cost of %.6g below the cost %.6g of '%s', "
            "the fit did not converge to the optimum.",
            problem.opid,
            result.cost_min,
            result.cost,
            parameter_set.sid,
        )
    display.print_identifiability(result.summary_df())
    return result


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
#: color of every identifiability in the figures
COLORS: dict[Identifiability, str] = {
    Identifiability.IDENTIFIABLE: "tab:green",
    Identifiability.NON_IDENTIFIABLE_LOWER: "tab:orange",
    Identifiability.NON_IDENTIFIABLE_UPPER: "tab:orange",
    Identifiability.NON_IDENTIFIABLE: "tab:red",
    Identifiability.STRUCTURAL: "tab:purple",
}


def _x_limits(
    values: np.ndarray, lower: float, upper: float, is_log: bool
) -> tuple[float, float]:
    """Get the limits of the axis of a profile.

    Args:
        values: the values of the scanned parameter.
        lower: lower bound of the parameter.
        upper: upper bound of the parameter.
        is_log: whether the axis is logarithmic.

    Returns:
        The finite bounds with a margin, the range of the scanned values with a
        margin on a side without a finite bound.
    """
    finite = values[np.isfinite(values)]
    lo = lower if np.isfinite(lower) and (lower > 0 or not is_log) else np.min(finite)
    hi = upper if np.isfinite(upper) else np.max(finite)
    lo, hi = float(lo), float(hi)
    if is_log:
        return lo / 1.5, hi * 1.5
    margin = 0.1 * (hi - lo) if hi > lo else 0.1 * max(abs(lo), 1.0)
    return lo - margin, hi + margin


def _profile_axes(
    result: IdentifiabilityResult, pid: str
) -> tuple[ChartAxis, ChartAxis]:
    """Get the axes of the profile of a parameter, in the scale of the parameter.

    The x axis spans the bounds with a margin, see `_x_limits`; the y axis the
    costs of the converged points and the threshold.
    """
    from sbmlsim.plot.chart import ChartAxis

    profile = result.profiles[pid]
    p = result.parameter(pid)
    is_log = result.scale(pid).is_log
    low, high = _x_limits(profile.values, p.lower_bound, p.upper_bound, is_log)
    ymin, ymax = _y_limits(result, pid)
    x = ChartAxis(
        label=f"{pid} [{p.unit or 'model'}]",
        scale="log" if is_log else "linear",
        min=low,
        max=high,
        grid=True,
    )
    return x, ChartAxis(label="cost", min=ymin, max=ymax, grid=True)


def _y_limits(result: IdentifiabilityResult, pid: str) -> tuple[float, float]:
    """Get the limits of the cost of a profile: the minimal cost to the threshold."""
    profile = result.profiles[pid]
    costs = profile.costs[profile.converged]
    ymax = max(result.threshold, float(np.max(costs)) if costs.size else 0.0)
    span = max(ymax - result.cost_min, 1e-12)
    return result.cost_min - 0.1 * span, ymax + 0.2 * span


def _plot_profile_axis(
    plot: ChartPlot, result: IdentifiabilityResult, pid: str
) -> None:
    """Draw the profile of a parameter into a plot of the axes of `_profile_axes`.

    The confidence interval as a band of the color of the identifiability
    (an open side extends to the bound or the axis), the threshold as a
    dashed line and the minimal cost as a dotted one, the profile, its failed
    points at the threshold, the optimum and the bounds.
    """
    from sbmlsim.plot.colors import normalize_color

    profile = result.profiles[pid]
    p = result.parameter(pid)
    is_log = result.scale(pid).is_log
    color = normalize_color(
        COLORS[profile.identifiability] if profile.identifiability else "tab:blue"
    )
    xlim = _x_limits(profile.values, p.lower_bound, p.upper_bound, is_log)
    ylim = _y_limits(result, pid)

    ci_lower = (
        max(p.lower_bound, xlim[0]) if profile.ci_lower is None else profile.ci_lower
    )
    ci_upper = (
        min(p.upper_bound, xlim[1]) if profile.ci_upper is None else profile.ci_upper
    )
    plot.area(
        [ci_lower, ci_upper],
        [ylim[0], ylim[0]],
        [ylim[1], ylim[1]],
        color=color,
        fill_opacity=0.12,
    )
    plot.rule(
        result.threshold,
        line="dashed",
        width=1.5,
        color="#000000",
        label="threshold",
        legend="threshold",
    )
    plot.rule(
        result.cost_min,
        line="dotted",
        width=1.5,
        color="#808080",
        label="minimal cost",
        legend="minimal cost",
    )
    converged = profile.converged
    plot.curve(
        profile.values[converged],
        profile.costs[converged],
        marker="circle",
        size=5,
        width=1.5,
        color=color,
        label="profile",
        legend="profile",
    )
    if np.any(~converged):
        plot.curve(
            profile.values[~converged],
            np.full(np.sum(~converged), result.threshold),
            line="none",
            marker="cross",
            size=7,
            color="#000000",
            label="failed",
            legend="failed",
        )
    plot.curve(
        [profile.value_optimum],
        [profile.cost_optimum],
        line="none",
        marker="diamond",
        size=11,
        color="#000000",
        label="optimum",
        legend="optimum",
    )
    for bound in (p.lower_bound, p.upper_bound):
        if np.isfinite(bound) and (bound > 0 or not is_log):
            plot.rule(bound, orientation="vertical", width=0.8, color="#808080")


def _profile_title(result: IdentifiabilityResult, pid: str) -> str:
    """Get the title of the panel of a profile, the parameter and its identifiability."""
    profile = result.profiles[pid]
    label = profile.identifiability.label if profile.identifiability else pid
    return f"{pid}: {label}"


def plot_profiles(
    result: IdentifiabilityResult, path: Path | None = None, ncols: int = 3
) -> Chart:
    """Plot the profiles of all parameters, one panel per parameter.

    Args:
        result: result of the analysis.
        path: file to save the figure to, `svg` or `png`, with its spec next
            to it, see `sbmlsim.report.figures.save_figure`.
        ncols: number of panels per row.

    Returns:
        The figure, a chart of the renderer of sbmlsim-core.
    """
    from sbmlsim.plot.chart import Chart

    pids = list(result.profiles)
    n = len(pids)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    chart = Chart(
        330.0 * ncols,
        270.0 * nrows,
        title=(
            f"Profile likelihood '{result.opid}' "
            f"({result.settings.alpha:.0%}, threshold {result.threshold:.4g})"
        ),
        rows=nrows,
        columns=ncols,
    )
    for k, pid in enumerate(pids):
        x, y = _profile_axes(result, pid)
        plot = chart.plot(
            k // ncols,
            k % ncols,
            title=_profile_title(result, pid),
            x=x,
            y=y,
            legend=k == 0,
        )
        _plot_profile_axis(plot, result, pid)
    if path is not None:
        _save(chart, path)
    return chart


def plot_profile(
    result: IdentifiabilityResult, pid: str, path: Path | None = None
) -> Chart:
    """Plot the profile of a parameter with the paths of the other parameters.

    The upper panel is the profile, the lower panel the other parameters along
    it, relative to their values at the optimum: `log10(parameter / optimum)`
    for a parameter on a logarithmic scale and `parameter - optimum` for one
    on the linear scale, the axis `log10(p / p_opt)` and `p - p_opt`. A
    parameter which changes along the profile is coupled to the scanned one
    (Maiwald et al. 2016).

    Args:
        result: result of the analysis.
        pid: id of the parameter.
        path: file to save the figure to, see `plot_profiles`.

    Returns:
        The figure, a chart of the renderer of sbmlsim-core.
    """
    from dataclasses import replace as replace_axis

    from sbmlsim.plot.chart import Chart, ChartAxis

    profile = result.profiles[pid]
    index = result.pids.index(pid)
    chart = Chart(432.0, 504.0, rows=5, columns=1, hspace=0.15)
    x, y = _profile_axes(result, pid)
    upper = chart.plot(
        0,
        0,
        row_span=3,
        title=_profile_title(result, pid),
        x=replace_axis(x, label_visible=False),
        y=y,
    )
    _plot_profile_axis(upper, result, pid)

    converged = profile.converged
    optimum = profile.paths[profile.index_optimum]
    linear: list[str] = []
    logarithmic: list[str] = []
    relatives: dict[str, np.ndarray] = {}
    for k, other in enumerate(result.pids):
        if k == index:
            continue
        along = profile.paths[converged, k]
        if result.scale(other).is_log:
            logarithmic.append(other)
            relatives[other] = np.log10(along / optimum[k])
        else:
            linear.append(other)
            relatives[other] = along - optimum[k]
    # short labels, the label of the lower panel is no longer than its height
    labels = []
    if logarithmic or not linear:
        labels.append("log10(p / p_opt)")
    if linear:
        labels.append("p - p_opt" + (" (linear)" if logarithmic else ""))
    lower = chart.plot(
        3,
        0,
        row_span=2,
        x=x,
        y=ChartAxis(label=", ".join(labels), grid=True),
        legend=len(result.pids) > 1,
    )
    for other, relative in relatives.items():
        lower.curve(
            profile.values[converged],
            relative,
            marker="circle",
            size=4,
            width=1.5,
            label=other,
            legend=other,
        )
    lower.rule(0.0, line="dotted", width=1.0, color="#808080")
    lower.rule(
        profile.value_optimum,
        orientation="vertical",
        line="dotted",
        width=1.0,
        color="#808080",
    )
    if path is not None:
        _save(chart, path)
    return chart


def _save(chart: Chart, path: Path, png_dpi: float | None = None) -> None:
    """Save a figure of the profiles with its spec, see `save_figure`."""
    from sbmlsim.report.figures import save_figure

    save_figure(chart, path, png_dpi=png_dpi)


def plot_all(
    result: IdentifiabilityResult,
    output_dir: Path,
    image_format: str = "svg",
    png_dpi: float | None = None,
) -> list[Path]:
    """Write the overview figure and the figure of every profile.

    Every figure is written with its spec, see
    `sbmlsim.report.figures.save_figure`.

    Args:
        result: result of the analysis.
        output_dir: directory of the figures.
        image_format: format of the figures, `svg` or `png`.
        png_dpi: resolution of a PNG copy of the overview figure, which is
            written if the format is another one, e.g. for LaTeX, which
            includes no SVG, and of the figures of the format `png`; no copy
            by default.

    Returns:
        The paths of the figures, the overview first; a PNG copy and a spec
        are not among them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"profiles.{image_format}"]
    _save(plot_profiles(result), paths[0], png_dpi=png_dpi)
    for pid in result.profiles:
        path = output_dir / f"profile_{pid}.{image_format}"
        figure = plot_profile(result, pid)
        _save(figure, path, png_dpi=png_dpi if image_format == "png" else None)
        paths.append(path)
    return paths
