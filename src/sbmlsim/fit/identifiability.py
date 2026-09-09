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

The scans run in logarithmic parameter space, as the optimization does. The
step along a profile is adaptive (Schälte et al. 2023): a step which raises the
cost by more than a fraction of the threshold is halved and repeated, a step
which raises it by little is enlarged, so the profile is resolved where it
changes. The other parameters start from the previous point of the profile
(Simpson & Maclaren 2023) and their paths are stored, so a parameter which is
coupled to the scanned one is seen in its path (Maiwald et al. 2016). A scan
stops when the profile crosses the threshold, at the bound of the parameter or
after `max_points`; a bound which is reached below the threshold is an open
interval (Borisov & Metelkin 2020). A scan which finds a lower cost than the
optimum reports it: the fit did not converge, and the threshold is taken
relative to the lowest cost of all profiles.

The scans are independent, so they run in the worker pool of the fit runner,
two scans per parameter. A parallel analysis needs the same
`if __name__ == "__main__":` guard as a parallel fit.

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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
from matplotlib.axes import Axes
from scipy.stats import chi2

from sbmlsim.fit import display, runner
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.optimization import FitTimeout, OptimizationProblem
from sbmlsim.fit.options import FitSettings
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.plot.serialization_matplotlib import plt

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
        initial_step: first step of a scan in decades of the parameter.
        min_step: smallest step in decades, a step is not halved below it.
        max_step: largest step in decades.
        step_factor: factor a step is reduced or enlarged by.
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
            optimization of the other parameters.
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
    optimizer_kwargs: Mapping[str, Any] = field(
        default_factory=lambda: {"diff_step": 0.05}
    )

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
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> ProfileSettings:
        """Create settings from a dictionary, i.e., from the stored JSON."""
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
        identifiability: classification of the parameter, set by `evaluate`.
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
        costs = self.costs[self.converged]
        return float(np.min(costs)) if len(costs) else self.cost_optimum

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

    def crossing(self, threshold: float, direction: int) -> float | None:
        """Get the value at which the profile crosses the threshold.

        The crossing is interpolated between the last point below and the
        first point above the threshold, in logarithmic space where the two
        values are positive, i.e. where the scan ran on a logarithmic scale,
        and linearly otherwise.

        Args:
            threshold: threshold on the cost.
            direction: `-1` for the crossing below the optimum, `+1` above.

        Returns:
            The value of the parameter at the crossing, `None` if the profile
            does not reach the threshold on that side.
        """
        values, costs = self.side(direction)
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
        if value_a > 0.0 and value_b > 0.0:
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

    def evaluate(self, threshold: float, flatness_cost: float) -> None:
        """Set the confidence interval and the classification.

        Args:
            threshold: threshold on the cost of the confidence intervals.
            flatness_cost: rise of the cost below which a side is flat.
        """
        self.ci_lower = self.crossing(threshold, direction=-1)
        self.ci_upper = self.crossing(threshold, direction=+1)

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
        parameter_set: parameters the profiles were computed around.
        parameters: fit parameters of the problem, with their bounds and units.
        settings: settings of the analysis.
        fit_settings: settings of the fit, which define the cost.
        cost: cost of the parameter set.
        profiles: profile of every analysed parameter by parameter id.
        duration: duration of the analysis in seconds.
    """

    opid: str
    parameter_set: ParameterSet
    parameters: list[FitParameter]
    settings: ProfileSettings
    fit_settings: FitSettings
    cost: float
    profiles: dict[str, ParameterProfile]
    duration: float = 0.0

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
        for profile in self.profiles.values():
            profile.evaluate(threshold=threshold, flatness_cost=flatness_cost)

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
            Table with the parameter, its value, unit and bounds, the
            confidence interval (`NaN` for an open side), the classification,
            the number of points of the profile and its lowest cost.
        """
        rows = []
        for pid, profile in self.profiles.items():
            p = self.parameter(pid)
            rows.append(
                {
                    "parameter": pid,
                    "value": profile.value_optimum,
                    "unit": p.unit,
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
                    "converged": bool(np.all(profile.converged)),
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
        )

    def to_json(self, path: Path | None = None) -> str | Path:
        """Store the result as JSON.

        Args:
            path: file to write, the JSON string is returned if it is `None`.

        Returns:
            The path or the JSON string.
        """
        info = self.to_dict()
        if path is None:
            return json.dumps(info, indent=2)
        with open(path, "w", encoding="utf-8") as f_json:
            json.dump(info, f_json, indent=2)
        return path

    @staticmethod
    def from_json(json_info: str | Path) -> IdentifiabilityResult:
        """Load a result from a JSON file or string.

        Args:
            json_info: path of the file or the JSON string.

        Returns:
            The result.
        """
        if isinstance(json_info, Path):
            with open(json_info, encoding="utf-8") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)
        return IdentifiabilityResult.from_dict(d)


# ----------------------------------------------------------------------------
# the scans
# ----------------------------------------------------------------------------
#: a point of a scan: the full parameter vector in logarithmic space, the cost
#: and whether the optimization of the other parameters converged
ProfilePoint = tuple[np.ndarray, float, bool]


def _scaled_bounds(problem: OptimizationProblem) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounds of the parameters in the space of the optimizer.

    The scans run in the space the fit searches, i.e. the
    `parameter_scale` of its settings, so that a step of the scan is a step of
    the optimizer.
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

    Returns:
        The point: the parameter vector the cost was reached with, the cost
        and whether the evaluation converged.
    """
    theta = np.array(theta, dtype=float)
    free = np.array([k != index for k in range(len(theta))])

    try:
        if not settings.reoptimize or not np.any(free):
            return theta, problem.cost_least_square(theta), True

        lower, upper = _scaled_bounds(problem)

        def residuals(theta_free: np.ndarray) -> np.ndarray:
            """Residuals as a function of the free parameters."""
            theta_full = theta.copy()
            theta_full[free] = theta_free
            return problem.residuals(theta_full)  # ty: ignore[invalid-return-type]

        # the start must be within the bounds, a warm start from the previous
        # point is, the optimum of the fit is by the checks of the problem
        x0 = np.clip(theta[free], lower[free], upper[free])
        opt_result = scipy.optimize.least_squares(
            fun=residuals,
            x0=x0,
            bounds=(lower[free], upper[free]),
            **settings.optimizer_kwargs,
        )
    except (RuntimeError, FitTimeout, ValueError) as err:
        logger.error(
            "'%s': the scan of '%s' failed at '%s': %s",
            problem.opid,
            problem.pids[index],
            problem.from_scale(theta)[index],
            err,
        )
        return theta, float("inf"), False

    theta_opt = theta.copy()
    theta_opt[free] = opt_result.x
    return theta_opt, float(opt_result.cost), bool(opt_result.success)


def _scan_direction(
    problem: OptimizationProblem,
    theta_optimum: np.ndarray,
    cost_optimum: float,
    index: int,
    direction: int,
    settings: ProfileSettings,
) -> list[ProfilePoint]:
    """Scan a parameter from the optimum in one direction.

    The step is adaptive: a step which raises the cost by more than
    `max_cost_fraction` of the distance to the threshold is reduced and
    repeated, a step which raises it by little is enlarged. The other
    parameters start from the previous point. The scan ends when the cost
    crosses the threshold, at the bound of the parameter, after `max_points`
    or when an evaluation fails.

    Args:
        problem: initialized optimization problem.
        theta_optimum: optimum in logarithmic space.
        cost_optimum: cost at the optimum.
        index: index of the scanned parameter.
        direction: `-1` towards the lower bound, `+1` towards the upper bound.
        settings: settings of the analysis.

    Returns:
        The points of the scan, from the optimum outward, without the optimum.
    """
    lower, upper = _scaled_bounds(problem)
    bound = upper[index] if direction > 0 else lower[index]
    threshold = settings.threshold(cost_optimum)
    max_rise = settings.max_cost_fraction * settings.delta / 2.0

    points: list[ProfilePoint] = []
    theta_prev, cost_prev = np.array(theta_optimum, dtype=float), cost_optimum
    step = settings.initial_step

    while len(points) < settings.max_points:
        if abs(theta_prev[index] - bound) <= BOUND_TOLERANCE:
            # the parameter is at its bound, the interval is open
            break

        while True:
            value = theta_prev[index] + direction * step
            if (direction > 0 and value >= bound) or (direction < 0 and value <= bound):
                value = bound
            theta_start = theta_prev.copy()
            theta_start[index] = value
            theta, cost, converged = _evaluate_point(
                problem, theta_start, index, settings
            )
            rise = cost - cost_prev
            if converged and rise > max_rise and step > settings.min_step:
                # the profile changes fast here, resolve it
                step = max(step / settings.step_factor, settings.min_step)
                continue
            break

        points.append((theta, cost, converged))
        if not converged or cost >= threshold:
            break
        if rise < max_rise / settings.step_factor:
            # the profile changes slowly, larger steps
            step = min(step * settings.step_factor, settings.max_step)
        theta_prev, cost_prev = theta, cost

    return points


def _scan_task(
    problem: OptimizationProblem, task: Mapping[str, Any]
) -> list[ProfilePoint]:
    """Run the scan of a task, see `_scan_direction`."""
    return _scan_direction(
        problem=problem,
        theta_optimum=np.asarray(task["theta_optimum"], dtype=float),
        cost_optimum=float(task["cost_optimum"]),
        index=int(task["index"]),
        direction=int(task["direction"]),
        settings=task["settings"],
    )


def _worker_scan(task: dict[str, Any]) -> tuple[int, int, list[ProfilePoint]]:
    """Run a scan in a worker process of the pool.

    Returns:
        The index of the parameter, the direction and the points of the scan.
    """
    problem = runner.worker_problem()
    return int(task["index"]), int(task["direction"]), _scan_task(problem, task)


def _assemble_profile(
    pid: str,
    theta_optimum: np.ndarray,
    cost_optimum: float,
    index: int,
    scans: Mapping[int, list[ProfilePoint]],
) -> ParameterProfile:
    """Combine the scans of both directions into the profile of a parameter."""
    lower = list(reversed(scans.get(-1, [])))
    upper = scans.get(+1, [])
    points = [*lower, (np.asarray(theta_optimum), cost_optimum, True), *upper]
    thetas = np.array([theta for theta, _, _ in points], dtype=float)
    return ParameterProfile(
        pid=pid,
        values=10.0 ** thetas[:, index],
        costs=np.array([cost for _, cost, _ in points], dtype=float),
        paths=10.0**thetas,
        converged=np.array([converged for _, _, converged in points], dtype=bool),
        index_optimum=len(lower),
    )


def profile_likelihood(
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    profile_settings: ProfileSettings | None = None,
    pids: Sequence[str] | None = None,
    n_cores: int = 1,
    serial: bool = False,
    show_progress: bool = True,
) -> IdentifiabilityResult:
    """Analyse the identifiability of the parameters of a fit.

    Computes the profile likelihood of every parameter around the parameter set
    and classifies the parameters, see the module documentation.

    Args:
        problem: optimization problem, initialized with the settings.
        settings: settings of the fit, which define the cost.
        parameter_set: optimal parameters, e.g., the best run of a fit.
        profile_settings: settings of the analysis, the defaults if `None`.
        pids: parameters to profile, all parameters of the problem by default.
        n_cores: number of worker processes, the scans of the parameters run
            in parallel. A parallel analysis needs the
            `if __name__ == "__main__":` guard like a parallel fit.
        serial: run the scans in this process, whatever `n_cores` says.
        show_progress: show the progress of the scans on the console.

    Returns:
        The profiles with the confidence intervals and the classification.

    Raises:
        KeyError: if a parameter id is not a parameter of the problem.
        ValueError: if the parameter set is outside of the bounds of the problem.
    """
    profile_settings = profile_settings or ProfileSettings()
    problem.initialize(settings)

    pids = list(pids) if pids is not None else list(problem.pids)
    unknown = [pid for pid in pids if pid not in problem.pids]
    if unknown:
        raise KeyError(
            f"'{problem.opid}': the parameters '{unknown}' are not parameters of "
            f"the problem '{problem.pids}'."
        )

    x = parameter_set.x(problem.pids)
    lower, upper = _scaled_bounds(problem)
    if problem.parameter_scale.is_log and np.any(x <= 0.0):
        raise ValueError(
            f"'{problem.opid}': the parameters must be positive, the scans run in "
            f"'{problem.parameter_scale.name}' space, got "
            f"'{dict(zip(problem.pids, x, strict=True))}'."
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
    tasks: list[dict[str, Any]] = [
        {
            "index": problem.pids.index(pid),
            "direction": direction,
            "theta_optimum": theta_optimum,
            "cost_optimum": cost_optimum,
            "settings": profile_settings,
        }
        for pid in pids
        for direction in (-1, +1)
    ]
    n_cores = min(runner.resolve_n_cores(n_cores), len(tasks))
    parallel = not serial and n_cores > 1

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
            "scans": f"{len(tasks)} on {n_cores if parallel else 1} core(s)",
        }
    )

    scans: dict[int, dict[int, list[ProfilePoint]]] = {
        task["index"]: {} for task in tasks
    }
    with runner.optimization_progress(
        "profiling", len(tasks), show_progress, unit="scans"
    ) as progress:
        if parallel:
            with runner.worker_pool(problem, settings, n_cores) as pool:
                async_results = [
                    pool.apply_async(_worker_scan, (task,)) for task in tasks
                ]
                for async_result in async_results:
                    index, direction, points = async_result.get()
                    scans[index][direction] = points
                    runner._advance(progress)
        else:
            for task in tasks:
                points = _scan_task(problem, task)
                scans[task["index"]][task["direction"]] = points
                runner._advance(progress)

    profiles = {
        pid: _assemble_profile(
            pid=pid,
            theta_optimum=theta_optimum,
            cost_optimum=cost_optimum,
            index=problem.pids.index(pid),
            scans=scans[problem.pids.index(pid)],
        )
        for pid in pids
    }
    result = IdentifiabilityResult(
        opid=problem.opid,
        parameter_set=parameter_set,
        parameters=problem.parameters,
        settings=profile_settings,
        fit_settings=settings,
        cost=cost_optimum,
        profiles=profiles,
        duration=time.time() - ts,
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


def _plot_profile_axis(
    ax: Axes, result: IdentifiabilityResult, pid: str, title: bool = True
) -> None:
    """Draw the profile of a parameter on an axis."""
    profile = result.profiles[pid]
    p = result.parameter(pid)
    color = COLORS[profile.identifiability] if profile.identifiability else "tab:blue"

    # the confidence interval, an open side extends to the bound
    ci_lower = p.lower_bound if profile.ci_lower is None else profile.ci_lower
    ci_upper = p.upper_bound if profile.ci_upper is None else profile.ci_upper
    ax.axvspan(ci_lower, ci_upper, color=color, alpha=0.12, linewidth=0)

    ax.axhline(result.threshold, color="black", linestyle="--", label="threshold")
    ax.axhline(result.cost_min, color="gray", linestyle=":", label="minimal cost")
    converged = profile.converged
    ax.plot(
        profile.values[converged],
        profile.costs[converged],
        marker="o",
        markersize=4,
        color=color,
        label="profile",
    )
    if np.any(~converged):
        ax.plot(
            profile.values[~converged],
            np.full(np.sum(~converged), result.threshold),
            linestyle="",
            marker="x",
            color="black",
            label="failed",
        )
    ax.plot(
        profile.value_optimum,
        profile.cost_optimum,
        marker="*",
        markersize=12,
        color="black",
        linestyle="",
        label="optimum",
    )
    for bound in (p.lower_bound, p.upper_bound):
        ax.axvline(bound, color="gray", linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlim(p.lower_bound / 1.5, p.upper_bound * 1.5)
    ymax = max(result.threshold, float(np.max(profile.costs[converged])))
    span = max(ymax - result.cost_min, 1e-12)
    ax.set_ylim(result.cost_min - 0.1 * span, ymax + 0.2 * span)
    ax.set_xlabel(f"{pid} [{p.unit or 'model'}]")
    ax.set_ylabel("cost")
    ax.grid(alpha=0.3)
    if title:
        label = profile.identifiability.label if profile.identifiability else pid
        ax.set_title(f"{pid}: {label}", fontsize="medium", color=color)


def plot_profiles(
    result: IdentifiabilityResult, path: Path | None = None, ncols: int = 3
) -> Any:
    """Plot the profiles of all parameters, one panel per parameter.

    Args:
        result: result of the analysis.
        path: file to save the figure to, the figure is returned otherwise.
        ncols: number of panels per row.

    Returns:
        The figure.
    """
    pids = list(result.profiles)
    n = len(pids)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 3.6 * nrows),
        layout="constrained",
        squeeze=False,
    )
    for k, pid in enumerate(pids):
        _plot_profile_axis(axes[k // ncols][k % ncols], result, pid)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    axes[0][0].legend(fontsize="x-small")
    fig.suptitle(
        f"Profile likelihood '{result.opid}' "
        f"({result.settings.alpha:.0%}, threshold {result.threshold:.4g})"
    )
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_profile(
    result: IdentifiabilityResult, pid: str, path: Path | None = None
) -> Any:
    """Plot the profile of a parameter with the paths of the other parameters.

    The upper panel is the profile, the lower panel the other parameters along
    it, relative to their values at the optimum. A parameter which changes
    along the profile is coupled to the scanned one (Maiwald et al. 2016).

    Args:
        result: result of the analysis.
        pid: id of the parameter.
        path: file to save the figure to, the figure is returned otherwise.

    Returns:
        The figure.
    """
    profile = result.profiles[pid]
    index = result.pids.index(pid)
    fig, (ax_profile, ax_paths) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6.0, 7.0),
        layout="constrained",
        sharex=True,
        height_ratios=[1.4, 1.0],
    )
    _plot_profile_axis(ax_profile, result, pid)
    ax_profile.legend(fontsize="x-small")
    ax_profile.set_xlabel("")

    converged = profile.converged
    optimum = profile.paths[profile.index_optimum]
    for k, other in enumerate(result.pids):
        if k == index:
            continue
        ax_paths.plot(
            profile.values[converged],
            np.log10(profile.paths[converged, k] / optimum[k]),
            marker=".",
            label=other,
        )
    ax_paths.axhline(0.0, color="gray", linestyle=":")
    ax_paths.axvline(profile.value_optimum, color="gray", linestyle=":")
    p = result.parameter(pid)
    ax_paths.set_xlabel(f"{pid} [{p.unit or 'model'}]")
    ax_paths.set_ylabel("log10(parameter / optimum)")
    ax_paths.grid(alpha=0.3)
    if len(result.pids) > 1:
        ax_paths.legend(fontsize="x-small", title="other parameters")
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_all(
    result: IdentifiabilityResult, output_dir: Path, image_format: str = "svg"
) -> list[Path]:
    """Write the overview figure and the figure of every profile.

    Args:
        result: result of the analysis.
        output_dir: directory of the figures.
        image_format: format of the figures.

    Returns:
        The paths of the figures, the overview first.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"profiles.{image_format}"]
    plot_profiles(result, path=paths[0])
    for pid in result.profiles:
        path = output_dir / f"profile_{pid}.{image_format}"
        plot_profile(result, pid, path=path)
        paths.append(path)
    return paths
