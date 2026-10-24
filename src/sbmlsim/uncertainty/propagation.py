"""The propagation of uncertain values through simulations.

`propagate` runs the variables of an `UncertaintyModel` through one
simulation or several scenarios: the draws of the variables (correlated,
quasi-random, see `sbmlsim.uncertainty.draws`) are one dimension `draw` of a
`Scan`, which `Simulator.run` simulates, so a model is compiled once and the
points run in the pool. The result does not depend on the number of workers.

**Scenarios with common random numbers.** Several simulations, e.g. a control
and a treatment, are run with the same draws, so a difference or a ratio of an
output between them is paired and its Monte Carlo variance is much smaller
than with independent draws (Glasserman 2003). An input variable is applied
to the scenarios which declare it, at the same quantile of the distribution
of every scenario, a parameter to every scenario.

**Adaptive number of draws.** Without `n` the propagation draws batches of
quasi-random points, doubling their number (powers of two keep the balance of
the Sobol points), until the half widths of the 95 % intervals of the mean,
the standard deviation and the 5 % and 95 % quantiles of every output are at
most `rel_tol` times the standard deviation of the output (0.1 by default,
which takes about 2000 draws of a normal output and more of a skewed one,
whose tail quantiles are less precise), after the adaptive Monte Carlo procedure of
the GUM (JCGM 101:2008, 7.9), with the errors of
`sbmlsim.uncertainty.statistics`, which are conservative for quasi-random
draws; with `hypotheses` it draws until every hypothesis is decided after the
adjustment for their multiplicity, which the anytime-valid statements of
`sbmlsim.uncertainty.testing` allow (more draws never reverse a decision). At
`n_max` (the last batch is cut to it) a warning says that the tolerance was
not reached or which hypotheses are undecided. The history of the batches is
recorded, and the result of the batches is the one of a single scan of all
the draws: the same draws of Sobol or random points, the labels, the values
and the errors of every batch.

The `UncertaintyResult` has the results of the scenarios, the variables and
the record (seed, design, number of draws, history, covariances), and gives
the summaries with their Monte Carlo errors, the values of an output over the
draws and the contrasts between scenarios.

Glasserman P (2003). Monte Carlo Methods in Financial Engineering. Springer.
JCGM 101:2008. Evaluation of measurement data, Supplement 1 to the Guide to
the expression of uncertainty in measurement: propagation of distributions
using a Monte Carlo method.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from sbmlsim.result.scan import ScanResult
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.scan import Design, Dimension, Scan
from sbmlsim.simulator.simulator import Simulator
from sbmlsim.simulator.worker import MAX_ERRORS
from sbmlsim.uncertainty.draws import draw
from sbmlsim.uncertainty.sources import UncertaintyModel, Variable
from sbmlsim.uncertainty.statistics import (
    first_order_given_data,
    mean_se,
    quantile_ci,
    sd_se,
)
from sbmlsim.uncertainty.testing import (
    Hypothesis,
    TestReport,
    TestResult,
    family_level,
    probability,
)
from sbmlsim.units import Quantity

logger = logging.getLogger(__name__)

#: the dimension of the draws in the results
DRAW: str = "draw"

#: the name of the scenario of a single simulation
DEFAULT_SCENARIO: str = "default"

#: the quantiles of a summary by default
QUANTILES: tuple[float, ...] = (0.025, 0.05, 0.5, 0.95, 0.975)


def _magnitudes(values: Any) -> np.ndarray:
    """Get the magnitudes of draws, the unit is the one of the variable."""
    if isinstance(values, Quantity):
        return np.asarray(values.magnitude, dtype=float)
    return np.asarray(values, dtype=float)


def _per_draw(values: xr.DataArray) -> np.ndarray:
    """Get the values of an output which is one value per draw.

    Raises:
        ValueError: for an output with another dimension, e.g. a timecourse,
            whose draws would be pooled over its time.
    """
    if values.dims != (DRAW,):
        raise ValueError(
            f"The output '{values.name}' is not one value per draw, it has the "
            f"dimensions {values.dims}; reduce the timecourse with an "
            f"observable, e.g. `at`, `max` or a `PK` parameter."
        )
    return np.asarray(values.values, dtype=float)


def _variable_dict(variable: Variable) -> dict[str, Any]:
    """Get a variable as a dictionary of JSON types."""
    return {
        "key": variable.key,
        "targets": list(variable.targets),
        "distribution": variable.distribution.to_dict(),
        "kind": variable.kind,
        "source": variable.source,
        "by_scenario": {s: d.to_dict() for s, d in variable.by_scenario.items()},
    }


@dataclass
class UncertaintyResult:
    """The result of a propagation, see the module.

    Attributes:
        results: scenario -> the result of the scan over the draws.
        variables: the variables of the analysis.
        record: what the propagation did, JSON types: the seed, the design,
            the number of draws, the history, the variables and the
            covariances.
    """

    results: dict[str, ScanResult]
    variables: list[Variable] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def scenarios(self) -> tuple[str, ...]:
        """Get the names of the scenarios."""
        return tuple(self.results)

    def _scenario(self, scenario: str | None) -> str:
        """Get the scenario of a call, the only one by default.

        Raises:
            ValueError: without a scenario of several, or for an unknown one.
        """
        if scenario is None:
            if len(self.results) != 1:
                raise ValueError(
                    f"The result has the scenarios {list(self.results)}, name one."
                )
            return next(iter(self.results))
        if scenario not in self.results:
            raise ValueError(
                f"The scenario '{scenario}' is none of {list(self.results)}."
            )
        return scenario

    def values(self, output: str, scenario: str | None = None) -> xr.DataArray:
        """Get the values of an output over the draws.

        Args:
            output: a variable of the result, e.g. `pk.cmax` or `[C]`.
            scenario: the scenario, the only one by default.

        Returns:
            The values with the dimension `draw` (and the time of a
            timecourse), the unit in `attrs["units"]`.

        Raises:
            ValueError: for an output which the result does not have.
        """
        result = self.results[self._scenario(scenario)]
        if output not in result.ds.data_vars:
            raise ValueError(
                f"The output '{output}' is none of {sorted(map(str, result.ds.data_vars))}."
            )
        values = result.ds[output].copy()
        values.attrs["units"] = result.units.get(output, "")
        return values

    def contrast(
        self,
        output: str,
        a: str,
        b: str,
        kind: Literal["ratio", "difference"] = "ratio",
    ) -> xr.DataArray:
        """Get the paired contrast of an output between two scenarios.

        The draws of the scenarios are the same, so the contrast of a draw is
        the effect of the scenarios for these values of the variables.

        Args:
            output: the output.
            a: the scenario of the numerator, or the minuend.
            b: the scenario of the denominator, or the subtrahend.
            kind: `ratio` (`a / b`) or `difference` (`a - b`).

        Returns:
            The contrast over the draws, named `<output>:<a>/<b>` or
            `<output>:<a>-<b>`.

        Raises:
            ValueError: for another kind.
        """
        x, y = self.values(output, a), self.values(output, b)
        if kind == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                # a zero denominator is an infinite ratio, 0 / 0 a failed draw
                contrast = x / y
            contrast.attrs["units"] = ""
            return contrast.rename(f"{output}:{a}/{b}")
        if kind == "difference":
            contrast = x - y
            contrast.attrs["units"] = x.attrs.get("units", "")
            return contrast.rename(f"{output}:{a}-{b}")
        raise ValueError(f"The kind {kind!r} is none of 'ratio' and 'difference'.")

    def probability(
        self,
        output: str | xr.DataArray,
        *,
        scenario: str | None = None,
        **kwargs: Any,
    ) -> TestResult:
        """Get the probability that an output satisfies a hypothesis.

        Args:
            output: an output, or values over the draws, e.g. a `contrast`.
            scenario: the scenario of an output, the only one by default.
            kwargs: the hypothesis and the threshold, see
                `sbmlsim.uncertainty.testing.probability`.

        Returns:
            The statement with its Monte Carlo interval, which names the
            output of a given scenario `<output>:<scenario>`, as a contrast
            names its scenarios.
        """
        values = (
            self._scenario_values(output, scenario)
            if isinstance(output, str)
            else output
        )
        return probability(values, **kwargs)

    def _scenario_values(self, output: str, scenario: str | None) -> xr.DataArray:
        """Get the values of an output, named `<output>:<scenario>` for a scenario."""
        values = self.values(output, scenario)
        return values if scenario is None else values.rename(f"{output}:{scenario}")

    def hypothesis_values(self, hypothesis: Hypothesis) -> xr.DataArray:
        """Get the values of the output of a hypothesis: of its scenario or contrast.

        The values are named `<output>:<scenario>` for a hypothesis with a
        scenario and as the `contrast` for one with a contrast, so its
        statement names them.
        """
        if hypothesis.contrast is not None:
            a, b = hypothesis.contrast
            return self.contrast(hypothesis.output, a, b, hypothesis.kind)
        return self._scenario_values(hypothesis.output, hypothesis.scenario)

    def test(
        self,
        hypotheses: Sequence[Hypothesis],
        adjust: Literal["bonferroni", "none"] = "bonferroni",
    ) -> TestReport:
        """Get the statements of hypotheses, adjusted for their multiplicity.

        Args:
            hypotheses: the hypotheses.
            adjust: the adjustment, see `sbmlsim.uncertainty.testing.TestReport`.

        Returns:
            The report of the statements.
        """
        return TestReport(
            [h.evaluate(self.hypothesis_values(h)) for h in hypotheses], adjust
        )

    def outputs(self) -> list[str]:
        """Get the outputs which are a value per draw, the values per simulation."""
        result = next(iter(self.results.values()))
        return [
            str(name)
            for name, data in result.ds.data_vars.items()
            if data.dims == (DRAW,) and str(name) != "status"
        ]

    def summary(
        self,
        outputs: Sequence[str] | None = None,
        quantiles: Sequence[float] = QUANTILES,
        level: float = 0.95,
    ) -> pd.DataFrame:
        """Get the summaries of outputs with their Monte Carlo errors.

        Args:
            outputs: the outputs, values per draw; every one by default.
            quantiles: the quantiles.
            level: the level of the intervals of the quantiles.

        Returns:
            One row per scenario and output: `scenario`, `output`, `unit`,
            `n` (the draws which did not fail), `n_failed`, `mean`, `mean_se`, `sd`,
            `sd_se` and per quantile `q<q>` with `q<q>_low` and `q<q>_high`,
            the ends of its distribution-free interval, see
            `sbmlsim.uncertainty.statistics`.
        """
        rows: list[dict[str, Any]] = []
        for scenario in self.scenarios:
            for output in outputs if outputs is not None else self.outputs():
                values = _per_draw(self.values(output, scenario))
                mean, mean_error, n = mean_se(values)
                sd, sd_error, _ = sd_se(values)
                row: dict[str, Any] = {
                    "scenario": scenario,
                    "output": output,
                    "unit": self.results[scenario].units.get(output, ""),
                    "n": n,
                    "n_failed": int(values.size - n),
                    "mean": mean,
                    "mean_se": mean_error,
                    "sd": sd,
                    "sd_se": sd_error,
                }
                for q in quantiles:
                    estimate, low, high, _ = quantile_ci(values, q, level)
                    row[f"q{q:g}"] = estimate
                    row[f"q{q:g}_low"] = low
                    row[f"q{q:g}_high"] = high
                rows.append(row)
        return pd.DataFrame(rows)

    def _draws_of(self, variable: Variable, scenario: str) -> np.ndarray | None:
        """Get the draws of a variable from the coordinates of a result."""
        coords = self.results[scenario].ds.coords
        for target in variable.targets:
            for name in (target, f"{DRAW}.{target}"):
                if name in coords:
                    return np.asarray(coords[name].values, dtype=float)
        return None

    def contributions(self, output: str, scenario: str | None = None) -> pd.DataFrame:
        """Get the contribution of every variable to the variance of an output.

        The first-order index of every variable from the draws of the
        propagation, see `statistics.first_order_given_data`; no further
        simulations. A variable of a covariance is flagged `correlated`: its
        index includes the effect of the variables it is correlated with.

        Args:
            output: an output which is a value per draw.
            scenario: the scenario, the only one by default.

        Returns:
            The variables, `variable`, `index` and `correlated`, the largest
            index first.
        """
        name = self._scenario(scenario)
        y = _per_draw(self.values(output, name))
        correlated = {
            target for c in self.record.get("covariance", []) for target in c["targets"]
        }
        rows = []
        for variable in self.variables:
            x = self._draws_of(variable, name)
            if x is None:
                continue
            rows.append(
                {
                    "variable": variable.key,
                    "index": first_order_given_data(x, y),
                    "correlated": variable.key in correlated,
                }
            )
        return (
            pd.DataFrame(rows, columns=pd.Index(["variable", "index", "correlated"]))
            .sort_values("index", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

    def to_netcdf(self, path: str | Path) -> None:
        """Write the result into a directory: a netCDF file per scenario and the record.

        Args:
            path: the directory, created if it does not exist.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for k, result in enumerate(self.results.values()):
            result.to_netcdf(directory / f"scenario_{k}.nc")
        (directory / "record.json").write_text(
            json.dumps(
                {
                    "scenarios": list(self.results),
                    "record": self.record,
                    "variables": [_variable_dict(v) for v in self.variables],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_netcdf(cls, path: str | Path) -> UncertaintyResult:
        """Read a result written by `to_netcdf`."""
        from sbmlsim.simulation.sampling.distributions import distribution_from_dict

        directory = Path(path)
        content = json.loads((directory / "record.json").read_text(encoding="utf-8"))
        results = {
            scenario: ScanResult.from_netcdf(directory / f"scenario_{k}.nc")
            for k, scenario in enumerate(content["scenarios"])
        }
        variables = [
            Variable(
                v["key"],
                tuple(v["targets"]),
                distribution_from_dict(v["distribution"]),
                v["kind"],
                v["source"],
                by_scenario={
                    s: distribution_from_dict(d)
                    for s, d in v.get("by_scenario", {}).items()
                },
            )
            for v in content["variables"]
        ]
        return cls(results, variables, content["record"])


def _scenario_values(
    variables: Sequence[Variable],
    draws: Mapping[str, Any],
    simulation: Simulation,
) -> dict[str, Any]:
    """Get the values of the targets of a scenario: its inputs and every parameter."""
    declared = {use.target for use in simulation.uncertain_values()}
    values: dict[str, Any] = {}
    for variable in variables:
        for target in variable.targets:
            if variable.source == "parameter" or target in declared:
                values[target] = draws[variable.key]
    return values


def _precise(values: np.ndarray, rel_tol: float) -> tuple[bool, float]:
    """Check whether the statistics of the draws of an output are precise enough.

    The half widths of the 95 % intervals of the mean, the standard deviation
    and the 5 % and 95 % quantiles must be at most `rel_tol` times the
    standard deviation of the output, see the module.

    Returns:
        Whether they are and the largest half width relative to the standard
        deviation.
    """
    finite = values[~np.isnan(values)]
    if finite.size < 8:
        return False, math.inf
    sd, sd_error, _ = sd_se(finite)
    if sd == 0.0:
        return True, 0.0
    widths = [1.96 * mean_se(finite)[1], 1.96 * sd_error]
    for q in (0.05, 0.95):
        _estimate, low, high, _n = quantile_ci(finite, q)
        widths.append(
            0.5 * (high - low) if np.isfinite(low) and np.isfinite(high) else math.inf
        )
    largest = max(widths) / sd
    return largest <= rel_tol, float(largest)


def _merged_attrs(dss: Sequence[xr.Dataset], n: int) -> dict[str, Any]:
    """Get the attributes of the batches of a scenario as the ones of one scan.

    The dimension of the draws has the labels and values of every batch and
    the options of its design the number of all the draws; the errors of the
    failed points are the ones of every batch.
    """
    attrs = copy.deepcopy(dss[0].attrs)
    scan = attrs.get("scan")
    if isinstance(scan, dict):
        (dimension,) = [d for d in scan["dimensions"] if d["id"] == DRAW]
        others = [
            next(d for d in ds.attrs["scan"]["dimensions"] if d["id"] == DRAW)
            for ds in dss[1:]
        ]
        for other in others:
            dimension["labels"] = list(dimension["labels"]) + list(other["labels"])
            for target, value in dimension["values"].items():
                more = other["values"][target]
                if isinstance(value, dict):
                    value["value"] = list(value["value"]) + list(more["value"])
                else:
                    dimension["values"][target] = list(value) + list(more)
        design = dimension.get("design")
        if isinstance(design, dict):
            design["options"] = design.get("options", {}) | {"start": 0, "n": n}
    if "errors" in attrs:
        attrs["errors"] = [e for ds in dss for e in ds.attrs.get("errors", [])][
            :MAX_ERRORS
        ]
    return attrs


def propagate(
    model: Any,
    simulation: Simulation | Mapping[str, Simulation],
    uq: UncertaintyModel,
    observables: Sequence[Any] = (),
    *,
    outputs: Sequence[str] | None = None,
    hypotheses: Sequence[Hypothesis] = (),
    adjust: Literal["bonferroni", "none"] = "bonferroni",
    n: int | None = None,
    rel_tol: float = 0.1,
    n_start: int = 256,
    n_max: int = 16384,
    design: Literal["sobol", "lhs", "random"] | None = None,
    seed: int | None = None,
    simulator: Simulator | None = None,
    keep: Sequence[str] | None = None,
    time: Any = None,
    on_error: Literal["raise", "flag"] = "flag",
) -> UncertaintyResult:
    """Propagate the uncertain values through a simulation or scenarios, see the module.

    Args:
        model: the model, loaded or a source.
        simulation: the simulation, or the scenarios by their name.
        uq: what is uncertain.
        observables: the observables of the run, e.g. a `PK`.
        outputs: the outputs of the adaptive stopping and of the summary,
            values per simulation; every value per simulation by default,
            none with hypotheses.
        hypotheses: the hypotheses of the adaptive stopping, which draws until
            every one is decided after the adjustment for their multiplicity.
        adjust: the adjustment of the hypotheses, see `TestReport`.
        n: the number of draws; `None` for the adaptive number.
        rel_tol: the tolerance of the adaptive stopping, the half width of the
            intervals of the statistics relative to the standard deviation of
            an output.
        n_start: the draws of the first batch of the adaptive stopping.
        n_max: the most draws of the adaptive stopping.
        design: `sobol`, `lhs` (only with `n`, a Latin hypercube has no
            continuation) or `random`; by default `random` with hypotheses,
            whose statements assume independent draws, else `sobol`.
        seed: the seed, `None` draws one, which the record keeps.
        simulator: the simulator, `Simulator()` by default.
        keep: the timecourses the results keep, see `Simulator.run`.
        time: a grid of times the timecourses are interpolated onto.
        on_error: `flag` (a failed draw is `NaN` and counted) or `raise`.

    Returns:
        The result.

    Raises:
        ValueError: without uncertain values, see `UncertaintyModel.variables`,
            for outputs which are no values per draw, without outputs or
            hypotheses for the adaptive stopping, for an output of the
            adaptive stopping without a draw which did not fail, or for a
            Latin hypercube of an adaptive number of draws.
    """
    scenarios = (
        {DEFAULT_SCENARIO: simulation}
        if isinstance(simulation, Simulation)
        else dict(simulation)
    )
    if design is None:
        design = "random" if hypotheses else "sobol"
    if n is None and design == "lhs":
        raise ValueError(
            "A Latin hypercube has no continuation, its points of more draws are "
            "another hypercube: give the number of draws `n`, or use 'sobol' or "
            "'random' for the adaptive number of draws."
        )
    for hypothesis in hypotheses:
        names = (
            hypothesis.contrast
            if hypothesis.contrast is not None
            else ([hypothesis.scenario] if hypothesis.scenario is not None else [])
        )
        unknown = [name for name in names if name not in scenarios]
        if unknown:
            raise ValueError(
                f"The hypothesis about '{hypothesis.output}' names the scenarios "
                f"{unknown}, which are none of {list(scenarios)}."
            )
    variables = uq.variables(scenarios)
    if not variables:
        raise ValueError(
            "The analysis has no uncertain values: declare inputs as `Uncertain` "
            "or give parameters with their distributions."
        )
    simulator = simulator if simulator is not None else Simulator()
    loaded = simulator.load(model)
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))

    batches: dict[str, list[xr.Dataset]] = {name: [] for name in scenarios}
    history: list[dict[str, Any]] = []

    def run_batch(start: int, size: int) -> None:
        for name, scenario in scenarios.items():
            # the same points in every scenario, each with its own marginals
            # and the references of its own pre-initialization
            draws = draw(
                variables,
                uq.covariance,
                size,
                design=design,
                seed=seed,
                start=start,
                model=loaded,
                simulation=scenario,
                scenario=name,
            )
            values = _scenario_values(variables, draws, scenario)
            dimension = Dimension(
                DRAW,
                values=values,
                labels=np.arange(start, start + size),
                design=Design(
                    method="propagation",
                    distributions={
                        v.key: v.marginal(name).to_dict() for v in variables
                    },
                    options={"seed": seed, "design": design, "start": start, "n": size},
                ),
            )
            result = simulator.run(
                loaded,
                Scan(scenario, [dimension]),
                list(observables) or None,
                time=time,
                keep=keep,
                on_error=on_error,
            )
            batches[name].append(result.ds)

    def combined(total: int) -> dict[str, ScanResult]:
        results = {}
        for name, dss in batches.items():
            if len(dss) == 1:
                results[name] = ScanResult(dss[0])
                continue
            ds = xr.concat(dss, dim=DRAW, combine_attrs="override", join="outer")
            ds.attrs = _merged_attrs(dss, total)
            results[name] = ScanResult(ds)
        return results

    def check_outputs() -> None:
        if outputs is None:
            return
        known = {
            str(name) for ds in next(iter(batches.values())) for name in ds.data_vars
        }
        unknown = [name for name in outputs if name not in known]
        if unknown:
            raise ValueError(
                f"The outputs {unknown} are no variables of the result: {sorted(known)}."
            )
        for name in outputs:
            _per_draw(next(iter(batches.values()))[0][name])

    if n is not None:
        run_batch(0, n)
        check_outputs()
        total = n
    else:
        total = min(n_start, n_max)
        run_batch(0, total)
        check_outputs()
        while True:
            results = combined(total)
            result = UncertaintyResult(results)
            names = (
                list(outputs)
                if outputs is not None
                else ([] if hypotheses else result.outputs())
            )
            if not names and not hypotheses:
                raise ValueError(
                    "The adaptive propagation needs outputs which are a value per "
                    "simulation, e.g. of a `PK` observable, or hypotheses; give "
                    "`outputs`, `hypotheses` or `n`."
                )
            checks = []
            for scenario, scenario_result in results.items():
                for name in names:
                    values = _per_draw(scenario_result.ds[name])
                    if np.isnan(values).all():
                        raise ValueError(
                            f"The output '{name}' of the scenario '{scenario}' has "
                            f"no draw which did not fail, see the attribute "
                            f"'errors' of its result."
                        )
                    if np.isinf(values).any():
                        raise ValueError(
                            f"The output '{name}' of the scenario '{scenario}' has "
                            f"infinite draws, whose mean and standard deviation "
                            f"the adaptive stopping cannot judge: give `n` or "
                            f"hypotheses about it."
                        )
                    checks.append(_precise(values, rel_tol))
            # the hypotheses at the levels of their report, evaluated once
            m = sum(h.at_least is not None for h in hypotheses)
            undecided = sum(
                h.evaluate(
                    result.hypothesis_values(h),
                    family_level(h.level, m, adjust)
                    if h.at_least is not None
                    else h.level,
                ).decision
                == "undecided"
                for h in hypotheses
            )
            converged = all(ok for ok, _ in checks) and undecided == 0
            entry: dict[str, Any] = {"n": total, "converged": converged}
            if checks:
                entry["precision"] = max(d for _, d in checks)
            if hypotheses:
                entry["undecided"] = undecided
            history.append(entry)
            if converged:
                break
            if total >= n_max:
                logger.warning(
                    "The propagation reached %s draws without its tolerance %s or "
                    "with %s undecided hypotheses, the largest half width of an "
                    "interval is %s standard deviations.",
                    total,
                    rel_tol,
                    undecided,
                    entry.get("precision"),
                )
                break
            size = min(total, n_max - total)
            run_batch(total, size)
            total += size

    record = {
        "seed": seed,
        "design": design,
        "n": total,
        "rel_tol": rel_tol if n is None else None,
        "history": history,
        "variables": [_variable_dict(v) for v in variables],
        "covariance": [c.to_dict() for c in uq.covariance],
        "scenarios": list(scenarios),
        "hypotheses": [h.to_dict() for h in hypotheses],
        "adjust": adjust,
    }
    return UncertaintyResult(combined(total), list(variables), record)
