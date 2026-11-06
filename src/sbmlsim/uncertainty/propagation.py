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

**Blocks and checkpoints.** A `Propagation` of a given number of draws runs
in blocks of draws (`block_size`), which run separately, e.g. as units on
several machines, and merge to the result of the whole propagation: every
block takes its rows of the points of the design of all the draws, so a Latin
hypercube stays one hypercube. With a `checkpoint` directory every block, and
every batch of the adaptive number of draws, is written as netCDF
(`<checkpoint>/<scenario>/draws-<start>-<stop>.nc`, the name of the scenario
percent-encoded) and read instead of simulated when it exists, so the blocks
of separate instances merge and an interrupted propagation resumes with the
same result; a block of another propagation (another model, engine,
integrator settings, observables, kept timecourses, grid of times or draws)
is refused. A ragged timecourse is padded to the longest block, as a run pads
it to its longest point.

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
import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
import xarray as xr

from sbmlsim.atomic import atomic_path
from sbmlsim.engine import Engine
from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.result.netcdf import json_attributes
from sbmlsim.result.scan import POINT, ScanResult, pad_points
from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.scan import Design, Dimension, Scan
from sbmlsim.simulator.simulator import Simulator, settings_record
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

#: the attribute of a checkpoint file with the configuration of its block
CHECKPOINT: str = "propagation"


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


def _directories(names: Sequence[str]) -> dict[str, str]:
    """Get the directories of the checkpoints of the scenarios.

    A directory is the name of its scenario percent-encoded, every character
    but letters, digits and `_.-~` (`urllib.parse.quote`), so the encoding
    is reversible and has no separator, and `.` and `..` are `%2E` and
    `%2E%2E`.

    Returns:
        Scenario -> its directory.

    Raises:
        ValueError: for an empty name or two names whose directories differ
            only in case, which a file system may not tell apart.
    """
    directories: dict[str, str] = {}
    folded: dict[str, str] = {}
    for name in names:
        if not name:
            raise ValueError(
                "A scenario with a checkpoint needs a name, its directory."
            )
        directory = quote(name, safe="")
        if directory in (".", ".."):
            directory = directory.replace(".", "%2E")
        other = folded.setdefault(directory.casefold(), name)
        if other != name:
            raise ValueError(
                f"The scenarios '{other}' and '{name}' differ only in case, their "
                f"directories of the checkpoints may be one directory."
            )
        directories[name] = directory
    return directories


def _model_fingerprint(model: SBMLModel) -> str:
    """Get the SHA-256 of what a loaded model simulates.

    Its engine, the SBML it loaded (`SBMLModel.sbml`), its added parameters,
    its changes, its selections and its optimization level.
    """
    content = repr(
        (
            str(model.engine),
            model.sbml,
            sorted((str(k), repr(v)) for k, v in (model.parameters or {}).items()),
            sorted((str(k), str(v)) for k, v in (model.changes or {}).items()),
            repr(model.selections),
            str(getattr(model, "jit_optimization", None)),
        )
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _time_record(time: Any) -> Any:
    """Get the grid of times of a run as JSON types, `None` without one."""
    if time is None:
        return None
    if isinstance(time, Quantity):
        return {
            "value": np.asarray(time.magnitude, dtype=float).ravel().tolist(),
            "unit": str(time.units),
        }
    return np.asarray(time, dtype=float).ravel().tolist()


class Propagation:
    """A propagation of uncertain values in blocks of draws, see the module.

    A propagation of a given number of draws `n` runs in blocks of draws,
    `blocks`, which run separately, e.g. on several machines, and merge to the
    result of the whole propagation (`run`): every block takes its rows of the
    `n` points of the design, so a Latin hypercube stays one hypercube, and the
    result of the blocks is the one of a single block of all the draws. The
    adaptive number of draws runs its rounds one after the other, every round
    a block of the continued sequence.

    With a `checkpoint` directory every block is written as
    `<checkpoint>/<scenario>/draws-<start>-<stop>.nc` once it ran, atomically,
    and a block whose files exist for every scenario is read instead of
    simulated, so the blocks of separate instances merge and a run resumes
    from the blocks it finished. The directory of a scenario is its name
    percent-encoded (`urllib.parse.quote`, `.` and `..` as `%2E` and
    `%2E%2E`). A file has the configuration of its block, which a read
    compares and removes, see `_read`. A checkpointed block is used as it is
    read, so its result does not depend on whether it was simulated or read.

    Attributes:
        scenarios: the simulations by the name of their scenario.
        variables: the uncertain variables.
        seed: the seed, drawn when none was given.
        design: the design of the draws.
        n: the number of draws, `None` for the adaptive number.
        block_size: the draws of a block, `None` for one block of `n`.
        checkpoint: the directory of the checkpoints, `None` for none.
    """

    def __init__(
        self,
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
        checkpoint: str | Path | None = None,
        block_size: int | None = None,
    ) -> None:
        """Check the propagation and load its model, see `propagate`.

        Args:
            model: the model, loaded or a source.
            simulation: the simulation, or the scenarios by their name.
            uq: what is uncertain.
            observables: the observables of the run, e.g. a `PK`.
            outputs: the outputs of the adaptive stopping and of the summary.
            hypotheses: the hypotheses of the adaptive stopping.
            adjust: the adjustment of the hypotheses, see `TestReport`.
            n: the number of draws; `None` for the adaptive number.
            rel_tol: the tolerance of the adaptive stopping.
            n_start: the draws of the first round of the adaptive stopping.
            n_max: the most draws of the adaptive stopping.
            design: `sobol`, `lhs` (only with `n`) or `random`.
            seed: the seed, `None` draws one, which `seed` keeps.
            simulator: the simulator, `Simulator()` by default.
            keep: the timecourses the results keep, see `Simulator.run`.
            time: a grid of times the timecourses are interpolated onto.
            on_error: `flag` (a failed draw is `NaN` and counted) or `raise`.
            checkpoint: the directory of the checkpoints of the blocks.
            block_size: the draws of a block of `n`; one block by default.

        Raises:
            ValueError: see `propagate`.
        """
        scenarios = (
            {DEFAULT_SCENARIO: simulation}
            if isinstance(simulation, Simulation)
            else dict(simulation)
        )
        if design is None:
            design = "random" if hypotheses else "sobol"
        if n is not None and n < 1:
            raise ValueError(f"The number of draws `n` {n} must be positive.")
        if n_start < 1:
            raise ValueError(f"The first round `n_start` {n_start} must be positive.")
        if n_max < 1:
            raise ValueError(f"The most draws `n_max` {n_max} must be positive.")
        if n is None and design == "lhs":
            raise ValueError(
                "A Latin hypercube has no continuation, its points of more draws "
                "are another hypercube: give the number of draws `n`, or use "
                "'sobol' or 'random' for the adaptive number of draws."
            )
        if block_size is not None:
            if block_size < 1:
                raise ValueError(f"The block size {block_size} must be positive.")
            if n is None:
                raise ValueError(
                    "A block size needs the number of draws `n`: the blocks of the "
                    "adaptive number of draws are its rounds."
                )
        self._directories = (
            _directories(list(scenarios)) if checkpoint is not None else {}
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
        self.scenarios: dict[str, Simulation] = scenarios
        self.variables: list[Variable] = list(variables)
        self.design: Literal["sobol", "lhs", "random"] = design
        self.n: int | None = n
        self.block_size: int | None = block_size
        self.checkpoint: Path | None = (
            Path(checkpoint) if checkpoint is not None else None
        )
        self._uq = uq
        self._observables = list(observables)
        self._outputs = list(outputs) if outputs is not None else None
        self._hypotheses = list(hypotheses)
        self._adjust: Literal["bonferroni", "none"] = adjust
        self._rel_tol = rel_tol
        self._n_start = n_start
        self._n_max = n_max
        self._keep = keep
        self._time = time
        self._on_error: Literal["raise", "flag"] = on_error
        self._simulator = simulator if simulator is not None else Simulator()
        self._loaded = self._simulator.load(model)
        # what a checkpoint of a block was run with besides its scan, see
        # `_read`: the observables, the kept timecourses, the grid of times
        # and the errors give the variables of a block
        self._configuration: dict[str, Any] = json_attributes(
            {
                "model": _model_fingerprint(self._loaded),
                "engine": self._simulator.engine.value,
                "integrator_settings": settings_record(
                    self._simulator.integrator_settings
                ),
                "observables": [o.to_dict() for o in self._observables],
                "keep": list(keep) if keep is not None else None,
                "time": _time_record(time),
                "on_error": on_error,
            }
        )
        # the references of the relative distributions are read with
        # roadrunner, see `sbmlsim.simulation.sampling.references`: a model of
        # another engine is loaded in roadrunner once, not once per block and
        # scenario, and only for a relative distribution
        relative = any(
            v.marginal(name).is_relative for v in self.variables for name in scenarios
        )
        self._reference_model = (
            load_model(self._loaded, Engine.ROADRUNNER) if relative else self._loaded
        )
        self.seed: int = (
            seed if seed is not None else int(np.random.default_rng().integers(2**31))
        )
        # the `n` points of the design per scenario, drawn once
        self._points: dict[str, dict[str, Any]] = {}
        # the results of the blocks which ran or were read, per scenario
        self._results: dict[tuple[int, int], dict[str, xr.Dataset]] = {}

    def blocks(self) -> list[range]:
        """Get the blocks of the draws of `n`.

        Returns:
            The draws `[k * block_size, min(n, (k + 1) * block_size))`, one
            block of all `n` draws without a block size.

        Raises:
            ValueError: without `n`, whose rounds are not known before they run.
        """
        if self.n is None:
            raise ValueError(
                "The blocks of a propagation need the number of draws `n`: the "
                "rounds of the adaptive number of draws are not known before "
                "they run."
            )
        size = self.block_size if self.block_size is not None else self.n
        return [range(k, min(self.n, k + size)) for k in range(0, self.n, size)]

    def _check_block(self, block: range) -> None:
        """Check that a block is consecutive draws of the propagation.

        Raises:
            ValueError: for a block which is empty, has a step or is not
                within the draws of `n`.
        """
        if block.step != 1 or len(block) == 0 or block.start < 0:
            raise ValueError(
                f"The block {block} is no range of consecutive draws from 0."
            )
        if self.n is not None and block.stop > self.n:
            raise ValueError(
                f"The block {block} is not within the {self.n} draws of the "
                f"propagation."
            )

    def _draws(self, block: range, name: str) -> dict[str, Any]:
        """Get the draws of a block of a scenario.

        With `n` the rows of the `n` points of the design, which are drawn
        once, so a Latin hypercube is one hypercube in every block; without
        `n` the points of the block continue the sequence.
        """
        arguments = {
            "design": self.design,
            "seed": self.seed,
            "model": self._reference_model,
            "simulation": self.scenarios[name],
            "scenario": name,
        }
        if self.n is None:
            return draw(
                self.variables,
                self._uq.covariance,
                len(block),
                start=block.start,
                **arguments,
            )
        if name not in self._points:
            # the same points in every scenario, each with its own marginals
            # and the references of its own pre-initialization
            self._points[name] = draw(
                self.variables, self._uq.covariance, self.n, start=0, **arguments
            )
        return {
            key: values[block.start : block.stop]
            for key, values in self._points[name].items()
        }

    def _scan(self, block: range, name: str) -> Scan:
        """Get the scan of the draws of a block of a scenario."""
        scenario = self.scenarios[name]
        values = _scenario_values(self.variables, self._draws(block, name), scenario)
        dimension = Dimension(
            DRAW,
            values=values,
            labels=np.arange(block.start, block.stop),
            design=Design(
                method="propagation",
                distributions={
                    v.key: v.marginal(name).to_dict() for v in self.variables
                },
                options={
                    "seed": self.seed,
                    "design": self.design,
                    "start": block.start,
                    "n": len(block),
                },
            ),
        )
        return Scan(scenario, [dimension])

    def _path(self, block: range, name: str) -> Path:
        """Get the checkpoint file of a block of a scenario."""
        if self.checkpoint is None:
            raise ValueError("The propagation has no checkpoint.")
        directory = self._directories[name]
        return self.checkpoint / directory / f"draws-{block.start}-{block.stop}.nc"

    def _read(self, block: range, name: str) -> xr.Dataset:
        """Read the checkpoint of a block of a scenario.

        The file has the configuration of its block in the attribute
        `CHECKPOINT`, which is removed: the model (`_model_fingerprint`), the
        engine, the integrator settings, the observables, the kept
        timecourses, the grid of times and `on_error`, which give its
        variables, and its scan, i.e. the simulation and the draws.

        Raises:
            ValueError: if the file is the block of another propagation,
                naming what differs.
        """
        path = self._path(block, name)
        ds = ScanResult.from_netcdf(path).ds
        found = ds.attrs.pop(CHECKPOINT, None)
        found = found if isinstance(found, dict) else {}
        expected = self._configuration | json_attributes(
            {"scan": self._scan(block, name).to_dict()}
        )
        found = found | {
            "scan": json_attributes({"scan": ds.attrs.get("scan")})["scan"]
        }
        differ = sorted(k for k in expected if expected[k] != found.get(k))
        if differ:
            raise ValueError(
                f"The checkpoint '{path}' is a block of another propagation, it "
                f"differs in {', '.join(differ)}: use another checkpoint directory."
            )
        return ds

    def run_block(self, block: range) -> None:
        """Run the draws of a block in every scenario.

        A block which ran or was read before is not run again. With a
        checkpoint a block whose files exist for every scenario is read, else
        it is simulated and written, see the class.

        Args:
            block: consecutive draws, within `n` if given.

        Raises:
            ValueError: for a block which is no consecutive draws of the
                propagation or a checkpoint of another propagation.
        """
        self._check_block(block)
        key = (block.start, block.stop)
        if key in self._results:
            return
        if self.checkpoint is not None and all(
            self._path(block, name).exists() for name in self.scenarios
        ):
            logger.info(
                "The draws %s-%s of the propagation are read from '%s'.",
                block.start,
                block.stop,
                self.checkpoint,
            )
            self._results[key] = {
                name: self._read(block, name) for name in self.scenarios
            }
            return
        results: dict[str, xr.Dataset] = {}
        for name in self.scenarios:
            result = self._simulator.run(
                self._loaded,
                self._scan(block, name),
                self._observables or None,
                time=self._time,
                keep=self._keep,
                on_error=self._on_error,
            )
            if self.checkpoint is not None:
                path = self._path(block, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with atomic_path(path) as temporary:
                    ScanResult(
                        result.ds.assign_attrs({CHECKPOINT: self._configuration})
                    ).to_netcdf(temporary)
                # the block as it is read, so it does not depend on whether it
                # was simulated or read
                results[name] = self._read(block, name)
            else:
                results[name] = result.ds
        self._results[key] = results

    def _combined(self, blocks: Sequence[range]) -> dict[str, ScanResult]:
        """Get the results of the scenarios of consecutive blocks from 0.

        The results of the blocks are the one of a single block of all the
        draws, see `_merged_attrs`; a ragged timecourse is padded to the
        longest block, as a run pads it to its longest draw.
        """
        total = blocks[-1].stop
        results = {}
        for name in self.scenarios:
            dss = [self._results[(b.start, b.stop)][name] for b in blocks]
            rows = max(ds.sizes.get(POINT, 0) for ds in dss)
            dss = [pad_points(ds, rows) for ds in dss]
            if len(dss) == 1:
                results[name] = ScanResult(dss[0])
                continue
            ds = xr.concat(dss, dim=DRAW, combine_attrs="override", join="outer")
            ds.attrs = _merged_attrs(dss, total)
            results[name] = ScanResult(ds)
        return results

    def _check_outputs(self, block: range) -> None:
        """Check the outputs on the results of a block.

        Raises:
            ValueError: for an output which is no variable or no value per draw.
        """
        if self._outputs is None:
            return
        dss = self._results[(block.start, block.stop)]
        known = {str(name) for ds in dss.values() for name in ds.data_vars}
        unknown = [name for name in self._outputs if name not in known]
        if unknown:
            raise ValueError(
                f"The outputs {unknown} are no variables of the result: {sorted(known)}."
            )
        first = next(iter(dss.values()))
        for name in self._outputs:
            _per_draw(first[name])

    def run(self) -> UncertaintyResult:
        """Run the propagation: every block of `n` or the adaptive rounds.

        Returns:
            The result, see `propagate`.

        Raises:
            ValueError: see `propagate`.
        """
        history: list[dict[str, Any]] = []
        if self.n is not None:
            blocks = self.blocks()
            for block in blocks:
                self.run_block(block)
            self._check_outputs(blocks[0])
        else:
            blocks = [range(min(self._n_start, self._n_max))]
            self.run_block(blocks[0])
            self._check_outputs(blocks[0])
            while True:
                total = blocks[-1].stop
                entry = self._judge(self._combined(blocks), total)
                history.append(entry)
                if entry["converged"]:
                    break
                if total >= self._n_max:
                    logger.warning(
                        "The propagation reached %s draws without its tolerance %s "
                        "or with %s undecided hypotheses, the largest half width "
                        "of an interval is %s standard deviations.",
                        total,
                        self._rel_tol,
                        entry.get("undecided", 0),
                        entry.get("precision"),
                    )
                    break
                blocks.append(range(total, total + min(total, self._n_max - total)))
                self.run_block(blocks[-1])
        record = {
            "seed": self.seed,
            "design": self.design,
            "n": blocks[-1].stop,
            "rel_tol": self._rel_tol if self.n is None else None,
            "history": history,
            "variables": [_variable_dict(v) for v in self.variables],
            "covariance": [c.to_dict() for c in self._uq.covariance],
            "scenarios": list(self.scenarios),
            "hypotheses": [h.to_dict() for h in self._hypotheses],
            "adjust": self._adjust,
        }
        return UncertaintyResult(self._combined(blocks), list(self.variables), record)

    def _judge(self, results: dict[str, ScanResult], total: int) -> dict[str, Any]:
        """Judge the draws of a round of the adaptive stopping.

        Returns:
            The entry of the history: the draws, whether the statistics are
            precise and the hypotheses decided, the largest relative half
            width and the number of undecided hypotheses.

        Raises:
            ValueError: without outputs and hypotheses, for an output without a
                draw which did not fail or with infinite draws.
        """
        result = UncertaintyResult(results)
        names = (
            self._outputs
            if self._outputs is not None
            else ([] if self._hypotheses else result.outputs())
        )
        if not names and not self._hypotheses:
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
                        f"The output '{name}' of the scenario '{scenario}' has no "
                        f"draw which did not fail, see the attribute 'errors' of "
                        f"its result."
                    )
                if np.isinf(values).any():
                    raise ValueError(
                        f"The output '{name}' of the scenario '{scenario}' has "
                        f"infinite draws, whose mean and standard deviation the "
                        f"adaptive stopping cannot judge: give `n` or hypotheses "
                        f"about it."
                    )
                checks.append(_precise(values, self._rel_tol))
        # the hypotheses at the levels of their report, evaluated once
        m = sum(h.at_least is not None for h in self._hypotheses)
        undecided = sum(
            h.evaluate(
                result.hypothesis_values(h),
                family_level(h.level, m, self._adjust)
                if h.at_least is not None
                else h.level,
            ).decision
            == "undecided"
            for h in self._hypotheses
        )
        entry: dict[str, Any] = {
            "n": total,
            "converged": all(ok for ok, _ in checks) and undecided == 0,
        }
        if checks:
            entry["precision"] = max(d for _, d in checks)
        if self._hypotheses:
            entry["undecided"] = undecided
        return entry


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
    checkpoint: str | Path | None = None,
    block_size: int | None = None,
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
        checkpoint: the directory the blocks are written into and read from,
            see `Propagation`; `None` keeps them in memory only.
        block_size: the draws of a block of `n`, see `Propagation`; one block
            of all the draws by default.

    Returns:
        The result.

    Raises:
        ValueError: without uncertain values, see `UncertaintyModel.variables`,
            for outputs which are no values per draw, without outputs or
            hypotheses for the adaptive stopping, for an output of the
            adaptive stopping without a draw which did not fail, for a Latin
            hypercube of an adaptive number of draws, for a number of draws or
            a block size, `n_start` or `n_max` which is not positive, for a
            block size without `n`, with a checkpoint for an empty scenario
            name or two which differ only in case, or for a checkpoint of
            another propagation.
    """
    return Propagation(
        model,
        simulation,
        uq,
        observables,
        outputs=outputs,
        hypotheses=hypotheses,
        adjust=adjust,
        n=n,
        rel_tol=rel_tol,
        n_start=n_start,
        n_max=n_max,
        design=design,
        seed=seed,
        simulator=simulator,
        keep=keep,
        time=time,
        on_error=on_error,
        checkpoint=checkpoint,
        block_size=block_size,
    ).run()
