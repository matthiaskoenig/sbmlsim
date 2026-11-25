"""Compare the four simulation engines of sbmlsim: the load, one simulation and a scan.

The engines of `sbmlsim.engine.Engine` - sbmlsim-core, roadrunner, COPASI
and AMICI, in the order of `ENGINES`, which the tables and the figure keep -
simulate the same models through the same simulator, so their speed and their
agreement are measured in one run on the four packaged models
(`repressilator`, `demo`, `midazolam`, `hctz`) and on the large model
`perfusion` of `benchmarks/models`, which is no part of the package, see
`MODELS`:

```bash
# the table of every model and every engine which is installed
uv run python scripts/engine_benchmark.py

# a quick look at one model, with the measurement as JSON next to the table
uv run python scripts/engine_benchmark.py --models hctz --points 200 \
    --out table.md --json benchmark.json

# the cold scans of one engine next to its warm ones, in a process of its own
uv run python scripts/engine_benchmark.py --models hctz --engines core --cold

# the warm scans on another output grid, 26 output times instead of 101
uv run python scripts/engine_benchmark.py --models hctz --steps 25

# the tables and the figure of the documentation
uv run python scripts/engine_benchmark.py --workers 8 \
    --out docs/benchmarks-tables.md --figure docs/images/benchmarks/engines.svg
```

The tables are printed and `--out` writes them, which is the copy to paste,
and `--figure` draws the comparison of the engines, see `figure`: a warning
of an engine goes to the console as well. Nothing regenerates
`docs/benchmarks.md`, so a new measurement is a run of this script and a
commit of its tables and its figure.

The numbers of a run belong to the machine it ran on: pin the process to a
fixed set of cores, one per worker of the scans
(`taskset -c 0,2,4,6,8,10,12,14 uv run python scripts/engine_benchmark.py
--workers 8`, the cores of `docs/benchmarks.md`), and leave the machine alone
while it runs, as `bench/RESULTS.md` of sbmlsim-core does, else the cells
move by tens of percent between runs. Pinning the process to one core while
its scans run on several workers measures nothing: every worker would run on
that core.

Every timed cell is a median of medians: a round is the median of its timed
calls and the cell is the median of its rounds (`Measurement`). A mean
carries a round which was slow because the machine was doing something else
into the cell; a median of enough rounds does not, which is what makes a
measurement on a machine which is not perfectly idle worth reading.
The engines take turns within a model and a model is measured on its own, so
a change of the load of the machine which outlasts their turns reaches every
engine of a comparison alike; one which is shorter than a phase of the run
can still move a single cell, which is why a cell against a measurement of
another day is read as such. A large model carries fewer points and shorter
rounds of its own
(`ModelBenchmark`, `ModelConfiguration`), since one of its simulations is
tens of seconds in some engines.

**What is measured.** Per model and engine:

- `load s`: `load_model` with the engine, `repetitions` loads of a round
  in this process, i.e. with the model already compiled; `first compile s` is
  the first load of that model in this process, the one which compiles it
  (the LLJIT of roadrunner, the WebAssembly module of sbmlsim-core). It is
  measured once per model and engine, while the runtime of sbmlsim-core is
  created once per process (51 ms, `sbmlsim.model.model_core.runtime` is
  cached), so of a run of several models only the first model of the core
  pays the runtime in this column. AMICI compiles a model into a cache on
  disk, once per machine and not once per process, which the measurement
  warms before it times anything: its compilation from cold, minutes per
  model, is not measured and not reported, and the first load of its process
  is the import of the compiled module.
- `compile s` and `instantiate s`: sbmlsim-core only, the compilation of
  the SBML into its module and the instantiation of that module in the
  runtime, which `load s` of the other engines does not break out. They are
  no part of `load s` of the core either, which finds the module compiled
  (`sbmlsim.model.model_core.compiled_model` keeps it per SBML and process),
  and they are part of its `first compile s`.
- `simulate ms`: one `Simulator.simulate` of the simulation of the model at
  the defaults of `Simulator`, `repetitions` simulations after `warmups` in
  a round and `rounds` rounds, the engines of the model taking turns within
  each of them, which is the method of `bench/RESULTS.md`. The
  defaults are the tolerances of a run and not the tighter ones of the
  agreement, which is measured in simulations of its own: every speed of the
  table is the speed of a run as a user gets it.
- `scan serial s` and `scan parallel s`: a scan of `points` doses or
  parameter values of that simulation, `Simulator(n_workers=1)` and
  `Simulator(n_workers=workers)`, each one run per round after
  one run which `--cold` reports. A parallel scan is measured for `workers > 1` only. The times
  are therefore warm: they exclude the pool of worker processes, which
  `sbmlsim.parallel` keeps per process and number of workers, the model
  every worker loads and compiles and the first batch of the threads of the
  sbmlsim-core, so a measurement which includes that first run measures
  something else and is not a contradiction: it is several times larger for
  an engine whose parallel scan runs in worker processes, which carries the
  pool and one compilation of the model per worker, while sbmlsim-core,
  which starts no pool, pays only its own compilation there. `--cold`
  reports that first run as `scan serial cold s` and `scan parallel cold s`,
  which is a cold figure of the first model and engine of its process only,
  since the pool is kept and the runtime of the core is created once per
  process: a cold figure of one engine is a run of its own
  (`--engines core --cold`), which is how `docs/benchmarks.md` measures the
  cold and the warm scan side by side on the machine it names.
- `max rel. diff.` and `max abs. diff.`: the agreement with roadrunner, see
  `agreement`.

The tables are narrow and the rows of every one of them are the same in the
same order (`markdown`): what every model was measured with, the loads, the
simulation and the scans, the ratio of every engine to sbmlsim-core, the
agreement, and the cells which were not measured. `--figure` draws the speeds of them, a panel per model and
measure, the dot of an engine its median (`figure`).

**The workers of a parallel scan are not the same thing in every engine**, so
the legend of the tables says what they are: python processes
for roadrunner, COPASI and AMICI, which load the model again in every worker
and hand the arrays of a chunk back, and threads of the engine inside one
python process for sbmlsim-core, which starts no pool at all. The two do
not scale alike, which is why the legend says so: a process runs the whole
point, its integration and the python around it, while a thread of the core
integrates and the rest of the run - the result of a chunk, the
interpolation, the observables - stays in the one python process of the run,
so the more output a point has, the less a scan of the core gains from its
threads (`--steps`, measured on the HCTZ model at 1000 points and 8 workers:
at 26 output times the core takes 308 ms against roadrunner's 413 ms, at 101
output times 417 ms against 494 ms and at 401 output times 905 ms against
795 ms, while its serial scan is half to three fifths of roadrunner's at
every one of them, see `docs/benchmarks.md`). A parallel scan of the other engines
starts worker processes, so a script which calls `measure` needs the
`if __name__ == "__main__":` guard, as this one has.

`--steps` scans every model on another equidistant output grid, which is
what the parallel scan of sbmlsim-core depends on most and what the
section The width of the output grid of `docs/benchmarks.md` measures; the
legend of the table names the grid of every model, so a run says which one it
measured.

**The rule of the agreement.** Every engine simulates the simulation of the
model at atol 1e-12 and rtol 1e-10, where the engines agree, and its output
is compared with the one of roadrunner over the columns both have, column by
column: the difference of a column is its largest difference divided by the
largest magnitude of that column in roadrunner, which is the metric the tests
of the engines use (`_scaled_error` of `tests/simulator/test_simulator_engines.py`),
and the largest of them is `max rel. diff.`. A column which is identically
zero in roadrunner has no relative difference - a naive `|c-u|/|c|` is
infinite or not a number there, and a value which passes through zero makes
it meaningless - so such a column is compared absolutely and enters
`max abs. diff.` instead. Both cells are therefore always numbers, and an
empty one means that roadrunner has no column of that kind. roadrunner is the
reference, its own differences are 0. A result of COPASI is not bitwise
reproducible across loads of a model, so its cells move a little between
runs, while roadrunner, AMICI and the core are bitwise.

An engine which is not installed is skipped and named below the table, and a
model an engine refuses (`UnsupportedModelError`) keeps its row with the
refusal as its note, so neither is read as a measurement which was slow. The
refusal can come at the load of the model or when the plan of its comparison
is checked, and an output of other rows than roadrunner's is noted the same
way (`agreement`): a cell which is noted after its load keeps what the load
measured and is not timed, so one awkward engine costs its own cells and not
the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Final, cast

import numpy as np

from sbmlsim import log
from sbmlsim.cgroup import available_cpus
from sbmlsim.engine import Engine, UnsupportedModelError, resolve_engine
from sbmlsim.model import SBMLModel, load_model
from sbmlsim.plot.chart import Chart, ChartAxis
from sbmlsim.resources import DEMO_SBML, HCTZ_SBML, MIDAZOLAM_SBML, REPRESSILATOR_SBML
from sbmlsim.result.timecourse import TimecourseResult
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import IntegratorSettings, Simulator

#: the engines in the order the table shows them, sbmlsim-core first, as
#: the documentation names them (`docs/simulation.md`); `Engine` declares them
#: in the order they were added, which is the order of `resolve_engine`
#: the name of an engine in the tables, the figure and the notes of this page.
#: The engine is named after its package, `sbmlsim-core`, as the documentation
#: names it; every other engine is named by its value.
ENGINE_NAMES: Final[dict[Engine, str]] = {Engine.CORE: "sbmlsim-core"}


def engine_label(engine: Engine) -> str:
    """Get the name of an engine in the tables and the figure.

    Args:
        engine: the engine.

    Returns:
        Its name, `sbmlsim-core` for the engine of that package and its value
        otherwise.
    """
    return ENGINE_NAMES.get(engine, engine.value)


ENGINES: Final[tuple[Engine, ...]] = (
    Engine.CORE,
    Engine.ROADRUNNER,
    Engine.COPASI,
    Engine.AMICI,
)

#: the engine every other one is compared with, see `agreement`: it is
#: simulated whether or not it is one of the measured engines, so its version
#: is part of every measurement, see `_versions`
REFERENCE: Final[Engine] = Engine.ROADRUNNER

#: the engine the ratios of the speeds are taken against, see `_ratio_table`:
#: sbmlsim-core, the first engine of `ENGINES` and the one the
#: documentation names first
REFERENCE_RATIO: Final[Engine] = Engine.CORE

#: the settings of the integrator at which the engines agree, see `agreement`
#: and the section Engines of `docs/simulation.md`
TIGHT: Final[IntegratorSettings] = {
    "absolute_tolerance": 1e-12,
    "relative_tolerance": 1e-10,
}

#: the distribution of an engine, whose version the measurement records
DISTRIBUTIONS: Final[dict[Engine, str]] = {
    Engine.ROADRUNNER: "libroadrunner",
    Engine.COPASI: "python-copasi",
    Engine.AMICI: "amici",
    Engine.CORE: "sbmlsim-core",
}

#: the timed simulations of a round, the median of this many after `WARMUPS`
REPETITIONS: Final[int] = 20

#: the rounds of the timed simulations and the scans, whose medians are the
#: sample the median of a cell is taken over, so a cell is `rounds`
#: measurements and never fewer than `MIN_ROUNDS`
ROUNDS: Final[int] = 5

#: the rounds below which a median has too small a sample to ignore a round
#: which was slow because the machine was doing something else
MIN_ROUNDS: Final[int] = 3

#: the simulations before the timed ones of a round, which are not measured
WARMUPS: Final[int] = 3

#: the points of the scans of a cell, which a model of its own number of
#: points replaces (`ModelBenchmark.points`)
POINTS: Final[int] = 1000

#: the colour of every engine in the figure, the slots of the categorical
#: theme the benchmark of sbmlsim-core draws with (`scripts/bench_figure.py`
#: of that repository): sbmlsim-core is its blue and roadrunner is the pink it
#: has there, so one engine is one colour across both projects, and COPASI and
#: AMICI follow in the order of the theme. The engine of every row is named
#: next to it, so a mark is never told apart by its colour alone
ENGINE_COLOURS: Final[dict[Engine, str]] = {
    Engine.CORE: "#2a78d6",
    Engine.ROADRUNNER: "#e87ba4",
    Engine.COPASI: "#eb6834",
    Engine.AMICI: "#1baf7a",
}

#: the resolution of the PNG of a figure, which `Chart.save` writes for a
#: `.png` file and ignores for an SVG
FIGURE_DPI: Final[int] = 200


@dataclass(frozen=True)
class _Measure:
    """One column of the figure, a timed field of `BenchmarkRow`.

    Attributes:
        field: the name of the field of the row.
        title: the title of the column.
        unit: the unit the field is drawn in.
        factor: what the seconds of the field are scaled by for that unit.
    """

    field: str
    title: str
    unit: str
    factor: float


#: the columns of the figure, a column which no cell was measured in left
#: out: the speed of a run, which is what the engines are compared on; the
#: loads of a model, which are paid once, are the table of `_loading_table`
MEASURES: Final[tuple[_Measure, ...]] = (
    _Measure("simulate_s", "one simulation", "ms", 1e3),
    _Measure("scan_serial_s", "a serial scan", "s", 1.0),
    _Measure("scan_parallel_s", "a parallel scan", "s", 1.0),
)

#: the models which are no part of the package, see `benchmarks/models/README.md`
BENCHMARK_MODELS: Final[Path] = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "models"
)

#: the liver of a machine perfusion, the large model of the benchmark
PERFUSION_SBML: Final[Path] = BENCHMARK_MODELS / "machine_perfusion_liver_flat.xml"


@dataclass(frozen=True)
class ModelBenchmark:
    """A model of the benchmark: what every engine loads, simulates and scans.

    Attributes:
        source: the SBML of the model, one of the packaged models.
        simulation: the simulation which is timed and compared.
        target: the entity the scan of the model sets, a dose or a parameter.
        low: the first value of the scan.
        high: the last value of the scan.
        points: the points of the scans of this model, `None` for the points
            of the measurement: a large model is scanned over fewer points,
            since one of its simulations is seconds in some engines.
        repetitions: the timed simulations of a round of this model, `None`
            for the ones of the measurement.
        warmups: the simulations before them, `None` for the ones of the
            measurement.
        rounds: the rounds of this model, `None` for the ones of the
            measurement; never fewer than `MIN_ROUNDS`, since the standard
            deviation of a cell is taken over them.
    """

    source: Path
    simulation: Simulation
    target: str
    low: float
    high: float
    points: int | None = None
    repetitions: int | None = None
    warmups: int | None = None
    rounds: int | None = None

    @property
    def output_times(self) -> int:
        """Get the number of output times of the simulation of the model.

        Every model of the benchmark has an equidistant output, `steps + 1`
        times, which every engine computes alike and which is what a point
        of a scan costs besides its integration, see `_output_line`.

        Returns:
            The number of output times.

        Raises:
            ValueError: if the simulation has no `steps`.
        """
        if self.simulation.steps is None:
            raise ValueError(
                f"The simulation which scans '{self.target}' has no steps, "
                "every model of the benchmark has an equidistant output."
            )
        return self.simulation.steps + 1

    def with_steps(self, steps: int) -> ModelBenchmark:
        """Get the model with another equidistant output grid, `--steps`.

        Every model of the benchmark has an equidistant output and no exact
        output times, see `MODELS`, so the grid is the one thing of its
        simulation which a measurement replaces.

        Args:
            steps: the steps of the output, `steps + 1` output times.

        Returns:
            The model with that output, the rest of its simulation kept: the
            pre-initialization is handed over as it was declared, an uncertain
            value as itself (`Simulation.uncertain_preinit`), so that a model
            of the benchmark which carries one keeps it.
        """
        simulation = Simulation(
            time_unit=self.simulation.time_unit,
            start=self.simulation.start,
            end=self.simulation.end,
            preinit_changes={
                **self.simulation.preinit_changes,
                **self.simulation.uncertain_preinit,
            },
            changes=list(self.simulation.changes),
            presimulation=self.simulation.presimulation,
            steps=steps,
            time_shift=self.simulation.time_shift,
        )
        return replace(self, simulation=simulation)

    def scan(self, points: int) -> Scan:
        """Get the scan of the model over `points` values of its target.

        Args:
            points: the points of the scan.

        Returns:
            The scan, the simulation of the model over one dimension.
        """
        values = np.linspace(self.low, self.high, points).tolist()
        return Scan(self.simulation, [Dimension("dim", values={self.target: values})])


#: the models of the benchmark, the packaged ones by their names: a day of an
#: oral dose of the two physiologically based models and 100 time units of
#: the two small ones, each with an equidistant output, which every engine
#: computes alike (the steps of the integrator are the engine's own)
MODELS: Final[dict[str, ModelBenchmark]] = {
    "repressilator": ModelBenchmark(
        source=REPRESSILATOR_SBML,
        simulation=Simulation(end=100, steps=100),
        target="n",
        low=1.5,
        high=4.0,
    ),
    "demo": ModelBenchmark(
        source=DEMO_SBML,
        simulation=Simulation(end=100, steps=100),
        target="Km_A",
        low=0.5,
        high=2.0,
    ),
    "midazolam": ModelBenchmark(
        source=MIDAZOLAM_SBML,
        simulation=Simulation(
            end=24 * 60, steps=100, preinit_changes={"PODOSE_mid": 10.0}
        ),
        target="PODOSE_mid",
        low=5.0,
        high=100.0,
    ),
    "hctz": ModelBenchmark(
        source=HCTZ_SBML,
        simulation=Simulation(
            end=24 * 60, steps=100, preinit_changes={"PODOSE_hctz": 25.0}
        ),
        target="PODOSE_hctz",
        low=5.0,
        high=100.0,
    ),
    # an hour of a normothermic machine perfusion of a liver, the large model:
    # 839 species and 1007 reactions against the 223 kB of HCTZ, scanned over
    # the flow of the pump, the input of such an experiment. One simulation of
    # it is tens of seconds in COPASI, so it is scanned over as many points as
    # the parallel scan has workers, one per worker, and measured in fewer and
    # shorter rounds; `MIN_ROUNDS` keeps the median a median of three
    "perfusion": ModelBenchmark(
        source=PERFUSION_SBML,
        simulation=Simulation(end=60, steps=100),
        target="Q_pump",
        low=0.0015,
        high=0.006,
        points=8,
        repetitions=3,
        warmups=1,
        rounds=MIN_ROUNDS,
    ),
}


@dataclass(frozen=True)
class Measurement:
    """A timing as the median of its rounds.

    A round is the median of the timed calls of that round (`_turns`), so one
    slow call does not move it, and the cell is the median of the rounds, so
    one slow round does not move it either. A measurement taken while the
    machine is doing something else is a few slow rounds among the others,
    which a mean carries into the cell and a median does not, so the median
    is what this benchmark reports, see the module.

    Attributes:
        median: the median of the rounds.
        rounds: the rounds it was taken over.
    """

    median: float
    rounds: int

    @classmethod
    def of(cls, values: Sequence[float]) -> Measurement:
        """Get the measurement of the times of the rounds.

        Args:
            values: the time of every round, at least one.

        Returns:
            The median of the rounds.

        Raises:
            ValueError: if there is not one round.
        """
        if not values:
            raise ValueError("A measurement needs the time of at least one round.")
        return cls(median=statistics.median(values), rounds=len(values))

    def scaled(self, factor: float) -> Measurement:
        """Get the measurement in another unit, e.g. milliseconds.

        Args:
            factor: what the median is multiplied by.

        Returns:
            The scaled measurement.
        """
        return replace(self, median=self.median * factor)


@dataclass(frozen=True)
class ModelConfiguration:
    """What one model was measured with, which a cell is read together with.

    A large model is scanned over fewer points and measured in fewer and
    shorter rounds than a small one (`ModelBenchmark`), so the configuration
    belongs to the model and not to the run, see the module.

    Attributes:
        output_times: the equidistant output times of its simulation.
        points: the points of its scans.
        repetitions: the timed simulations of a round.
        warmups: the simulations before them.
        rounds: the rounds the median is over.
    """

    output_times: int
    points: int
    repetitions: int
    warmups: int
    rounds: int


@dataclass(frozen=True)
class BenchmarkRow:
    """What was measured of one model and engine, see the module.

    A field is `None` for a measurement which was not made, which the table
    renders as an empty cell.

    Attributes:
        load_s: `load_model` with the model already compiled, seconds.
        compile_s: the compilation of the SBML by sbmlsim-core, seconds.
        instantiate_s: the instantiation of its module, seconds.
        first_compile_s: the first load of the process, which compiles,
            seconds.
        simulate_s: one simulation, seconds.
        scan_serial_s: the scan on one worker, seconds.
        scan_serial_cold_s: the first run of that scan in this process,
            `--cold`, seconds.
        scan_parallel_s: the scan on the workers of the run, seconds.
        scan_parallel_cold_s: the first run of that scan in this process,
            `--cold`, seconds.
        max_relative_difference: the agreement with roadrunner over the
            columns which are not zero in roadrunner, see `agreement`.
        max_absolute_difference: the agreement over the columns which are.
        note: why the cell was not measured, e.g. a model the engine
            refuses; a cell noted after its load keeps its load and has no
            simulation, no scan and no agreement, see the module.
    """

    load_s: Measurement | None = None
    compile_s: Measurement | None = None
    instantiate_s: Measurement | None = None
    first_compile_s: float | None = None
    simulate_s: Measurement | None = None
    scan_serial_s: Measurement | None = None
    scan_serial_cold_s: float | None = None
    scan_parallel_s: Measurement | None = None
    scan_parallel_cold_s: float | None = None
    max_relative_difference: float | None = None
    max_absolute_difference: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """The measurement of every model and engine, see the module.

    Attributes:
        rows: what was measured, by the model and the engine.
        repetitions: the timed simulations of a round.
        rounds: the rounds of the simulations and the scans.
        warmups: the simulations before the timed ones of a round.
        points: the points of the scans.
        workers: the workers of the parallel scan, processes or threads.
        cold: whether the first run of every scan is reported, `--cold`.
        steps: the output grid every model was scanned on, `--steps`, `None`
            for the grid of the model, see `MODELS`.
        configurations: what every model was measured with, by its name: a
            large model carries fewer points and shorter rounds of its own,
            so `points`, `repetitions`, `warmups` and `rounds` above are the
            defaults of the run and the configuration is the cell's, see
            `ModelConfiguration`.
        skipped: why an engine was not measured, by the engine.
        versions: the versions of python and of the engines.
    """

    rows: dict[tuple[str, Engine], BenchmarkRow]
    repetitions: int
    rounds: int
    warmups: int
    points: int
    workers: int
    cold: bool = False
    steps: int | None = None
    configurations: dict[str, ModelConfiguration] = field(default_factory=dict)
    skipped: dict[Engine, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Cell:
    """A model and an engine which is installed, prepared for the measurement.

    Attributes:
        name: the name of the model.
        engine: the engine.
        benchmark: the model of the benchmark.
        model: the model, loaded in the engine by the first, timed load.
        first_compile_s: that load, which compiled the model, seconds.
        scan: the scan of the model over the points of the measurement.
        serial: the simulator of the simulations and of the serial scan.
        parallel: the simulator of the parallel scan, `None` for one worker.
    """

    name: str
    engine: Engine
    benchmark: ModelBenchmark
    model: SBMLModel
    first_compile_s: float
    scan: Scan
    serial: Simulator
    parallel: Simulator | None

    @property
    def key(self) -> tuple[str, Engine]:
        """Get the key of the cell in the rows of a result."""
        return self.name, self.engine


def agreement(
    result: TimecourseResult, reference: TimecourseResult
) -> tuple[float | None, float | None]:
    """Compare the output of an engine with the one of roadrunner, see the module.

    The columns both results have are compared one by one: a column which is
    not identically zero in roadrunner by its largest difference divided by
    its largest magnitude in roadrunner, and a column which is by its largest
    absolute difference, since a relative difference of it is undefined. A
    value which is not finite in both results is left out of its column.

    Args:
        result: the output of the engine.
        reference: the output of roadrunner.

    Returns:
        The largest relative difference and the largest absolute difference,
        each `None` when the reference has no column of that kind.

    Raises:
        ValueError: if a common column has a different number of rows, i.e.
            the two engines did not simulate the same output.
    """
    relative: list[float] = []
    absolute: list[float] = []
    columns = set(result.columns)
    for column in reference.columns:
        if column not in columns:
            continue
        values, expected = result[column], reference[column]
        if values.shape != expected.shape:
            raise ValueError(
                f"The column '{column}' has {values.shape[0]} rows and "
                f"{expected.shape[0]} rows in roadrunner."
            )
        finite = np.isfinite(values) & np.isfinite(expected)
        if not finite.any():
            continue
        values, expected = values[finite], expected[finite]
        difference = float(np.max(np.abs(values - expected)))
        scale = float(np.max(np.abs(expected)))
        if scale > 0.0:
            relative.append(difference / scale)
        else:
            absolute.append(difference)
    return (
        max(relative) if relative else None,
        max(absolute) if absolute else None,
    )


def measure(
    models: Sequence[str],
    engines: Sequence[Engine],
    *,
    repetitions: int,
    rounds: int,
    warmups: int,
    points: int,
    workers: int,
    cold: bool = False,
    steps: int | None = None,
) -> BenchmarkResult:
    """Measure every model with every engine which is installed, see the module.

    Args:
        models: the names of the models, see `MODELS`.
        engines: the engines; one which is not installed is skipped.
        repetitions: the timed simulations of a round.
        rounds: the rounds of the simulations and of the scans.
        warmups: the simulations before the timed ones of a round.
        points: the points of the scans.
        workers: the workers of the parallel scan, processes or, for the
            sbmlsim-core, threads of the engine; `1` measures no parallel scan.
        cold: whether the first run of every scan is reported next to the
            warm median, see `_scans`.
        steps: the equidistant output grid of every model, `steps + 1` output
            times; `None` keeps the grid of the model, see `MODELS`.

    Returns:
        The measurement, see `markdown` for its table.

    Raises:
        ValueError: for a name which is no model of the benchmark, or an
            argument which is not positive.
    """
    _check(repetitions=repetitions, rounds=rounds, points=points, workers=workers)
    if warmups < 0:
        raise ValueError(f"The warm ups must not be negative, not {warmups}.")
    if steps is not None:
        _check(steps=steps)
    benchmarks = {name: _benchmark(name, steps) for name in models}
    available, skipped = _installed(engines)
    rows: dict[tuple[str, Engine], BenchmarkRow] = {}
    configurations: dict[str, ModelConfiguration] = {}
    # one model after the other, its engines taking turns within it: a
    # comparison is read down a model, so that is where a change of the load
    # of the machine has to hit every engine alike, and a model carries its
    # own points and rounds, which a measurement of all models at once could
    # not give it, see `ModelConfiguration`
    for name, benchmark in benchmarks.items():
        configuration = ModelConfiguration(
            output_times=benchmark.output_times,
            points=points if benchmark.points is None else benchmark.points,
            repetitions=(
                repetitions if benchmark.repetitions is None else benchmark.repetitions
            ),
            warmups=warmups if benchmark.warmups is None else benchmark.warmups,
            rounds=rounds if benchmark.rounds is None else benchmark.rounds,
        )
        configurations[name] = configuration
        prepared: list[_Cell] = []
        for engine in available:
            try:
                cell = _prepare(name, benchmark, engine, configuration.points, workers)
            except UnsupportedModelError as err:
                rows[name, engine] = BenchmarkRow(note=str(err))
                continue
            prepared.append(cell)
        if not prepared:
            continue
        rows.update(_loads(prepared, configuration.repetitions, configuration.rounds))
        reference = _reference(benchmark)
        cells: list[_Cell] = []
        for cell in prepared:
            try:
                relative, absolute = agreement(_tight(cell), reference)
            except ValueError as err:
                # a plan the engine refuses (`UnsupportedModelError`, which is
                # a `ValueError`) and an output of other rows than roadrunner's
                # (`agreement`) are this cell and not the run: the cell keeps
                # its note and is not timed, as a model an engine refuses at
                # its load is, since the measurement would raise again
                rows[cell.key] = replace(rows[cell.key], note=str(err))
                continue
            rows[cell.key] = replace(
                rows[cell.key],
                max_relative_difference=relative,
                max_absolute_difference=absolute,
            )
            cells.append(cell)
        if not cells:
            continue
        simulations = _simulations(
            cells,
            configuration.repetitions,
            configuration.warmups,
            configuration.rounds,
        )
        for key, measurement in simulations.items():
            rows[key] = replace(rows[key], simulate_s=measurement)
        for key, (first, warm) in _scans(
            cells, configuration.rounds, parallel=False
        ).items():
            rows[key] = replace(
                rows[key],
                scan_serial_s=warm,
                scan_serial_cold_s=first if cold else None,
            )
        if workers > 1:
            for key, (first, warm) in _scans(
                cells, configuration.rounds, parallel=True
            ).items():
                rows[key] = replace(
                    rows[key],
                    scan_parallel_s=warm,
                    scan_parallel_cold_s=first if cold else None,
                )
    return BenchmarkResult(
        rows=rows,
        repetitions=repetitions,
        rounds=rounds,
        warmups=warmups,
        points=points,
        workers=workers,
        cold=cold,
        steps=steps,
        configurations=configurations,
        skipped=skipped,
        versions=_versions(available),
    )


def markdown(result: BenchmarkResult) -> str:
    """Render a measurement as the tables of the documentation.

    Every measurement of a cell in one table is thirteen columns of numbers
    which no page displays well, so the measurements are narrow tables of the
    same rows in the same order, sbmlsim-core first: what every model was
    measured with, the loads, the simulation and the scans, the agreement,
    and the cells which were not measured. The lines after them say with what
    the measurement was made and what every column is, see the module.

    Args:
        result: the measurement.

    Returns:
        The tables and their legend.
    """
    return (
        "\n".join(
            [
                *_configuration_table(result),
                *_loading_table(result),
                *_simulating_table(result),
                *_ratio_table(result),
                *_agreement_table(result),
                *_notes_table(result),
                *_legend(result),
            ]
        )
        + "\n"
    )


def _table(
    header: Sequence[str], aligned: Sequence[str], rows: Sequence[Sequence[str]]
) -> list[str]:
    """Render one markdown table and the empty line after it.

    Args:
        header: the names of the columns.
        aligned: the alignment of every column, `---` or `---:`.
        rows: the cells of every row.

    Returns:
        The lines of the table.
    """
    return [
        f"| {' | '.join(header)} |",
        f"|{'|'.join(aligned)}|",
        *(f"| {' | '.join(row)} |" for row in rows),
        "",
    ]


def _configuration_table(result: BenchmarkResult) -> list[str]:
    """Render what every model was measured with, see `ModelConfiguration`.

    Args:
        result: the measurement.

    Returns:
        The lines of the table.
    """
    rows = [
        [
            name,
            str(configuration.output_times),
            str(configuration.points),
            str(configuration.repetitions),
            str(configuration.warmups),
            str(configuration.rounds),
        ]
        for name, configuration in result.configurations.items()
    ]
    return [
        "### What every model was measured with",
        "",
        *_table(
            [
                "model",
                "output times",
                "scan points",
                "simulations",
                "warm ups",
                "rounds",
            ],
            ["---", "---:", "---:", "---:", "---:", "---:"],
            rows,
        ),
    ]


def _loading_table(result: BenchmarkResult) -> list[str]:
    """Render the loads of every cell.

    Args:
        result: the measurement.

    Returns:
        The lines of the table.
    """
    rows = [
        [
            name,
            engine_label(engine),
            _measurement(row.load_s),
            _number(row.first_compile_s),
            _measurement(row.compile_s),
            _measurement(row.instantiate_s),
        ]
        for (name, engine), row in result.rows.items()
    ]
    return [
        "### Loading a model",
        "",
        *_table(
            [
                "model",
                "engine",
                "load s",
                "first compile s",
                "compile s",
                "instantiate s",
            ],
            ["---", "---", "---:", "---:", "---:", "---:"],
            rows,
        ),
    ]


def _simulating_table(result: BenchmarkResult) -> list[str]:
    """Render one simulation and the scans of every cell.

    Args:
        result: the measurement.

    Returns:
        The lines of the table.
    """
    header = ["model", "engine", "simulate ms", "scan serial s", "scan parallel s"]
    aligned = ["---", "---", "---:", "---:", "---:"]
    if result.cold:
        header[4:4] = ["scan serial cold s"]
        header.append("scan parallel cold s")
        aligned.extend(["---:", "---:"])
    rows = []
    for (name, engine), row in result.rows.items():
        cells = [
            name,
            engine_label(engine),
            _measurement(row.simulate_s, 1e3),
            _measurement(row.scan_serial_s),
            *([_number(row.scan_serial_cold_s)] if result.cold else []),
            _measurement(row.scan_parallel_s),
            *([_number(row.scan_parallel_cold_s)] if result.cold else []),
        ]
        rows.append(cells)
    return ["### One simulation and a scan", "", *_table(header, aligned, rows)]


#: the measures of the table of the ratios, the field of a row and its column
RATIO_MEASURES: Final[tuple[tuple[str, str], ...]] = (
    ("load_s", "load"),
    ("simulate_s", "one simulation"),
    ("scan_serial_s", "a serial scan"),
    ("scan_parallel_s", "a parallel scan"),
)


def _ratio(value: Measurement | None, reference: Measurement | None) -> str:
    """Render the ratio of a measurement to the one of sbmlsim-core.

    Args:
        value: the measurement of the engine, `None` where it was not measured.
        reference: the measurement of sbmlsim-core, `None` where the core
            was not measured or its median is 0.

    Returns:
        The ratio with three significant digits, `1` for the core itself and
        an empty cell where either side is missing.
    """
    if value is None or reference is None or not reference.median:
        return ""
    return f"{value.median / reference.median:.3g}"


def _ratio_table(result: BenchmarkResult) -> list[str]:
    """Render every measure of every engine as its ratio to sbmlsim-core.

    The time of an engine divided by the time of sbmlsim-core on the same
    model and measure, so the core is 1, a number above it is that many times
    the time of the core and a number below it is a cell the core loses. The
    ratios are what carries over from one machine to another, which the
    seconds of the tables above do not.

    Args:
        result: the measurement.

    Returns:
        The lines of the table, empty where sbmlsim-core was not measured.
    """
    if not any(engine is REFERENCE_RATIO for _name, engine in result.rows):
        return []
    header = ["model", "engine", *(column for _field, column in RATIO_MEASURES)]
    aligned = ["---", "---", *("---:" for _measure in RATIO_MEASURES)]
    rows = []
    for (name, engine), row in result.rows.items():
        reference = result.rows.get((name, REFERENCE_RATIO))
        if reference is None:
            continue
        rows.append(
            [
                name,
                engine_label(engine),
                *(
                    _ratio(getattr(row, field), getattr(reference, field))
                    for field, _column in RATIO_MEASURES
                ),
            ]
        )
    if not rows:
        return []
    return [
        f"### Against {engine_label(REFERENCE_RATIO)}",
        "",
        *_table(header, aligned, rows),
    ]


def _agreement_table(result: BenchmarkResult) -> list[str]:
    """Render the agreement of every cell with roadrunner.

    Args:
        result: the measurement.

    Returns:
        The lines of the table.
    """
    rows = [
        [
            name,
            engine_label(engine),
            _difference(row.max_relative_difference),
            _difference(row.max_absolute_difference),
        ]
        for (name, engine), row in result.rows.items()
    ]
    return [
        "### The agreement with roadrunner",
        "",
        *_table(
            ["model", "engine", "max rel. diff.", "max abs. diff."],
            ["---", "---", "---:", "---:"],
            rows,
        ),
    ]


def _notes_table(result: BenchmarkResult) -> list[str]:
    """Render the cells which were not measured, nothing where every one was.

    Args:
        result: the measurement.

    Returns:
        The lines of the table, empty for a measurement without a note.
    """
    rows = [
        [name, engine_label(engine), row.note]
        for (name, engine), row in result.rows.items()
        if row.note
    ]
    if not rows:
        return []
    return [
        "### What was not measured",
        "",
        *_table(["model", "engine", "why"], ["---", "---", "---"], rows),
    ]


def _measurement(value: Measurement | None, factor: float = 1.0) -> str:
    """Render a measurement as its median.

    Args:
        value: the measurement, `None` for one which was not made.
        factor: what it is scaled by, e.g. `1e3` for milliseconds.

    Returns:
        The median of the rounds,
        and an empty cell for a measurement which was not made.
    """
    if value is None:
        return ""
    scaled = value.scaled(factor)
    return f"{scaled.median:.3g}"


def figure_paths(path: Path, models: Sequence[str]) -> list[Path]:
    """Get the file of every model a figure is written to.

    Args:
        path: the file `--figure` names, whose stem the files share; they are
            SVG files.
        models: the models in order.

    Returns:
        The files, a model each, in the order they are written.
    """
    return [path.with_name(f"{path.stem}-{model}.svg") for model in models]


def figure(result: BenchmarkResult, path: Path) -> None:
    """Write the figures which compare the engines, one per model.

    A model gets a figure of its own, since the models of this benchmark are
    five orders of magnitude apart and are not comparable with each other
    anyway: each is its own simulation over its own number of points. Within
    a figure a plot is a measure and a row is an engine, on a logarithmic
    axis of its own, so the engines of a model are read against each other,
    which is what the figure is for.

    The dot of an engine is the median of its rounds, a `Measurement` keeps
    no spread, see the section The run of `docs/benchmarks.md`. A dot and not
    a bar, since a bar on a logarithmic axis encodes its length as the
    logarithm of its value; the engine of every row is named on the axis, so the identity of a mark is never its colour alone, and
    the numbers are the tables of `markdown`. The figures are drawn by the
    renderer of sbmlsim-core and written as SVG, `figure_paths`.

    Args:
        result: the measurement.
        path: the file `--figure` names; the files of the models are named
            after its stem, see `figure_paths`, and its directory is created.

    Raises:
        ValueError: for a measurement without a cell which was measured.
    """
    measures = [
        measure
        for measure in MEASURES
        if any(getattr(row, measure.field) for row in result.rows.values())
    ]
    models = [
        name
        for name in result.configurations
        if any(name == key[0] for key in result.rows)
    ]
    if not measures or not models:
        raise ValueError("The measurement has no cell to draw.")
    engines = list(dict.fromkeys(engine for _, engine in result.rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    for model, target in zip(models, figure_paths(path, models), strict=True):
        _model_figure(result, model, engines, measures).save(target, dpi=FIGURE_DPI)


def _model_figure(
    result: BenchmarkResult,
    model: str,
    engines: Sequence[Engine],
    measures: Sequence[_Measure],
) -> Chart:
    """Draw the figure of one model.

    Args:
        result: the measurement.
        model: the model of the figure.
        engines: the engines in the order of the rows, the first one on top.
        measures: the measures, a plot each.

    Returns:
        The chart, a plot per measure.
    """
    configuration = result.configurations.get(model)
    points = f", {configuration.points} points" if configuration else ""
    chart = Chart(
        width=100 + 300 * len(measures),
        height=170 + 34 * len(engines),
        title=f"{model}{points}: the median of the rounds of a cell",
        columns=len(measures),
    )
    names = [engine_label(engine) for engine in engines]
    for column, measure in enumerate(measures):
        plot = chart.plot(
            0,
            column,
            title=measure.title,
            x=ChartAxis(label=f"{measure.unit}, logarithmic", scale="log", grid=True),
            y=ChartAxis(
                scale="category",
                categories=names,
                reverse=True,
                label_visible=False,
            ),
            legend=False,
        )
        for engine in engines:
            value = getattr(
                result.rows.get((model, engine), BenchmarkRow()), measure.field
            )
            if value is None:
                continue
            plot.curve(
                [value.scaled(measure.factor).median],
                [engine_label(engine)],
                line="none",
                marker="circle",
                size=8,
                color=ENGINE_COLOURS[engine],
            )
    return chart


def json_payload(result: BenchmarkResult) -> dict[str, object]:
    """Get the measurement as the JSON of `--json`.

    Args:
        result: the measurement.

    Returns:
        The settings of the run, the versions, the skipped engines and a row
        per model and engine.
    """
    return {
        "repetitions": result.repetitions,
        "rounds": result.rounds,
        "warmups": result.warmups,
        "points": result.points,
        "workers": result.workers,
        "cold": result.cold,
        "steps": result.steps,
        "configurations": {
            name: asdict(configuration)
            for name, configuration in result.configurations.items()
        },
        "versions": result.versions,
        "skipped": {engine.value: reason for engine, reason in result.skipped.items()},
        "rows": [
            {"model": name, "engine": engine.value, **asdict(row)}
            for (name, engine), row in result.rows.items()
        ],
    }


def result_from_json(payload: dict[str, object]) -> BenchmarkResult:
    """Read a measurement back from the JSON of `--json`.

    The tables and the figure of a run are written from its result, so a
    measurement which was stored is rendered again without measuring again.

    Args:
        payload: what `json_payload` wrote.

    Returns:
        The measurement.

    Raises:
        ValueError: if a row names an engine which is not known.
    """

    def measurement(value: object) -> Measurement | None:
        if value is None:
            return None
        fields = cast(dict[str, object], value)
        return Measurement(
            median=float(cast(float, fields["median"])),
            rounds=int(cast(int, fields["rounds"])),
        )

    def number(value: object) -> float | None:
        return None if value is None else float(cast(float, value))

    rows: dict[tuple[str, Engine], BenchmarkRow] = {}
    for item in cast(list[dict[str, object]], payload["rows"]):
        model = str(item["model"])
        name = str(item["engine"])
        try:
            engine = Engine(name)
        except ValueError as error:
            raise ValueError(f"{name} is no engine of the benchmark") from error
        note = item.get("note")
        rows[(model, engine)] = BenchmarkRow(
            load_s=measurement(item.get("load_s")),
            compile_s=measurement(item.get("compile_s")),
            instantiate_s=measurement(item.get("instantiate_s")),
            first_compile_s=number(item.get("first_compile_s")),
            simulate_s=measurement(item.get("simulate_s")),
            scan_serial_s=measurement(item.get("scan_serial_s")),
            scan_serial_cold_s=number(item.get("scan_serial_cold_s")),
            scan_parallel_s=measurement(item.get("scan_parallel_s")),
            scan_parallel_cold_s=number(item.get("scan_parallel_cold_s")),
            max_relative_difference=number(item.get("max_relative_difference")),
            max_absolute_difference=number(item.get("max_absolute_difference")),
            note=None if note is None else str(note),
        )
    configurations = {
        name: ModelConfiguration(**cast(dict[str, int], fields))
        for name, fields in cast(dict[str, object], payload["configurations"]).items()
    }
    skipped = {
        Engine(name): str(reason)
        for name, reason in cast(dict[str, object], payload["skipped"]).items()
    }
    steps = payload["steps"]
    return BenchmarkResult(
        rows=rows,
        repetitions=int(cast(int, payload["repetitions"])),
        rounds=int(cast(int, payload["rounds"])),
        warmups=int(cast(int, payload["warmups"])),
        points=int(cast(int, payload["points"])),
        workers=int(cast(int, payload["workers"])),
        cold=bool(payload["cold"]),
        steps=None if steps is None else int(cast(int, steps)),
        configurations=configurations,
        versions=cast(dict[str, str], payload["versions"]),
        skipped=skipped,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command line tool.

    Args:
        argv: the arguments, `sys.argv[1:]` by default.

    Returns:
        The exit code, 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="the models of the benchmark, all of them by default",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=[engine.value for engine in Engine],
        default=[engine.value for engine in ENGINES],
        help="the engines in the order of the table, every one which is "
        "installed by default (sbmlsim-core first, see ENGINES)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS,
        help=f"the timed simulations of a round (default {REPETITIONS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help=f"the rounds of the simulations and the scans (default {ROUNDS})",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=POINTS,
        help=f"the points of the scans (default {POINTS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpus(),
        help="the workers of the parallel scan (default: every CPU)",
    )
    parser.add_argument(
        "--cold",
        action="store_true",
        help="report the first run of every scan next to its warm median",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="scan every model on an output of this many steps "
        "(default: the output of the model)",
    )
    parser.add_argument(
        "--json", type=Path, help="write the measurement as JSON into this file"
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        help="write the table and the figure of a measurement which "
        "`--json` stored, without measuring again",
    )
    parser.add_argument("--out", type=Path, help="write the table into this file")
    parser.add_argument(
        "--figure",
        type=Path,
        help="draw the comparison of the engines into this file, e.g. "
        "docs/images/benchmarks/engines.svg",
    )
    args = parser.parse_args(argv)
    if args.from_json:
        result = result_from_json(
            json.loads(args.from_json.read_text(encoding="utf-8"))
        )
    else:
        result = measure(
            args.models,
            [resolve_engine(engine) for engine in args.engines],
            repetitions=args.repetitions,
            rounds=args.rounds,
            warmups=WARMUPS,
            points=args.points,
            workers=args.workers,
            cold=args.cold,
            steps=args.steps,
        )
    table = markdown(result)
    print(table)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    if args.json:
        args.json.write_text(
            json.dumps(json_payload(result), indent=2) + "\n", encoding="utf-8"
        )
    if args.figure:
        figure(result, args.figure)
    return 0


def _check(**arguments: int) -> None:
    """Refuse an argument of the measurement which is not positive.

    Args:
        **arguments: the arguments by their names.

    Raises:
        ValueError: for an argument below 1.
    """
    for name, value in arguments.items():
        if value < 1:
            raise ValueError(
                f"The {name} of the benchmark must be 1 or more, not {value}."
            )


def _benchmark(name: str, steps: int | None = None) -> ModelBenchmark:
    """Get the model of the benchmark of a name.

    Args:
        name: the name of the model.
        steps: the output grid of `--steps`, the one of the model if `None`.

    Returns:
        The model.

    Raises:
        ValueError: for a name which is no model of the benchmark.
    """
    benchmark = MODELS.get(name)
    if benchmark is None:
        raise ValueError(
            f"'{name}' is no model of the benchmark, the models are: "
            f"{', '.join(MODELS)}."
        )
    return benchmark if steps is None else benchmark.with_steps(steps)


def _installed(
    engines: Sequence[Engine],
) -> tuple[list[Engine], dict[Engine, str]]:
    """Split the engines into the ones which are installed and the ones which are not.

    A simulator of an engine imports it when it is created, which is where an
    engine whose extra or bindings are missing names its install.

    Args:
        engines: the engines of the measurement, a repeated one once.

    Returns:
        The engines which are installed, in the order they were given, and
        why every other one was skipped.
    """
    available: list[Engine] = []
    skipped: dict[Engine, str] = {}
    for engine in dict.fromkeys(engines):
        try:
            Simulator(n_workers=1, engine=engine)
        except ImportError as err:
            skipped[engine] = str(err)
            continue
        available.append(engine)
    return available, skipped


def _prepare(
    name: str,
    benchmark: ModelBenchmark,
    engine: Engine,
    points: int,
    workers: int,
) -> _Cell:
    """Prepare a model and an engine for the measurement, which loads the model.

    The load is timed here: it is the first load of the model in this engine
    and in this process, the one which compiles it, and it is the load an
    engine which refuses the model raises in.

    Args:
        name: the name of the model.
        benchmark: the model of the benchmark.
        engine: the engine.
        points: the points of the scans.
        workers: the workers of the parallel scan.

    Returns:
        The cell.

    Raises:
        UnsupportedModelError: for a model the engine refuses.
    """
    if engine is Engine.AMICI:
        # AMICI compiles a model into its cache on disk, which takes minutes
        # and happens once per machine and not once per process: the cache is
        # warmed here, so that the timed load below is the import of the
        # compiled module, as the load of the other engines is the one of a
        # compiled model. It is announced because it is minutes in which the
        # logs of AMICI say nothing and the measurement looks like a hang
        print(
            f"warming the cache of AMICI for the model '{name}', which "
            "compiles it on the first run and takes minutes",
            file=sys.stderr,
            flush=True,
        )
        load_model(benchmark.source, engine)
    start = time.perf_counter()
    model = load_model(benchmark.source, engine)
    first = time.perf_counter() - start
    return _Cell(
        name=name,
        engine=engine,
        benchmark=benchmark,
        model=model,
        first_compile_s=first,
        scan=benchmark.scan(points),
        serial=Simulator(n_workers=1, engine=engine),
        parallel=None if workers <= 1 else Simulator(n_workers=workers, engine=engine),
    )


def _loads(
    cells: Sequence[_Cell], repetitions: int, rounds: int
) -> dict[tuple[str, Engine], BenchmarkRow]:
    """Measure the load of the model of every cell.

    The model of a cell was loaded once when it was prepared, the first load
    of the process and the one which compiles the model; the engines take
    turns over the loads after it, which find the model compiled.

    Args:
        cells: the cells of one model.
        repetitions: the loads of a cell in a round.
        rounds: the rounds the median is over.

    Returns:
        The row of every cell with its loads.
    """
    warm = _rounds(
        {
            cell.key: lambda cell=cell: load_model(cell.benchmark.source, cell.engine)
            for cell in cells
        },
        repetitions=repetitions,
        warmups=0,
        rounds=rounds,
    )
    rows: dict[tuple[str, Engine], BenchmarkRow] = {}
    for cell in cells:
        compile_s, instantiate_s = (
            _core_compilation(cell, repetitions, rounds)
            if cell.engine is Engine.CORE
            else (None, None)
        )
        rows[cell.key] = BenchmarkRow(
            load_s=warm[cell.key],
            compile_s=compile_s,
            instantiate_s=instantiate_s,
            first_compile_s=cell.first_compile_s,
        )
    return rows


def _core_compilation(
    cell: _Cell, repetitions: int, rounds: int
) -> tuple[Measurement, Measurement]:
    """Measure the compilation and the instantiation of a model by sbmlsim-core.

    The engine compiles the SBML of a model into a WebAssembly module and the
    runtime instantiates that module, which `CoreSBMLModel` does once per
    SBML and process, see `sbmlsim.model.model_core.compiled_model`; the two
    are measured here with the bindings, on the SBML of the model (which is
    the SBML the engine compiled: the benchmark adds no parameters) and at
    the compile options of `CoreSBMLModel`.

    Args:
        cell: the cell of sbmlsim-core.
        repetitions: the compilations of a round.
        rounds: the rounds the median is over.

    Returns:
        The compilation and the instantiation.
    """
    # imported here: nothing of the benchmark of another engine imports the
    # bindings of sbmlsim-core
    from sbmlsim.engine import import_core
    from sbmlsim.model.model_core import JACOBIAN, OPTIMIZE, runtime

    core = import_core()
    sbml = cell.model.sbml
    compiled = core.compile_sbml(sbml, optimize=OPTIMIZE, jacobian=JACOBIAN)
    times = _rounds(
        {
            "compile": lambda: core.compile_sbml(
                sbml, optimize=OPTIMIZE, jacobian=JACOBIAN
            ),
            "instantiate": lambda: runtime().load(compiled.wasm, compiled.description),
        },
        repetitions=repetitions,
        warmups=0,
        rounds=rounds,
    )
    return times["compile"], times["instantiate"]


def _reference(benchmark: ModelBenchmark) -> TimecourseResult:
    """Simulate a model with roadrunner at the tolerances of the agreement.

    Args:
        benchmark: the model of the benchmark.

    Returns:
        The output of roadrunner, the reference of every engine.
    """
    simulator = Simulator(n_workers=1, engine=REFERENCE, **TIGHT)
    return simulator.simulate(benchmark.source, benchmark.simulation)


def _tight(cell: _Cell) -> TimecourseResult:
    """Simulate the model of a cell at the tolerances of the agreement.

    Args:
        cell: the cell.

    Returns:
        The output of its engine.
    """
    simulator = Simulator(n_workers=1, engine=cell.engine, **TIGHT)
    return simulator.simulate(cell.model, cell.benchmark.simulation)


def _simulations(
    cells: Sequence[_Cell], repetitions: int, warmups: int, rounds: int
) -> dict[tuple[str, Engine], Measurement]:
    """Measure one simulation of every cell, the method of the module.

    Args:
        cells: the cells of one model, which take turns.
        repetitions: the timed simulations of a round.
        warmups: the simulations before them.
        rounds: the rounds the median is over.

    Returns:
        The measurement of one simulation of every cell.
    """
    functions = {
        cell.key: lambda cell=cell: cell.serial.simulate(
            cell.model, cell.benchmark.simulation
        )
        for cell in cells
    }
    return _rounds(functions, repetitions=repetitions, warmups=warmups, rounds=rounds)


def _scans(
    cells: Sequence[_Cell], rounds: int, *, parallel: bool
) -> dict[tuple[str, Engine], tuple[float, Measurement]]:
    """Measure the scan of every cell, serially or on the workers of the run.

    A scan is run once before the timed ones, which pays the pool of the
    processes of a parallel run of roadrunner, COPASI or AMICI and the models
    its workers load, and the threads of a run of sbmlsim-core. That first
    run is timed as well and returned next to the warm measurement, so the
    two measurements of a run are the same whether or not `--cold` reports
    the first one.

    Args:
        cells: the cells of one model, which take turns.
        rounds: the timed runs, the sample of the mean and the deviation.
        parallel: whether the scan runs on the workers of the measurement.

    Returns:
        The seconds of the first run and the measurement of the warm scan of
        every cell; a cell without a parallel simulator, i.e. a measurement
        of one worker, is left out.
    """
    functions: dict[tuple[str, Engine], Callable[[], object]] = {}
    for cell in cells:
        simulator = cell.parallel if parallel else cell.serial
        if simulator is None:
            continue
        functions[cell.key] = lambda simulator=simulator, cell=cell: simulator.run(
            cell.model, cell.scan, progress=False
        )
    # one run of every scan before the timed ones, and not one per round: a
    # scan is seconds; it is timed too, which is the cold run of `--cold`
    cold = {key: _timed(function) for key, function in functions.items()}
    warm = _rounds(functions, repetitions=1, warmups=0, rounds=rounds)
    return {key: (cold[key], measurement) for key, measurement in warm.items()}


def _rounds[K](
    functions: Mapping[K, Callable[[], object]],
    *,
    repetitions: int,
    warmups: int,
    rounds: int,
) -> dict[K, Measurement]:
    """Get the median of the rounds of every function.

    A round is the median of its timed calls (`_turns`), so one slow call does
    not move it, and the rounds are the sample, see `Measurement`.

    Args:
        functions: the functions by their keys.
        repetitions: the timed calls of a round.
        warmups: the calls before them, which are not timed.
        rounds: the rounds.

    Returns:
        The measurement of every function.
    """
    medians: dict[K, list[float]] = {key: [] for key in functions}
    for _ in range(rounds):
        for key, median in _turns(
            functions, repetitions=repetitions, warmups=warmups
        ).items():
            medians[key].append(median)
    return {key: Measurement.of(values) for key, values in medians.items()}


def _turns[K](
    functions: Mapping[K, Callable[[], object]], *, repetitions: int, warmups: int
) -> dict[K, float]:
    """Get the median time of every function, the functions taking turns.

    The functions are called one after the other, so a change of the load of
    the machine hits every one of them alike, as the benchmarks of
    `tests/simulator/test_benchmark.py` and of `bench/RESULTS.md` measure it.

    Args:
        functions: the functions by their keys.
        repetitions: the timed calls of every function.
        warmups: the calls before them, which are not timed.

    Returns:
        The median of the timed calls of every function.
    """
    times: dict[K, list[float]] = {key: [] for key in functions}
    for repetition in range(warmups + repetitions):
        for key, function in functions.items():
            seconds = _timed(function)
            if repetition >= warmups:
                times[key].append(seconds)
    return {key: statistics.median(values) for key, values in times.items()}


def _timed(function: Callable[[], object]) -> float:
    """Get the seconds one call of a function takes.

    Args:
        function: the function.

    Returns:
        The seconds.
    """
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def _versions(engines: Sequence[Engine]) -> dict[str, str]:
    """Get the versions of python, of sbmlsim and of the engines of a measurement.

    The reference engine is always one of them, also when it was not
    measured: it simulates the reference of the agreement of every table,
    see `REFERENCE`, so a measurement of sbmlsim-core alone still names
    the libroadrunner its agreement column was computed against.

    Args:
        engines: the engines which were measured.

    Returns:
        The version by name, without a distribution which is not installed.
    """
    names = [
        "sbmlsim",
        DISTRIBUTIONS[REFERENCE],
        *(DISTRIBUTIONS[engine] for engine in engines),
    ]
    versions = {"python": sys.version.split()[0]}
    for name in dict.fromkeys(names):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _number(value: float | None) -> str:
    """Render a time of the table, an empty cell for a measurement which was not made.

    Args:
        value: the time.

    Returns:
        The cell.
    """
    return "" if value is None else f"{value:.3g}"


def _difference(value: float | None) -> str:
    """Render a difference of the agreement, see `agreement`.

    Args:
        value: the difference.

    Returns:
        The cell, `0` for an engine which agrees bitwise.
    """
    if value is None:
        return ""
    return "0" if value == 0.0 else f"{value:.2e}"


def _count(number: int, word: str) -> str:
    """Render a number of things, the word in the plural for every other number.

    Args:
        number: the number.
        word: the word in the singular.

    Returns:
        The number and the word, e.g. `1 round` or `3 rounds`.
    """
    return f"{number} {word}" if number == 1 else f"{number} {word}s"


def _scan_line() -> str:
    """Get the legend line which says that the scan times are warm.

    The scans of a cell run once before they are timed, see
    `_scans`, and the pool of a parallel run is kept per process and number
    of workers (`sbmlsim.parallel.pool`), so the timed runs pay neither the
    pool, nor the model every worker loads and compiles, nor the first batch
    of the threads of sbmlsim-core. A reader who compares a figure of
    another measurement, which includes the first run, with a column of this
    table would otherwise read the difference as a contradiction. The line
    says this qualitatively and names no number: it is rendered into the
    table of every run, whatever its points, workers, models and machine,
    while a cold figure belongs to one of them, see `docs/benchmarks.md`.

    Returns:
        The line.
    """
    return (
        "The scan times are warm: every scan runs once before it is "
        "timed, so they exclude the pool of worker processes, which is kept "
        "per process and number of workers, the model every worker loads and "
        "compiles and the first batch of the threads of sbmlsim-core. A "
        "measurement which includes that first run is not the same "
        "measurement: it is several times larger for an engine whose "
        "parallel scan runs in worker processes, which carries the pool and "
        "one compilation of the model per worker, while sbmlsim-core, "
        "which starts no pool, pays only its own compilation there."
    )


def _cold_line() -> str:
    """Get the legend line of the cold columns of `--cold`.

    The first run of a scan in a process is a cold run only for the first
    model and engine which runs one: the pool of worker processes is kept per
    process and number of workers (`sbmlsim.parallel.pool`) and the runtime
    of sbmlsim-core is created once per process, so a later cell finds
    them and pays only what is its own. The line says so, since a reader
    would otherwise compare the cold cells of two engines of one run.

    Returns:
        The line.
    """
    return (
        "`scan serial cold s` and `scan parallel cold s` are that first run, "
        "timed: what the scan costs in a process which has not run one yet. "
        "They are a cold measurement of the first model and engine of the "
        "run only, since the pool of worker processes is kept per process "
        "and number of workers and the runtime of sbmlsim-core is created "
        "once per process, so every later cell finds them started. A cold "
        "figure of one engine is therefore a run of its own, with that "
        "engine alone."
    )


def _output_line(result: BenchmarkResult) -> str:
    """Get the legend line of the simulation and the scan of every model.

    What a point of the table costs depends on the horizon and the output of
    its simulation, which the parallel scan of sbmlsim-core depends on
    most: the line names them, so a row can be placed and reproduced.

    Args:
        result: the measurement.

    Returns:
        The line.
    """
    parts: list[str] = []
    for name in dict.fromkeys(name for name, _ in result.rows):
        benchmark = MODELS.get(name)
        if benchmark is None:
            continue
        if result.steps is not None:
            benchmark = benchmark.with_steps(result.steps)
        parts.append(
            f"`{name}` to {benchmark.simulation.end} in "
            f"{_count(benchmark.output_times, 'output time')}, its scan over "
            f"`{benchmark.target}` from {benchmark.low} to {benchmark.high}"
        )
    return (
        "The simulation of a model is its horizon with an equidistant "
        "output, which every engine computes alike (the steps of the "
        "integrator are each engine's own), in the time unit of the model: "
        + "; ".join(parts)
        + ". How much output a point has matters for the parallel scan of "
        "sbmlsim-core, see the workers of a parallel scan below."
    )


def _legend(result: BenchmarkResult) -> list[str]:
    """Get the lines which follow the table of a measurement.

    Args:
        result: the measurement.

    Returns:
        What it was measured with, what every column is and which engines
        were not measured.
    """
    versions = ", ".join(
        f"{name} {version}" for name, version in result.versions.items()
    )
    lines = [
        f"Measured with {versions}: every cell is the median of its rounds, "
        f"a round the median of its timed calls after its warm ups, and the "
        f"engines take turns within a "
        f"model, so a change of the load of the machine which outlasts their "
        f"turns reaches every engine of a comparison alike, and one which is "
        f"shorter than a phase of the run can still move a single cell. The "
        f"run measured "
        f"{_count(result.repetitions, 'simulation')} after "
        f"{_count(result.warmups, 'warm up')} in "
        f"{_count(result.rounds, 'round')} and scanned "
        f"{_count(result.points, 'point')} unless the model above says "
        f"otherwise, and the parallel scans ran on "
        f"{_count(result.workers, 'worker')}.",
        "",
        "- A cell is the median of the rounds of its model, each round the "
        "median of its timed calls: a round which was slow because the "
        "machine was doing something else moves a mean and not a median, "
        "which is why the tables report medians and no deviation.",
        f"- The table against the engine {REFERENCE_RATIO.value} is the "
        f"median of an engine divided by the median of the engine "
        f"{REFERENCE_RATIO.value} on the same model and measure, so the "
        f"engine {REFERENCE_RATIO.value} is 1, a number above it is that "
        "many times its time and a number below it is a cell it loses. The "
        "ratios are what carries from one machine to another; the seconds "
        "of the tables above belong to this one.",
        "- `load s` is `load_model` with the model already compiled, "
        "`first compile s` the first load of that model in this process, "
        "the one which compiles it (the LLJIT of roadrunner, the module of "
        "sbmlsim-core); the runtime of sbmlsim-core is created once "
        "per process, so only the first model of a run pays it in this "
        "column. AMICI is loaded from a cache which was warmed before the "
        "measurement, so its compilation from cold, minutes per model, is "
        "not measured, and COPASI, which compiles the model on every load, "
        "has the same measurement twice.",
        "- `compile s` and `instantiate s` are the compilation of the SBML "
        "into the module of sbmlsim-core and its instantiation in the "
        "runtime, which `load s` of the other engines does not break out. "
        "They are no part of `load s` of the core either, which finds the "
        "module compiled (the engine keeps it per SBML and process), and "
        "they are part of its `first compile s`.",
        "- `simulate ms` is one `Simulator.simulate` at the defaults of "
        "`Simulator`, i.e. the tolerances a run has and not the tighter ones "
        "of the agreement below, which is measured in simulations of its "
        "own: every speed of this table is the speed of a run as a user gets "
        "it. `scan serial s` and `scan parallel s` are that simulation over "
        "the points of the scan on one worker and on the workers of "
        "the parallel scan below.",
        "- " + _scan_line(),
        *(["- " + _cold_line()] if result.cold else []),
        "- " + _output_line(result),
        f"- The workers of a parallel scan are {result.workers} python "
        "processes for roadrunner, COPASI and AMICI, which load the model in "
        "every worker, and for sbmlsim-core the threads of the engine "
        f"inside one python process: `Simulator.run` hands the points of a "
        "scan to the engine in batches which it integrates on "
        f"{result.workers} threads of its own (`worker._run_chunk_core`, at "
        "most `CORE_MAX_CHUNK` points per batch), so a run of the engine "
        "core starts no pool of processes at all and loads the model once. "
        "The two do not scale alike: a process runs the whole point, its "
        "integration and the python around it, while a thread of the engine "
        "integrates only and the one python process of the run assembles the "
        "result of every point itself, so the more output a point has, the "
        "less the threads gain.",
        "- `max rel. diff.` and `max abs. diff.` are the agreement with "
        "roadrunner at atol 1e-12 and rtol 1e-10: the largest difference of a "
        "column divided by the largest magnitude of that column in "
        "roadrunner, and, for a column which is identically zero in "
        "roadrunner, where a relative difference is undefined, the largest "
        "absolute difference. roadrunner is the reference, COPASI is not "
        "bitwise reproducible across loads of a model.",
        "- An empty cell is a measurement which was not made.",
    ]
    if any(row.note for row in result.rows.values()):
        lines.append(
            "- A `note` is why a row is missing measurements, e.g. a model "
            "the engine refuses."
        )
    for engine, reason in result.skipped.items():
        lines.append(
            f"- The engine `{engine_label(engine)}` was not measured: {reason}"
        )
    return lines


if __name__ == "__main__":
    # the warnings of the engines, e.g. a setting of roadrunner another engine
    # ignores, and not the messages of every one of the many loads of a model
    log.enable_rich_logging(level=logging.WARNING)
    sys.exit(main())
