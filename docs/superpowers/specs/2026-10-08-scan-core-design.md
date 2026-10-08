# Scans and observables: one core for many simulations (#249, #28)

A scan is the one way to run many similar simulations: a base `Simulation` and dimensions of changes, run serially or in a pool of processes, answered with an `xarray` result whose coordinates are the changed values. Observables are what a scan computes from every simulation: formulas over the timecourses, reductions over time (`max`, `min`, `mean`, `at`), the parameters of a non-compartmental analysis with pkpdutils, and custom functions. The parameter scans, the sensitivity analyses, the uncertainty analysis and the experiments build on this core and contain no simulation loop and no pool of their own.

This design covers the core, i.e. sub-projects 1 (scan, execution, result) and 2 (observables). Sub-project 3 (the analyses: sampling, sensitivity, uncertainty) and sub-project 4 (observables in the simulation experiments, plots over scan dimensions) get their own designs; their outline is at the end, so that the core serves them.

## The goal

- One scan definition, `Scan(simulation, dimensions)`, with product dimensions and coupled values within a dimension. A dimension varies the values of targets (the vocabulary of `Simulation`: before the initialization or as a `Change` at a time), whole simulations or models.
- One runner, `Simulator.run(model, scan, observables)`, the same code serially and in a pool of processes, with `n_workers` as the only switch. A point is a compiled plan with other values, not a compiled simulation.
- Observables are evaluated in the worker on the native solution of every simulation, before any interpolation; only the kept observables travel back.
- One result, `ScanResult`: the scan dimensions with their changed values as coordinates, the units of every variable, `summary` over dimensions, netCDF, and the handover to pkpdutils.
- Scans of 1e4 to 1e5 points on one workstation, with the memory bounded by what is kept.
- No API compatibility: `ScanSim`, `XResult` and `SimulatorSerial` are replaced, callers, examples, tests and docs are migrated.

## The starting point

Measured on `develop` (`4d8fd17`, release 0.8.4):

| | |
| --- | --- |
| engines | three independent ways to run many simulations: `SimulatorSerial.run_scan` (serial, units, engine semantics), `sensitivity/` (raw roadrunner, `resetAll()` plus `setValue`, its own `Pool`, which pickles the roadrunner instance into every task, static chunks), the pool of the fit (`fit/runner.py:615`, repeats and profiles); the SBML Test Suite has a fourth (`testsuite/runner.py:143`) |
| compile per point | `run_scan` builds a `Simulation` per point with `with_values` and runs a full `compile_simulation` for each (`simulation_serial.py:133,153`); `Plan.with_values` exists and is what the fit uses |
| errors | the first failing point aborts the whole scan |
| `Dimension` | stores `changes` by reference, does not check `index` against the values, uses `index` both as positions and as coordinates (labels other than `0..n-1` break), fails on a scalar and counts a formula string as its characters (`len(Dimension("d", changes={"a": "k1*2"})) == 4`) |
| coordinates | a scan dimension has the coordinates `0..n-1` (`xresult.py:334`), the changed values are lost; examples dig them out of the scan |
| `XResult` | `__getattr__` recurses for `scan` and on unpickling (`xresult.py:64`), so a result cannot leave a worker; `to_dataframe` flattens without the scan coordinates |
| reductions | `max`/`min` with one argument in a `Data` formula reduce over the whole array, i.e. over all simulations of a scan, not per simulation (`data.py:90`) |
| observables | a scalar per simulation (Cmax, AUC, a value at a time) exists only inside the user's `simulate` of `sensitivity/`, written by hand (`examples/sensitivity/sensitivity_example.py:68`) |
| pkpdutils | removed as a dependency in #250 because nothing imported it; `uv.lock` still lists it, `sbml4humans` and `python-libsedml` |
| `ModelSensitivity` | the difference scan has no reference point, the distribution scan draws from the global random state without a seed, the reference values ignore the model changes and initialize the shared model as a side effect |
| `Data` of type TASK | `Quantity(xres[selection].values)`, the names and coordinates of the dimensions are dropped (`data.py:344`) |

## Decisions

| | |
| --- | --- |
| compatibility | none; `ScanSim`, `XResult`, `SimulatorSerial` and `Dimension(index=...)` are replaced, release 0.9.0 |
| scan | `Scan(simulation, dimensions)`, immutable, validated at construction; a `Simulation` is a scan without dimensions |
| model | not part of the scan: a run is `(model, scan)`, as a `Task` pairs a model and a simulation; a dimension over models replaces the model of the run |
| dimension kinds | `values` (targets to arrays), `simulations` (categorical `Simulation`s), `models` (categorical models) |
| coordinates | the labels of a dimension, and every changed target as a coordinate along its dimension |
| point | the plan of its simulation and model, compiled once, with the values of the point applied by `Plan.with_values`; no compile per point |
| execution | the same worker function serially and in the pool; a pool of processes from `sbmlsim/parallel.py`, kept per process; models loaded once per worker |
| output grid | ragged native time points by default (as the engine has today); a common grid when every point has the same `times`/`steps`; `time=` interpolates in the worker |
| observables | `Formula`, `PK`, `Custom`; composition by id; evaluated in the worker on the native solution |
| reductions | `max`, `min`, `mean`, `at` in formulas, over time and per simulation; one pre-pass in `simulator/formula.py` for observables and `Data` |
| PK | `pkpdutils.nca` on the chunk; pkpdutils is a dependency again, imported lazily |
| units | derived for every observable; a `Formula` takes an optional `unit` |
| result | `ScanResult`, an `xarray.Dataset` with the units in `attrs`, netCDF; no `to_dataframe`, `to_tsv` or `to_mean_dataframe` |
| errors | `on_error="raise"` (default) or `"flag"`: `NaN` and a `status` variable |
| fit | keeps its plans and its evaluation, which are the engine already; its pool moves to `parallel.py` |

## The scan

```python
from sbmlsim import Q
from sbmlsim.simulation import Change, Dimension, Scan, Simulation

sim = Simulation(
    time_unit="hr",
    end=48,
    preinit_changes={"BW": Q(70, "kg")},
    changes=[Change([0, 24], {"PODOSE_hctz": Q(10, "mg")})],
    steps=480,
)

scan = Scan(
    simulation=sim,
    dimensions=[
        Dimension("dose", values={"PODOSE_hctz": Q([5, 10, 20], "mg")}),
        Dimension("regimen", simulations={"single": sim_single, "multiple": sim_multiple}),
        Dimension("genotype", models={"wt": model_wt, "pm": model_pm}),
        Dimension("glc", values={"[glc_ext]": Q([3, 5, 8], "mM")}, at=Q(12, "hr")),
        Dimension("sample", values={"k1": Q(k1, "1/min"), "k2": Q(k2, "1/min")}),
    ],
)
```

`Dimension(id, *, values=None, simulations=None, models=None, at=None, labels=None)`, exactly one of `values`, `simulations` and `models`:

- `values` maps a target to an array (a `Quantity`, or numbers in the unit of the target in the model). All arrays of one dimension have the same length; they are coupled, i.e. point `k` sets the `k`-th value of every target. A value replaces its target wherever the simulation sets it and is added to the `preinit_changes` otherwise, the rule of `Simulation.with_values` and `Plan.with_values`. With `at`, the values are a `Change` at that time instead. Values are numbers; a formula per point is a `simulations` dimension.
- `simulations` maps a label to a `Simulation`. Every point simulates its own simulation; their times, outputs and changes may differ.
- `models` maps a label to a model (`AbstractModel`, `RoadrunnerSBMLModel` or a path). Every point simulates its own model.
- `labels` are the coordinate of the dimension: integers `0..n-1` for `values` by default, the keys for `simulations` and `models`.

The values are copied into read-only numpy arrays at construction; a `Dimension` and a `Scan` never change after it, and running a scan never changes the objects of the user. Validation at construction:

- unique dimension ids, which are none of `time`, `_point`, `statistic`, the id of an observable or a changed target;
- equal lengths within a dimension, `len(labels)` equal to them, unique labels;
- a scalar or a string where an array is expected raises;
- a target in at most one of the dimensions without `at`, and in at most one of the dimensions with the same `at`;
- the `at` of a `values` dimension inside `[start, end]` of every simulation it applies to;
- at most one `simulations` and at most one `models` dimension.

Several dimensions span their cartesian product; the points are enumerated in C order of the dimensions, the last dimension fastest. A sampled design (Sobol, Latin hypercube, a virtual population, an ensemble of fits) is one dimension with coupled values; the samplers of sub-project 3 return such a `Dimension`.

`ScanSim`, `Dimension.index`, `Dimension.changes`, `simulation/range.py` and `indices_from_dimensions` are removed; `Scan` and `Dimension` live in `simulation/scan.py`. `Scan.to_dict` serializes the dimensions with their values, which is what the result stores as its provenance.

## Observables

```python
from sbmlsim.simulation import Custom, Formula, PK

observables = [
    Formula("ins", "[ins]"),
    Formula("ins0", "at(ins, 0)"),
    Formula("ins_rel", "ins / ins0"),
    Formula("glc_max", "max([glc])"),
    Formula("score", "piecewise(1, glc_max > 10, 0)", unit="dimensionless"),
    PK("hctz", "[Cve_hctz]", dose="PODOSE_hctz", route="po"),
    Custom("ratio", ratio, unit="dimensionless", symbols=["[glc]", "ins_rel"]),
]
```

`Observable` is the base class with `id`, `unit` and `kind` (`TIMECOURSE` or `SCALAR`); the definitions are frozen and picklable and live in `simulation/observables.py`, their evaluation without pint in `simulator/observables.py`. An observable reads the selections of the model and other observables by id; the dependencies are a directed acyclic graph, a cycle or an unknown symbol raises when the scan is compiled.

### `Formula(id, formula, unit=None)`

The math of PEtab compiled by `compile_formula`, over roadrunner selections (`S` amount, `[S]` concentration, `time`) and the ids of other observables, extended by four reductions over the time of one simulation:

- `max(x)`, `min(x)`: the largest and smallest value, ignoring `NaN`;
- `mean(x)`: the time weighted mean, i.e. the trapezoidal integral divided by the time between the first and the last time point of the simulation, which does not depend on the step sizes of the integrator;
- `at(x, t)`: the value at the time `t` in the time unit of the model, linearly interpolated on the native solution; for a time of a change it is the value after the change, outside of the simulated times it is `NaN`.

`max` and `min` with two or more arguments stay the elementwise functions of PEtab. The reductions are found by the pre-pass which `data.py` has today (`_replace_reductions`), which moves to `simulator/formula.py`, gains `mean` and `at`, and reduces along the time axis of each simulation, not over the whole array; `Data` formulas use the same pre-pass and so reduce per simulation as well. A formula whose free symbols are all scalars is a scalar, otherwise a timecourse; a scalar in a timecourse formula broadcasts over time, which is how `ins / ins0` normalizes on a baseline (#28).

The unit is derived by applying the compiled formula to quantities of one in the units of its symbols; the reductions keep the unit of their argument. With `unit` given and a derived unit, the values are converted into `unit`, and an incompatible unit raises. Where the unit cannot be derived (e.g. `piecewise`, a comparison, a function pint does not know), `unit` is taken as declared; without it the compile step raises with the formula and asks for `unit=`.

### `PK(id, selection, *, dose=None, route=None, options=None, parameters=None)`

A non-compartmental analysis of one selection with `pkpdutils.nca`, vectorized over the simulations of a chunk: the chunk is handed over as `Timecourses.from_arrays(time, values, time_unit=..., unit=..., dims=("_sim",), dose=..., route=...)` with the times per simulation (ragged, padded with `NaN`), which pkpdutils supports.

- `dose` is a target of the model. Its value in every simulation is read from the plan of the point: the dose amounts are the values the plan assigns to the target, the dose times the times of these assignments (`start` for a `preinit_changes` value), so a multiple dosing by `Change([0, 24, 48], ...)` and a dose scanned by a dimension are found without further input. A formula value of the target raises. A `Quantity` is a fixed dose at `start`. Without `dose` the parameters which depend on it are `NaN`, so `PK` also serves a variable which is not a drug (the AUC of glucose).
- `route` is a pkpdutils `Route` or its name; `options` a pkpdutils `NCAOptions`; `parameters` the subset of the parameters to keep, all by default.
- Every parameter pkpdutils derives from a timecourse is a scalar observable `<id>.<parameter>` (`hctz.cmax`, `hctz.tmax`, `hctz.auc_inf`, `hctz.thalf`, ...) with the unit pkpdutils gives it; formulas reference them by that name.

pkpdutils (>= 1.3, which needs nothing sbmlsim does not depend on already) is a dependency again and imported only when a `PK` observable is compiled.

### `Custom(id, function, unit, *, symbols, kind=SCALAR)`

`function(time: np.ndarray, values: dict[str, np.ndarray]) -> float | np.ndarray` is called once per simulation with its native time points and the selections and observables named in `symbols`; it returns a float for a scalar and an array of the length of `time` for a timecourse. It must be a function of a module, so that it pickles for the workers; a lambda or a closure raises when the scan is compiled.

### Selections

The selections a run asks roadrunner for are `time` and the union of what the observables read. A run without observables keeps today's behavior: the selections of the model, each a timecourse observable of its own name.

## Execution

```python
from sbmlsim.simulator import Simulator

simulator = Simulator(n_workers=None, integrator_settings={"absolute_tolerance": 1e-10})
res = simulator.run(model, scan, observables, keep=["ins_rel", "hctz.cmax"])
res = simulator.run(model, sim)                 # a single simulation
res = simulator.run(None, scan_over_models)     # a models dimension supplies the model
tc = simulator.simulate(model, sim)             # one TimecourseResult, for the fit and the test suites
```

`Simulator(n_workers=None, integrator_settings=None)`; `run(model, scan, observables=None, *, time=None, keep=None, on_error="raise", progress=None) -> ScanResult`.

1. **Compile.** One plan per combination of the values of the `simulations` and `models` dimensions (one plan without them), with `compile_simulation`. The `values` dimensions are converted into the units of their targets in the model once per model (a vector of floats per target). The observables are ordered, compiled and checked against the symbols of every model; an observable id which is a symbol of a model or a changed target raises.
2. **Points.** A point is a tuple of indices; nothing is built per point in the parent. The points are cut into chunks which share one model, at most `ceil(n_points / (4 * n_workers))` and at most 1000 points each.
3. **Worker.** For every point of a chunk: `plan.with_values(values of the point)`, a `values` dimension with `at` adds or merges its event through `Plan.with_values(values, at=time)`, then `execute(plan, model, selections)`. The model is loaded once per worker and kept under `model_key` together with the integrator settings. The native solutions of the chunk are stacked into a padded array `(n_sim, n_time_max)`, the observables are evaluated on it, the kept timecourses are interpolated onto `time` if it is given, and the worker answers with numpy arrays and the status per point.
4. **Assembly.** The parent writes the arrays of each chunk into the arrays of the result (scalars preallocated, ragged timecourses padded to the longest chunk at the end) and reshapes them into the dimensions.

Output grid:

- `time=None` and every plan with the same `times` (a `Simulation` with `times` or `steps`): the result has a dimension `time` with these times;
- `time=None` otherwise: the ragged layout of today, a dimension `_point` and the variable `time` over `(*dims, _point)`;
- `time` given (`Quantity` or numbers in the time unit of the model): every kept timecourse is interpolated linearly in the worker onto it; the value at the time of a change is the value after it.

The observables are always computed on the native solution, so `PK` and the reductions are as exact as the output of the simulation, independent of `time`.

`keep` is the list of observables in the result, all by default; the others are evaluated as intermediates and dropped. It is the control of the memory: 1e5 points, 481 times and two timecourses are 0.8 GB, the same points with ten scalars 8 MB.

`on_error="raise"` raises the error of the first failing point with its labels and values. `on_error="flag"` sets every observable of a failing point to `NaN` and records it in the variable `status` over the scan dimensions (`0` ok, `1` failed), logs one warning with the count and keeps the first ten messages in `attrs["errors"]`.

`progress` shows a rich progress bar in the parent; `None` shows it for runs which go to the pool.

The result does not depend on `n_workers` or the chunk size; the values of a scan are fixed before the run, so a sampled scan is reproducible by the seed of its sampler alone.

### `sbmlsim/parallel.py`

One module for every pool of sbmlsim:

- `process_context()` moves here from `utils.py` (forkserver instead of fork);
- `pool(n_workers)` gives a `ProcessPoolExecutor` which is kept per process and size and shut down at exit, so that the start of the workers (imports, roadrunner) is paid once per process and not once per run;
- `resolve_workers(n_workers, n_tasks)`: `1` is serial in the calling process, a number is taken as given, `None` is `os.process_cpu_count()` for 64 points or more and serial below; the threshold is a constant of the module, set by the benchmark test;
- `worker_cache(key, factory)` keeps objects in a worker (models, initialized problems) and builds one only when the key is new;
- the probe of `fit/runner.py` which reports a missing `if __name__ == "__main__":` guard when the workers die while they start.

The scan runner, the fit (`worker_pool`, also used by the identifiability) and `testsuite.map_cases` use it; the sensitivity analyses move to it with sub-project 3.

## The result

```python
res["hctz.cmax"]                                   # DataArray over (dose, regimen, genotype, sample)
res["ins_rel"].sel(regimen="single")               # (dose, genotype, sample, time)
res["hctz.auc_inf"].plot(x="PODOSE_hctz")          # a changed value is a coordinate
res.quantity("hctz.cmax")                          # pint Quantity
res.summary("sample", quantiles=[0.05, 0.5, 0.95]) # mean, sd, cv, min, max, quantiles over sample
res.interpolate(times)                             # ragged to a grid
res.nca("hctz")                                    # pkpdutils NCAResult
res.to_timecourses("[Cve_hctz]")                   # pkpdutils Timecourses
res.to_netcdf(path); ScanResult.from_netcdf(path)
```

`ScanResult` (`result/scan.py`) wraps one `xarray.Dataset`:

- dimensions: the scan dimensions in their order, then `time` or `_point`, i.e. `(*dims, time)`, the layout of pkpdutils `Timecourses`, so the handover needs no transpose;
- variables: one per kept observable, scalars over the scan dimensions and timecourses over `(*dims, time)` or `(*dims, _point)`, `time` over `(*dims, _point)` in the ragged layout, `status` with `on_error="flag"`;
- coordinates: the labels of every dimension, every changed target as a coordinate along its dimension (`PODOSE_hctz(dose)`, `k1(sample)`), `time` on a grid;
- `attrs["units"]` of every variable and coordinate, the convention of pkpdutils; the dataset carries the serialized scan and the observables as provenance.

Methods: `__getitem__` (a `DataArray`), `quantity`, `sel`/`isel` (a `ScanResult`), `summary(dims, statistics, quantiles)` (a `ScanResult` with a dimension `statistic`; a timecourse in the ragged layout is interpolated onto the union of its time points first), `interpolate(times)`, `nca(id)` (the `NCAResult` pkpdutils computed, collected from the chunks), `to_timecourses(id)`, `to_netcdf`/`from_netcdf`. A `ScanResult` pickles.

`XResult`, `XResult.from_timecourses`, `dim_mean`, `dim_std`, `dim_min`, `dim_max`, `to_dataframe`, `to_tsv`, `to_mean_dataframe` and the `_dfs` dimension are removed; xarray's own `res.ds.to_dataframe()` is the export of a table. `TimecourseResult` stays as the native solution of one simulation.

## The callers

Removing `ScanSim`, `XResult` and `SimulatorSerial` breaks every caller; this design migrates them to the core without new behavior, the richer integration is sub-project 4:

- `experiment/`: a `Task` pairs a model with a `Simulation` or a `Scan`, the runner calls `Simulator.run`, the results are `ScanResult` and are written as netCDF instead of TSV (the report links only the datasets as TSV, `report/experiment_report.py:111`).
- `data.py`: a `Data` of type TASK returns a `DataArray` with its dimensions and coordinates and its unit; the reductions of a FUNCTION are per simulation.
- `plot/`: reads `ScanResult` as it reads `XResult` today (the first simulation of a scan, `plot/padding.py:17`); curves over the scan dimensions are sub-project 4.
- `fit/`: `_simulate_groups` keeps `execute` on its plans; `worker_pool` and the identifiability use `parallel.py`; `Simulator.simulate` replaces `SimulatorSerial.simulate`.
- `testsuite/`: `Simulator.simulate` and `parallel.pool`.
- `comparison/`: `DataSetsComparison` takes `xarray.Dataset`s, `examples/comparison/diff_example.py` passes `res.ds`.
- `simulation/sensitivity.py`: `ModelSensitivity` returns a `Dimension`; its behavior is unchanged until sub-project 3 replaces it.
- `sensitivity/`: unchanged until sub-project 3.
- examples: `scan.py`, `model_sensitivity.py`, `units.py`, `demo/demo.py`, `repressilator/repressilator_scans.py`, `glucose/experiments/dose_response.py` (which loses its workaround for the coordinates), `examples/README.md`.
- docs: `scans.md` (rewritten for `Scan`, `Dimension`, the parallel run and the result), a new `observables.md`, `simulation.md`, `units.md`, `experiments.md`, `data.md`, `index.md`, the API pages (`simulation.scan`, `simulation.observables`, `simulator.simulator`, `result.scan`, `parallel`, without `simulation.range` and `result.xresult`), `CLAUDE.md`.

pkdb_models uses `ModelSensitivity` and `sensitivity/` and is already stale against the engine (`Timecourse` was removed in `bc190b2`); its migration is the issue the core cleanup opened and is not part of this design.

## Bugs fixed on the way

- `Dimension`: changes by reference, no length check, `index` as positions, scalars, formula strings counted as characters.
- `XResult.__getattr__` recursion and its pickling (gone with `ScanResult`, which is tested to pickle).
- `max`/`min` of a `Data` formula reduce over all simulations of a scan.
- A compile per scan point; the first failing point aborts a scan.
- `uv.lock` lists the removed `pkpdutils`, `sbml4humans` and `python-libsedml` (regenerated when pkpdutils is added again).

## Testing

- `Dimension` and `Scan`: every validation rule, C order, labels and coordinates, read-only values, objects of the user unchanged after a run.
- Points: `Plan.with_values` and `with_values(..., at=...)` give the same result as compiling the `Simulation` the point describes, on the probe model of the engine, for `preinit_changes`, a `Change` at a time, a presimulation and a compartment.
- Dimensions over simulations and models: different outputs per point, ragged and on a grid.
- Observables against analytic solutions on a one-compartment model with first-order absorption: `max`, `min`, `mean`, `at` (also at the time of a change), a baseline formula, units derived, a declared unit, the error without one, a cycle, an unknown symbol.
- `PK` against a direct `pkpdutils.nca` call on the same arrays, single and multiple dosing, a scanned dose, without a dose; Cmax, tmax and AUC against the analytic solution with a `steps` grid.
- `Custom`: scalar and timecourse, a lambda refused.
- Equivalence: the result of a scan is identical for `n_workers` 1, 2 and 4 and for chunk sizes 1 and 1000.
- Errors: a point which fails in the integrator with `"raise"` (the message names the labels and values) and with `"flag"` (`NaN`, `status`, the warning).
- `ScanResult`: coordinates, units, `summary`, `interpolate`, `nca`, `to_timecourses`, netCDF round trip, pickle.
- Regression: the values of the scan examples (`examples/scan.py`, `repressilator_scans.py`, `glucose/dose_response.py`) are recorded on `develop` before the change and compared after it.
- Speed: a scan does not call `compile_simulation` per point (asserted with a mock in a normal test). With the `benchmark` marker, deselected by default: `Simulator.simulate` is not slower than `SimulatorSerial.simulate` today, the serial time of a scan of 1e3 points is reported against today's `run_scan`, and a scan of 1e4 points on 4 workers is at least 2.5 times faster than on 1.

## Phases

1. **The scan core**: `parallel.py`, `Scan` and `Dimension`, `Plan.with_values(..., at=...)`, `Simulator` with the serial and the pooled run, `ScanResult` with the selections as observables, the migration of the callers, examples and docs. Verification: all tests pass, the regression values of the scan examples agree, the equivalence and speed tests pass.
2. **The observables**: `Formula` with the reductions (the pre-pass moved from `data.py`), `PK` with pkpdutils, `Custom`, `keep`, `nca`/`to_timecourses`, `observables.md`. Verification: the analytic tests and the comparison with pkpdutils.

Each phase has its own plan and pull request.

## Sub-projects 3 and 4 (outline)

- **The analyses** (sub-project 3): `simulation/sampling.py` is the one sampler and returns a `Dimension`: grids, relative changes `±delta` with the reference point, distributions (normal, lognormal, uniform, truncated) with correlations, the designs of LHS, Sobol, FAST and Morris (scipy `qmc`, SALib), parameters of a fit (the covariance of `fisher.py` in the space of the `parameter_scale`, the parameter sets of fit repeats) and virtual populations (a function of a module which maps covariates to parameters). It replaces `ModelSensitivity`, the samplers of `sensitivity/` and `fit/sampling.py`. `sensitivity/` drops `SensitivitySimulation.simulate(r, changes)`: an analysis is a base `Scan`, scalar observables and parameters, run by `Simulator`, the indices computed on the arrays of the result. The uncertainty analysis is a sampler, a run and `summary`, which gives the prediction bands of timecourses and the distributions of scalars.
- **The experiments** (sub-project 4): observables of a `SimulationExperiment`, `Data("hctz.cmax", task=...)`, plots with one curve per point of the scan dimensions (replacing `first_curve`).

## Risks

- The integrator steps can be coarse in a smooth elimination phase, which makes an AUC by the trapezoidal rule less exact. The tests compare `PK` with the analytic solution on the integrator steps and on a `steps` grid; the documentation recommends `steps` or `times` for a non-compartmental analysis, as it is done for data.
- A pool pays the import of sbmlsim and the load of a model per worker; the pool is kept per process and the automatic mode stays serial for small scans.
- Ragged timecourses of 1e5 points with the integrator steps can be large; `keep` and `time` bound them, the documentation says so.
- Breaking changes for pkdb_models, which is already stale against the engine.

## Out of scope

- A distributed backend (dask, ray, a cluster) and results on disk during a run.
- Streaming quantiles; a summary needs the kept points.
- The observables of the fit (`fit/objects.py`, with their placeholders and noise) stay as they are; they share `compile_formula`.
- Gradients, steady state scans other than `presimulation`.
