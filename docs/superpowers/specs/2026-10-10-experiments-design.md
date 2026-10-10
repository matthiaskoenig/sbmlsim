# Scans and observables in simulation experiments (#249, sub-project 4)

A simulation experiment declares observables next to its simulations, its data are labelled arrays which keep the dimensions of a scan, its figures draw one curve per point of a named dimension or a band over draws, and its fit mappings compare scalar observables such as a `cmax` with data. The `ScanResult` of the scan core already holds all of this; the experiment layer drops it today.

This design is sub-project 4 of `2026-10-08-scan-core-design.md`, whose outline it follows; the scan core (phases 1 and 2) and the analyses (sub-project 3, phases 1 and 2) are merged into `develop` (`c583315d`).

## The goal

- Observables of an experiment: `SimulationExperiment.observables()` returns the `Formula`, `PK` and `Custom` observables of the scan core, and `Data("hctz.cmax", task=...)` reads one. A task computes only the observables its data reference.
- Labelled data: `Data.get_data` returns an `xarray.DataArray` with the dimensions of the scan, their coordinates and the unit, so nothing downstream guesses which point of a scan it holds.
- Figures over scan dimensions: a curve names the dimensions it draws one line per point of (`over=`), a scalar can be drawn over a dimension (cmax over dose), and `Plot.band` draws the median and a quantile band over a dimension of draws. Nothing is dropped silently.
- Fits of scalar observables: a fit mapping compares a value per simulation, or a value over one scan dimension, with data, next to the timecourse mappings of today.
- No API compatibility: `Data.get_data` returns arrays instead of quantities, `first_curve` is removed; callers, examples, tests and docs are migrated.

## The starting point

Measured on `develop` (`c583315d`).

| | |
| --- | --- |
| tasks | `Task(model, simulation)` holds keys only; `_run_tasks` (`experiment/experiment.py:444`) calls `simulator.run(model, scan)` without observables, `time` or `keep` |
| selections | `_selections_of_model` (`experiment/experiment.py:502`) adds the index of every task `Data` of figures, data and fit mappings to the roadrunner selections, so `Data("hctz.cmax", task=...)` would become a selection and fail |
| `Data` | a TASK resolves to `ScanResult.quantity(selection)`, a pint `Quantity` in the layout of the result, `(*dims, time)`, `(*dims, _point)` or `(*dims)`; the dimension names and labels are dropped; a FUNCTION reduces `max`/`min` over the last axis |
| figures | `plot/padding.py` `first_curve` reshapes the data to `(-1, n_time)[0]`: a curve of a scan draws its first point and a scalar over a dimension its first element, silently; both serializers call it; no colour per point, no band over a scan |
| scans in experiments | `examples/demo` (a two-dimensional scan whose figure draws the first point), `examples/glucose/experiments/dose_response.py` (a scan with `steps=1` and a hand-made matplotlib figure of `res.ds.isel(time=0)`), tests of scans and ragged scans |
| pharmacokinetics | no experiment computes a cmax or an AUC; `examples/observables.py` shows `PK` with `Simulator.run` outside an experiment |
| fits | `FitData` builds `Data` for x, y and their errors and the problem uses `.magnitude` as 1-D arrays; `OptimizationProblem._simulate_groups` executes one plan per group with `execute` and interpolates the selections at the reference times; scalar observables and scan tasks are not possible |
| serialization | `Task.to_dict`, `Data.to_dict` and `Curve.to_dict` know nothing of observables, selections of points or dimensions |

## Decisions

| Question | Decision |
| --- | --- |
| scope | curves per scan point, scalars over a dimension, bands over draws, and scalar observables compared with data and fitted |
| declaration of observables | a method `observables()` of the experiment, `{id: Observable}`; every task computes the observables its data reference |
| representation of data | labelled arrays end to end: `xarray.DataArray` with dims, coords and `attrs["units"]`, the convention of `ScanResult` and `SensitivityResult` |
| dimensions in a figure | explicit: a curve names its `over` dimensions; a dimension which is neither the axis nor in `over` raises |
| fits of scalars | one rule: the observable has at most one dimension after `sel`, none (a value per simulation) or one (matched to the dataset by its coordinate) |
| PEtab | scalar observables and scan tasks are gaps of PEtab v2; `export` raises for them |

## Observables in an experiment

`SimulationExperiment.observables() -> dict[str, Observable]`, where `Observable` is `Formula | PK | Custom` of `sbmlsim.simulation.observables`, the base returns `{}`. `initialize()` calls it after `simulations()`; the order of the methods of an experiment becomes `datasets`, `models`, `simulations`, `observables`, `tasks`, `data`, `fit_mappings`, `figures`. `_check_keys` requires every key to equal the `id` of its observable and no id to clash with a dataset, simulation, task or data key; `_check_types` checks the types.

What a task computes. `_run_tasks` collects the task `Data` of the figures, the data and the fit mappings of every task, as `_selections_of_model` does today, and classifies the index of each:

- an observable id, or `<id>.<parameter>` of a `PK`: an output to keep;
- a dimension id of the scan, `<dimension>.<target>` of a target the dimension changes (e.g. `dose.PODOSE_hctz`) or a coordinate of a dimension: read from the coordinates of the result, never selected;
- `time`: the time of the result;
- otherwise a selection of the model, a timecourse, also when the scan of the task changes it: the plain name of a model symbol always means its timecourse (the timecourse of a scanned initial concentration is what a figure of it draws), and the scan core keeps the coordinate of such a target as `<dimension>.<target>` next to the variable.

A task without a referenced observable runs as today (`simulator.run(model, scan)` with the selections of the model set from its data, or all of them without `reduced_selections`). A task with referenced observables runs `simulator.run(model, scan, observables, keep=...)`: the observables are the ones the kept outputs need, and `keep` also names every referenced selection, so that the timecourses and the values per simulation come from one run; without `reduced_selections` it names every selection of the model. For this the scan core accepts a selection of the model in `keep` of a run with observables: it is an output of the graph, a timecourse of its own name in the unit of the model, as in a run without observables (an observable cannot stand in for it, since an observable shares no name with a selection). An index which is neither an observable, a coordinate, `time` nor a symbol of the model raises at `initialize()`, naming the task, the index and the observable ids of the experiment.

The results are unchanged: `self._results[task]` is the `ScanResult`, written as netCDF with the definitions of its observables in `attrs["observables"]`. `Task.to_dict` gains the observables the task computes and the experiment JSON the observables of the experiment (`Observable.to_dict` exists).

## Data as labelled arrays

`Data.get_data(experiment) -> xr.DataArray`, named by the sid of the data, with its dimensions, their coordinates and `attrs["units"]` (the unit as a string pint parses). `sbmlsim.data.to_quantity(array, ureg) -> Quantity` gives the pint quantity where pint is the point (the unit conversions of a fit, a matplotlib figure of a user); `Data.unit` stays and is set when the data is resolved.

Task data. A variable of the `ScanResult` with its coordinates: a timecourse over `(*dims, time)`, or `(*dims, _point)` for a ragged result with the padding `NaN`; a value per simulation over `(*dims)`; for a plain `Simulation` over `(time,)` or a 0-d array. `Data("time", task=...)` is the coordinate `time`, for a ragged result the padded time over `(*dims, _point)`. `Data("dose.PODOSE_hctz", task=...)` reads the values a dimension sets to a target, over the dimension, e.g. `(dose,)`, in their unit, a coordinate of a dimension by its name, and a dimension id its labels; `Data("PODOSE_hctz", task=...)` is the timecourse of the symbol, as for any selection.

Selection. `Data(..., sel={"dose": "25mg"})` selects by label; a scalar label drops the dimension, a list keeps it. An unknown dimension or label raises a `ValueError` naming the dimensions or labels which exist. `sel` applies to task and function data and is part of `Data.to_dict`.

Dataset data. A column of a `DataSet` becomes an array over the dimension `row`, whose coordinate is the index of the dataset, in the unit of the column in `DataSet.uinfo`. `sel={"group": "25mg"}` keeps the rows whose column `group` equals the value or one of a list of values; a dataset array always keeps `row`, a selected study group is an array of length 1.

Function data. The variables are arrays broadcast by dimension name, so `y / dose` over `(dose, time)` and `(dose,)` needs no reshaping; the unit is derived by evaluating the formula on quantities, as today. `max(x)` and `min(x)` of a single argument reduce over the time dimension of `x` (`time` or `_point`) by name, ignoring `NaN`, and keep the other dimensions; `mean` and `at` stay reserved for observables, which have the times.

Padding. `first_curve` is removed. `without_padding` becomes the mask of the padding over `_point`, which the figures apply to every line; it never chooses a point.

## Figures over scan dimensions

A curve names the dimensions it draws one line per point of: `Curve(x, y, ..., over="dose")` or `over=("dose", "condition")`, also in `Plot.curve` and `Plot.add_data`. After `sel`, x, y, `xerr` and `yerr` are broadcast by dimension name; the dimensions of x which are not in `over` must be exactly one, the axis along which a line is drawn: `time` or `_point` for a timecourse, `dose` for a scalar over a dimension (x `Data("dose.PODOSE_hctz", task=...)` over `(dose,)`, y `Data("hctz.cmax", task=...)`). Every point of the `over` dimensions is one line, in the order of their labels. A dimension of y which is neither the axis nor in `over` raises, e.g. "y of curve 'cve' has the dimension 'dose'; name it with over='dose' or select a label with Data(sel=...)". The dimensions of task data follow from the scan of the task and the kind of its observable, so the check runs at `initialize()`, before any simulation; function data is checked when it is evaluated.

Colours and legend. With one `over` dimension the lines take colours along its points: shades of the colour of the curve's style if it sets one, else the colour map `viridis`. With two, the colour follows the first and the line style (solid, dashed, dotted, dash-dot) the second; more than four points of the second raise. A legend entry names its point: the value and unit of the changed target if the dimension changes exactly one target (`PODOSE_hctz = 25 mg`), else its label. Above ten points the curve gets a colour bar of the dimension instead of legend entries.

Bands. `Plot.band(x, y, across="draw", quantiles=(0.05, 0.95), median=True, over=None, style=None, name=None)` reduces the dimension `across` of y to the two quantiles (ignoring `NaN`) and draws the shaded area between them, with the median as a line if `median`; the remaining dimensions follow the rules of a curve, `over` gives one band per point in the colours of a curve. The quantiles are computed when the figure is drawn, from the resolved data; the user calls no `summary`. `Band` is an element of the figure model (`plot/plotting.py`) next to `Curve` and `ShadedArea`.

Backends. Both draw the same: matplotlib one `plot`/`errorbar` per line and `fill_between` per band; plotly one `Scatter` per line, grouped in the legend by curve, and a pair of filled traces per band, with the same colours, legend entries and colour bars. `Curve.to_dict` and `Band.to_dict` record `over`, `across` and `quantiles`.

## Fits of scalar observables

`FitData(..., sel=None)` passes `sel` to every `Data` it builds (x, y, their sd and se); `xid` may be `None` when the observable is 0-d. At `initialize()` every fit mapping is one of three kinds, decided from the dimensions of the observable after `sel`, anything else raises with its dimensions and the fix:

- timecourse: y over `(time,)` with x the time; the prediction is the observable interpolated at the reference times, or read at the output times, as today;
- scalar, 0-d: a value per simulation of a plain `Simulation`, or of a scan with every dimension selected; the reference is an array over `row`, usually one row (a study group); several rows (individuals) each compare with the one value; no x;
- scalar over one dimension: y over `(dose,)` of a scan task with x the values of that dimension (`dose.PODOSE_hctz`); the x column of the reference, in a convertible unit, is matched by linear interpolation of y along the coordinate, with the error of the time for a reference outside the range; the coordinate must be monotonic.

The simulation of a fit. `OptimizationProblem.initialize` compiles the observables each simulation group needs (`compile_observables`, once) and keeps the `ObservableGraph`, which is picklable, so the workers of a parallel fit get it with the problem. The evaluation of a graph on the native solutions and plans of points is factored out of `simulator/worker.py` (`_observe`) into one function, which the workers of `Simulator.run` and the fit share. A group of a mapping over a dimension executes the plan of every point of the dimension (`Plan.with_values`, with the fitted values of the group applied as now) and evaluates the graph on their solutions; a cmax over doses costs one simulation per dose and evaluation.

Residuals and weights are those of the timecourses (the sd, else the se, of the reference). The baseline residuals `ABSOLUTE_TO_BASELINE` and `NORMALIZED_TO_BASELINE` raise for a scalar mapping, which has no baseline point. The `DV`/`PRED` table, the metrics, the goodness of fit and the Bland-Altman plot take the scalar points like the points of a timecourse; the figures of a mapping draw a 0-d mapping as the data with their error bars next to the simulated value, and a mapping over a dimension as the simulated curve over the coordinate with the data points.

PEtab. PEtab v2 has neither reductions over the time nor scans: `petab_v2.export` raises for a mapping with a scalar observable or a scan task, naming the mappings, and `gaps.py` catalogues both gaps. Such a fit runs in sbmlsim and is not exchanged as PEtab. `ObservableModel`, the formula observables a PEtab problem brings, stays as it is.

## The callers

- `plot/serialization_matplotlib.py`, `plot/serialization_plotly.py`, `plot/plotting.py` (`Plot.add_data`), `fit/objects.py`, `fit/optimization.py`, `fit/report.py` and the experiment JSON move to the arrays.
- `examples/demo` names `over` for its two-dimensional scan instead of drawing its first point.
- `examples/glucose/experiments/dose_response.py` replaces its matplotlib figure by an observable over the dimension of doses.
- The HCTZ studies (`examples/hctz_fitting/experiments`) get `observables()` with a `PK` of `[Cve_hctz]` and a figure of cmax and AUC; there is no table of pharmacokinetic parameters of HCTZ to fit, so their fit stays on the timecourses.
- A new example `examples/experiment_scans.py` on the packaged midazolam model shows the four uses in one experiment: curves per dose, cmax over dose, a band over LHS draws and a fit mapping of cmax over dose against a small dataset, labelled in the file as synthetic.

## Errors

An unknown index of task data, an observable id which clashes with another key, a dimension of a curve neither on the axis nor in `over`, more than four points of a second `over` dimension, an unknown dimension or label of `sel`, a fit mapping with more than one dimension or with a baseline residual on a scalar, a non-monotonic coordinate of a mapping over a dimension, and the PEtab export of a scalar or scan mapping raise a `ValueError` with the experiment, the object and what exists or what to do. The checks which depend only on definitions run at `initialize()`.

## Phases

1. Observables in experiments and `Data` as labelled arrays (observables, data and the migrated callers). A curve whose data keeps a scan dimension raises in this phase, so the demo selects a label until phase 2.
2. Figures over scan dimensions: `over=`, colours, legend and colour bars, `Plot.band`, both backends, the figures of the demo, the glucose example and the new example, `docs/plotting.md`.
3. Fits of scalar observables: the three kinds of mappings, observables and scan points in the simulation of a fit, the figures of the report, the PEtab gaps, the fit of the new example, the docs of fitting.

## Testing

- Data: task variables, `time` (common and ragged), coordinates and dimension labels, `sel` (scalar and list, errors), dataset columns and dataset `sel`, function data broadcast by name, `max`/`min` over the time by name, units.
- Experiments: a task computes only the referenced observables, referenced selections kept next to observables in one run, `reduced_selections=False` with observables, the errors at `initialize()`, the JSON of observables and tasks, netCDF of the results.
- Figures: one and two `over` dimensions, the error for a dimension which is not named, legend labels from changed values and from labels, the colour bar from eleven points, band quantiles against numpy with `NaN`, the padding of ragged results, both backends rendered and inspected.
- Fits: a 0-d and a 1-d scalar mapping recover the parameters of a synthetic problem; the cost of the HCTZ timecourse problem is bit-identical to before; baseline residuals raise for scalars; a parallel fit with scalar mappings gives the result of the serial one; the PEtab export raises with the mapping names.
- Examples: the new example and the migrated ones run in `tests/examples/test_example_scripts.py`; their figures are inspected.

## Out of scope

- One kind of observable for experiments and PEtab, i.e. replacing `ObservableModel` by the observables of the scan core.
- An extension of PEtab for scalar observables or scans.
- Sensitivity analyses and their plots inside a simulation experiment.
