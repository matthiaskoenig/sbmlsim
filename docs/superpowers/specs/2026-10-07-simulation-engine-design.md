# The simulation engine: one way to set up simulations, exact PEtab v2 semantics

A new simulation definition and a compiled engine on roadrunner, which the simulation experiments, the scans, the parameter fitting and the PEtab v2 layer all use. The definition carries units, the engine runs without them. The semantics of a simulation are the ones of PEtab v2, so the PEtab test suite passes and the problems of the PEtab benchmark collection are read, simulated and optimized with the PEtab objective.

## The goal

- There is one way to set up a simulation, `Simulation`, used by the simulation experiments, the scans, the fits and the problems read from PEtab. `Timecourse` and `TimecourseSim` are removed.
- Multiple dosing and every other change at a time is one concept, `Change`, which takes one time or a vector of times.
- The cases of the PEtab v2 test suite (`petabtests/cases/v2.0.0/sbml`) pass: log-likelihood, chi2, simulations and log-prior within the tolerances of each case. The math cases (`v2.0.0/math`) pass.
- Every problem of the benchmark collection is read after `petab1to2`, simulated at its nominal parameters and agrees with its `simulations.tsv`; its log-likelihood agrees with the reference values of AMICI where the conversion keeps the noise model; every problem runs a short optimization of the PEtab objective. A complete benchmark on a server is a later step and only needs the runner of this design.
- An evaluation of the objective spends its time in roadrunner: no pint, no interpolation, no model rewrite, no data frame in the hot path.

## The starting point

Measured on `develop` (`2d6799e`) with the PEtab test suite at `4c17947` and the benchmark collection converted with `petab.v2.petab1to2`:

| | |
| --- | --- |
| test suite | 8 of 31 cases pass |
| benchmark | 13 of 35 problems are read and simulated, 22 fail when they are read or initialized; the 13 are not checked against their `simulations.tsv` |
| initial assignments | `SimulatorSerial._timecourse` calls `resetToOrigin()` and sets the changes of the first timecourse as `r[key] = value`, so an initial assignment which depends on a changed parameter keeps the value of the model (case 0001: `B = b0` stays `1` for `b0 = 0`). A fit parameter which feeds an initial assignment is ignored without a message. This is a bug of `sbmlsim`, not only of the PEtab layer |
| objective | a fit is a weighted least squares fit. An estimated parameter which is not an entity of the model, i.e. a scaling, an offset or a standard deviation of `observableParameters` or `noiseParameters`, is dropped with a warning, so most problems of the benchmark collection cannot fit their data |
| observables | a formula observable is written into a copy of the SBML as a parameter with an assignment rule. An observable with placeholders produces invalid SBML (15 of the 22 failures), and the copy costs 25 s for `Froehlich_CellSystems2018` |
| conditions | a `targetValue` which is a formula is refused (cases 0016, 0020, 0026 to 0028, 0030, 0031) |
| steady state | a measurement at `time = inf` is refused (case 0023), a pre-equilibration is a simulation of the duration of the experiment |
| output | the simulation runs on a grid of 100 steps and the residuals interpolate linearly on it |
| smaller | `LOG10` is the scale of every parameter and refuses a lower bound `<= 0`; a start value outside of the bounds raises; unit strings with `**` do not parse; the experiment class built at runtime cannot be pickled, so a fit of a problem read from PEtab runs serially |

roadrunner 2.10, probed for this design:

| operation | effect |
| --- | --- |
| `r["init(p)"] = v; r.reset()` | the initial assignments which depend on `p` are evaluated again, also the ones of parameters |
| `r["p"] = v; r.reset()` | species whose initial assignment depends on `p` follow, parameters with an initial assignment on `p` do not |
| `r["init(p)"] = v` for `p` with an initial assignment | raises |
| `r["init([S])"] = v` for `S` with an initial assignment | overrides the initial assignment |
| `r.resetToOrigin()` after `init(...)` | keeps the values set with `init(...)` |
| `r["C"] = v` for a compartment | keeps the amounts of its species, i.e. the semantics of an SBML event |
| `r.simulate(times=[...])` | exact output times with a fixed step integrator; with `variable_step_size=True` the times are ignored |
| `r.simulate(start, end)` with `variable_step_size=True` | the steps of the integrator |

## Decisions

| | |
| --- | --- |
| scope | one design, four phases, each with its own plan, pull request and verification |
| definition | `Simulation` with `time_unit`, `start`, `end`, `preinit_changes`, `changes` (a list of `Change`), optional `presimulation`, `times`, `steps` and `time_shift` |
| removed | `Timecourse`, `TimecourseSim`, `AbstractSim` and the `time_offset`; examples and tests are migrated, release 0.9.0 |
| semantics | the initialization and the changes of PEtab v2 for every simulation, not only for PEtab problems |
| output | the steps of the integrator by default, exact `times` or an equidistant grid of `steps` on request |
| results | ragged: every simulation keeps its own time points, interpolation onto a common grid on request |
| units | one registry, `sbmlsim.units.ureg`, with `Q = ureg.Quantity`. Units are converted to the units of the model once, when a simulation is compiled |
| symbols | targets, selections and formulas use the convention of roadrunner: `S` is the amount of a species, `[S]` its concentration. The PEtab layer translates the ids of PEtab, which mean what the model math means, with `petab_v2/symbols.py` |
| formulas | strings of the math of PEtab over the symbols of the model, in the units of the model; a formula carries no units |
| observables | formulas compiled to numpy once and evaluated on the selections after the simulation; the SBML is not rewritten |
| objectives | `WEIGHTED_LEAST_SQUARES`, the objective of `sbmlsim` and the default of a fit defined in python, and `LIKELIHOOD`, the negative log-likelihood of PEtab with priors, the default of a problem read from PEtab |
| fit parameters | a parameter targets an entity of the model, a parameter of an observable or a parameter of the noise |
| optimizer | scipy `least_squares` for both objectives; the likelihood is written as residuals, see below |
| steady state | integration until the rates of change are below a tolerance, with events active |

## The definition

```python
from sbmlsim import Q
from sbmlsim.simulation import Change, Simulation, SteadyState

# a single dose, the output are the steps of the integrator
sim = Simulation(
    time_unit="hr",
    start=0,
    end=48,
    preinit_changes={"BW": Q(70, "kg"), "PODOSE_hctz": Q(10, "mg")},
)

# multiple dosing and other changes at times, starting before the time 0
sim = Simulation(
    time_unit="hr",
    start=-72,
    end=48,
    preinit_changes={"BW": Q(70, "kg")},
    changes=[
        Change(-72, {"[glc_ext]": Q(5, "mM")}),
        Change([-72, -48, -24, 0], {"PODOSE_hctz": Q(10, "mg")}),
        Change(10, {"[glc_ext]": "[glc_ext] + 5"}),
    ],
)

# pre-equilibration and exact output times
sim = Simulation(
    time_unit="min",
    start=0,
    end=60,
    presimulation=SteadyState(preinit_changes={"insulin": Q(0, "pM")}),
    changes=[Change(0, {"insulin": Q(100, "pM")})],
    times=[0, 5, 10, 30, 60],
)
```

`Simulation`:

| field | type | meaning |
| --- | --- | --- |
| `time_unit` | `str` | unit of `start`, `end`, `times`, `time_shift` and the times of a `Change` given as numbers. A `Quantity` is accepted wherever a time is and converted |
| `start`, `end` | `float \| Quantity` | the interval of the simulation, `start < end`, negative times allowed |
| `preinit_changes` | `dict[str, Quantity \| float]` | applied to the model before it is initialized, see the semantics. Numbers only, no formulas |
| `changes` | `list[Change]` | changes at times in `[start, end]` |
| `presimulation` | `SteadyState \| None` | a pre-equilibration before `start` |
| `times` | `Sequence[float] \| Quantity \| None` | exact output times in `[start, end]` |
| `steps` | `int \| None` | an equidistant grid of `steps + 1` points; excludes `times` |
| `time_shift` | `float \| Quantity` | added to the time of the result. Rarely needed, a correct `start` is the better way |

`Change(times, values)`: `times` is one time or a sequence of times, `values` maps a target to a `Quantity`, a number in the unit of the target, or a formula. A `Change` with several times is the same change at each of them, which is what a multiple dosing is. Two changes at the same time are merged; two values for one target at one time raise.

`SteadyState(preinit_changes=None, tolerance=1e-8, max_time=1e8)`: a pre-equilibration before `start`. The model is initialized once, with the `preinit_changes` of the `Simulation` and of the `SteadyState` together (a target in both raises), integrated until it is at steady state, and the time is set to `start`. The state at steady state is the state the `Simulation` starts from, so what differs between the pre-equilibration and the simulation is a `Change` at `start`. This is the order of PEtab: the parameter table and the conditions of the period at `-inf` are the pre-initialization, the conditions of the next period are applied to the steady state.

`ScanSim(simulation, dimensions)` stays. A `Dimension` maps targets to vectors of values. The value of a scan replaces the value of the same target wherever the simulation sets it, i.e. in `preinit_changes` and in every `Change`; a target which the simulation does not set is added to `preinit_changes`. A dimension with `at=<time>` applies its values as a `Change` at that time instead.

## The semantics

The order follows PEtab v2, "Initialization and parameter application" and "Reinitialization semantics".

1. Pre-initialization. The values of `preinit_changes` are applied to the uninitialized model: no initial assignment has been evaluated. For a problem read from PEtab they are the values of the parameter table and the changes of the conditions of the first period.
2. Initialization. The model is initialized at `start`: the initial assignments are evaluated, except the ones of the targets of step 1, whose values replace them.
3. Changes at a time `t`, including `t = start`: every value is evaluated first, with the current state and `time = t`; then all values are assigned at once. A species is set as amount (`S`) or concentration (`[S]`). A compartment keeps the amount of a species with `hasOnlySubstanceUnits=true` and the concentration of every other species of it, which is the rule of PEtab and not the rule of an SBML event. Then the assignment rules are updated and the events whose trigger became true fire, with the state after the assignment.
4. Integration to the next time of a change or to `end`. An output time which is the time of a change gives the state after the change.
5. Steady state. A `SteadyState` replaces step 2 by an initialization at the time `0` of the pre-equilibration followed by an integration from it until `max |dx/dt| <= atol + rtol |x|` for every differential entity, in growing horizons up to `max_time`, with events active. If it is not reached, the simulation fails with `SteadyStateError`; the PEtab objective is then `inf`. A measurement at `time = inf` is the steady state of the last period, which the engine reaches by the same integration after `end`.

## The engine

`sbmlsim.simulator` holds three units.

- `compile.py`: `compile_simulation(simulation, model) -> Plan`. It converts every quantity into the unit of its target (`UnitsInformation.normalize_changes`), resolves every target to its kind (parameter, compartment, species as amount or concentration, `hasOnlySubstanceUnits`), merges the changes into a sorted list of events `(time, targets, values)` and compiles every formula to a numpy function of the symbols it reads. A `Plan` is a frozen dataclass of numbers, strings and compiled functions: no `Quantity`, no `libsbml`, picklable (the formulas are kept as strings and compiled again after unpickling).
- `executor.py`: `execute(plan, r, selections, overrides=None) -> TimecourseResult`. It runs a plan on a loaded roadrunner instance. `overrides` are the values of the parameters of a fit, which replace values of the plan without compiling it again.
- `simulation_serial.py`: `SimulatorSerial` loads a model once, compiles the simulations of a task and runs them; `run_simulation` and `run_scan` answer with an `XResult`.

The executor implements the semantics on roadrunner as follows:

- The pre-initialization sets `init(...)` and calls `reset()`. Because `resetToOrigin()` keeps values set with `init(...)`, the executor records the original initial value of every target the first time it touches it on an instance and restores the ones a plan does not set before every simulation. `resetToOrigin()` is not used.
- A parameter with an initial assignment which is a target of `preinit_changes` cannot be set with `init(...)`. The model of such a problem is derived once, when it is loaded: the initial assignment of the parameter moves to a new parameter `<p>__initial`, and the executor sets `init(p)` to the value of the plan, or to the value of `<p>__initial` after a first `reset()`, followed by a second `reset()`.
- A compartment changed at a time is set and the concentration species in it are scaled back to their concentration.
- Output: the steps of the integrator, `simulate(times=...)` with a fixed step integrator for `times`, or the grid of `steps`. Every interval between two changes is one call of `simulate`, the results are concatenated, and the state at the time of a change appears once, after the change.
- Integrator settings stay the settings of `RoadrunnerSBMLModel`, the switch between variable and fixed step is per interval.

## Results

`TimecourseResult` is the result of one simulation, the array of roadrunner and the names of its columns, as today.

`XResult` keeps the results of a scan ragged: its `xarray.Dataset` has the dimensions of the scan and a dimension `_point`, every selection and `time` are variables over both, and a simulation with fewer points is padded with `NaN`. A simulation which is not a scan has no scan dimension.

- `XResult.interpolate(times)` gives the result on a common grid, with the dimension `_time`, which is what `XResult` holds today.
- `dim_mean`, `dim_std`, `dim_min`, `dim_max` take an optional `times` and interpolate onto it; without it onto the union of the time points of the simulations.
- A simulation with `times` or `steps` has the same time points in every simulation of a scan, and `interpolate` is the identity.
- A plot draws every curve on its own time points.

## Units

`sbmlsim.units.ureg` is the registry of the package and `Q` its `Quantity`, exported as `sbmlsim.Q`. Every model, experiment, runner and fit uses it; `UnitsInformation._default_ureg()` and the registries created per runner and per experiment are removed.

`UnitsInformation.from_sbml` defines the id of every unit definition of a model in the registry today (`ureg.define(f"{uid} = {unit_str}")`, also `substance`, `volume` and the other predefined units of SBML Level 2), and the registry ignores a redefinition. With one registry for every model, two models which define one id differently would convert with the definition of the first. A unit of a model is therefore stored as the expression of its unit definition in units pint knows (`Units.udef_to_str`), and no id of a model is defined in the registry; the units which `sbmlsim` defines itself (`percent`, `IU`, ...) are defined once when the registry is created.

`SimulationExperiment.Q_` is removed, the examples use `Q`.

## The fit

`OptimizationProblem` compiles the simulations of its fit mappings into plans once, in `initialize`, with `times` = the times of the reference data (the union for a group of mappings which share a simulation). An evaluation of the residuals runs every plan with the values of the parameters as `overrides`, evaluates the observables on the result and compares them with the data. There is no interpolation and no unit conversion in an evaluation.

An observable is a formula over the symbols of the model (a selection is the simplest formula) and over the parameters of the observable. The parameters of an observable and of the noise can have a value per measurement, which is how the placeholders of PEtab are represented.

A `FitParameter` has a `target` and a `kind`: `MODEL` (an entity of the model, applied as a pre-initialization change, or as a change of a period for a versioned parameter), `OBSERVABLE` and `NOISE`. A parameter which PEtab estimates and which is not an entity of the model is therefore fitted, the warning and the `noise-parameters` gap are removed.

`FitSettings.objective`:

- `WEIGHTED_LEAST_SQUARES`: the residuals, weights and loss functions of today.
- `LIKELIHOOD`: the negative log-likelihood of the training data plus the negative log-prior. It is handed to `least_squares` as residuals whose half squared sum is the objective up to a constant: `(m - y) / σ` and `sqrt(2 log σ + c)` per measurement for `normal`, the same on `log m - log y` with `log m` in the constant for `log-normal`, `sqrt(2 |m - y| / σ)` and `sqrt(2 log σ + c)` for `laplace` and `log-laplace`, and `sqrt(2 (-log p(θ) + c))` per prior. `c` is a fixed offset which keeps the radicands positive within the bounds; the objective reports `-llh` and `-log posterior` without it. The likelihood module of today becomes the evaluation of this objective and loses its own simulation.

`FitParameter.scale` is per parameter. A problem read from PEtab uses `LIN` for a parameter whose lower bound is `<= 0` and `LOG10` otherwise, unless the extension says otherwise. A start value outside of the bounds is clipped to the bounds with a warning.

A problem read from PEtab is built from plans and plain data, not from a class created at runtime, so it is picklable and a fit runs in the worker pool of `fit/runner.py`.

The reports, the identifiability and the Fisher information use the problem as today; they gain the parameters of observables and of the noise.

## The PEtab v2 layer

- `reader.py` builds a `Simulation` per experiment: the parameter table and the conditions of the first period are `preinit_changes`, the later periods are `Change`s with their formulas, a period at `-inf` is the `SteadyState` with the conditions of that period as its `preinit_changes` and the next period as the `Change` at `start`, a measurement at `inf` is a steady state output, `times` are the times of the measurements. The ids of PEtab are translated with `symbols.py`. Observables, placeholders, noise formulas and noise distributions become the observables and noise models of the fit, the parameters of the parameter table become fit parameters of the three kinds, and the priors become the priors of the objective. `petab_v2/observables.py`, which writes the observables into the SBML, is removed.
- `export.py` writes a `Simulation` as an experiment: the `preinit_changes` are the condition of the first period, every time of a `Change` is a period, a `SteadyState` is the period at `-inf`. What the tables cannot hold, i.e. the units, the output grid, the `time_shift` and the settings, stays in the `sbmlsim` extension, whose version increases.
- `gaps.py` is updated: `noise-parameters`, `presimulation` and `selections` are no longer gaps.
- `math`: the formulas are parsed with `petab.v2.math.sympify_petab`, which the math cases of the test suite check.

## The test suites

Both suites follow the pattern of the PEtab SciML suite (`sbmlsim/sciml/testsuite.py`, `scripts/sciml_testsuite.py`): a pinned commit, a cache under `~/.cache/sbmlsim/`, an environment variable which points elsewhere, a baseline in `tests/data/` which lists the cases which do not pass with their reason, one test per case, a pytest marker which a normal run deselects and a tox environment which downloads first.

- `sbmlsim/fit/petab_v2/testsuite.py`: the cases of `v2.0.0/sbml` compared on `llh`, `chi2`, the simulations and `log_prior`/`unnorm_log_posterior` where a case gives them, and the cases of `v2.0.0/math`. Marker `petab_testsuite`, tox `petab`, `SBMLSIM_PETAB_SUITE_PATH`, baseline `tests/data/petab_baseline.json`, script `scripts/petab_testsuite.py`.
- `sbmlsim/fit/petab_v2/benchmark.py`: a problem of the collection is converted with `petab1to2` into the cache, read, simulated at its nominal parameters and compared with its `simulations.tsv` (relative and absolute tolerance `1e-3`, the simulations of the collection are not more exact), its log-likelihood is compared with the reference values of AMICI where the noise distribution survives the conversion, and a smoke optimization (one start, a bounded number of evaluations) checks that the objective decreases. The timings of reading, of one evaluation and of the smoke optimization are written into the report. Marker `petab_benchmark`, tox `benchmark`, `SBMLSIM_BENCHMARK_PATH`, baseline `tests/data/benchmark_baseline.json`, script `scripts/petab_benchmark.py` with `download`, `run`, `report`, `baseline` and `optimize --problem ... --starts ... --cores ...`, which is what the complete benchmark runs on a server.
- The SBML Test Suite runner and the PEtab SciML runner move to the engine; their baselines must not get worse.

## Speed

- A model is loaded once per process and model.
- The compile step does everything which does not depend on the parameters; an evaluation sets values by index, runs `simulate` and evaluates compiled numpy functions.
- No pint, `deepcopy`, `xarray` or `pandas` in an evaluation of the objective.
- The benchmark report states the time of an evaluation of the objective and the time of the simulations alone; the target is an overhead below 20 % for models whose simulation takes more than 1 ms.

## Migration

- `Timecourse`, `TimecourseSim`, `AbstractSim`, `time_offset`, `Timecourse.model_changes` and `TimecourseSim.selections` are removed. The model changes of an `AbstractModel` are merged into the `preinit_changes`, the ones of the simulation win.
- `model_manipulations` (`ModelChange.clamp_species`) become a property of the model a task loads, not of a simulation.
- The changes of a first `Timecourse` were applied after the initialization and without the initial assignments; they become `preinit_changes`, so a change of a parameter now reaches the initial assignments which depend on it. This changes results where that bug was hit, which the release notes of 0.9.0 say.
- Every example and test is migrated; `examples/hctz_fitting` keeps its fit and its cost within the tolerance of its tests, recomputed where the initial assignment bug changed it.
- `sensitivity/` builds its simulations with `Simulation`.

## Phases

1. The engine: `Simulation`, `Change`, `SteadyState`, the compile step, the executor, ragged results, `ScanSim`, `SimulatorSerial`, the simulation experiments, the units registry, the migration of the examples and tests, the SBML Test Suite on the engine. Verification: all tests pass, the SBML Test Suite baseline does not get worse, the semantics are tested on the probe model of this design and on the cases 0001 to 0032 at the level of the simulations.
2. PEtab on the engine: the reader and the export, observables and noise as compiled formulas, the log-likelihood, chi2 and log-prior, the PEtab test suite with its baseline, the benchmark runner with its simulations and log-likelihood. Verification: every case of the test suite passes or is in the baseline with a reason, every problem of the collection is read and simulated.
3. The fit on the engine: `OptimizationProblem` with plans, the parameters of observables and of the noise, `LIKELIHOOD`, priors, scales per parameter, a picklable problem read from PEtab, the smoke optimization of every problem of the collection, the PEtab SciML suite on the engine. Verification: the HCTZ fit, the SciML baseline, the benchmark baseline.
4. Speed: profiling of the evaluation on the problems of the collection, the timings in the benchmark report, the documentation of the test suites.

## Risks

- roadrunner semantics of `init(...)`, `reset()` and events at the time of a change differ between versions; the executor is tested on a probe model which covers every row of the table above and pins the behavior.
- The steady state by integration is slow for stiff models and may not converge; the tolerance and `max_time` are settings, and the benchmark report shows the problems which need them.
- The migration changes the results of models whose first timecourse changed a parameter used in an initial assignment; the HCTZ example is checked explicitly.
- The `LIKELIHOOD` residual trick needs an offset `c` which keeps `2 log σ + c` positive; `c` is derived from the lower bounds of the noise parameters and the smallest numeric noise value, and the objective raises if a radicand becomes negative.
- Ragged results change the shape of `XResult`, which the plots and the reports read; phase 1 migrates them.

## Out of scope

- Gradients by sensitivities (forward or adjoint); the fit keeps finite differences.
- PEtab v1; v1 problems are converted with `petab1to2`.
- Models in other languages than SBML (PySB cases of the test suite).
- The complete benchmark on a server; this design delivers the runner it uses.
