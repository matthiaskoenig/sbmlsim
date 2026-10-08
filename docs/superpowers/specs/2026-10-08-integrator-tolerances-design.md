# Robust integrator tolerances: a tolerance per state and restarts in local time

## Goal

The absolute tolerances of CVODE are a vector with one value per state, set by sbmlsim from the kind of the state, so that they are reproducible, understandable and of a sensible size in every model, with or without units. A restart of the integration at a late time (a dose, a reset, a change) no longer drops the first step below the resolution of the time.

## Findings this design is based on

Measured on the models of pkdb_models with sbmlsim 0.8.4 (hctz, albuterol) and libroadrunner 2.10.0:

- sbmlsim multiplies the scalar absolute tolerance by the smallest finite positive initial volume of the model (`RoadrunnerSBMLModel._tolerance_volume_factor`). roadrunner then turns the scalar into a vector by its own rule: it scales the value by the initial value of a state, or by the volume of its compartment when the value is 0, also for a species with `hasOnlySubstanceUnits=true`. The vector is computed when the setting is applied and does not follow later changes of the state.
- hctz started its urine with a volume of 1e-12 l. The scalar became 1e-10 * 1e-12 = 1e-22 and the vector reached 1e-34 for the urine states (minimum 1e-34, maximum 1.9e-19). Plain roadrunner with 1e-10 gives 1e-22 to 1.9e-7, a spread of 15 orders of magnitude which depends on the initial values.
- With such tolerances CVODE estimates a first step of about 1e-19 after a restart. At a late time (`t = 1440 min`) the step is below the resolution of the time and CVODE warns "Internal t = ... and h = ... are such that t + h = t on the next step": 185 warnings in the simulation experiments of hctz, about 1000 in a fit and 213 000 in the profiles of its identifiability. albuterol, whose volumes are not degenerate, warns 240 times after a late inhaled dose into a state which is 0.
- The engine of sbmlsim 0.8.3 restarted every timecourse at the time 0 and never hit the warning; the engine of 0.8.4 integrates in absolute time.
- `initial_time_step`, which reaches roadrunner since the fix on this branch, is no remedy: 1e-10 min removes the warnings of hctz, 1e-8 min fails the error test of CVODE in two of its experiments, and 1e-10 min fails it in albuterol.
- A probe which sets one value per state by id (1e-10 for an amount and a rate rule state, 1e-10 times the volume for a concentration species, no volume factor) gives tolerances of 3e-13 to 6e-9 for hctz, 0 warnings for albuterol and 4 for hctz (the first dose into a state which is 0 at `t = 1440 min`); the results agree with the old ones to 1.4e-5 relative to the largest value of a variable (a ratio of two small amounts of albuterol in the urine), to 4e-6 otherwise.
- The four models of pkdb_models which run on 0.8.4 (albuterol, hctz, lenvatinib, rapamycin) have no event and no math which reads the time.

## Decisions

- The tolerances are plain numbers in the units of the model, no quantities: many models have no units.
- A tolerance is given per kind of state, with overrides per id.
- A concentration species gets the tolerance of a concentration times a reference volume of its compartment, with a floor for degenerate compartments.
- The relative tolerance stays a scalar; CVODE has only one.
- A model which does not read the time is integrated in local time; a model which reads the time keeps the absolute time and has `initial_time_step` as its remedy. There is no automatic floor of the first step: a floor which is too large fails the error test, so it has no safe default.

## A. The tolerance model

The states which CVODE integrates are of three kinds:

| kind | states | absolute tolerance |
| --- | --- | --- |
| `AMOUNT` | species with `hasOnlySubstanceUnits=true` | `amount` |
| `CONCENTRATION` | species with `hasOnlySubstanceUnits=false`, integrated as amounts | `concentration * V_ref` of the compartment |
| `OTHER` | parameters and compartments with a rate rule, e.g. a dose in mg or a volume in l | `other` |

A species is a state if it is not the variable of an assignment rule and is not a boundary species without a rate rule; a species with a rate rule is a state of the kind its `hasOnlySubstanceUnits` gives. Every number is in the unit of the model of its state, i.e. a plain number which works for a model without units.

The reference volume `V_ref` of a compartment is its initial volume, raised to `1e-6` times the largest finite positive initial volume of the model when it is smaller, not finite or not positive. A compartment whose volume is raised is logged as a warning once per model, with its id and both volumes. The reference volume is the initial volume of the model as loaded, so the vector does not depend on the state of an earlier simulation or on the changes before the initialization.

The setting is a float, the same tolerance for every kind as today, or an `AbsoluteTolerance`:

```python
AbsoluteTolerance(
    amount=1e-10, concentration=1e-10, other=1e-10, ids={"Aurine_hctz": 1e-12}
)
```

`ids` overrides the tolerance of single states, in the unit of the model of the state; an override of a concentration species is its tolerance as an amount, i.e. it is not multiplied by the volume. The defaults are the ones of today: 1e-10 for every kind in `SimulatorSerial`, 1e-6 in `FitSettings`.

## B. Components and data flow

- `sbmlsim/model/tolerances.py`, pure functions without roadrunner:
  - `StateKind`, the enum of the three kinds.
  - `AbsoluteTolerance`, a frozen dataclass with `amount`, `concentration`, `other` and `ids` (a sorted tuple of pairs, so it is hashable), `AbsoluteTolerance.of(value)` for a float or an `AbsoluteTolerance`, `to_dict` and `from_dict`, which reads a float as well.
  - `state_kinds(symbols) -> dict[str, StateKind]` from the `ModelSymbols`.
  - `absolute_tolerances(symbols, initial_volumes, tolerance) -> dict[str, float]`, the vector by id, with the reference volumes of A.
- `ModelSymbols` gets `boundary` (the boundary species) where `state_kinds` needs it, and `time_dependent` (see C).
- `RoadrunnerSBMLModel.set_integrator_settings` becomes a method of the model, since it needs the symbols. `absolute_tolerance` is resolved into the vector, which is set with `Integrator.setIndividualTolerance(id, value)` after the scalar fallback; every other setting is passed on to roadrunner and a name the integrator does not have is an error. The vector is computed once per model and tolerance and set again whenever the model gets a new roadrunner instance, i.e. when it is loaded and when a model is derived for a change before the initialization. `_tolerance_volume_factor` is removed.
- `SimulatorSerial(absolute_tolerance=...)` takes a float or an `AbsoluteTolerance`; its settings apply to every model it runs.
- `FitSettings.absolute_tolerance` is an `AbsoluteTolerance`, normalized from a float in `__post_init__`; `to_dict` writes the kinds, `from_dict` reads a float of stored settings as well, and the PEtab extension carries it through the settings. `OptimizationProblem.initialize` hands it to its simulator.
- Unchanged: `relative_tolerance` is a scalar; `SteadyState.absolute_tolerance` is the criterion on the rates of change of a steady state and not a setting of CVODE; the SBML Test Suite and the PEtab v2 test suite pass a float.

The flow: the settings of a simulator or a fit, `SimulatorSerial.integrator_settings`, `set_model` and the initialization of a model, `RoadrunnerSBMLModel.set_integrator_settings`, `absolute_tolerances(symbols, initial volumes)`, `setIndividualTolerance` for every state.

## C. Restarts in local time

`ModelSymbols.time_dependent` is true if any math of the model reads the time or a delay: rules, kinetic laws, initial assignments, function definitions, and triggers, delays, priorities and assignments of events. The test is on the csymbols `time` and `delay` of the math, not on identifiers, so a parameter named `time` (case 01820 of the SBML Test Suite) does not count.

For a model which does not read the time the executor integrates every segment `[a, b]` of a plan as `[0, b - a]`: it sets the time of roadrunner to 0 at the restart, asks for the output times `times - a`, adds `a` to the column of the time, and carries the state over. A value or a change does not read the time either (a formula of a change which reads `time` is evaluated with the absolute time of the change, as now), so the mathematics is the one of the absolute time, and CVODE always starts its first step at 0. A negative `start` is harmless as a consequence.

The events of a model which do not read the time work in local time: the handling of the events by the executor (`_triggers`, `_fire_events`, the firing after a change) gets the absolute time of the segment as an argument instead of reading `r.model.getTime()`. A steady state is integrated from 0 as well: the presimulation before the start as now, the steady state after the end (the measurement of PEtab at `inf`) in local time like a segment.

A model which reads the time is integrated in absolute time, as now; its remedy against the warning is `initial_time_step`, which the documentation describes with the risk of a step which is too large.

The results agree with the ones of the absolute time within the tolerances, not bit for bit, because the first step of CVODE depends on the time; they are closer to the engine of 0.8.3, which restarted at 0.

## D. Diagnostics and errors

- `RoadrunnerSBMLModel.tolerances()` returns a `pandas.DataFrame` with id, kind, compartment, reference volume and absolute tolerance of every state. The console of a fit and the section of the settings of a fit report show it, with whether the model is integrated in local time.
- A `ValueError` names what is wrong: a setting the integrator does not have (already on this branch), an override whose id is not a state (with the states), a tolerance which is not finite or not positive.
- A concentration species in a compartment whose volume is `NaN` or 0 gets the floored reference volume and the warning of A.

## E. Tests

sbmlsim:

- `tests/model/test_tolerances.py` on the probe model of `tests/simulator/models.py`, which has amount, concentration and rate rule states: the kinds, the vector, the floor of the reference volume and its warning, the overrides, the errors, the round trip of `AbsoluteTolerance` with the float of stored settings.
- The simulator: the vector reaches roadrunner by id, again after `set_model` and for a model derived for a change before the initialization (`getAbsoluteToleranceVector` matched to the state ids).
- The executor: local time agrees with absolute time within the tolerances for the probe model, a model with events which do not read the time and a negative start; a model whose rule reads `time` is integrated in absolute time and is right; on the HCTZ model of `examples/hctz_fitting` a dose at a late time prints no "t + h = t" (stderr captured at the file descriptor).
- The fit: the round trip of the settings, the round trip of the PEtab extension, the table of the tolerances in the report.
- The suites: `pytest`, the SBML Test Suite (`pytest -m testsuite`, no regression against `tests/data/testsuite_baseline.json`), the PEtab v2 test suite (`pytest -m petab_testsuite`, every case) and the benchmark collection (`pytest -m petab_benchmark`, the 28 of 35 problems which agree keep agreeing).

pkdb_models, against the branch: the simulation experiments of hctz and albuterol give no warning and agree with the baseline of sbmlsim 0.8.3; the workaround `initial_time_step=1e-10` of hctz is removed.

## F. Compatibility and documentation

- `RoadrunnerSBMLModel.set_integrator_settings` is a method of the model instead of a static method which takes the roadrunner instance, a breaking change for a caller outside of sbmlsim.
- The simulations change within their tolerances and the costs of fits change slightly; the release notes say so.
- Documentation: `docs/simulation.md` (selections and integrator settings), `docs/models.md` (replaces the paragraph on the scaling by the smallest volume), `docs/fitting.md` (the settings of the integrator of a fit), the architecture of `CLAUDE.md`; the release notes of 0.8.5 at the release.

## G. Delivery

One pull request from the branch `fix/pkdb-models-084`, released as sbmlsim 0.8.5. It holds the three commits already on the branch (every setting of the integrator reaches roadrunner, `FitSettings.initial_time_step`, no compartment without a size in the default selections) and the work of this design. pkdb_models then requires sbmlsim 0.8.5.

## Out of scope

- Tolerances relative to the typical size of a state from a reference simulation. They are the most physical choice but need an extra simulation and depend on the conditions; they are the next step if the tolerances of A are not enough.
- A relative tolerance per state, which CVODE does not have.
- Units of tolerances as quantities.
