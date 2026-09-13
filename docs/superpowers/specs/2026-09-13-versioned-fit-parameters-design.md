# Versioned fit parameters

A parameter of a fit which is estimated separately for parts of the data, e.g. an absorption rate fitted once for the tablet arms and once for the solution arms of the same optimization problem.

## The problem

Every `FitParameter` of `sbmlsim` is an entity of the model and is estimated once for the whole problem. `initialize()` builds one `changes` dict from the parameter vector (`optimization.py:1304`) and `_simulate_groups` applies it to every simulation (`optimization.py:1218`), so a parameter has one value everywhere.

Data does not always work that way. A study which gives hydrochlorothiazide as a tablet and a study which gives it as a solution have different dissolution, so `Ka_dis_hctz` is one number for the first and another for the second, and both are estimated in the same fit against all of the data. Today this needs two optimization problems, which loses the parameters the two share.

PEtab expresses this natively: a condition assigns `Ka_dis_hctz = Ka_dis_tablet` and an experiment references the condition. `sbmlsim` cannot read such a problem back, which is a large part of the PEtab benchmark collection.

This design is the first of two. The second is the error model, i.e. estimating the standard deviation of an observable instead of weighting its data, which is the `noise-parameters` gap (`petab_v2/gaps.py:184`). The two are independent: a version is a model entity applied selectively, the error model is not a model entity at all. The error model is not part of this spec.

## Decisions

| | |
| --- | --- |
| what a subset is | a selector over the fit mappings, i.e. the `MappingFilter` the training and validation selection already uses |
| a simulation no version selects | keeps the value of the model, and the problem reports it |
| two versions selecting one mapping | an error at `initialize()` |
| where the binding lives | `ParameterMapping`, a first class object |
| PEtab | in this pass, both directions |

## The objects

### `FitParameter` gains two fields

Both default so that every existing definition means what it means today.

- `target: str | None = None`, the entity of the model the parameter sets. `None` means `target == pid`, which is the current behaviour: the parameter is the entity.
- `mappings: MappingFilter | Iterable[MappingFilter] | None = None`, where it applies. `None` means everywhere, again the current behaviour. Several filters are combined with and, as `helpers._filters` already does.

```python
FitParameter("Ka_dis_hctz", 0.35, 0.01, 10.0, "1/hr")                        # unchanged
FitParameter("Ka_dis_tablet", 0.35, 0.01, 10.0, "1/hr",
             target="Ka_dis_hctz", mappings=filter_tablet)
FitParameter("Ka_dis_solution", 0.35, 0.01, 10.0, "1/hr",
             target="Ka_dis_hctz", mappings=filter_solution)
```

`pid` stays the name of the estimated quantity: it is what appears in the parameter vector, in `ParameterSets`, in the parameter table and in the profiles. `target` is where the value is written. The two were the same string until now, which is why this is additive.

### `ParameterMapping`, new, in `fit/parameter_mapping.py`

`objects.py` is 636 lines and holds the description of a fit; a resolution of it is a different job and gets its own module.

It is built from the fit parameters, the resolved mappings and the simulation groups of an initialized problem, and owns:

- **the bindings**, per parameter its `target` and the simulation groups it applies to,
- **`changes_for(group, quantities)`**, the `{target: quantity}` a group is simulated with, which is the one thing the hot path needs,
- **`coverage()`**, per parameter the groups it reaches and per target the groups no version reaches.

It validates on construction, so an inconsistent problem fails at `initialize()` and not halfway through a fit.

It exists as an object rather than a dict inside `OptimizationProblem` because it is the same knowledge PEtab keeps in its condition table: the export reads it and the reader builds it, instead of both re-deriving the binding from selectors that cannot be serialized. It also gives the coverage report a home. `__getstate__` reduces an initialized problem to its definition for the workers, and a resolution which is rebuilt by `initialize()` fits that lifecycle unchanged.

## Resolution and validation

### Where

The filters need the `FitMapping`, which the problem does not keep: it stores `mapping_keys`, `experiment_keys` and `mapping_kinds`, not the objects. Resolution therefore happens in the mapping loop of `initialize()`, where `mapping: FitMapping` is in scope (`optimization.py:497`). Each versioned parameter's filters are evaluated against `(mapping_id, mapping)` and the hit is recorded against the index of the mapping. Only indices are retained, so the problem stays as light and as picklable as it is.

After `_group_mappings()` (`optimization.py:804`) the hits per mapping are lifted to hits per group and handed to `ParameterMapping`.

### What is checked

**Overlap is an error.** Two parameters with the same `target` selecting the same mapping is contradictory, the simulation would need two values for one entity. The message names both parameters and the mapping.

A parameter without a selector covers every mapping, so it overlaps with any version of its target and the same error is raised. There is no fallback: a target is either estimated once for everything or estimated per subset, and the subsets which are not covered keep the value of the model. This was chosen over a version without a selector standing for the rest, because a silent fallback and a reported gap cannot both be the answer.

**A subset must not split a simulation.** A group is one simulation, so every mapping of it must agree on which parameter drives a given target: for each group and target the bound parameters across the mappings of the group are at most one.

This constraint is invisible until it is hit and is worth stating. It holds automatically for tablet against solution, because mappings share a simulation only when they come from the same dosing arm and an arm has one application form. It bites when a selector is written on something which varies inside an arm, e.g. an observable: `Ka_dis_urine` and `Ka_dis_plasma` would select two mappings of one simulation, and the error says so instead of letting one of them win silently.

**`target` must exist in the model.** `_validate_parameters` (`optimization.py:771`) and `_store_model_parameters` (`optimization.py:826`) read `model.r[pid]` and move to `model.r[target]`. Several versions of one target start from the same value of the model, which is what the reference parameter set should show.

**The versions of a target agree on their unit.** The unit of a parameter is the unit its value is given to the model in, so two versions of one entity which disagree are a mistake and raise. Bounds and start values are per version and may differ, which is the point of having versions.

**Coverage is reported and not enforced.** A group no version reaches keeps the value of the model, which is right for `Ka_dis` on the intravenous arms, where the parameter has no meaning. `coverage()` produces the table:

```
Parameters
  Ka_dis_tablet    tablet    6 of 9 simulations
  Ka_dis_solution  solution  1 of 9 simulations
  not covered:     2 simulations (Beermann1976 iv1_5, iv35_4)
                   -> Ka_dis_hctz keeps the model value 0.35 1/hr
```

`fit/display.py` prints it with the parameters and the report shows it. Without it a versioned fit looks like an ordinary one with oddly named parameters.

## The fit

### The hot path

`residuals` builds the quantities once per evaluation, as it does today, and `changes_for` assembles a small dict of references per group. `_simulate_groups` stops taking a `changes` argument and asks the mapping for the group it is about to run. Ten groups times a handful of entries per evaluation is nothing against the integrator, which profiling puts at 63% of a fit.

`simulation.timecourses[0].changes.update(changes)` never removes keys, which looks like a contamination trap and is not one: `_group_mappings` keys groups on `id(simulation)`, so no two groups share a simulation object, and the binding is fixed at `initialize()`. Every simulation therefore always receives the same set of targets and only their values change, and a target no version covers is never written, so it genuinely keeps the value of the model.

### What follows for free

A version is an ordinary parameter with its own `pid`, its own bounds and its own entry in the parameter vector, so:

- `FitMetrics` counts `k = len(parameters)` and the AIC and the BIC charge for every version,
- `profile_likelihood` scans per `pid`, so the versions are profiled separately and re-optimized against each other,
- `fisher_information` gets them as columns of the jacobian, and two versions the data cannot separate show up as a correlation near ±1,
- `ParameterSets` is keyed by `pid`, so stored parameters, `report_cli` and the fit report are untouched.

### What has to be written

`fit/display.py` gains a `target` column in the parameter table, shown only when some parameter has a target different from its `pid`, and the coverage table underneath it. The parameter table of the fit report gains the same.

## PEtab

### Export

A version becomes two things:

- a row of the parameters table, `Ka_dis_tablet`, `estimate=1`, with its bounds and its scale as any parameter, and the value of its target in the model as the nominal value,
- a change in the condition of period 0 of every experiment the version covers, `target_id = Ka_dis_hctz` and `target_value = Ka_dis_tablet`.

Period 0 because a fit parameter is applied to `timecourses[0].changes`, so the mapping is exact. `_periods` (`petab_v2/export.py:343`) builds a condition when `tc.changes` is not empty and now also when `ParameterMapping` binds a version to the group, merging the two when both apply. The export asks the mapping which versions cover an experiment instead of re-deriving it from the selectors, which is what the object is for.

A parameter without versions is unchanged: an estimated model entity with no condition.

### Reader

The inverse. A change of a condition whose `target_value` is the id of an estimated parameter is a versioned parameter: `target` is the `target_id` and the selector is the mappings of the experiments whose conditions carry it. A change whose value is a number stays a timecourse change, so the two are told apart by whether the value resolves to an estimated parameter.

This is also what lets the reader handle problems of other tools which use condition specific parameters, which today come back with the binding lost.

### The selector does not round trip, the binding does

A `MappingFilter` is a python callable and cannot be written to a TSV. PEtab stores the resolved conditions, so a problem which is read back has parameters selected by an explicit set of mapping ids and not by `application_form == TABLET`. The resolution is identical and the fit, its cost and its parameters are unchanged, i.e. the round trip is exact in effect and not in source form. `docs/petab.md` says so.

`fit/helpers.py` gains `filter_keys(ids) -> MappingFilter` so that this is expressible; the reader builds its selectors with it, and the hand rolled copy in `examples/hctz_fitting/fitting/mapping_collections.py` uses it instead.

### Extension and gaps

Nothing new in the extension: PEtab expresses the binding natively and the units per parameter which the extension already carries cover a version like any other parameter. No gap is created, which is the reason for doing PEtab in the same pass. The round trip test records that selectors come back as sets of ids.

## Out of scope

- the error model, i.e. an estimated standard deviation, which is the second design,
- observable scaling and offset, `y = s·f(x) + b`, which is the other classical nuisance parameter and needs the hierarchical optimization of Loos et al. 2018 to be worth having,
- a condition target which is an expression: PEtab allows a formula, this supports a parameter or a number,
- nesting, i.e. a version of a version,
- automatic partitioning of the data; the selectors are written by hand.

## Testing

- a versioned parameter resolves to the simulations its filter selects, and to nothing else,
- two versions of a target on one mapping raise at `initialize()`, naming both,
- a selector which splits a simulation group raises, naming the group,
- a simulation no version covers is simulated with the value of the model, and `coverage()` reports it,
- an unversioned problem produces exactly the changes it produces today, i.e. the existing HCTZ fits are unchanged to the last digit,
- a fit of the HCTZ problem with `Ka_dis_tablet` and `Ka_dis_solution` gives two different values and a lower cost than the shared parameter,
- the metrics count both versions in `k`, and the profiles cover both,
- a problem with versions is written as PEtab, validated with `petab`, read back, and its cost agrees with the original within the tolerance the round trip test already uses,
- the selectors of a problem which is read back are sets of ids and resolve to the same simulations.

## Files

| file | change |
| --- | --- |
| `fit/objects.py` | `FitParameter.target`, `FitParameter.mappings` |
| `fit/parameter_mapping.py` | new, `ParameterMapping` |
| `fit/optimization.py` | resolution in `initialize`, `changes_for` in `_simulate_groups`, `target` in `_validate_parameters` and `_store_model_parameters` |
| `fit/helpers.py` | `filter_keys` |
| `fit/display.py` | the `target` column and the coverage table |
| `fit/report.py` | the same in the parameter table |
| `fit/petab_v2/export.py` | the condition of a version |
| `fit/petab_v2/reader.py` | a condition whose value is an estimated parameter |
| `docs/fitting.md`, `docs/petab.md` | the feature and the round trip |
| `examples/hctz_fitting/fitting/parameters.py` | the tablet and solution versions as the worked example |
