# Cleanup of sbmlsim to its core (#250)

## Goal

sbmlsim is reduced to its core: simulating timecourses, scans, sensitivity and uncertainty analysis, parameter fitting, plotting and reporting of results. The leftovers of SED-ML and the dead code are removed, the remaining code has one way to do one thing, and the continuous integration gets faster. The result is one pull request for #250 with one commit per group below, so it is reviewed commit by commit.

## Boundaries

The decision what stays is the user's, not a usage count:

- Stays, although pkdb_models does not use it: everything of PEtab (`fit/petab_v2/` with export, reader, extension, gaps, the PEtab test suite and the benchmark collection, `fit/petab_omex.py`), PEtab SciML (`sciml/`, `fit/petab_v2/sciml*.py`, `fit/derived.py`, `model/provenance.py`), `comparison/` and `examples/comparison`, the SBML Test Suite (`testsuite/`).
- Stays, because pkdb_models uses it: `simulation/sensitivity.py` (`ModelSensitivity`, 9 model directories), `data.load_pkdb_dataframes_by_substance` (9 directories), the whole `sensitivity/` package, `ExperimentRunner`, `ExperimentReport`, the figure model.
- Goes: the leftovers of SED-ML, the dead code, the shims and the examples which are not about the core.
- Out of scope: the migration of pkdb_models to the API of the simulation engine (1232 of its files still use `TimecourseSim`/`Timecourse`, removed in `bc190b2`, and 233 files are stale against 0.8.3 already). It gets an issue of its own. No removal of this cleanup touches what live code of pkdb_models imports.

## Findings this design is based on

- The sources are 43.5k lines. Of 463 public symbols 393 are never used by pkdb_models, most of them are internals of the core and stay.
- A pull request takes 6.5 to 8.5 minutes, set by the test job on Windows (pytest 430 s, 172 s on Linux). Of the test time `tests/fit` is 51 %, `tests/examples` 28 % (21 examples, each a subprocess) and `tests/sciml` 12 %.
- `import sbmlsim.simulator` costs 2.2 s: `simulator/formula.py` imports `petab.v2.math`, whose package imports `petab_sciml` and with it `torch`. Every xdist worker, every example subprocess and every worker of a parallel fit pays it.

## A. Leftovers of SED-ML

Removed:

- `simulation/base.py`, `simulation/algorithm.py`, `simulation/calculation.py`, `simulation/change.py`. The integrator is set by `RoadrunnerSBMLModel.set_integrator_settings`, the changes by `simulation.definition.Change`. `tests/simulation/test_algorithm.py` and the mention in `fit/petab_v2/gaps.py` go with them.
- In `simulation/range.py` everything except `Dimension`: `Range`, `VectorRange`, `UniformRangeType`, `UniformRange`, `DataRange`, `FunctionalRange` and the `__main__` block.
- `result/datagenerator.py` with `examples/datagenerator.py` and `tests/result/test_datagenerator.py`.
- `result/report.py` and the hook `SimulationExperiment.reports()` with `_reports`; the reports are collected but never rendered or serialized, and no experiment of pkdb_models defines the hook. `examples/curve_types/experiment.py` loses its report.
- The libsedml half of `mathml.py` (`formula_to_astnode`, `astnode_to_formula`, `parse_mathml_str`, `evaluate`, `replace_piecewise`, `_get_variables`, the `__main__` block) and the dependency `python-libsedml`.
- The fixtures of SED-ML and NuML which nothing references: `tests/data/data/{numlData*,reading-*,oscli*,parameter-from-data-csv}.xml`, `tests/data/models/{asedml3repeat,asedmlComplex,app2sim,BorisEJB,curien,lorenz,oscli,BioModel1_*}.xml`, `tests/data/petab/icg_example1/`, and the constants of `tests/test_data.py` pointing at the missing `RESOURCES_DIR/testdata`. Each is checked for references before it is deleted.

Moved: the sbmlmath half of `mathml.py` (`formula_expression`, `evaluate_formula`, `expression_to_*`) is used only by `sciml/` and moves into `sciml/` as a module of it; `mathml.py` disappears.

### Formulas of `Data`

A `Data` of type FUNCTION is compiled with `simulator.formula.compile_formula`, so sbmlsim has one formula language, the math of PEtab, for changes, observables and data. Formulas like `x / y`, `x^2`, `ln(x)` or `piecewise(...)` read the same as before.

PEtab math has no reduction over an array; `sympify_petab("Y/max(Y)")` fails with "Unexpected number of arguments: 1 in max(Y)". The examples normalize curves with `Y/max(Y)` (`examples/repressilator/repressilator.py:65`, `repressilator_scans.py:97,106`). This is kept as the one extension of PEtab math for data: a call of `max` or `min` with a single argument reduces its argument over the data (`nanmax`, `nanmin`). A pre-pass finds these calls (balanced parentheses, nested calls inside out), compiles and evaluates the argument on the data, reduces it and replaces the call by a placeholder symbol whose value is the reduction; the rest is compiled by `compile_formula`. A call with two or more arguments is the elementwise `max`/`min` of PEtab. The values stay pint quantities, the evaluation keeps their units as it does now; tests cover units, the reduction, nested reductions and the error for invalid math. `docs/data.md` documents the syntax and the extension.

## B. Dead code and shims

Removed, each verified to have no caller in `src/`, `examples/`, `docs/` and in pkdb_models; the check of pkdb_models searches every file for the name, including method calls, not only the imports:

- In `model/model_resources.py` the sources from URNs, URLs and BioModels (they import `requests`, which is not a dependency) with `tests/models/test_biomodels.py`, which needs the network and gets 403 in CI; `Source.is_path`. In `model/model.py` `AbstractModel.SourceType` and `LanguageType.CELLML`.
- `model/model_change.py` (`ModelChange.clamp_species`) with `examples/model_change.py` and `tests/test_model_change.py`.
- `RoadrunnerSBMLModel.copy_roadrunner_model`, `Figure.num_subplots`, `Figure.from_plots`, `XResult.from_netcdf`, `XResult.is_ragged`, `XResult.from_dfs`, `ScanSim.get_dimension`, `SimulationExperiment.from_json` (raises `NotImplementedError`), `simulation.sensitivity.DistributionType`, `utils.function_name`, `ObjectJSONEncoder.to_json`.
- The helpers of `fit/` which only tests call: `helpers.mapping_kinds_info`, `OptimizationResult.run_result`. A helper which pkdb_models calls stays although sbmlsim does not (`OptimizationResult.xopt_fit_parameters`, `SensitivityParameter.parameters_set_bounds`, `SensitivityParameter.parameter_to_latex`), and so does a helper which documents a result of the method (`identifiability.cost_threshold`, `likelihood.stencil`, `metrics.aic`/`bic`).
- The `__main__` blocks of library modules: `units.py`, `simulation/sensitivity.py`, `fit/cli.py` (the ones of `mathml.py`, `range.py`, `calculation.py` go with A).
- Shims: the guard for the removed parameters `fitting_type` and `weighting_local` in `fit/runner.py`, the alias `TEMPLATE_PATH` and the FIXME for "old outputs" in `report/experiment_report.py`, the backwards compatibility of `Data(index="[X]")` next to `symbol` in `data.py`.
- `tests/test_sensitivity.py::test_sensitivity_example`, skipped with "no sensitivity support", and the second run of an example which `tests/examples/test_example_scripts.py` already runs (`tests/test_units.py::test_example_units`). `tests/sensitivity/test_sensitivity_example.py` stays: it tests every analysis class on the definitions of the example, not the example script.
- `examples/julia/` and `examples/interpolation/` with `tests/examples/test_interpolation.py`; neither is about the core.
- The untracked directories which hold only `__pycache__` (`src/sbmlsim/combine/`, `src/sbmlsim/interpolation/`, `tests/{combine,interpolation,processing,comparison}/`, `tests/data/{combine,diff,data/omex}`) are deleted locally; they are not in git.

## C. Dependencies

- Removed: `pkpdutils` and `sbml4humans` (nothing imports them), `python-libsedml` (with A).
- Declared: `pyyaml`, which `fit/petab_v2/testsuite.py`, `benchmark.py` and `sciml/` import directly and today get only through petab.
- `pymetadata` stays for `fit/petab_omex.py`, `seaborn` for `sensitivity/plots.py` and `comparison/diff.py`.

## D. Speed of the tests and of CI

- `simulator/formula.py` imports `petab.v2.math` when the first formula is compiled, not when the module is imported. A test asserts in a subprocess that `import sbmlsim`, `sbmlsim.simulator` and `sbmlsim.experiment` do not import `torch` or `petab`.
- `fit/runner.py` runs the repeats serially when `n_cores=1`; it starts a `multiprocessing.Pool` only for two or more workers.
- The sensitivity plots take the resolution as a parameter (`dpi`, default 300 as now, hardcoded today in `sensitivity/plots.py`, `sensitivity_sampling.py`, `sensitivity_morris.py`); the quick mode of the sensitivity example, which the tests run, saves at a low resolution.
- The test of `examples.petab.benchmark` runs without the identifiability analysis.
- The fixtures of `tests/fit` which no test mutates become module scoped.
- The two assertions on wall time which can fail on a loaded runner (`tests/model/test_model_roadrunner_init.py`, 5 ms per `initialize`; `tests/sciml/test_compiler.py`, a model load under 10 s which takes 7 to 11 s under load) are made robust: they compare against a reference measured in the same test or get a bound with a margin which the runners keep.
- The cache of uv is saved by one job of the matrix per key, which ends the warnings "Unable to reserve cache".
- The operating system matrix stays: the job on Windows found the bug of the flattening of hierarchical models.

The time of the default suite is measured before and after on 4 CPUs (`taskset -c 0-3 pytest -n 4`, 2:53 before) and the pull request reports both, together with the wall time of the jobs of CI.

## Documentation

- `zensical.toml` excludes `docs/superpowers/` with the `exclude` plugin, so the plans and specs stay in the repository but are no longer published.
- The pages in `docs/api/` of removed modules are removed, `docs/api/index.md` no longer describes the SED-ML `Change`, the pages which use removed API are updated.
- `CLAUDE.md` is updated where it describes removed code (the hook `reports()`, `mathml.py`, the dependencies, `model_resources` sources).

## Verification

- `pytest`, `ruff check`, `ruff format --check`, `uv run ty check`, `uv run zensical build --clean` pass; the examples of `tests/examples` run.
- The test suites deselected by default run before and after with the same result: `pytest -m testsuite` against `tests/data/testsuite_baseline.json`, `pytest -m petab_testsuite tests/fit`, the SciML suite.
- pkdb_models: the live files (those whose imports resolve against the present sbmlsim) still import; a script in the scratchpad imports every module of them before and after.
- Each commit passes the checks on its own, so the history can be bisected.
