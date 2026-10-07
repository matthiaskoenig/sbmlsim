# Simulation engine, phase 2: PEtab on the engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read a PEtab v2 problem with its full semantics onto the engine of phase 1, i.e. conditions with formulas, ids translated into selections, steady state measurements, observables and noise with placeholders, and evaluate it exactly: simulations, log-likelihood, chi2 and log-prior. Add the PEtab v2 test suite and the benchmark collection as suites with baselines.

**Architecture:** The reader builds a `Simulation` per experiment as in phase 1, now with formula changes and the translation of PEtab ids. Every parameter of the parameter table which is not an entity of the model is added to the model as a constant parameter, so a value of the parameter table, an observable parameter and a noise parameter are model parameters like any other: conditions read them, the fit sets them before the initialization, and the observables and the noise formulas read them from the simulation. An observable is an `ObservableModel` (formula, placeholders and their values per measurement), compiled to numpy and evaluated on the selections of the simulation at the measurements; the SBML is no longer rewritten for observables. A measurement at `time = inf` is a steady state output after the end of the simulation.

**Tech Stack:** python 3.13, libroadrunner 2.10, petab 0.9 (`petab.v2`, `sympify_petab`), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-simulation-engine-design.md`

## Spec amendment

The spec gave a `FitParameter` a kind (`MODEL`, `OBSERVABLE`, `NOISE`). Adding the parameters of the parameter table which are not entities to the model makes every parameter a model parameter: one kind, one way to set a value (a pre-initialization change), one way to read it (a selection). The spec is amended in Task 1.

## Global Constraints

- The constraints of phase 1 hold (no em dash, no agent attribution, annotations and docstrings, ty clean, ruff clean, markdown without hard wraps).
- The test suite is pinned by a commit of `PEtab-dev/petab_test_suite`, the benchmark collection by a commit of `Benchmarking-Initiative/Benchmark-Models-PEtab`; both are cached under `~/.cache/sbmlsim/` like the SBML Test Suite and the PEtab SciML suite, with `SBMLSIM_PETAB_SUITE_PATH` and `SBMLSIM_BENCHMARK_PATH`.
- Every case and every problem which does not pass is listed in its baseline with a reason; a normal `pytest` deselects them (markers `petab_testsuite`, `petab_benchmark`).

## Review Focus

- A condition `A = A + 5` on a concentration species at a time with a measurement: the measurement sees the state after the change (cases 0028, 0031).
- A compartment and a species in it set by one condition: the species gets its value, the other concentration species keep their concentration (case 0022).
- Two measurements of one observable with different placeholder values in one experiment: each is evaluated with its own values (case 0006).
- A noise formula which reads the observable (case 0021).
- A measurement at `inf` and one at a finite time of the same observable and experiment (case 0023).

---

### Task 1: Spec amendment and the parameters of the parameter table in the model

**Files:**
- Modify: `docs/superpowers/specs/2026-10-07-simulation-engine-design.md` (section "Amendments after phase 1")
- Modify: `src/sbmlsim/model/model.py` (`AbstractModel(parameters: Mapping[str, float] | None = None)`), `src/sbmlsim/model/model_roadrunner.py` (the parameters are added to the SBML with the helpers, before the model is loaded, refusing an id which is an entity)
- Test: `tests/model/test_model_parameters.py`

Steps: test that a model loaded with `parameters={"scale": 2.0}` has `r["scale"] == 2.0`, that `initialize([Assignment("scale", PARAMETER, 3.0)])` sets it, that it is not in the default selections, and that an id of an entity raises; implement; commit.

### Task 2: Formula changes before the initialization

**Files:**
- Modify: `src/sbmlsim/simulation/definition.py` (allow formulas in `preinit_changes`; a formula reads only parameters, checked at compile time), `src/sbmlsim/simulator/plan.py`, `src/sbmlsim/model/model_roadrunner.py` (`initialize` sets the values first, then evaluates the formulas on the current values, then the initial assignments)
- Test: `tests/simulator/test_executor.py`

Steps: test case-0026-like `preinit_changes={"A": "a1 + a2"}` with `a1`, `a2` model parameters; a formula reading a species raises at compile time; implement; commit.

### Task 3: Steady state output at `inf`

**Files:**
- Modify: `src/sbmlsim/simulation/definition.py` (`times` may contain `inf`), `src/sbmlsim/simulator/plan.py` (`Plan.steady_state_output: SteadyStatePlan | None`), `src/sbmlsim/simulator/executor.py` (after the last segment integrate to steady state and append a row with `time = inf`)
- Modify: `src/sbmlsim/fit/optimization.py` (`x_references` may hold `inf` for a time observable; `_compile_plans` keeps them; `_interpolate` takes the row at `inf`)
- Test: `tests/simulator/test_executor.py`, `tests/fit/test_steady_state_data.py`

### Task 4: PEtab ids as selections

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/symbols.py` (`selection_of_target(target, model) -> str`, `selections_of_formula(formula, model) -> str` which rewrites every concentration based species `S` of a math expression to `[S]`)
- Modify: `src/sbmlsim/fit/petab_v2/reader.py` (`_period_changes` translates targets and formulas; a formula is kept as a formula change; the first period's formulas are pre-initialization formulas, Task 2)
- Test: `tests/fit/test_petab_v2_symbols.py`, `tests/fit/test_petab_v2_reader.py`

### Task 5: Observables as formulas with placeholders

**Files:**
- Create: `ObservableModel` in `src/sbmlsim/fit/objects.py` (formula in the selection convention, placeholders, placeholder values per measurement as numbers or formulas of parameters), `FitMapping(observable_model=...)`
- Modify: `src/sbmlsim/fit/optimization.py` (the selections of a mapping are the symbols of its observable model; `predictions` and `residuals` evaluate it at the rows of the data)
- Modify: `src/sbmlsim/fit/petab_v2/reader.py` (an observable is an `ObservableModel`, the measurements of a mapping keep their placeholder values), remove `src/sbmlsim/fit/petab_v2/observables.py` and its use, `export.py` writes the observable model
- Test: `tests/fit/test_observable_model.py`, `tests/fit/test_petab_v2_reader.py`

### Task 6: Noise and log-likelihood on the engine

**Files:**
- Modify: `src/sbmlsim/fit/petab_v2/likelihood.py` (noise formulas read the parameters from the simulation, i.e. from the model; `chi2`; `log_prior` of the priors of PEtab v2 with truncation at the bounds; the gradient is unchanged)
- Test: `tests/fit/test_petab_v2_likelihood.py`

### Task 7: The PEtab v2 test suite

**Files:**
- Create: `src/sbmlsim/fit/petab_v2/testsuite.py` (`PetabSuite` download and cache of a pinned commit, `PetabCase.from_directory`, `run` comparing `llh`, `chi2`, the simulations, `log_prior` and `unnorm_log_posterior` with their tolerances, `CaseResult`/`CaseStatus` like the SciML suite; the math cases of `v2.0.0/math/math_tests.yaml` against `compile_formula`)
- Create: `scripts/petab_testsuite.py` (`download`, `run`, `baseline`), `tests/fit/test_petab_testsuite.py` (one test per case, marker `petab_testsuite`), `tests/data/petab_baseline.json`, the tox environment `petab`, the marker in `pyproject.toml`
- Modify: `docs/petab_testsuite.md`, `docs/testsuites.md`

### Task 8: The benchmark collection

**Files:**
- Create: `src/sbmlsim/fit/petab_v2/benchmark.py` (download of a pinned commit, conversion with `petab1to2` into the cache, `BenchmarkProblem.run`: read, simulate at the nominal parameters, compare with `simulations.tsv` with the tolerance `1e-3`, compare the log-likelihood with the reference of AMICI where the noise distribution survives the conversion, and the timings), `scripts/petab_benchmark.py` (`download`, `run`, `report`, `baseline`), `tests/fit/test_petab_benchmark.py` (marker `petab_benchmark`), `tests/data/benchmark_baseline.json`, tox `benchmark`
- Modify: `docs/petab_benchmark.md`, `docs/testsuites.md`, `examples/petab/benchmark.py`

### Task 9: Documentation and verification

Update `docs/petab.md` (the reading of a problem, the observables, the noise, what is not supported), `CLAUDE.md` (the PEtab paragraph), run every suite, open the pull request.
