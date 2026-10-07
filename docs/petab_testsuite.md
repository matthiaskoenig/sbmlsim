# PEtab Test Suite

The [PEtab test suite](https://github.com/PEtab-dev/petab_test_suite) is the conformance suite of [PEtab](petab.md). Every case is a small PEtab problem, `XXXX/_XXXX.yaml`, with the values a correct tool computes at the nominal parameters in `XXXX/_XXXX_solution.yaml`, and each case exercises one feature of the format. The cases of PEtab v2 are in `petabtests/cases/v2.0.0/sbml`, the expressions of the math of PEtab with the values they parse to in `petabtests/cases/v2.0.0/math/math_tests.yaml`.

## What a case checks

A case states three values, and the cases with priors two more, each with its tolerance:

| value | tolerance | compared |
| --- | --- | --- |
| `simulation_files` | `tol_simulations` | the simulated observables at every measurement, a table like the measurement table with the column `simulation` |
| `chi2` | `tol_chi2` | the sum of the squared residuals, each divided by the standard deviation of its measurement |
| `llh` | `tol_llh` | the log-likelihood of the measurements under the noise model of their observables |
| `log_prior` | `tol_log_prior` | the log density of the prior of every estimated parameter |
| `unnorm_log_posterior` | `tol_unnorm_log_posterior` | the log-likelihood plus the log priors |

The simulations test the semantics of the problem, i.e. the initialization of the model with the parameter table, the conditions of the periods of an experiment, the pre-equilibration and the observables with their placeholders, and the two other values add the noise model. A tool covers a case if its values agree with the reference values within the tolerances.

## The features of the cases

The cases of PEtab v2 cover, one or a few at a time:

- the conditions of an experiment: numeric and parametric overrides, several conditions at one time, conditions which change a compartment and the species in it, a condition at a time without measurements, the start of an experiment at a time other than zero, and target values which are math expressions,
- the initialization: initial assignments overridden by the parameter table and by conditions, initial values which are estimated,
- pre-equilibration to a steady state, with species which are reinitialized after it and with events during it,
- the placeholders of the observables and of the noise per measurement, replicate measurements, the noise distributions and a noise formula which depends on the observable,
- events of the model which trigger together with a condition,
- priors on the parameters, truncated and not truncated.

The `README.md` of `petabtests/cases/v2.0.0/sbml` describes every case in one line.

## sbmlsim and the suite

`sbmlsim.fit.petab_v2.testsuite` runs the suite: `PetabSuite` is a commit of it with the download and the cache, `PetabCase` is a case of a model and `MathCase` a case of the math. A case is read with `PetabReader.from_yaml`, simulated with the integrator tolerances `1e-12` at the nominal values of its parameter table (`PetabReader.nominal_parameters`, which may lie outside of the bounds of a parameter) and compared with the values of its solution: `log_likelihood`, `chi2`, `log_prior` and `unnorm_log_posterior` of `sbmlsim.fit.petab_v2.likelihood`, and the predictions of the optimization problem at the measurements, see [PEtab](petab.md#reading-a-petab-problem) and [the log-likelihood](petab.md#the-log-likelihood). A math case compiles its expression with `sbmlsim.simulator.formula.compile_formula`, which evaluates the formulas of the conditions, the observables and the noise, and compares its value; a symbolic expected value is compared at values of its symbols. The outcome of a case is `pass`, `tolerance` (a value outside of its tolerance, the message names it) or `error` (the case cannot be read or simulated).

The suite is pinned by a commit, `PETAB_SUITE_COMMIT`, and cached under `~/.cache/sbmlsim/petab-test-suite/<commit>/`, i.e. the directory `petabtests/cases/v2.0.0` of the suite; `SBMLSIM_PETAB_SUITE_PATH` points at a checkout instead. Every case is a test with the marker `petab_testsuite`, which a normal `pytest` deselects:

```bash
uv run python scripts/petab_testsuite.py download  # fetch the pinned commit
uv run pytest -m petab_testsuite tests/fit          # every case is a test
uv run python scripts/petab_testsuite.py run        # the outcome against the baseline
uv run python scripts/petab_testsuite.py baseline   # refresh the baseline
tox r -e petab                                      # download and run
```

The cases which do not pass are recorded in `tests/data/petab_baseline.json` with their status and their reason, and a run fails in both directions: a case which passed and fails is a regression, a case which is recorded and passes is a baseline which is out of date. All 31 cases of the models of SBML and all 98 math cases of the pinned commit pass.
