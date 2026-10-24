# PEtab Test Suite

The [PEtab test suite](https://github.com/PEtab-dev/petab_test_suite) is the conformance suite of [PEtab](petab.md). Each case is a small PEtab problem that tests one feature of the format, together with the values a correct tool computes at the nominal parameters. Running it tells you whether `sbmlsim` reads and simulates PEtab v2 problems the way the standard defines them.

All 31 SBML cases and all 98 math cases of the pinned commit pass.

## Run it

```bash
uv run python scripts/petab_testsuite.py download   # download the pinned commit
uv run python scripts/petab_testsuite.py run        # run every case, compare with the baseline
uv run pytest -m petab_testsuite tests/fit          # the same, one test per case
tox r -e petab                                      # download, then run the tests
```

The suite is pinned by a commit (`PETAB_SUITE_COMMIT` in `sbmlsim.fit.petab_v2.testsuite`) and cached in `~/.cache/sbmlsim/petab-test-suite/<commit>/`. Set `SBMLSIM_PETAB_SUITE_PATH` to use a checkout somewhere else; it points at the directory `petabtests/cases/v2.0.0` of the checkout.

A normal `pytest` skips these tests. `tests/data/petab_baseline.json` lists the cases that do not pass (none at the moment), and a run fails both when a case breaks and when a listed case starts to pass. After a change that fixes or breaks cases, refresh the list with `uv run python scripts/petab_testsuite.py baseline`.

## What a case checks

The model cases are in `petabtests/cases/v2.0.0/sbml` of the suite, `XXXX/_XXXX.yaml` with the expected values in `XXXX/_XXXX_solution.yaml`. Each case states three values, and the cases with priors two more, each with its own tolerance:

| value | tolerance | what it is |
| --- | --- | --- |
| `simulation_files` | `tol_simulations` | the simulated observable at every measurement |
| `chi2` | `tol_chi2` | the sum of the squared residuals, each divided by the standard deviation of its measurement |
| `llh` | `tol_llh` | the log-likelihood of the measurements under the noise model |
| `log_prior` | `tol_log_prior` | the log density of the prior of every estimated parameter |
| `unnorm_log_posterior` | `tol_unnorm_log_posterior` | the log-likelihood plus the log priors |

The simulations test how a problem is interpreted. `chi2` and `llh` add the noise model, and the last two the priors.

The cases cover, one or a few at a time:

- conditions: numeric and parametric overrides, several conditions at one time, a condition that changes a compartment and the species in it, a condition at a time without measurements, an experiment that does not start at zero, and target values that are math expressions,
- initialization: initial assignments overridden by the parameter table and by conditions, and estimated initial values,
- pre-equilibration to a steady state, with species reinitialized after it and with events during it,
- placeholders in observables and in the noise, replicate measurements, the noise distributions, and a noise formula that depends on the observable,
- model events that trigger together with a condition,
- priors on the parameters, truncated and not truncated.

The `README.md` of `petabtests/cases/v2.0.0/sbml` describes every case in one line. The math cases are in `petabtests/cases/v2.0.0/math/math_tests.yaml`: expressions of PEtab math with the values they evaluate to. The suite also has cases with PySB models, which `sbmlsim` does not run.

## How sbmlsim runs a case

The code is in `sbmlsim.fit.petab_v2.testsuite`. A model case is read with `PetabReader.from_yaml` and simulated with integrator tolerances of `1e-12` at the nominal values of its parameter table, which may lie outside the bounds of a parameter. The results are compared with the solution: the predictions at the measurements, and `log_likelihood`, `chi2`, `log_prior` and `unnorm_log_posterior` of `sbmlsim.fit.petab_v2` (see [Reading a PEtab problem](petab.md#reading-a-petab-problem) and [The log-likelihood](petab.md#the-log-likelihood)).

A math case compiles its expression with `sbmlsim.simulator.formula.compile_formula`, the same compiler that evaluates the formulas of conditions, observables and noise. An expected value that is itself an expression is compared at values of its symbols.

The outcome of a case is `pass`, `tolerance` (a value is outside its tolerance and the message names it) or `error` (the case cannot be read or simulated).
