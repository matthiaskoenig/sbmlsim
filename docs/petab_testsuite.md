# PEtab Test Suite

The [PEtab test suite](https://github.com/PEtab-dev/petab_test_suite) is the conformance suite of [PEtab](petab.md). Every case is a small PEtab problem, `XXXX/_XXXX.yaml`, with the values a correct tool computes at the nominal parameters in `XXXX/_XXXX_solution.yaml`, and each case exercises one feature of the format. The cases of PEtab v2 are in `petabtests/cases/v2.0.0/sbml`, the expressions of the math of PEtab with the values they parse to in `petabtests/cases/v2.0.0/math/math_tests.yaml`.

## What a case checks

A case states three values, each with its tolerance:

| value | tolerance | compared |
| --- | --- | --- |
| `simulation_files` | `tol_simulations` | the simulated observables at every measurement, a table like the measurement table with the column `simulation` |
| `chi2` | `tol_chi2` | the sum of the squared residuals, each divided by the standard deviation of its measurement |
| `llh` | `tol_llh` | the log-likelihood of the measurements under the noise model of their observables |

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

`sbmlsim` does not run the suite as part of its tests yet. A case is read with `PetabReader.from_yaml`, its log-likelihood is `sbmlsim.fit.petab_v2.likelihood.log_likelihood` at the nominal parameters and its simulations are the predictions of the optimization problem at the measurements, see [PEtab](petab.md#reading-a-petab-problem) and [the log-likelihood](petab.md#the-log-likelihood).
