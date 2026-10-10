# Examples

Runnable examples for sbmlsim. They are **not** part of the package: they are not installed with `pip install sbmlsim`, they are read and run from a checkout of the repository.

Every example is a module of the `examples` package, so it is run from the root of the repository with

```bash
python -m examples.timecourse
python -m examples.scan
python -m examples.sensitivity.sensitivity_example
```

An example writes what it creates into the current working directory: figures are saved as `png` next to it, simulation experiments write their results into `results/`. No example opens a window: matplotlib figures are saved to a file, never shown, so that the examples also run on a machine without a display. The models and data an example reads stay next to the example.

## What is where

| path | content |
| --- | --- |
| `examples/timecourse.py` | simulations of the repressilator: the steps of the integrator, changes before the initialization and a change at a time on an equidistant grid |
| `examples/scan.py` | parameter scans of dimension 0, 1 and 2, including a scan over a distribution of parameter values |
| `examples/observables.py` | observables of a scan: a formula of the mass concentration, the PK analysis of pkpdutils and a custom function of midazolam over three doses |
| `examples/experiment_scans.py` | scans and observables in a simulation experiment: the PK parameters of midazolam over three doses as labelled arrays, read by the labels of the doses, drawn as curves per dose, cmax over the dose and a band over Latin hypercube draws |
| `examples/fit_sampling.py` | sampling of the start values of a parameter fit, uniform and logarithmic, with and without latin hypercube sampling |
| `examples/units.py` | units of a model and changes with pint quantities |
| `examples/model_sensitivity.py` | a local design and lognormal draws of all parameters of the repressilator, with their mean and range |
| `examples/curve_types/` | a two reaction model (created with sbmlutils) and a simulation experiment showing the curve types of the plots |
| `examples/initial_assignment/` | a simulation experiment on a model with initial assignments and changes of the assigned parameters |
| `examples/glucose/` | dose response experiment of the hepatic glucose model with data from PK-DB, the hormones drawn as observables over the glucose dimension |
| `examples/demo/` | the demo model with scans and sensitivity simulations as a simulation experiment, with the initial values drawn as curves with a colour bar |
| `examples/repressilator/` | the repressilator as a simulation experiment, with post processing functions and scans |
| `examples/hctz_fitting/` | hydrochlorothiazide pharmacokinetics: a whole body model, simulation experiments against the data of Beermann 1976 and Patel 1984 (with the cmax and AUC of HCTZ over the dose) and the parameter fitting problems built on them (`examples/hctz_fitting/fitting/`) |
| `examples/sensitivity/` | the local, Sobol, FAST and Morris analyses of a simple chain model, each a design of the sampler, a run of the scan core with observables and a dimension of conditions, and the indices on the result |
| `examples/petab/` | PEtab parameter estimation problems of the [benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab). `benchmark.py` converts `Perelson_Science1996` or `Boehm_JProteomeRes2014` to PEtab v2, fits it with sbmlsim, analyses the identifiability and reports it. The PEtab v2 layer on the HCTZ problem is `examples/hctz_fitting/fitting/petab_problem.py` |
| `examples/sciml/` | hybrid problems of PEtab SciML: `lotka_volterra_fit.py` reads the case 001 of the test suite (`examples/sciml/lotka_volterra/`, a network in the right hand side), improves it with a short fit in one process, reports the fit and writes the fitted problem as PEtab SciML again; `neural_ode/` is a neural ODE defined in python, fitted once from the values of its network and in parallel from random starts, with validation data after the training data, written as PEtab SciML |
| `examples/comparison/` | comparison of simulation results between simulators. `diff_example.py` compares roadrunner with [JWS Online](https://jjj.bio.vu.nl) on the repressilator with `sbmlsim.comparison.diff`; `simulate_amici.py`, `simulate_copasi.py` and `example_comparison.py` run the same conditions with AMICI and COPASI (not installed with sbmlsim) |

## Tests

`tests/examples/test_example_scripts.py` runs the examples which work offline and without optional dependencies as `python -m examples.<module>` in a temporary working directory, so an example which breaks fails the test suite.

A curve draws one line: the point of a scan it shows is selected with `Data(sel=...)`; `examples/demo`, `examples/glucose` and `examples/repressilator` run and are tested.

`examples/hctz_fitting` is the reference problem of the parameter fitting: `python -m examples.hctz_fitting.simulations` runs the simulation experiments, `python -m examples.hctz_fitting.fitting.fitting` the fit, and `python -m examples.hctz_fitting.fitting.run_report <parameters.json>` creates the report of a finished fit again, or of several fits at once, without optimizing. `python -m examples.hctz_fitting.fitting.identifiability` runs a global optimization followed by the profile likelihood of the best parameter set, and `python -m examples.hctz_fitting.fitting.identifiability_report <parameters.json>` computes the profiles of stored parameters. `python -m examples.hctz_fitting.fitting.petab_problem` writes the fit as a PEtab v2 problem, validates it with `petab` and reads it back, and reports what PEtab cannot express about the fit. These are the general tools of `sbmlsim.fit.cli` and `sbmlsim.fit.petab_v2` on the `FitDefinition` objects of `examples/hctz_fitting/fitting/fitting.py`, which is all the example has to provide. The tests in `tests/fit/` use it.
