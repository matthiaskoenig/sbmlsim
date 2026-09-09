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
| `examples/timecourse.py` | timecourse simulations of the repressilator: single, with parameter changes and concatenated timecourses |
| `examples/scan.py` | parameter scans of dimension 0, 1 and 2, including a scan over a distribution of parameter values |
| `examples/fit_sampling.py` | sampling of the start values of a parameter fit, uniform and logarithmic, with and without latin hypercube sampling |
| `examples/model_change.py` | clamping species with `ModelChange`, manually on the roadrunner instance and in a `TimecourseSim` |
| `examples/units.py` | units of a model and changes with pint quantities |
| `examples/model_sensitivity.py` | sensitivity scans of all parameters, by relative differences and by sampling from distributions |
| `examples/datagenerator.py` | reducing scan results with a `DataGenerator` |
| `examples/interpolation.py` | interpolation of data points as an SBML model |
| `examples/curve_types/` | a two reaction model (created with sbmlutils) and a simulation experiment showing the curve types of the plots |
| `examples/initial_assignment/` | a simulation experiment on a model with initial assignments and changes of the assigned parameters |
| `examples/glucose/` | dose response experiment of the hepatic glucose model with data from PK-DB |
| `examples/demo/` | the demo model with scans and sensitivity simulations as a simulation experiment |
| `examples/repressilator/` | the repressilator as a simulation experiment, with post processing functions and scans |
| `examples/hctz/` | hydrochlorothiazide pharmacokinetics: a whole body model, simulation experiments against the data of Beermann 1976 and Patel 1984 and the parameter fitting problems built on them (`examples/hctz/fitting/`) |
| `examples/sensitivity/` | local and global sensitivity analysis (sampling, Sobol, FAST, Morris) of a simple chain model |
| `examples/petab/` | PEtab parameter estimation problems of the [benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab). `benchmark.py` converts `Perelson_Science1996` or `Boehm_JProteomeRes2014` to PEtab v2, fits it with sbmlsim, analyses the identifiability and reports it. The PEtab v2 layer on the HCTZ problem is `examples/hctz/fitting/petab_problem.py` |
| `examples/julia/` | notes and an example on calling julia from python via juliacall (not installed with sbmlsim) |

## Tests

`tests/examples/test_example_scripts.py` runs the examples which work offline and without optional dependencies as `python -m examples.<module>` in a temporary working directory, so an example which breaks fails the test suite.

The simulation experiments with post processing functions and multi-dimensional scans in `examples/demo` and `examples/repressilator` currently fail while the experiment pipeline is reworked, see the skipped tests in `tests/experiment/`; they are not part of the example tests.

`examples/hctz` is the reference problem of the parameter fitting: `python -m examples.hctz.simulations` runs the simulation experiments, `python -m examples.hctz.fitting.fitting` the fit, and `python -m examples.hctz.fitting.run_report <parameters.json>` creates the report of a finished fit again, or of several fits at once, without optimizing. `python -m examples.hctz.fitting.identifiability` runs a global optimization followed by the profile likelihood of the best parameter set, and `python -m examples.hctz.fitting.identifiability_report <parameters.json>` computes the profiles of stored parameters. `python -m examples.hctz.fitting.petab_problem` writes the fit as a PEtab v2 problem, validates it with `petab` and reads it back, and reports what PEtab cannot express about the fit. These are the general tools of `sbmlsim.fit.cli` and `sbmlsim.fit.petab_v2` on the `FitDefinition` objects of `examples/hctz/fitting/fitting.py`, which is all the example has to provide. The tests in `tests/fit/` use it.
