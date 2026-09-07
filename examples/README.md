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
| `examples/midazolam/` | midazolam pharmacokinetics: simulation experiments against the data of Kupferschmidt 1995 and Mandema 1992, parameter fitting problems and their serialization to SED-ML |
| `examples/covid/` | COVID-19 models from BioModels as simulation experiments and COMBINE archives |
| `examples/sedml/` | execution of SED-ML files (`execute_sedml.py`) and COMBINE archives (`execute_omex.py`), with the SED-ML L1V4 example files under `l1v4/` |
| `examples/sensitivity/` | local and global sensitivity analysis (sampling, Sobol, FAST, Morris) of a simple chain model |
| `examples/petab/` | PEtab parameter estimation problems, with pypesto and AMICI (both not installed with sbmlsim) |
| `examples/julia/` | notes and an example on calling julia from python via juliacall (not installed with sbmlsim) |

## Tests

`tests/examples/test_example_scripts.py` runs the examples which work offline and without optional dependencies as `python -m examples.<module>` in a temporary working directory, so an example which breaks fails the test suite.

The simulation experiments with post processing functions and multi-dimensional scans in `examples/demo`, `examples/repressilator`, `examples/midazolam`, `examples/covid` and `examples/sedml/execute_omex.py` currently fail while the experiment pipeline is reworked, see the skipped tests in `tests/experiment/` and `tests/fit/`; they are not part of the example tests.
