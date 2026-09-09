![](images/favicon/sbmlsim-100x100-300dpi.png)

# sbmlsim: SBML simulation made easy
[![GitHub Actions CI/CD Status](https://github.com/matthiaskoenig/sbmlsim/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/matthiaskoenig/sbmlsim/actions/workflows/ci-cd.yml) [![Documentation](https://img.shields.io/badge/docs-sbmlsim-3f51b5.svg)](https://matthiaskoenig.github.io/sbmlsim) [![Version](https://img.shields.io/pypi/v/sbmlsim.svg)](https://pypi.org/project/sbmlsim/) [![Python Versions](https://img.shields.io/pypi/pyversions/sbmlsim.svg)](https://pypi.org/project/sbmlsim/) [![MIT License](https://img.shields.io/pypi/l/sbmlsim.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/55952847.svg)](https://zenodo.org/badge/latestdoi/55952847)

`sbmlsim` is a collection of python utilities for the simulation of models in the [Systems Biology Markup Language](https://sbml.org) (SBML), built on [libroadrunner](https://libroadrunner.org). The source code is available from [https://github.com/matthiaskoenig/sbmlsim](https://github.com/matthiaskoenig/sbmlsim).

## Background

SBML is the exchange format for computational models in systems biology ([Keating *et al.* 2020](references.md#standards)) and libroadrunner is a fast simulator for it ([Welsh *et al.* 2023](references.md#simulation)). Simulating a model is a few lines with roadrunner; a simulation *experiment* is more: the model comes with changes of parameters and initial conditions, timecourses are concatenated into dosing protocols, parameters are scanned over ranges, the results are compared to experimental data in the units of the model, plotted and reported, and all of that has to be reproducible.

`sbmlsim` is the layer above the simulator which describes these experiments. A `Timecourse` is a period of a simulation with its changes, a `TimecourseSim` concatenates them, a `ScanSim` runs a simulation over the dimensions of parameter changes, and a `SimulationExperiment` collects models, datasets, simulations, tasks, data and figures into one python object which is executed and reported by an `ExperimentRunner`. Results are `XResult` objects, labeled N-dimensional arrays with units, so the mean over a scan dimension or the conversion to the units of a dataset is one call.

Around this core the package collects the tasks which come with simulation experiments: fitting parameters to data, exchanging a fit as a PEtab problem, and analysing the sensitivity of a model to its parameters.

## Features

- **[Models](models.md)** — SBML models are loaded into roadrunner with their units, parameter changes and selections; species can be clamped and model sources can be files, URNs or URLs.
- **[Timecourse simulations](simulation.md)** — `Timecourse` and `TimecourseSim`, concatenated periods with changes of parameters and initial conditions, for dosing protocols and perturbations.
- **[Parameter scans](scans.md)** — `ScanSim` runs a simulation over the dimensions of parameter changes, the result is an N-dimensional `XResult`.
- **[Units](units.md)** — the units of the model are read from the SBML and all changes and results carry [pint](https://pint.readthedocs.io) quantities, so values are converted instead of assumed.
- **[Simulation experiments](experiments.md)** — `SimulationExperiment` and `ExperimentRunner`, the reproducible description of an experiment with models, datasets, simulations, tasks, data and figures.
- **[Data](data.md)** — `Data` references simulation results and experimental datasets, with functions computed from them.
- **[Plots and reports](plotting.md)** — figures described independent of the backend and rendered with matplotlib, HTML and markdown reports of experiments.
- **[Parameter fitting](fitting.md)** — `FitParameter`, `FitMapping` and `OptimizationProblem` with local and global optimizers, analysis of the results and PEtab archives.
- **[Sensitivity analysis](sensitivity.md)** — local sensitivities by finite differences and the global Morris, Sobol and FAST methods of [SALib](https://salib.readthedocs.io), with classification and plots.

The standards and methods behind the package are cited in [References](references.md).

## Quickstart

A model is simulated with a `TimecourseSim`, the result is an `XResult`:

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial

simulator = SimulatorSerial(model=REPRESSILATOR_SBML)
simulation = TimecourseSim(
    [
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10}),
    ]
)
xres = simulator.run_timecourse(simulation)
print(xres["X"])
```

Continue with [Installation](installation.md) and the [timecourse simulation guide](simulation.md).

## How to cite

[![DOI](https://zenodo.org/badge/55952847.svg)](https://zenodo.org/badge/latestdoi/55952847)

If you use `sbmlsim` please cite the archived software on [Zenodo](https://zenodo.org/badge/latestdoi/55952847):

> König, M. (2026). *sbmlsim: SBML simulation made easy* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.597149

## License

- Source Code: [MIT](https://opensource.org/license/MIT)
- Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

## Funding

Matthias König is supported by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 "QuaLiPerF (Quantifying Liver Perfusion-Function Relationship in Complex Resection — A Systems Medicine Approach)" by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA).

Matthias König was supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054). Matthias König has received funding from the EOSCsecretariat.eu which has received funding from the European Union's Horizon Programme call H2020-INFRAEOSC-05-2018-2019, grant Agreement number 831644.
