![sbmlsim logo](https://github.com/matthiaskoenig/sbmlsim/raw/develop/docs/images/favicon/sbmlsim-100x100-300dpi.png)

# sbmlsim: SBML simulation made easy
[![GitHub Actions CI/CD Status](https://github.com/matthiaskoenig/sbmlsim/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/matthiaskoenig/sbmlsim/actions/workflows/ci-cd.yml)
[![Documentation](https://img.shields.io/badge/docs-sbmlsim-3f51b5.svg)](https://matthiaskoenig.github.io/sbmlsim)
[![Version](https://img.shields.io/pypi/v/sbmlsim.svg)](https://pypi.org/project/sbmlsim/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sbmlsim.svg)](https://pypi.org/project/sbmlsim/)
[![MIT License](https://img.shields.io/pypi/l/sbmlsim.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/55952847.svg)](https://zenodo.org/badge/latestdoi/55952847)

`sbmlsim` is a collection of python utilities for the simulation of models in the [Systems Biology Markup Language](https://sbml.org) (SBML), built on [libroadrunner](https://libroadrunner.org).

Features include

- **timecourse simulations** — concatenated timecourses with changes of parameters and initial conditions, in the units of the model
- **parameter scans** — simulations over the dimensions of parameter changes, with results as labeled N-dimensional arrays ([xarray](https://xarray.dev))
- **simulation experiments** — models, datasets, simulations, tasks and figures of an experiment as one reproducible python object, with HTML and markdown reports
- **parameter fitting** — optimization problems from experimental data with local and global optimizers, and PEtab archives
- **sensitivity analysis** — local sensitivities and the global Morris, Sobol and FAST methods
- **SED-ML and COMBINE archives** — execution of simulation experiments described in SED-ML

The documentation is available at [https://matthiaskoenig.github.io/sbmlsim](https://matthiaskoenig.github.io/sbmlsim).

If you have any questions or issues please [open an issue](https://github.com/matthiaskoenig/sbmlsim/issues).

## How to cite
[![DOI](https://zenodo.org/badge/55952847.svg)](https://zenodo.org/badge/latestdoi/55952847)

If you use `sbmlsim` please cite the archived software on [Zenodo](https://zenodo.org/badge/latestdoi/55952847):

> König, M. (2026). *sbmlsim: SBML simulation made easy* (Version 0.6.0) [Computer software]. Zenodo. https://zenodo.org/badge/latestdoi/55952847

```bibtex
@software{konig_sbmlsim,
  author    = {König, Matthias},
  title     = {sbmlsim: SBML simulation made easy},
  year      = {2026},
  month     = sep,
  version   = {0.6.0},
  publisher = {Zenodo},
  url       = {https://zenodo.org/badge/latestdoi/55952847},
}
```

## License
- Source Code: [MIT](https://opensource.org/license/MIT)
- Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

## Funding
Matthias König is supported by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 "QuaLiPerF (Quantifying Liver Perfusion-Function Relationship in Complex Resection - A Systems Medicine Approach)" by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA).

Matthias König was supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (LiSyM, grant number 031L0054). Matthias König has received funding from the EOSCsecretariat.eu which has received funding from the European Union's Horizon Programme call H2020-INFRAEOSC-05-2018-2019, grant Agreement number 831644.

© 2019-2026 Matthias König, [https://livermetabolism.com](https://livermetabolism.com)
