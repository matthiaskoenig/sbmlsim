# Installation

`sbmlsim` requires python >= 3.13 and is available from [pypi](https://pypi.python.org/pypi/sbmlsim). It is tested on Linux, macOS and Windows. The simulations run on [libroadrunner](https://libroadrunner.org), which ships binary wheels for all three platforms, so no compiler is needed. On python 3.15 h5py, which reads and writes the netCDF files of the results, publishes no wheels yet: pip builds it from source, which needs a C compiler and HDF5 (e.g. `libhdf5-dev` on Debian and Ubuntu, `brew install hdf5` on macOS).

## With uv

[uv](https://docs.astral.sh/uv/) is the recommended way to install the package. In a project it is added as a dependency, which resolves and locks it together with the rest of the environment:

```bash
uv add sbmlsim
```

Into an existing virtual environment it is installed through the pip interface of uv:

```bash
uv venv --python 3.14
uv pip install sbmlsim
```

## With pip

```bash
pip install sbmlsim
```

## Development version

The current state of the `develop` branch is installed directly from GitHub:

```bash
uv add "sbmlsim @ git+https://github.com/matthiaskoenig/sbmlsim.git@develop"
```

or, with pip,

```bash
pip install git+https://github.com/matthiaskoenig/sbmlsim.git@develop
```

To work on the repository itself, with the test and documentation tooling, see [Development](development.md).

## Dependencies

`sbmlsim` builds on the packages of the COMBINE ecosystem and the scientific python stack. They are installed with it:

| package | used for |
| --- | --- |
| [libroadrunner](https://libroadrunner.org) | simulation of the SBML models |
| [sbmlutils](https://github.com/matthiaskoenig/sbmlutils), [python-libsbml](https://sbml.org/software/libsbml/) | reading, validating and changing SBML models |
| [pymetadata](https://github.com/matthiaskoenig/pymetadata) | COMBINE archives of the PEtab problems |
| [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [xarray](https://xarray.dev), [scipy](https://scipy.org), [sympy](https://www.sympy.org) | numerics, data and results |
| [h5netcdf](https://h5netcdf.org), [h5py](https://www.h5py.org) | the netCDF files of the results |
| [sbmlmath](https://github.com/dweindl/sbmlmath) | the math of SBML as sympy expressions |
| [pint](https://pint.readthedocs.io), [pydantic](https://docs.pydantic.dev) | units and unit conversions, validated data |
| [petab](https://petab.readthedocs.io), [pyyaml](https://pyyaml.org), [SALib](https://salib.readthedocs.io) | parameter fitting problems, the files of the PEtab test suites and global sensitivity analysis |
| [matplotlib](https://matplotlib.org), [seaborn](https://seaborn.pydata.org), [jinja2](https://jinja.palletsprojects.com), [rich](https://rich.readthedocs.io) | plots, reports and the output of scripts |
| [certifi](https://github.com/certifi/python-certifi) | the CA bundle of requests, with which sbmlutils and pymetadata reach the network, at least 2026.10.19, which removes the emSign root of the compromised intermediate keys ([GHSA-6q48-6c7w-9gw2](https://github.com/advisories/GHSA-6q48-6c7w-9gw2)) |

## Logging

`sbmlsim` does not configure logging. It logs to loggers below the `sbmlsim` logger and leaves handlers, levels and formatting to the application, so the messages of the package stay under your control:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("sbmlsim").setLevel(logging.WARNING)
```

For scripts and interactive work the rich output of the package can be turned on explicitly:

```python
from sbmlsim import log

log.enable_rich_logging()
```
