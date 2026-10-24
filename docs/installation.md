# Installation

`sbmlsim` needs python 3.13 or newer and runs on Linux, macOS and Windows. Install it from [PyPI](https://pypi.org/project/sbmlsim/) with pip:

```bash
pip install sbmlsim
```

The simulator [libroadrunner](https://libroadrunner.org) comes as a binary wheel for all three platforms, so you need no compiler. The one exception is python 3.15: h5py, which writes the result files, has no wheels for it yet, so pip builds it from source. That needs a C compiler and the HDF5 library (`libhdf5-dev` on Debian and Ubuntu, `brew install hdf5` on macOS).

## With uv

With [uv](https://docs.astral.sh/uv/), add `sbmlsim` to your project, which locks it together with the rest of your environment:

```bash
uv add sbmlsim
```

or install it into a virtual environment:

```bash
uv venv --python 3.14
uv pip install sbmlsim
```

## Check the installation

Import the package and run a short simulation of one of the models that ship with it:

```python
import sbmlsim
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator

print(sbmlsim.__version__)
res = Simulator().run(REPRESSILATOR_SBML, Simulation(end=100, steps=10))
print(res)
```

```text title="Output"
0.13.0
ScanResult(time: 11; PX, PY, PZ, X, Y, Z, eff, n, ... (41 variables))
```

Now continue with the [Tutorial](tutorial.md).

## Optional packages

Two features need packages which are not installed with `sbmlsim`:

| feature | install |
| --- | --- |
| interactive HTML figures of experiments (`figure_formats=["html"]`, see [Plots and reports](plotting.md)) | `pip install plotly` |
| hybrid models with neural networks ([PEtab SciML](petab_sciml.md)) | `pip install "sbmlsim[sciml]"` |

## Development version

Install the current state of the `develop` branch from GitHub:

```bash
pip install git+https://github.com/matthiaskoenig/sbmlsim.git@develop
```

or with uv:

```bash
uv add "sbmlsim @ git+https://github.com/matthiaskoenig/sbmlsim.git@develop"
```

To work on `sbmlsim` itself, with the tests and the documentation, see [Development](development.md).

## Dependencies

pip installs these packages together with `sbmlsim`:

| package | used for |
| --- | --- |
| [libroadrunner](https://libroadrunner.org) | simulating the SBML models |
| [sbmlutils](https://github.com/matthiaskoenig/sbmlutils), [python-libsbml](https://sbml.org/software/libsbml/) | reading, validating and changing SBML models |
| [pymetadata](https://github.com/matthiaskoenig/pymetadata) | COMBINE archives of PEtab problems |
| [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [xarray](https://xarray.dev), [scipy](https://scipy.org), [sympy](https://www.sympy.org) | numerics, data and results |
| [h5netcdf](https://h5netcdf.org), [h5py](https://www.h5py.org) | the netCDF files of the results |
| [sbmlmath](https://github.com/dweindl/sbmlmath) | the math of SBML as sympy expressions |
| [pint](https://pint.readthedocs.io), [pydantic](https://docs.pydantic.dev) | units and validated data |
| [petab](https://petab.readthedocs.io), [pyyaml](https://pyyaml.org), [SALib](https://salib.readthedocs.io) | PEtab problems and global sensitivity analysis |
| [pkpdutils](https://github.com/matthiaskoenig/pkpdutils) | non-compartmental pharmacokinetic analysis |
| [matplotlib](https://matplotlib.org), [seaborn](https://seaborn.pydata.org), [jinja2](https://jinja.palletsprojects.com), [rich](https://rich.readthedocs.io) | figures, reports and console output |
| [certifi](https://github.com/certifi/python-certifi) | the certificates with which sbmlutils and pymetadata reach the network; at least 2026.10.19, which fixes [GHSA-6q48-6c7w-9gw2](https://github.com/advisories/GHSA-6q48-6c7w-9gw2) |

## Logging

`sbmlsim` logs its messages with the `logging` module of the standard library, under the logger `sbmlsim`, and never configures logging itself. Your application decides what is shown:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("sbmlsim").setLevel(logging.WARNING)
```

In scripts and interactive sessions you can switch on the colored output of [rich](https://rich.readthedocs.io):

```python
from sbmlsim import log

log.enable_rich_logging()
```
