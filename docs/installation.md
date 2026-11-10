# Installation

`sbmlsim` needs python 3.14 or newer and runs on Linux, macOS and Windows. It is on [PyPI](https://pypi.org/project/sbmlsim/), its simulator [libroadrunner](https://libroadrunner.org) comes from the builds of the `develop` branch of libroadrunner: `sbmlsim` relies on fixes of libroadrunner which are not released yet, so it depends on `libroadrunner>=2.11.1.dev655`, a development version which is not on PyPI. The builds are served by the package index [roadrunner-wheels](https://github.com/matthiaskoenig/roadrunner-wheels), which every install names as an additional index:

```bash
pip install --extra-index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/ sbmlsim
```

Without the index the install stops with `No matching distribution found for libroadrunner>=2.11.1.dev655`. The index serves the wheels of libroadrunner (and of rrplugins) only, every other package comes from PyPI. Once libroadrunner is released with these fixes, the release on PyPI satisfies the requirement as well and the index is no longer needed.

libroadrunner comes as a binary wheel for all three platforms, so you need no compiler. The one exception is python 3.15: h5py, which writes the result files, has no wheels for it yet, so pip builds it from source. That needs a C compiler and the HDF5 library (`libhdf5-dev` on Debian and Ubuntu, `brew install hdf5` on macOS).

On python 3.15, libroadrunner 2.11 takes the symbols of the GDB JIT interface from the interpreter, which not every python 3.15 exports: with the python 3.15 of Ubuntu, for example, `import roadrunner` fails with `undefined symbol: __jit_debug_register_code`. The python of uv exports them, so use it for python 3.15, e.g. `uv venv --python 3.15 --managed-python`. Python 3.14 works with any interpreter.

## With uv

With [uv](https://docs.astral.sh/uv/), add the index to the `pyproject.toml` of your project and take libroadrunner from it:

```toml
[tool.uv.sources]
libroadrunner = { index = "roadrunner-dev" }

[[tool.uv.index]]
name = "roadrunner-dev"
url = "https://matthiaskoenig.github.io/roadrunner-wheels/simple/"
explicit = true
```

and add `sbmlsim` together with `libroadrunner`, which locks them with the rest of your environment:

```bash
uv add sbmlsim libroadrunner
```

uv applies a source to the direct dependencies of a project only, so `libroadrunner` is one of them; the index is `explicit`, uv asks it for libroadrunner only and every other package comes from PyPI. A project which already depends on `sbmlsim` adds the lines and runs `uv add libroadrunner`; `uv lock --upgrade-package libroadrunner` moves the lock to the latest build. The sources are not part of a wheel built from the project: a package which depends on `sbmlsim` is installed with the index as well.

To install into a virtual environment instead:

```bash
uv venv --python 3.14
uv pip install --extra-index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/ sbmlsim
```

The index keeps every build: a lock or a pinned version installs the same build later.

## Check the installation

Import the package and run a short simulation of one of the models that ship with it:

```python
import sbmlsim
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator

print(sbmlsim.__version__)
res = Simulator().run(REPRESSILATOR_SBML, Simulation(end=100))
print(res)
```

```text title="Output"
0.21.0
ScanResult(_point: 1084; PX, PY, PZ, X, Y, Z, eff, n, ... (41 variables))
```

Now continue with the [Tutorial](tutorial.md).

## Optional packages

Five features need packages which are not installed with `sbmlsim`; install them into the environment of `sbmlsim`, with the index of the builds of libroadrunner as above (`--extra-index-url`, or a uv project with the index):

| feature | install |
| --- | --- |
| interactive HTML figures of experiments (`figure_formats=["html"]`, see [Plots and reports](plotting.md)) | `pip install plotly` |
| hybrid models with neural networks ([PEtab SciML](petab_sciml.md)) | `pip install "sbmlsim[sciml]"` |
| COPASI as a second simulation engine (`Simulator(engine="copasi")`, see [COPASI](simulation.md#copasi)) | `pip install "sbmlsim[copasi]"` |
| AMICI as a third simulation engine (`Simulator(engine="amici")`, see [AMICI](simulation.md#amici)) | `pip install "sbmlsim[amici]"` |
| the analyses as task types of [simrun](https://github.com/matthiaskoenig/simrun), which runs them on a cluster (see [simrun](run.md#simrun)) | `pip install "sbmlsim[simrun]"` |

AMICI has no wheels and is built from source when it is installed. It needs a C++ compiler, a BLAS library with its `cblas` header and the headers of HDF5 from the system, e.g. on Debian or Ubuntu:

```bash
sudo apt install build-essential libopenblas-dev libhdf5-dev
```

CMake and SWIG come from PyPI with the build of AMICI, and the extra installs `ninja`, with which AMICI compiles a model.

## Development version

Install the current state of the `develop` branch from GitHub, with the index of the builds of libroadrunner:

```bash
pip install --extra-index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/ git+https://github.com/matthiaskoenig/sbmlsim.git@develop
```

or with uv, in a project with the index of [With uv](#with-uv):

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
| [h5netcdf](https://h5netcdf.org), [h5py](https://www.h5py.org), [hdf5plugin](https://github.com/silx-kit/hdf5plugin) | the netCDF files of the results, compressed with zstd |
| [sbmlmath](https://github.com/dweindl/sbmlmath) | the math of SBML as sympy expressions |
| [pint](https://pint.readthedocs.io), [pydantic](https://docs.pydantic.dev) | units and validated data |
| [petab](https://petab.readthedocs.io), [pyyaml](https://pyyaml.org), [SALib](https://salib.readthedocs.io) | PEtab problems and global sensitivity analysis |
| [pkpdutils](https://github.com/matthiaskoenig/pkpdutils) | non-compartmental pharmacokinetic analysis |
| [matplotlib](https://matplotlib.org), [seaborn](https://seaborn.pydata.org), [jinja2](https://jinja.palletsprojects.com), [rich](https://rich.readthedocs.io) | figures, reports and console output |
| [certifi](https://github.com/certifi/python-certifi) | the certificates with which sbmlutils and pymetadata reach the network |

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
