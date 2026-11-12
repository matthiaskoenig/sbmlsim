# Installation

`sbmlsim` needs python 3.14 or newer and runs on Linux, macOS and Windows. It is on [PyPI](https://pypi.org/project/sbmlsim/), its simulator [libroadrunner](https://libroadrunner.org) is not: `sbmlsim` relies on fixes of libroadrunner which are not released yet, so it needs a build of the `develop` branch of libroadrunner, `libroadrunner>=2.11.1.dev655`, which the package index [roadrunner-wheels](https://github.com/matthiaskoenig/roadrunner-wheels) serves. Install the build from that index first and then `sbmlsim` from PyPI, whose requirement the installed build satisfies:

```bash
pip install --no-deps --index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/ "libroadrunner>=2.11.1.dev655"
pip install sbmlsim
```

The index serves the wheels of libroadrunner (and of rrplugins) only, so the first command takes nothing else from it (`--no-deps`, the numpy of libroadrunner comes from PyPI with `sbmlsim`) and every other package comes from PyPI with the second one. `pip install sbmlsim` alone stops with `No matching distribution found for libroadrunner>=2.11.1.dev655`; a single `pip install --extra-index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/ sbmlsim` installs the build as well, it asks the index for every package of the install. Once libroadrunner is released with these fixes, the release on PyPI satisfies the requirement as well and the index is no longer needed.

`import sbmlsim` checks the installed libroadrunner against this requirement and raises an `ImportError` with these commands when another one is installed, e.g. the release of PyPI: such a libroadrunner loads the models and simulates them with other rows at the events and the changes of a simulation, so a wrong install shows up as wrong results and not as an error. `SBMLSIM_SKIP_ROADRUNNER_CHECK=1` switches the check off, e.g. for a libroadrunner built from source.

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

To install into a virtual environment instead, install libroadrunner from the index first and then `sbmlsim`, whose requirement the installed build satisfies:

```bash
uv venv --python 3.14
uv pip install --no-deps --index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/ "libroadrunner>=2.11.1.dev655"
uv pip install sbmlsim
```

`uv pip install --extra-index-url ... sbmlsim` asks the index for every package of the install, and GitHub Pages, which serves it, answers a request for a package it does not have with a `503` now and then, which fails the install; pip skips such an answer, and a uv project with the explicit index asks it for libroadrunner only.

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

Five features need packages which are not installed with `sbmlsim`; install them into the environment of `sbmlsim`, where the installed build of libroadrunner satisfies the requirement, so the index is needed only when the install upgrades libroadrunner (`--extra-index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/`, or a uv project with the index as in [With uv](#with-uv)):

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

Install the current state of the `develop` branch from GitHub, the build of libroadrunner from the index of the builds first:

```bash
pip install --no-deps --index-url https://matthiaskoenig.github.io/roadrunner-wheels/simple/ "libroadrunner>=2.11.1.dev655"
pip install git+https://github.com/matthiaskoenig/sbmlsim.git@develop
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
| [packaging](https://packaging.pypa.io) | the check of the installed libroadrunner at the import |
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
