# PEtab SciML phase 1: networks and their forward pass implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A network of the PEtab SciML NN YAML format is read into `sbmlsim.sciml.Network`, evaluated with numpy without torch, and the cases `ml_model_import` 001 to 054 and `initialization` 001 to 003 of the PEtab SciML test suite pass or are listed in a baseline with their reason.

**Architecture:** The new package `sbmlsim.sciml` knows nothing of PEtab. One interpreter (`interpreter.evaluate`) walks the nodes of the forward pass, every layer and function is implemented once against a `Backend` and registered under its PyTorch name together with the backends it supports. Phase 1 ships the `NumpyBackend`; the `SympyBackend` of phase 3 is a subclass of `Backend` and needs no change of a layer. The download and the cache of a test suite move out of `SemanticSuite` into `sbmlsim.testsuite.cache`, which the new `sbmlsim.sciml.testsuite` shares.

**Tech Stack:** python 3.13+, numpy, scipy (`scipy.special.erf`), `petab-sciml` 0.0.3 (`NNModel`, `ArrayData`, brings `h5py`), `petab` 0.9.0 (`ProblemConfig`, `ParameterTable`, `MappingTable`), `torch` (CPU build, tests only), pytest with pytest-xdist, tox with tox-uv, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-09-30-petab-sciml-design.md` (phase 1 of the table "Phases")

## Global Constraints

- python >= 3.13. The repository environment (`.venv`) is python 3.14.
- Every module, class and function in `src/` carries full type annotations and a google style docstring (ruff `D`). `tests/` and `scripts/` are exempt from the docstring rules only where `.ruff.toml` says so (`tests/**`), `scripts/` is not exempt.
- `uv run ruff check`, `uv run ruff format` and `uvx ty check` stay at zero diagnostics. ty runs with `error-on-warning = true`. Suppress only with a rule specific `# ty: ignore[rule-name]`, never a blanket `# type: ignore`.
- Library code logs with `logging.getLogger(__name__)` and lazy `%s` formatting (ruff `G`), it never prints. `scripts/` print with `sbmlsim.console.console`.
- Never use the em dash character in any file. Use the plain dash "-" instead.
- No attribution of agents anywhere: no `Co-Authored-By` lines, no "Generated with" lines in commits, code, comments or documentation.
- Never edit `CHANGELOG.md` or a file which is marked as generated.
- Markdown carries no hard line wraps: a paragraph, a list item or a table row is a single line.
- Tests run with `uv run pytest -q -x <path>`.
- Commits are made on the current branch (`petab-sciml-phase1`). Nothing is pushed.
- The extra is exactly `sciml = ["petab-sciml>=0.0.3"]`. `torch` is only in the `dev` extra. No module below `src/` imports `torch`.
- `sbmlsim.sciml` imports `petab_sciml` at the top of its modules. No other module of the package imports `sbmlsim.sciml` at import time.
- The tests which need no download run in every test session: the forward pass against torch for every layer and function, with random parameters. The tests of the test suite carry the marker `sciml_testsuite`, which `addopts` deselects like the marker `testsuite`.
- The arrays of `sbmlsim` are in the PyTorch layout, row major, in double precision. Networks are evaluated in evaluation mode, training mode is not supported.
- The test suite is pinned by the commit `0622bbfc5e12eb9b482659eabd1756ca0e87dfc8`.
- Every code block of this plan was run against this repository state: the tests pass, and ruff and ty are at zero diagnostics after every task. Transcribe the blocks as they are. If a block does not pass, stop and report, do not improvise a different design.

## Review Focus

- **An array file which does not fit the architecture** (an array which is missing or empty, has the wrong shape, holds `nan` or `inf`, belongs to another network or to a layer the network does not have): the import raises `NetworkImportError` which names the network, the layer and the array, and never evaluates with values it made up. Pinned in task 4 by `test_arrays_which_do_not_fit`, `test_an_empty_array_is_an_array_without_values`, `test_an_array_file_of_another_network` and in task 8 by `test_an_array_without_values_is_an_error`.
- **An input whose shape the layer cannot take** (no batch axis for `BatchNorm`, the wrong number of axes, a kernel which is larger than the input, arrays of `cat` which do not fit): a `ValueError` which names the network and the node, not an empty array and not a bare numpy message. Pinned in task 2 by `test_an_input_of_the_wrong_size`, in task 5 by `test_a_kernel_larger_than_the_input` and `test_an_input_with_the_wrong_number_of_axes`, in task 6 by `test_a_window_larger_than_the_input`, in task 7 by `test_a_batch_norm_needs_a_batch`.
- **Inputs of a large magnitude** (`|x| = 800`, a constant input of a normalization layer): every activation returns a finite value without a numpy warning, a variance of zero does not divide by zero. Pinned in task 3 by `VALUES` in `test_function` and by `test_the_values_are_finite`, in task 7 by `test_a_constant_input_is_normalized_to_zero`.
- **Layers of nested modules** (a layer id such as `block.0`): the id of an element stays a valid SBML `SId`, a key `net1.block.0.weight` resolves to the layer `block.0`, and two elements which would get one id are an error. Pinned in task 4 by `test_an_id_is_an_sid` and `test_two_elements_with_one_id`, in task 8 by `test_the_arrays_an_entry_covers`.
- **A download which fails or is not the suite** (no network, a HTML page instead of a zip archive, an archive without cases): an `OSError` or `BadZipFile`, and neither the cache directory nor the staging directory exist afterwards, so the next run does not find a half filled cache. Pinned in task 9 by `test_a_failed_download_leaves_nothing`, `test_a_file_which_is_not_an_archive`, `test_an_archive_without_cases` and in task 10 by `test_an_archive_which_is_not_the_suite`.

---

## Findings of the research

These were found by running the reference values of the test suite and the APIs. Where they differ from the spec, the plan follows the finding and the difference is stated.

| # | finding | what the plan does |
| --- | --- | --- |
| 1 | The `solutions.yaml` of `ml_model_import` has no `tol` (the spec names `tol`). The suite checks its own reference values with `atol=1e-3`, and `atol=1e-2` for the cases with dropout (`pysrc/ml_import_helper.py`) | `MODEL_IMPORT_TOLERANCE = 1e-3`, `DROPOUT_TOLERANCE = 1e-2` for a case with a `dropout` field. `initialization` states `tol: 0.001` and uses it |
| 2 | The reference values of `BatchNorm` (042 to 045) are calculated in training mode, i.e. with the statistics of the batch, and the array files hold `weight` and `bias` only, no `running_mean` and `running_var`. PyTorch in evaluation mode with its initial statistics (0 and 1) differs by up to 3.4 | A normalization layer uses its stored statistics when the arrays hold them (the spec), and calculates them from its input when they hold none, which is what PyTorch does for `track_running_stats=False`. 042 to 045 pass |
| 3 | The reference values of the dropout cases (019 to 022) are the mean of 40000 forward passes in training mode. For `Dropout`, `Dropout1d` and `Dropout2d` in front of a linear layer the mean is the evaluation mode, the identity passes within `1e-2`. For `AlphaDropout` (020) the mean is not the input, the difference is 0.5 | 020 is listed in the baseline with this reason, status `tolerance` |
| 4 | `petab.v2.Problem.from_yaml` needs `torch` for a SciML problem: `SciMLExt.from_config` calls `NNModel.to_pytorch_module()` and fails with `NameError: name 'nn' is not defined` without it. The spec assumes that `petab` reads the problem and `torch` is a test dependency | Phase 1 reads an `initialization` problem with `ProblemConfig`, `ParameterTable` and `MappingTable`, which need no `torch`. The reader of phase 3 has to do the same |
| 5 | `NNModel.to_pytorch_module()` fails for `log_sigmoid` (case 030), PyTorch names the function `logsigmoid` | The tests compare with the layers and functions of `torch` directly. `log_sigmoid` and `logsigmoid` are both registered |
| 6 | The layout of an array file with `metadata/pytorch_format` false is not defined by the standard | It is read as the column major layout, i.e. the axes are reversed, which is the permutation the Julia code of the suite applies. The shape is checked against the layer afterwards |
| 7 | `Layers` and `ActivationFunctions` of `petab_sciml.constants` do not list the normalization layers and `log_sigmoid`, which the suite uses | The registry is keyed by the PyTorch names as strings, not by the enums |
| 8 | `FitParameter(unit=None)` logs a warning, which a network would repeat for every element | The fit parameters of a network carry the unit `dimensionless` |
| 9 | `layers.py` of the spec would hold 1500 lines | It is the package `sbmlsim/sciml/layers/` with one module per family, the import path `sbmlsim.sciml.layers` is the one of the spec. The interpreter is `sbmlsim/sciml/interpreter.py`, the errors are `sbmlsim/sciml/errors.py` |
| 10 | `torch` 2.14.0 installs for python 3.13 and 3.14. The default build for linux brings 2 GB of CUDA libraries | `[tool.uv.sources]` takes the CPU build for linux, tox installs it from the CPU index in `commands_pre` |
| 11 | `gelu` of the suite (029) differs from the error function by `3e-4`, i.e. the reference is an approximation | `gelu` is exact by default (`approximate="none"`) as in PyTorch, 029 passes within `1e-3` |

Not part of phase 1, although the section "Dependencies" of the spec names it: `sbmlmath` as a declared dependency. Phase 3 adds it together with the compiler which uses it.

## File structure

| file | responsibility | task |
| --- | --- | --- |
| `pyproject.toml`, `tox.ini` | the extra `sciml`, `torch` in `dev`, the marker `sciml_testsuite`, the tox environment `sciml` | 1, 10 |
| `src/sbmlsim/sciml/__init__.py` | the import guard which names the extra, the exports | 1, 4 |
| `src/sbmlsim/sciml/errors.py` | `NetworkImportError`, `UnsupportedLayerError` | 1 |
| `src/sbmlsim/sciml/backend.py` | `BackendKind`, `Backend`, `NumpyBackend`, `ALL_BACKENDS`, `NUMPY_ONLY` | 1 |
| `src/sbmlsim/sciml/layers/registry.py` | `LAYERS`, `FUNCTIONS`, `ArraySpec`, the decorators `layer`, `layer_nd`, `function` | 2 |
| `src/sbmlsim/sciml/layers/core.py` | `Linear`, `Bilinear`, `Flatten`, the dropout layers | 2 |
| `src/sbmlsim/sciml/interpreter.py` | `evaluate`, the one interpreter of the forward pass | 2 |
| `src/sbmlsim/sciml/layers/functions.py` | the activation functions, `flatten`, `cat` | 3 |
| `src/sbmlsim/sciml/network.py` | `Network`, `NetworkParameters`, `element_id`, `load_array_data` | 4 |
| `src/sbmlsim/sciml/layers/windows.py` | the sliding windows the convolution and the pooling share | 5 |
| `src/sbmlsim/sciml/layers/convolution.py` | `Conv1-3d`, `ConvTranspose1-3d` | 5 |
| `src/sbmlsim/sciml/layers/pooling.py` | `MaxPool`, `AvgPool`, `LPPool`, `AdaptiveMaxPool`, `AdaptiveAvgPool`, each `1-3d` | 6 |
| `src/sbmlsim/sciml/layers/normalization.py` | `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm` | 7 |
| `src/sbmlsim/sciml/parameters.py` | `covered_arrays`, `resolve_entries`, `nominal_parameters`, `network_fit_parameters` | 8 |
| `src/sbmlsim/testsuite/cache.py` | `cache_root`, `cache_path`, `fetch` | 9 |
| `src/sbmlsim/testsuite/cases.py` | `SemanticSuite` uses the cache and keeps its interface | 9 |
| `src/sbmlsim/sciml/testsuite.py` | `SCIML_SUITE_COMMIT`, `SciMLSuite`, `ModelImportCase`, `InitializationCase`, `ProblemImportCase` | 10 |
| `scripts/sciml_testsuite.py` | the command line `download`, `run`, `baseline` | 10 |
| `tests/data/sciml_baseline.json` | the cases which do not pass, with their reason | 10 |
| `tests/sciml/` | the tests, `conftest.py` holds the comparison with PyTorch | 1 to 10 |

The interfaces the tasks share, in one place:

```python
# backend.py
class BackendKind(StrEnum): NUMPY = "numpy"; SYMPY = "sympy"
ALL_BACKENDS: frozenset[BackendKind]; NUMPY_ONLY: frozenset[BackendKind]
class Backend(ABC):
    kind: ClassVar[BackendKind]; dtype: ClassVar[type]
    def asarray(self, values: Any) -> np.ndarray
    def exp / log / tanh / sqrt / erf / absolute (self, x: np.ndarray) -> np.ndarray
    def select(self, x, threshold: float, below, above) -> np.ndarray   # above where x > threshold
    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float
class NumpyBackend(Backend)

# layers/registry.py
@dataclass(frozen=True) class ArraySpec: shape: tuple[int, ...]; required: bool = True; trainable: bool = True
@dataclass(frozen=True) class LayerType: name; forward; arrays; backends
@dataclass(frozen=True) class FunctionType: name; function; backends
LAYERS: dict[str, LayerType]; FUNCTIONS: dict[str, FunctionType]
def layer(*names, arrays=no_arrays, backends=ALL_BACKENDS)       # forward(backend, args, arrays, *inputs)
def layer_nd(template, arrays=None, backends=ALL_BACKENDS)      # forward(n, backend, args, arrays, *inputs), arrays(n, args)
def function(*names, backends=ALL_BACKENDS)                     # implementation(backend, *inputs, **kwargs)
def as_tuple(value: Any, n: int) -> tuple[int, ...]

# interpreter.py
def evaluate(model: NNModel, parameters: Mapping[str, Mapping[str, np.ndarray]], inputs: Sequence[ArrayLike], backend: Backend, on_node: NodeHook | None = None) -> tuple[np.ndarray, ...]

# network.py
NetworkParameters = dict[str, dict[str, np.ndarray]]
def element_id(network: str, layer: str, array: str, index: tuple[int, ...]) -> str
def copy_parameters(parameters) -> NetworkParameters
def load_array_data(path: Path) -> ArrayData
@dataclass class Network:
    sid: str; model: NNModel; parameters: NetworkParameters
    from_files(yaml_path, array_path=None, sid=None); read_arrays(array_path); array_specs(); used_layers(); backends()
    check_arrays(parameters, complete=True); forward(*inputs, parameters=None); parameter_ids(); with_values(values)

# parameters.py
def covered_arrays(network: Network, key: str) -> tuple[int, list[tuple[str, str]]]
def resolve_entries[T](network: Network, entries: Mapping[str, T]) -> dict[tuple[str, str], T]
def nominal_parameters(network: Network, values: Mapping[str, float] | None = None) -> NetworkParameters
def network_fit_parameters(network, estimate, bounds, values=None) -> list[FitParameter]

# testsuite/cache.py
def cache_root() -> Path
def cache_path(variable: str, *parts: str) -> Path
def fetch(url: str, path: Path, select: Callable[[Path], Path]) -> Path
```

---

### Task 1: The extra, the errors and the numpy backend

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`, `[tool.pytest.ini_options]`, new `[tool.uv.sources]` and `[[tool.uv.index]]`)
- Modify: `tox.ini` (`[testenv]`, `[testenv:ty]`)
- Create: `src/sbmlsim/sciml/__init__.py`
- Create: `src/sbmlsim/sciml/errors.py`
- Create: `src/sbmlsim/sciml/backend.py`
- Create: `tests/sciml/__init__.py` (empty)
- Test: `tests/sciml/test_package.py`
- Test: `tests/sciml/test_backend.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `sbmlsim.sciml.backend.BackendKind` (`NUMPY`, `SYMPY`), `Backend` (abstract, methods `asarray`, `exp`, `log`, `tanh`, `sqrt`, `erf`, `absolute`, `select(x, threshold, below, above)`, `stabilizer(x, axis)`, class attributes `kind` and `dtype`), `NumpyBackend`, `ALL_BACKENDS`, `NUMPY_ONLY`. `sbmlsim.sciml.errors.NetworkImportError(ValueError)` and `UnsupportedLayerError(NotImplementedError)` with `__init__(network: str, node: str, target: str, reason: str)` and the attributes of the same names. The pytest marker `sciml_testsuite`.

The design constraint of this task: `Backend` is everything which differs between the forward pass on numbers and the compilation into expressions. A layer uses the array operations of numpy (`@`, `reshape`, `sum`, `concatenate`, `tensordot`), which work on `object` arrays of sympy expressions as well, and takes the elementwise functions from the backend. `select` is the only function with a condition: numpy implements it as `np.where`, the sympy backend of phase 3 as a `Piecewise`. `stabilizer` is the shift of `softmax`, the maximum in numpy and zero in sympy. Do not add a method to `Backend` which has no meaning on expressions (a maximum over an axis, a sort), such a layer declares `NUMPY_ONLY` and uses numpy directly.

- [ ] **Step 1: Add the extra and the test dependency**

In `pyproject.toml` replace

```toml
[project.optional-dependencies]
dev = [
	"bump-my-version>=1.5.1",
```

with

```toml
[project.optional-dependencies]
# neural networks of hybrid problems (PEtab SciML), see sbmlsim.sciml
sciml = [
	"petab-sciml>=0.0.3",
]
dev = [
	"sbmlsim[sciml]",
	# the reference of the forward pass in the tests, never imported by the package
	"torch>=2.14.0",
	"bump-my-version>=1.5.1",
```

Replace

```toml
addopts = "-n auto -m 'not testsuite'"
markers = [
    "testsuite: a semantic case of the SBML Test Suite, run before a release",
]
```

with

```toml
addopts = "-n auto -m 'not testsuite and not sciml_testsuite'"
markers = [
    "testsuite: a semantic case of the SBML Test Suite, run before a release",
    "sciml_testsuite: a case of the PEtab SciML test suite, run before a release",
]
```

Append at the end of the file

```toml

[tool.uv.sources]
# the CPU build of torch: the default build for linux brings 2 GB of CUDA
# libraries, which the comparison of a forward pass does not need
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

The comment lines above `addopts` in `pyproject.toml` stay as they are.

- [ ] **Step 2: Install and check the environment**

Run: `uv sync --extra dev`

Run: `uv run python -c "import torch, petab_sciml, h5py; print(torch.__version__)"`
Expected on linux: a version which ends with `+cpu`, e.g. `2.14.0+cpu`. No `nvidia` package is installed: `uv pip list | grep -ci nvidia` prints `0`.

- [ ] **Step 3: Install the extra and torch in the tox environments**

`[tool.uv.sources]` is read by `uv sync`, not by tox, so tox installs the CPU build itself. In `tox.ini` replace the section `[testenv]` with

```ini
[testenv]
package = wheel
wheel_build_env = .pkg
# the networks of `sbmlsim.sciml` need `petab-sciml`
extras = sciml
# the semantic cases of the SBML Test Suite are cached in the home directory,
# and `SBMLSIM_TEST_SUITE_PATH` points at them when they live elsewhere. The
# same holds for the PEtab SciML test suite and `SBMLSIM_SCIML_SUITE_PATH`
passenv =
    HOME
    USERPROFILE
    XDG_CACHE_HOME
    SBMLSIM_TEST_SUITE_PATH
    SBMLSIM_SCIML_SUITE_PATH
deps =
    pytest
    pytest-xdist
# the reference of the forward pass, the CPU build: the default build for
# linux brings 2 GB of CUDA libraries
allowlist_externals = uv
commands_pre =
    uv pip install --python {env_python} --index-url https://download.pytorch.org/whl/cpu torch
commands =
    pytest
```

and in the section `[testenv:ty]` add two lines, `extras = sciml` and an empty `commands_pre =`, so the section reads

```ini
[testenv:ty]
passenv =
    TY_OUTPUT_FORMAT
# ty checks `src`, `tests`, `examples` and `scripts`, so the environment needs
# what they import beyond the runtime dependencies: pytest, and plotly, which is
# in the `dev` extra and which `sbmlsim.plot.serialization_plotly` imports.
# The extra `sciml` brings `petab-sciml` and `h5py`. torch is not needed, the
# tests import it with `pytest.importorskip`
extras = sciml
deps =
    ty
    pytest
    plotly
commands_pre =
commands =
    ty check
```

- [ ] **Step 4: Write the failing tests**

Create the empty file `tests/sciml/__init__.py`.

Create `tests/sciml/test_package.py`:

````python
"""Tests of the package `sbmlsim.sciml` and its optional dependency."""

import re
import subprocess
import sys
from pathlib import Path

import sbmlsim


def _python(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in a new interpreter, the imports of the tests do not count."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_the_missing_extra_is_named() -> None:
    """Without `petab_sciml` the import says how to install it."""
    result = _python(
        "import sys; sys.modules['petab_sciml'] = None; import sbmlsim.sciml"
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert "pip install sbmlsim[sciml]" in result.stderr


def test_the_package_does_not_import_the_networks() -> None:
    """`sbmlsim` works without the extra, nothing imports `sbmlsim.sciml`."""
    result = _python(
        "import sys; sys.modules['petab_sciml'] = None\n"
        "import sbmlsim, sbmlsim.fit, sbmlsim.testsuite, sbmlsim.fit.petab_v2\n"
        "assert 'sbmlsim.sciml' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_the_package_does_not_import_torch() -> None:
    """`torch` is a dependency of the tests, no module of the package names it."""
    pattern = re.compile(r"^\s*(import|from)\s+torch\b", re.MULTILINE)
    modules = sorted(Path(sbmlsim.__file__).parent.rglob("*.py"))
    assert modules
    assert [str(path) for path in modules if pattern.search(path.read_text())] == []
````

Create `tests/sciml/test_backend.py`:

````python
"""Tests of the backend on arrays of numbers."""

import inspect

import numpy as np

from sbmlsim.sciml.backend import (
    ALL_BACKENDS,
    NUMPY_ONLY,
    Backend,
    BackendKind,
    NumpyBackend,
)


def test_the_kinds_of_backends() -> None:
    """A layer declares its backends with these sets."""
    assert {BackendKind.NUMPY, BackendKind.SYMPY} == ALL_BACKENDS
    assert {BackendKind.NUMPY} == NUMPY_ONLY
    assert NumpyBackend.kind == BackendKind.NUMPY
    assert NumpyBackend.dtype is float


def test_a_backend_implements_the_elementwise_functions() -> None:
    """The abstract class names what a backend on expressions implements."""
    assert inspect.isabstract(Backend)
    assert Backend.__abstractmethods__ == {
        "exp",
        "log",
        "tanh",
        "sqrt",
        "erf",
        "absolute",
        "select",
        "stabilizer",
    }
    assert not inspect.isabstract(NumpyBackend)


def test_the_array_of_the_backend() -> None:
    """Values are converted to arrays of numbers in double precision."""
    backend = NumpyBackend()
    for values in ([1, 2], np.array([1, 2]), np.array([1.0, 2.0], dtype="f4")):
        array = backend.asarray(values)
        assert array.dtype == float
        np.testing.assert_array_equal(array, [1.0, 2.0])
    assert backend.asarray(3).shape == ()


def test_the_elementwise_functions() -> None:
    """The functions are the ones of numpy and scipy."""
    backend = NumpyBackend()
    x = np.array([[0.25, 1.0], [4.0, 9.0]])
    np.testing.assert_allclose(backend.exp(x), np.exp(x))
    np.testing.assert_allclose(backend.log(x), np.log(x))
    np.testing.assert_allclose(backend.tanh(x), np.tanh(x))
    np.testing.assert_allclose(backend.sqrt(x), [[0.5, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(backend.absolute(-x), x)
    np.testing.assert_allclose(backend.erf(np.array([0.0, 1.0])), [0.0, 0.8427007929])


def test_the_selection_by_a_condition() -> None:
    """`above` holds for `x > threshold`, the threshold itself is `below`."""
    backend = NumpyBackend()
    x = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(backend.select(x, 0.0, -5.0, 5.0), [-5.0, -5.0, 5.0])
    np.testing.assert_array_equal(backend.select(x, 0.0, 0.0, x), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(backend.select(x, 0.0, 2 * x, x), [-2.0, 0.0, 1.0])


def test_an_overflow_is_not_a_warning() -> None:
    """The branch of a selection which is not chosen may overflow."""
    backend = NumpyBackend()
    with np.errstate(over="raise"):
        assert backend.exp(np.array([800.0]))[0] == np.inf


def test_the_shift_of_softmax() -> None:
    """The maximum along the axis keeps its axis, so it broadcasts."""
    backend = NumpyBackend()
    x = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 4.0]])
    np.testing.assert_array_equal(backend.stabilizer(x, 1), [[5.0], [7.0]])
    np.testing.assert_array_equal(backend.stabilizer(x, 0), [[7.0, 5.0, 4.0]])
````

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'sbmlsim.sciml'`.

- [ ] **Step 6: Write the errors**

Create `src/sbmlsim/sciml/errors.py`:

````python
"""Errors of the neural networks."""

from __future__ import annotations


class NetworkImportError(ValueError):
    """A network or its arrays cannot be read.

    The message names the network and, where it applies, the layer and the
    array.
    """


class UnsupportedLayerError(NotImplementedError):
    """A node of the forward pass has no implementation.

    Attributes:
        network: id of the network.
        node: name of the node of the forward pass.
        target: the layer type, function or method of the node.
        reason: why the node is not evaluated.
    """

    def __init__(self, network: str, node: str, target: str, reason: str) -> None:
        """Initialize the error.

        Args:
            network: id of the network.
            node: name of the node of the forward pass.
            target: the layer type, function or method of the node.
            reason: why the node is not evaluated.
        """
        self.network = network
        self.node = node
        self.target = target
        self.reason = reason
        super().__init__(
            f"Network '{network}', node '{node}': '{target}' is not supported, {reason}"
        )
````

- [ ] **Step 7: Write the backend**

Create `src/sbmlsim/sciml/backend.py`:

````python
"""The backends the layers and functions of a network are written against.

A layer is implemented once. It uses the array operations of numpy (`@`,
`reshape`, `sum`, `concatenate`), which work on arrays of numbers and on
`object` arrays of expressions alike, and takes everything which differs
between the two from a `Backend`: the elementwise functions and the functions
with a condition.

`NumpyBackend` works on `float` arrays and is the forward pass. A backend on
sympy expressions is the first half of the compilation of a network into an
SBML model; it subclasses `Backend`, sets `kind` to `BackendKind.SYMPY` and
implements the abstract methods, the layers do not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
from scipy import special


class BackendKind(StrEnum):
    """The kinds of backends a layer declares its support for."""

    NUMPY = "numpy"
    SYMPY = "sympy"


#: the backends of a layer which only uses the methods of `Backend`
ALL_BACKENDS: frozenset[BackendKind] = frozenset(BackendKind)

#: the backends of a layer which needs numbers, e.g. to find a maximum
NUMPY_ONLY: frozenset[BackendKind] = frozenset({BackendKind.NUMPY})


class Backend(ABC):
    """The elementwise functions and the array type of a forward pass.

    Every method takes and returns arrays of the `dtype` of the backend and
    works elementwise, with the broadcasting of numpy.

    Attributes:
        kind: the kind of the backend, which a layer declares its support for.
        dtype: the dtype of the arrays, `float` or `object`.
    """

    kind: ClassVar[BackendKind]
    dtype: ClassVar[type]

    def asarray(self, values: Any) -> np.ndarray:
        """Convert values to an array of the backend.

        Args:
            values: an array, a nested sequence or a scalar.

        Returns:
            The array with the dtype of the backend.
        """
        return np.asarray(values, dtype=self.dtype)

    @abstractmethod
    def exp(self, x: np.ndarray) -> np.ndarray:
        """Calculate the exponential function."""

    @abstractmethod
    def log(self, x: np.ndarray) -> np.ndarray:
        """Calculate the natural logarithm."""

    @abstractmethod
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hyperbolic tangent."""

    @abstractmethod
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the square root."""

    @abstractmethod
    def erf(self, x: np.ndarray) -> np.ndarray:
        """Calculate the error function."""

    @abstractmethod
    def absolute(self, x: np.ndarray) -> np.ndarray:
        """Calculate the absolute value."""

    @abstractmethod
    def select(
        self,
        x: np.ndarray,
        threshold: float,
        below: np.ndarray | float,
        above: np.ndarray | float,
    ) -> np.ndarray:
        """Choose between two values by a condition on `x`.

        This is the one function with a condition, every piecewise activation
        is written with it.

        Args:
            x: the values the condition is evaluated on.
            threshold: the threshold of the condition.
            below: the result where `x <= threshold`.
            above: the result where `x > threshold`.

        Returns:
            `above` where `x > threshold` and `below` elsewhere.
        """

    @abstractmethod
    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float:
        """Get a shift which keeps the exponentials of `softmax` finite.

        `softmax` does not change when a value which is constant along `axis`
        is subtracted from `x`. A backend on numbers returns the maximum along
        the axis, a backend on expressions returns zero.

        Args:
            x: the input of `softmax`.
            axis: the axis `softmax` normalizes over.

        Returns:
            The shift, which broadcasts against `x`.
        """


class NumpyBackend(Backend):
    """The backend on arrays of numbers, i.e. the forward pass."""

    kind: ClassVar[BackendKind] = BackendKind.NUMPY
    dtype: ClassVar[type] = float

    def exp(self, x: np.ndarray) -> np.ndarray:
        """Calculate the exponential function.

        An overflow is `inf` and not a warning: the branch of a `select` which
        is not chosen is evaluated as well.
        """
        with np.errstate(over="ignore"):
            return np.exp(x)

    def log(self, x: np.ndarray) -> np.ndarray:
        """Calculate the natural logarithm."""
        return np.log(x)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hyperbolic tangent."""
        return np.tanh(x)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the square root."""
        return np.sqrt(x)

    def erf(self, x: np.ndarray) -> np.ndarray:
        """Calculate the error function."""
        return special.erf(x)

    def absolute(self, x: np.ndarray) -> np.ndarray:
        """Calculate the absolute value."""
        return np.abs(x)

    def select(
        self,
        x: np.ndarray,
        threshold: float,
        below: np.ndarray | float,
        above: np.ndarray | float,
    ) -> np.ndarray:
        """Choose between two values by a condition on `x`."""
        return np.where(x > threshold, above, below)

    def stabilizer(self, x: np.ndarray, axis: int) -> np.ndarray | float:
        """Get the maximum along the axis."""
        return np.max(x, axis=axis, keepdims=True)
````

- [ ] **Step 8: Write the package with the import guard**

Create `src/sbmlsim/sciml/__init__.py`:

````python
"""Neural networks of hybrid problems.

A hybrid problem combines a mechanistic model in SBML with neural networks,
which is what [PEtab SciML](https://github.com/PEtab-dev/petab_sciml)
describes. This package is the native half of the support: `Network` is the
architecture and the arrays of one network and `Network.forward` evaluates it
with numpy. The package knows nothing of PEtab, `sbmlsim.fit.petab_v2`
translates.

The architecture is read and written with `petab_sciml`, which is not a
dependency of `sbmlsim` but the `sciml` extra:

```bash
pip install sbmlsim[sciml]
```
"""

try:
    import petab_sciml  # noqa: F401
except ModuleNotFoundError as err:
    raise ImportError(
        "sbmlsim.sciml requires the package 'petab_sciml', which is installed "
        "with the extra 'sciml': pip install sbmlsim[sciml]"
    ) from err

from sbmlsim.sciml.backend import Backend, BackendKind, NumpyBackend
from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError

__all__ = [
    "Backend",
    "BackendKind",
    "NetworkImportError",
    "NumpyBackend",
    "UnsupportedLayerError",
]
````

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `10 passed`.

- [ ] **Step 10: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: `All checks passed!` three times (ruff format prints `... files already formatted`).

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml tox.ini src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the extra, the errors and the numpy backend"
```

---

### Task 2: The registry, the interpreter and `Linear`, `Bilinear`, `Flatten`, dropout

**Files:**
- Create: `src/sbmlsim/sciml/layers/__init__.py`
- Create: `src/sbmlsim/sciml/layers/registry.py`
- Create: `src/sbmlsim/sciml/layers/core.py`
- Create: `src/sbmlsim/sciml/interpreter.py`
- Create: `tests/sciml/conftest.py`
- Test: `tests/sciml/test_interpreter.py`
- Test: `tests/sciml/test_layers_core.py`

**Interfaces:**
- Consumes: `Backend`, `BackendKind`, `NumpyBackend`, `ALL_BACKENDS`, `NUMPY_ONLY` of `sbmlsim.sciml.backend`; `UnsupportedLayerError(network, node, target, reason)` of `sbmlsim.sciml.errors`; `NNModel`, `Node`, `Layer`, `Input` of `petab_sciml` (`NNModel.nn_model_id: str`, `.inputs: list[Input]`, `.layers: list[Layer]` with `layer_id`, `layer_type`, `args: dict | None`, `.forward: list[Node]` with `name`, `op`, `target`, `args: list | None`, `kwargs: dict | None`).
- Produces: `sbmlsim.sciml.layers.LAYERS`, `FUNCTIONS`, `ArraySpec`, `LayerType`, `FunctionType`; `sbmlsim.sciml.layers.registry.layer`, `layer_nd`, `function`, `no_arrays`, `as_tuple`; `sbmlsim.sciml.layers.core.flatten_array(x, start_dim=0, end_dim=-1)`; `sbmlsim.sciml.interpreter.evaluate(model, parameters, inputs, backend, on_node=None) -> tuple[np.ndarray, ...]` and the constants `PLACEHOLDER`, `CALL_MODULE`, `CALL_FUNCTION`, `CALL_METHOD`, `OUTPUT`. The fixtures of `tests/sciml/conftest.py`: `layer_model(layer_type, args, n_inputs=1) -> NNModel` (network `net1`, layer `layer1`), `function_model(target, kwargs, op="call_function") -> NNModel` (network `net1`, node `f`), `forward(model, parameters, *inputs)`, `rng`, `compare_layer(layer_type, args, *shapes)`, `compare_function(target, kwargs, x, torch_name=None, op="call_function")`.

The semantics of the layers, with the argument names and defaults of PyTorch:

| layer | arguments | arrays | output |
| --- | --- | --- | --- |
| `Linear` | `in_features`, `out_features`, `bias=True` | `weight (out_features, in_features)`, `bias (out_features,)` | `x @ weight.T + bias` on the last axis, input `(*, in_features)`, output `(*, out_features)` |
| `Bilinear` | `in1_features`, `in2_features`, `out_features`, `bias=True` | `weight (out_features, in1_features, in2_features)`, `bias (out_features,)` | `y_k = x1^T weight_k x2 + bias_k`, inputs `(*, in1_features)` and `(*, in2_features)` |
| `Flatten` | `start_dim=1`, `end_dim=-1` | none | the axes `start_dim` to `end_dim` as one axis, row major |
| `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`, `AlphaDropout`, `FeatureAlphaDropout` | `p`, `inplace`, not used | none | the input |

The semantics of the interpreter:

- The inputs are given in the order of the `placeholder` nodes and are converted with `backend.asarray`.
- A positional argument of a node which is a string and the name of a node which was evaluated is the value of that node, a list is resolved entry by entry (`cat`), everything else is a literal. The keyword arguments are literals, `petab_sciml` writes the nodes a function is called with as positional arguments only.
- The keyword arguments `inplace` and `_stacklevel` are dropped, `dtype` is dropped when it is `None`. Any other keyword argument which the implementation does not have is an `UnsupportedLayerError`, it is never dropped silently.
- A `call_method` node is the function of the same name (`x.tanh()` is `tanh(x)`).
- A `ValueError` of a layer is raised again with the network and the node in front of its message.
- `on_node(node, value)` is called after every `call_module`, `call_function` and `call_method` node, the following nodes see what it returns. The compiler of phase 3 replaces the expressions of a node by symbols there.

- [ ] **Step 1: Write the fixtures**

Create `tests/sciml/conftest.py`:

````python
"""Fixtures of the tests of the neural networks.

The forward pass of `sbmlsim` is compared with PyTorch, for every layer and
function and with random arrays. `torch` is only in the `dev` extra, the
tests which need it are skipped without it.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml.backend import NumpyBackend
from sbmlsim.sciml.interpreter import evaluate

#: absolute and relative tolerance of the comparison with PyTorch, which runs
#: in double precision
TOLERANCE = 1e-10


def build_layer_model(
    layer_type: str, args: dict[str, Any], n_inputs: int = 1
) -> NNModel:
    """Build the network `net1` which is the single layer `layer1`."""
    names = [f"net_input{k}" for k in range(n_inputs)]
    forward = [
        Node(name=name, op="placeholder", target=name, args=[], kwargs={})
        for name in names
    ]
    forward.append(
        Node(name="layer1", op="call_module", target="layer1", args=names, kwargs={})
    )
    forward.append(
        Node(name="output", op="output", target="output", args=["layer1"], kwargs={})
    )
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id=f"input{k}") for k in range(n_inputs)],
        layers=[Layer(layer_id="layer1", layer_type=layer_type, args=args)],
        forward=forward,
    )


def build_function_model(
    target: str, kwargs: dict[str, Any], op: str = "call_function"
) -> NNModel:
    """Build the network `net1` which is the single function `f` of its input."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=[],
        forward=[
            Node(
                name="net_input",
                op="placeholder",
                target="net_input",
                args=[],
                kwargs={},
            ),
            Node(name="f", op=op, target=target, args=["net_input"], kwargs=kwargs),
            Node(name="output", op="output", target="output", args=["f"], kwargs={}),
        ],
    )


def run_forward(
    model: NNModel,
    parameters: dict[str, dict[str, np.ndarray]],
    *inputs: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Evaluate a network with numpy."""
    return evaluate(model, parameters, inputs, NumpyBackend())


@pytest.fixture
def layer_model() -> Callable[..., NNModel]:
    """Get the function which builds a network of a single layer."""
    return build_layer_model


@pytest.fixture
def function_model() -> Callable[..., NNModel]:
    """Get the function which builds a network of a single function."""
    return build_function_model


@pytest.fixture
def forward() -> Callable[..., tuple[np.ndarray, ...]]:
    """Get the function which evaluates a network with numpy."""
    return run_forward


@pytest.fixture
def rng() -> np.random.Generator:
    """Get a seeded random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def compare_layer(rng: np.random.Generator) -> Callable[..., None]:
    """Get the comparison of a layer with the layer of PyTorch.

    The arrays of the layer, the running statistics of a normalization layer
    and the inputs are random. PyTorch evaluates in evaluation mode.
    """
    torch = pytest.importorskip("torch")

    def compare(
        layer_type: str, args: dict[str, Any], *shapes: tuple[int, ...]
    ) -> None:
        module = getattr(torch.nn, layer_type)(**args).double()
        state = {}
        for name, tensor in module.state_dict().items():
            if name == "num_batches_tracked":
                state[name] = tensor
            elif name == "running_var":
                state[name] = torch.from_numpy(rng.uniform(0.5, 2.0, tensor.shape))
            else:
                state[name] = torch.from_numpy(rng.normal(size=tuple(tensor.shape)))
        module.load_state_dict(state)
        module.eval()

        parameters = {
            "layer1": {
                name: tensor.numpy().copy()
                for name, tensor in state.items()
                if name != "num_batches_tracked"
            }
        }
        inputs = [rng.normal(size=shape) for shape in shapes]
        with torch.no_grad():
            expected = module(*(torch.from_numpy(x) for x in inputs)).numpy()

        model = build_layer_model(layer_type, args, n_inputs=len(shapes))
        (observed,) = run_forward(model, parameters, *inputs)
        assert observed.shape == expected.shape
        np.testing.assert_allclose(observed, expected, rtol=TOLERANCE, atol=TOLERANCE)

    return compare


@pytest.fixture
def compare_function() -> Callable[..., None]:
    """Get the comparison of a function with the function of PyTorch."""
    torch = pytest.importorskip("torch")

    def compare(
        target: str,
        kwargs: dict[str, Any],
        x: np.ndarray,
        torch_name: str | None = None,
        op: str = "call_function",
    ) -> None:
        name = target if torch_name is None else torch_name
        reference = getattr(torch.nn.functional, name, None) or getattr(torch, name)
        expected = reference(torch.from_numpy(x), **kwargs).numpy()

        model = build_function_model(target, kwargs, op=op)
        (observed,) = run_forward(model, {}, x)
        assert observed.shape == expected.shape
        np.testing.assert_allclose(observed, expected, rtol=TOLERANCE, atol=TOLERANCE)

    return compare
````

- [ ] **Step 2: Write the failing tests**

Create `tests/sciml/test_interpreter.py`:

````python
"""Tests of the interpreter of the forward pass."""

from collections.abc import Callable

import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import UnsupportedLayerError
from sbmlsim.sciml.backend import NUMPY_ONLY, Backend, BackendKind, NumpyBackend
from sbmlsim.sciml.interpreter import evaluate
from sbmlsim.sciml.layers import FUNCTIONS, LAYERS
from sbmlsim.sciml.layers.registry import FunctionType, function

WEIGHT = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
BIAS = np.array([0.1, 0.2, 0.3])
ARRAYS = {"layer1": {"weight": WEIGHT, "bias": BIAS}}


def _node(name: str, op: str, target: str, args: list, **kwargs: object) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs=kwargs)


def _model(*nodes: Node, layer_type: str = "Linear") -> NNModel:
    """Build a network with the layer `layer1` and the nodes as forward pass."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type=layer_type,
                args={"in_features": 2, "out_features": 3},
            )
        ],
        forward=list(nodes),
    )


X = _node("x", "placeholder", "x", [])
LAYER1 = _node("layer1", "call_module", "layer1", ["x"])


@pytest.fixture
def scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the function `scale(x, factor=2.0)` for a test."""

    def implementation(backend: Backend, x: np.ndarray, factor: float = 2.0):
        return factor * x

    monkeypatch.setitem(
        FUNCTIONS,
        "scale",
        FunctionType(
            name="scale",
            function=implementation,
            backends=frozenset({BackendKind.NUMPY}),
        ),
    )


@pytest.mark.usefixtures("scale")
@pytest.mark.parametrize("op", ["call_function", "call_method"])
def test_the_nodes_are_evaluated_in_their_order(op: str) -> None:
    """A layer and a function, with a positional and a keyword argument."""
    model = _model(
        X,
        LAYER1,
        _node("scale", op, "scale", ["layer1"]),
        _node("scale_1", op, "scale", ["scale", 5.0]),
        _node("scale_2", op, "scale", ["scale_1"], factor=0.1, inplace=False),
        _node("output", "output", "output", ["scale_2"]),
    )
    x = np.array([0.5, -0.25])
    (y,) = evaluate(model, ARRAYS, [x], NumpyBackend())
    np.testing.assert_allclose(y, WEIGHT @ x + BIAS)
    assert y.dtype == float


def test_an_input_is_converted_to_the_array_of_the_backend() -> None:
    """Single precision, integers and lists are inputs."""
    model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    expected = WEIGHT @ [1.0, 2.0] + BIAS
    for x in (np.array([1, 2]), np.array([1.0, 2.0], dtype="f4"), [1.0, 2.0]):
        (y,) = evaluate(model, ARRAYS, [x], NumpyBackend())
        np.testing.assert_allclose(y, expected)
        assert y.dtype == float


@pytest.mark.usefixtures("scale")
def test_several_outputs() -> None:
    """An output node with a list returns one array per entry."""
    model = _model(
        X,
        LAYER1,
        _node("scale", "call_function", "scale", ["layer1"]),
        _node("output", "output", "output", [["layer1", "scale"]]),
    )
    hidden, y = evaluate(model, ARRAYS, [np.array([0.5, -0.25])], NumpyBackend())
    np.testing.assert_allclose(y, 2.0 * hidden)


@pytest.mark.usefixtures("scale")
def test_the_hook_replaces_the_value_of_a_node() -> None:
    """The hook sees every layer and function, not the inputs and the output."""
    seen: list[str] = []

    def on_node(node: Node, value: np.ndarray) -> np.ndarray:
        seen.append(node.name)
        return np.ones_like(value) if node.name == "layer1" else value

    model = _model(
        X,
        LAYER1,
        _node("scale", "call_function", "scale", ["layer1"]),
        _node("output", "output", "output", ["scale"]),
    )
    (y,) = evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend(), on_node)
    assert seen == ["layer1", "scale"]
    np.testing.assert_array_equal(y, [2.0, 2.0, 2.0])


@pytest.mark.parametrize("n_inputs", [0, 2])
def test_the_number_of_inputs(n_inputs: int) -> None:
    """The inputs are the placeholders of the forward pass."""
    model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    with pytest.raises(ValueError, match=rf"{n_inputs} inputs were given for the 1"):
        evaluate(model, ARRAYS, [np.zeros(2)] * n_inputs, NumpyBackend())


def test_an_input_of_the_wrong_size() -> None:
    """The error of numpy names the network and the node."""
    model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    with pytest.raises(ValueError, match=r"Network 'net1', node 'layer1'"):
        evaluate(model, ARRAYS, [np.zeros(3)], NumpyBackend())


def test_a_forward_pass_without_an_output() -> None:
    """A forward pass which ends without an output node is an error."""
    with pytest.raises(ValueError, match=r"'net1'.*no output node"):
        evaluate(_model(X, LAYER1), ARRAYS, [np.zeros(2)], NumpyBackend())


def test_a_layer_which_the_network_does_not_have() -> None:
    """A node which calls an unknown layer names the layer."""
    model = _model(
        X,
        _node("layer2", "call_module", "layer2", ["x"]),
        _node("output", "output", "output", ["layer2"]),
    )
    with pytest.raises(ValueError, match=r"node 'layer2'.*no layer 'layer2'"):
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())


def test_an_unknown_opcode() -> None:
    """`get_attr` of `torch.fx` is not part of the NN YAML."""
    model = _model(
        X,
        _node("w", "get_attr", "w", []),
        _node("output", "output", "output", ["w"]),
    )
    with pytest.raises(ValueError, match=r"node 'w'.*'get_attr' is not known"):
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())


def test_a_layer_without_an_implementation() -> None:
    """An unknown layer type names the network, the node and the type."""
    model = _model(
        X,
        LAYER1,
        _node("output", "output", "output", ["layer1"]),
        layer_type="LSTM",
    )
    with pytest.raises(UnsupportedLayerError, match=r"'net1'.*'layer1'.*'LSTM'") as e:
        evaluate(model, ARRAYS, [np.zeros(2)], NumpyBackend())
    assert (e.value.network, e.value.node, e.value.target) == (
        "net1",
        "layer1",
        "LSTM",
    )


class ExpressionBackend(NumpyBackend):
    """A backend which says that it is the one on expressions."""

    kind = BackendKind.SYMPY


@pytest.mark.usefixtures("scale")
def test_a_node_which_the_backend_does_not_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layer and a function declare the backends they are evaluated by."""
    function_model = _model(
        X,
        _node("scale", "call_function", "scale", ["x"]),
        _node("output", "output", "output", ["scale"]),
    )
    with pytest.raises(UnsupportedLayerError, match=r"'scale'.*backend 'sympy'"):
        evaluate(function_model, {}, [np.zeros(2)], ExpressionBackend())

    layer_model = _model(X, LAYER1, _node("output", "output", "output", ["layer1"]))
    evaluate(layer_model, ARRAYS, [np.zeros(2)], ExpressionBackend())
    monkeypatch.setitem(
        LAYERS,
        "Linear",
        LAYERS["Linear"].__class__(
            name="Linear",
            forward=LAYERS["Linear"].forward,
            arrays=LAYERS["Linear"].arrays,
            backends=NUMPY_ONLY,
        ),
    )
    with pytest.raises(UnsupportedLayerError, match=r"'Linear'.*backend 'sympy'"):
        evaluate(layer_model, ARRAYS, [np.zeros(2)], ExpressionBackend())


def test_a_function_is_registered_under_every_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decorator registers the function and returns it unchanged."""
    monkeypatch.setattr("sbmlsim.sciml.layers.registry.FUNCTIONS", {})
    from sbmlsim.sciml.layers import registry

    @function("one", "uno", backends=NUMPY_ONLY)
    def one(backend: Backend, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    assert set(registry.FUNCTIONS) == {"one", "uno"}
    assert registry.FUNCTIONS["uno"].function is one
    assert registry.FUNCTIONS["uno"].backends == NUMPY_ONLY
    assert isinstance(one, Callable)
````

Create `tests/sciml/test_layers_core.py`:

````python
"""The layers of both backends against PyTorch."""

from collections.abc import Callable

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml.backend import ALL_BACKENDS
from sbmlsim.sciml.layers import LAYERS


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(3,), (4, 3), (2, 4, 3)])
def test_linear(
    compare_layer: Callable[..., None], bias: bool, shape: tuple[int, ...]
) -> None:
    """`Linear` works on the last axis, with and without a batch."""
    compare_layer("Linear", {"in_features": 3, "out_features": 5, "bias": bias}, shape)


def test_linear_has_a_bias_by_default(compare_layer: Callable[..., None]) -> None:
    """`bias` is `True` when the YAML does not give it."""
    compare_layer("Linear", {"in_features": 3, "out_features": 5}, (3,))


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch", [(), (4,)])
def test_bilinear(
    compare_layer: Callable[..., None], bias: bool, batch: tuple[int, ...]
) -> None:
    """`Bilinear` takes two inputs."""
    args = {"in1_features": 3, "in2_features": 4, "out_features": 2, "bias": bias}
    compare_layer("Bilinear", args, (*batch, 3), (*batch, 4))


@pytest.mark.parametrize(
    ("args", "shape"),
    [
        ({}, (2, 3, 4)),
        ({"start_dim": 1, "end_dim": -1}, (2, 3, 4, 5)),
        ({"start_dim": 0, "end_dim": -1}, (2, 3, 4)),
        ({"start_dim": 1, "end_dim": 2}, (2, 3, 4, 5)),
        ({"start_dim": -2, "end_dim": -1}, (2, 3, 4)),
    ],
)
def test_flatten(
    compare_layer: Callable[..., None], args: dict, shape: tuple[int, ...]
) -> None:
    """`Flatten` flattens in row major order, from the axis 1 by default."""
    compare_layer("Flatten", args, shape)


@pytest.mark.parametrize(
    ("layer_type", "shape"),
    [
        ("Dropout", (5,)),
        ("AlphaDropout", (5,)),
        ("FeatureAlphaDropout", (2, 3, 4)),
        ("Dropout1d", (2, 3, 4)),
        ("Dropout2d", (2, 3, 4, 5)),
        ("Dropout3d", (2, 3, 4, 5, 6)),
    ],
)
def test_dropout_is_the_identity(
    compare_layer: Callable[..., None], layer_type: str, shape: tuple[int, ...]
) -> None:
    """A dropout layer in evaluation mode returns its input."""
    compare_layer(layer_type, {"p": 0.5, "inplace": False}, shape)


def test_the_layers_of_both_backends() -> None:
    """The layers a compiled network may use declare both backends."""
    for name in ("Linear", "Bilinear", "Flatten", "Dropout", "AlphaDropout"):
        assert LAYERS[name].backends == ALL_BACKENDS


def test_flatten_with_the_start_behind_the_end(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """A range of axes which is empty is an error which names the node."""
    model = layer_model("Flatten", {"start_dim": 2, "end_dim": 1})
    with pytest.raises(ValueError, match=r"node 'layer1'.*start_dim"):
        forward(model, {}, np.zeros((2, 3, 4)))


def test_a_scalar_is_flattened_to_one_element(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """An array without axes becomes an array with one element, as in PyTorch."""
    model = layer_model("Flatten", {"start_dim": 0, "end_dim": -1})
    (y,) = forward(model, {}, np.array(2.0))
    assert y.shape == (1,)
````

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'sbmlsim.sciml.interpreter'`.

- [ ] **Step 4: Write the registry**

Create `src/sbmlsim/sciml/layers/registry.py`:

````python
"""The registry of the layers and functions of a forward pass.

A layer is a function `(backend, args, arrays, *inputs) -> output`, with the
arguments of the layer in the NN YAML (`args`, the keyword arguments of the
PyTorch class) and the arrays of the layer in the PyTorch layout. A function
is a function `(backend, *inputs, **kwargs) -> output` with the keyword
arguments of the PyTorch function. Both are registered under their PyTorch
name together with the backends they support.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import ALL_BACKENDS, BackendKind


@dataclass(frozen=True)
class ArraySpec:
    """An array of a layer.

    Attributes:
        shape: the shape of the array in the PyTorch layout.
        required: whether the layer cannot be evaluated without the array.
        trainable: whether the elements are parameters of a fit. The running
            statistics of a normalization layer are arrays but not parameters.
    """

    shape: tuple[int, ...]
    required: bool = True
    trainable: bool = True


#: the arrays of a layer from its arguments
ArraysFunction = Callable[[Mapping[str, Any]], dict[str, ArraySpec]]


def no_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a layer without arrays.

    Args:
        args: the arguments of the layer.

    Returns:
        An empty dictionary.
    """
    return {}


@dataclass(frozen=True)
class LayerType:
    """A type of layer, e.g. `Linear`.

    Attributes:
        name: the name of the PyTorch class.
        forward: the implementation `(backend, args, arrays, *inputs)`.
        arrays: the arrays of a layer of this type from its arguments.
        backends: the backends the implementation supports.
    """

    name: str
    forward: Callable[..., np.ndarray]
    arrays: ArraysFunction
    backends: frozenset[BackendKind]


@dataclass(frozen=True)
class FunctionType:
    """A function or method of the forward pass, e.g. `relu`.

    Attributes:
        name: the name of the PyTorch function.
        function: the implementation `(backend, *inputs, **kwargs)`.
        backends: the backends the implementation supports.
    """

    name: str
    function: Callable[..., np.ndarray]
    backends: frozenset[BackendKind]


#: the layers by the name of their PyTorch class
LAYERS: dict[str, LayerType] = {}

#: the functions and methods by their PyTorch name
FUNCTIONS: dict[str, FunctionType] = {}


def layer(
    *names: str,
    arrays: ArraysFunction = no_arrays,
    backends: frozenset[BackendKind] = ALL_BACKENDS,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register the implementation of one or more layer types.

    Args:
        *names: the names of the PyTorch classes the function implements.
        arrays: the arrays of a layer from its arguments.
        backends: the backends the implementation supports.

    Returns:
        The decorator, which returns the function unchanged.
    """

    def register(forward: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        for name in names:
            LAYERS[name] = LayerType(
                name=name, forward=forward, arrays=arrays, backends=backends
            )
        return forward

    return register


def layer_nd(
    template: str,
    arrays: Callable[[int, Mapping[str, Any]], dict[str, ArraySpec]] | None = None,
    backends: frozenset[BackendKind] = ALL_BACKENDS,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register the implementation of a layer type for 1, 2 and 3 dimensions.

    The implementation is `(n, backend, args, arrays, *inputs)` with the
    number of spatial dimensions `n`, and is registered once per dimension
    with `n` bound.

    Args:
        template: the name of the PyTorch classes with `{n}` for the number of
            dimensions, e.g. `Conv{n}d`.
        arrays: the arrays of a layer from `n` and its arguments.
        backends: the backends the implementation supports.

    Returns:
        The decorator, which returns the function unchanged.
    """

    def register(forward: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        for n in (1, 2, 3):
            name = template.format(n=n)
            LAYERS[name] = LayerType(
                name=name,
                forward=partial(forward, n),
                arrays=no_arrays if arrays is None else partial(arrays, n),
                backends=backends,
            )
        return forward

    return register


def function(
    *names: str,
    backends: frozenset[BackendKind] = ALL_BACKENDS,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register the implementation of one or more functions.

    Args:
        *names: the names of the PyTorch functions the function implements.
        backends: the backends the implementation supports.

    Returns:
        The decorator, which returns the function unchanged.
    """

    def register(
        implementation: Callable[..., np.ndarray],
    ) -> Callable[..., np.ndarray]:
        for name in names:
            FUNCTIONS[name] = FunctionType(
                name=name, function=implementation, backends=backends
            )
        return implementation

    return register


def as_tuple(value: Any, n: int) -> tuple[int, ...]:
    """Expand an argument of a layer to one value per spatial dimension.

    Args:
        value: an integer, which holds for every dimension, or a sequence of
            `n` integers.
        n: the number of spatial dimensions.

    Returns:
        The values of the dimensions.

    Raises:
        ValueError: if a sequence does not have `n` entries.
    """
    if isinstance(value, (int, np.integer)):
        return (int(value),) * n
    values = tuple(int(v) for v in value)
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"'{value}' does not have {n} entries")
    return values
````

- [ ] **Step 5: Write the layers of both backends**

Create `src/sbmlsim/sciml/layers/core.py`:

````python
"""The layers which both backends support.

`Linear`, `Bilinear` and `Flatten` are array operations of numpy, which work
on arrays of numbers and of expressions. The dropout layers are the identity,
because a network is evaluated in evaluation mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import Backend
from sbmlsim.sciml.layers.registry import ArraySpec, layer


def flatten_array(x: np.ndarray, start_dim: int = 0, end_dim: int = -1) -> np.ndarray:
    """Flatten a range of axes of an array in row major order.

    Args:
        x: the array.
        start_dim: the first axis which is flattened.
        end_dim: the last axis which is flattened.

    Returns:
        The array with the axes `start_dim` to `end_dim` as one axis.

    Raises:
        ValueError: if `start_dim` is behind `end_dim`.
    """
    if x.ndim == 0:
        return x.reshape(1)
    start = start_dim % x.ndim
    end = end_dim % x.ndim
    if start > end:
        raise ValueError(
            f"flatten: start_dim '{start_dim}' is behind end_dim '{end_dim}'"
        )
    return x.reshape((*x.shape[:start], -1, *x.shape[end + 1 :]))


def linear_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `Linear` layer."""
    arrays = {"weight": ArraySpec((args["out_features"], args["in_features"]))}
    if args.get("bias", True):
        arrays["bias"] = ArraySpec((args["out_features"],))
    return arrays


@layer("Linear", arrays=linear_arrays)
def linear(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `Linear`, i.e. `y = x W^T + b` on the last axis.

    Args:
        backend: the backend.
        args: `in_features`, `out_features`, `bias` (default `True`).
        arrays: `weight` of shape `(out_features, in_features)` and `bias` of
            shape `(out_features,)`.
        x: input of shape `(*, in_features)`.

    Returns:
        The output of shape `(*, out_features)`.
    """
    y = x @ arrays["weight"].T
    if "bias" in arrays:
        y = y + arrays["bias"]
    return y


def bilinear_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `Bilinear` layer."""
    arrays = {
        "weight": ArraySpec(
            (args["out_features"], args["in1_features"], args["in2_features"])
        )
    }
    if args.get("bias", True):
        arrays["bias"] = ArraySpec((args["out_features"],))
    return arrays


@layer("Bilinear", arrays=bilinear_arrays)
def bilinear(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
    """Evaluate `Bilinear`, i.e. `y_k = x1^T A_k x2 + b_k` on the last axis.

    Args:
        backend: the backend.
        args: `in1_features`, `in2_features`, `out_features`, `bias` (default
            `True`).
        arrays: `weight` of shape `(out_features, in1_features, in2_features)`
            and `bias` of shape `(out_features,)`.
        x1: first input of shape `(*, in1_features)`.
        x2: second input of shape `(*, in2_features)`.

    Returns:
        The output of shape `(*, out_features)`.
    """
    # (*, in1) . (out, in1, in2) over in1 -> (*, out, in2)
    left = np.tensordot(x1, arrays["weight"], axes=([-1], [1]))
    y = (left * x2[..., np.newaxis, :]).sum(axis=-1)
    if "bias" in arrays:
        y = y + arrays["bias"]
    return y


@layer("Flatten")
def flatten_layer(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `Flatten`.

    Args:
        backend: the backend.
        args: `start_dim` (default `1`) and `end_dim` (default `-1`).
        arrays: no arrays.
        x: the input.

    Returns:
        The input with the axes `start_dim` to `end_dim` as one axis, in row
        major order.
    """
    return flatten_array(x, args.get("start_dim", 1), args.get("end_dim", -1))


@layer(
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
)
def dropout(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate a dropout layer in evaluation mode, which is the identity.

    Args:
        backend: the backend.
        args: `p` and `inplace`, which are not used.
        arrays: no arrays.
        x: the input.

    Returns:
        The input.
    """
    return x
````

- [ ] **Step 6: Write the package of the layers**

Create `src/sbmlsim/sciml/layers/__init__.py`. The table of the docstring names the modules of the following tasks already, the imports grow with the tasks:

````python
"""The layers and functions of the forward pass of a network.

Every layer and every function of the NN YAML is implemented once, against a
backend, and registered under its PyTorch name in `LAYERS` or `FUNCTIONS`
with the backends it supports. Importing this package registers all of them.

| module | content | backends |
| --- | --- | --- |
| `core` | `Linear`, `Bilinear`, `Flatten`, the dropout layers | numpy, sympy |
| `functions` | the activation functions, `flatten`, `cat` | numpy, sympy |
| `convolution` | `Conv1-3d`, `ConvTranspose1-3d` | numpy |
| `pooling` | `MaxPool`, `AvgPool`, `LPPool` and the adaptive pools, `1-3d` | numpy |
| `normalization` | `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm` | numpy |
"""

from sbmlsim.sciml.layers import core
from sbmlsim.sciml.layers.registry import (
    FUNCTIONS,
    LAYERS,
    ArraySpec,
    FunctionType,
    LayerType,
)

__all__ = [
    "FUNCTIONS",
    "LAYERS",
    "ArraySpec",
    "FunctionType",
    "LayerType",
    "core",
]
````

- [ ] **Step 7: Write the interpreter**

Create `src/sbmlsim/sciml/interpreter.py`:

````python
"""The interpreter of the forward pass of a network.

The forward pass of the NN YAML is the list of the nodes of a `torch.fx`
graph: `placeholder` (an input), `call_module` (a layer), `call_function` and
`call_method` (a function), and `output`. `evaluate` walks the list once with
a backend, so the forward pass in numpy and the compilation into expressions
are the same interpreter and the same layers.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from petab_sciml import NNModel, Node

from sbmlsim.sciml.backend import Backend
from sbmlsim.sciml.errors import UnsupportedLayerError
from sbmlsim.sciml.layers import FUNCTIONS, LAYERS

#: the opcodes of the nodes
PLACEHOLDER = "placeholder"
CALL_MODULE = "call_module"
CALL_FUNCTION = "call_function"
CALL_METHOD = "call_method"
OUTPUT = "output"

#: keyword arguments of the functions of PyTorch which do not change a value
IGNORED_KWARGS: frozenset[str] = frozenset({"inplace", "_stacklevel"})

#: called with a node and its value, returns the value the next nodes see
NodeHook = Callable[[Node, np.ndarray], np.ndarray]


def _resolve(value: Any, state: Mapping[str, np.ndarray]) -> Any:
    """Replace the names of nodes in an argument by the values of the nodes.

    Args:
        value: an argument of a node: the name of a node, a list of arguments
            or a literal.
        state: the values of the nodes which were evaluated.

    Returns:
        The argument with the values of the nodes.
    """
    if isinstance(value, (list, tuple)):
        return [_resolve(v, state) for v in value]
    if isinstance(value, str) and value in state:
        return state[value]
    return value


def _call_function(
    network: str, node: Node, backend: Backend, args: list[Any], kwargs: dict[str, Any]
) -> np.ndarray:
    """Evaluate a `call_function` or `call_method` node.

    Args:
        network: id of the network.
        node: the node.
        backend: the backend.
        args: the resolved positional arguments.
        kwargs: the keyword arguments.

    Returns:
        The value of the node.

    Raises:
        UnsupportedLayerError: if the function is not implemented, not
            available in the backend or called with an argument the
            implementation does not have.
    """
    function_type = FUNCTIONS.get(node.target)
    if function_type is None:
        raise UnsupportedLayerError(
            network, node.name, node.target, "the function is not implemented"
        )
    if backend.kind not in function_type.backends:
        raise UnsupportedLayerError(
            network,
            node.name,
            node.target,
            f"the function is not available in the backend '{backend.kind}'",
        )
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in IGNORED_KWARGS and not (key == "dtype" and value is None)
    }
    try:
        inspect.signature(function_type.function).bind(backend, *args, **kwargs)
    except TypeError as err:
        raise UnsupportedLayerError(
            network, node.name, node.target, f"the arguments do not fit: {err}"
        ) from err
    return function_type.function(backend, *args, **kwargs)


def evaluate(
    model: NNModel,
    parameters: Mapping[str, Mapping[str, np.ndarray]],
    inputs: Sequence[ArrayLike],
    backend: Backend,
    on_node: NodeHook | None = None,
) -> tuple[np.ndarray, ...]:
    """Evaluate the forward pass of a network.

    Args:
        model: the architecture of the network.
        parameters: the arrays of the layers, layer id -> array name -> array,
            in the PyTorch layout.
        inputs: the inputs, one per `placeholder` node in the order of the
            nodes.
        backend: the backend the layers and functions are evaluated with.
        on_node: called after every layer and function with the node and its
            value, the next nodes see what it returns. The compilation uses it
            to replace the expressions of a node by symbols.

    Returns:
        The outputs of the network.

    Raises:
        UnsupportedLayerError: if a layer or function is not implemented or
            not available in the backend.
        ValueError: if the number of inputs is not the number of placeholders,
            if a node has an unknown opcode or if a layer cannot be evaluated
            on its input; the message names the network and the node.
    """
    network = model.nn_model_id
    layers = {layer.layer_id: layer for layer in model.layers}
    placeholders = [node for node in model.forward if node.op == PLACEHOLDER]
    if len(placeholders) != len(inputs):
        raise ValueError(
            f"Network '{network}': {len(inputs)} inputs were given for the "
            f"{len(placeholders)} inputs {[node.name for node in placeholders]}"
        )

    state: dict[str, np.ndarray] = {}
    remaining = iter(inputs)
    for node in model.forward:
        args = _resolve(node.args or [], state)
        # the keyword arguments are literals: `petab_sciml` writes the nodes a
        # function is called with as positional arguments only
        kwargs = dict(node.kwargs or {})

        if node.op == PLACEHOLDER:
            state[node.name] = backend.asarray(next(remaining))
            continue
        if node.op == OUTPUT:
            output = args[0]
            return tuple(output) if isinstance(output, list) else (output,)

        try:
            if node.op == CALL_MODULE:
                value = _call_module(network, node, layers, parameters, backend, args)
            elif node.op in (CALL_FUNCTION, CALL_METHOD):
                value = _call_function(network, node, backend, args, kwargs)
            else:
                raise ValueError(f"the opcode '{node.op}' is not known")
        except UnsupportedLayerError:
            raise
        except ValueError as err:
            raise ValueError(f"Network '{network}', node '{node.name}': {err}") from err

        state[node.name] = value if on_node is None else on_node(node, value)

    raise ValueError(f"Network '{network}': the forward pass has no output node")


def _call_module(
    network: str,
    node: Node,
    layers: Mapping[str, Any],
    parameters: Mapping[str, Mapping[str, np.ndarray]],
    backend: Backend,
    args: list[Any],
) -> np.ndarray:
    """Evaluate a `call_module` node, i.e. a layer.

    Args:
        network: id of the network.
        node: the node.
        layers: the layers of the network by their id.
        parameters: the arrays of the layers.
        backend: the backend.
        args: the resolved positional arguments, i.e. the inputs of the layer.

    Returns:
        The value of the node.

    Raises:
        UnsupportedLayerError: if the layer type is not implemented or not
            available in the backend.
        ValueError: if the network has no layer of the id.
    """
    if node.target not in layers:
        raise ValueError(f"the network has no layer '{node.target}'")
    layer = layers[node.target]
    layer_type = LAYERS.get(layer.layer_type)
    if layer_type is None:
        raise UnsupportedLayerError(
            network, node.name, layer.layer_type, "the layer is not implemented"
        )
    if backend.kind not in layer_type.backends:
        raise UnsupportedLayerError(
            network,
            node.name,
            layer.layer_type,
            f"the layer is not available in the backend '{backend.kind}'",
        )
    arrays = {
        name: backend.asarray(array)
        for name, array in parameters.get(layer.layer_id, {}).items()
    }
    return layer_type.forward(backend, layer.args or {}, arrays, *args)
````

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `49 passed`. If torch is not installed the comparisons with PyTorch are skipped, which is not the state this plan expects: run `uv sync --extra dev` first.

- [ ] **Step 9: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 10: Commit**

```bash
git add src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the interpreter of the forward pass and the layers Linear, Bilinear, Flatten and dropout"
```

---

### Task 3: The activation functions and the tensor operations

**Files:**
- Create: `src/sbmlsim/sciml/layers/functions.py`
- Modify: `src/sbmlsim/sciml/layers/__init__.py` (import `functions`)
- Test: `tests/sciml/test_functions.py`

**Interfaces:**
- Consumes: `Backend` (`exp`, `log`, `tanh`, `erf`, `absolute`, `select`, `stabilizer`); `function(*names, backends=ALL_BACKENDS)` of `sbmlsim.sciml.layers.registry`; `flatten_array(x, start_dim, end_dim)` of `sbmlsim.sciml.layers.core`; the fixtures `compare_function`, `function_model`, `forward` of `tests/sciml/conftest.py`.
- Produces: the entries of `FUNCTIONS`: `tanh`, `sigmoid`, `relu`, `relu6`, `hardtanh`, `hardsigmoid`, `hardswish`, `leaky_relu`, `elu`, `celu`, `selu`, `gelu`, `softplus`, `log_sigmoid`, `logsigmoid`, `mish`, `silu`, `softsign`, `tanhshrink`, `softmax`, `log_softmax`, `flatten`, `cat`, `concat`, `concatenate`. All of them support both backends.

The semantics, with the argument names and defaults of `torch.nn.functional`. `select(x, t, a, b)` is `b` where `x > t` and `a` elsewhere, `clip(x, low, high)` is `select(x, low, low, select(x, high, x, high))`.

| function | arguments | value |
| --- | --- | --- |
| `tanh` | | `tanh(x)` |
| `sigmoid` | | `1 / (1 + exp(-x))` |
| `relu` | | `select(x, 0, 0, x)` |
| `relu6` | | `clip(x, 0, 6)` |
| `hardtanh` | `min_val=-1.0`, `max_val=1.0` | `clip(x, min_val, max_val)` |
| `hardsigmoid` | | `clip(x / 6 + 1 / 2, 0, 1)` |
| `hardswish` | | `x * clip(x + 3, 0, 6) / 6` |
| `leaky_relu` | `negative_slope=0.01` | `select(x, 0, negative_slope * x, x)` |
| `elu` | `alpha=1.0` | `select(x, 0, alpha * (exp(x) - 1), x)` |
| `celu` | `alpha=1.0` | `select(x, 0, alpha * (exp(x / alpha) - 1), x)` |
| `selu` | | `1.0507009873554805 * elu(x, alpha=1.6732632423543772)` |
| `gelu` | `approximate="none"` | `0.5 * x * (1 + erf(x / sqrt(2)))`, for `"tanh"`: `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))` |
| `softplus` | `beta=1.0`, `threshold=20.0` | `select(beta * x, threshold, log(1 + exp(beta * x)) / beta, x)` |
| `log_sigmoid` | | `select(x, 0, x - log(1 + exp(x)), -log(1 + exp(-x)))`, the same function in both branches |
| `mish` | | `x * tanh(softplus(x))` |
| `silu` | | `x * sigmoid(x)` |
| `softsign` | | `x / (1 + abs(x))` |
| `tanhshrink` | | `x - tanh(x)` |
| `softmax` | `dim` (required) | `e / sum(e, dim)` with `e = exp(x - stabilizer(x, dim))` |
| `log_softmax` | `dim` (required) | `s - log(sum(exp(s), dim))` with `s = x - stabilizer(x, dim)` |
| `flatten` | `start_dim=0`, `end_dim=-1` | `flatten_array` (the default of the function is `0`, the one of the layer `Flatten` is `1`) |
| `cat` | `tensors` (a list), `dim=0` | `np.concatenate(tensors, axis=dim)` |

- [ ] **Step 1: Write the failing test**

Create `tests/sciml/test_functions.py`:

````python
"""The activation functions and tensor operations against PyTorch."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import Input, NNModel, Node

from sbmlsim.sciml import UnsupportedLayerError
from sbmlsim.sciml.backend import ALL_BACKENDS
from sbmlsim.sciml.layers import FUNCTIONS

#: values on both sides of every threshold of the piecewise functions, the
#: thresholds themselves and values which overflow a naive exponential
VALUES = np.array(
    [-800.0, -30.0, -6.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 6.0, 20.0, 21.0, 800.0]
)

FUNCTION_CASES: list[tuple[str, dict[str, Any]]] = [
    ("tanh", {}),
    ("sigmoid", {}),
    ("relu", {}),
    ("relu", {"inplace": False}),
    ("relu6", {"inplace": False}),
    ("hardtanh", {}),
    ("hardtanh", {"min_val": -2.0, "max_val": 0.5, "inplace": False}),
    ("hardsigmoid", {"inplace": False}),
    ("hardswish", {"inplace": False}),
    ("leaky_relu", {}),
    ("leaky_relu", {"negative_slope": 0.2, "inplace": False}),
    ("elu", {}),
    ("elu", {"alpha": 2.0, "inplace": False}),
    ("celu", {}),
    ("celu", {"alpha": 2.0, "inplace": False}),
    ("selu", {"inplace": False}),
    ("gelu", {}),
    ("gelu", {"approximate": "none"}),
    ("gelu", {"approximate": "tanh"}),
    ("softplus", {}),
    ("softplus", {"beta": 2.0, "threshold": 5.0}),
    ("mish", {"inplace": False}),
    ("silu", {"inplace": False}),
    ("softsign", {}),
    ("tanhshrink", {}),
    ("softmax", {"dim": 0}),
    ("softmax", {"dim": -1, "_stacklevel": 3, "dtype": None}),
    ("log_softmax", {"dim": 0}),
    ("log_softmax", {"dim": -1, "_stacklevel": 3, "dtype": None}),
    ("flatten", {}),
    ("flatten", {"start_dim": 1, "end_dim": -1}),
]


@pytest.mark.parametrize(("target", "kwargs"), FUNCTION_CASES)
def test_function(
    compare_function: Callable[..., None], target: str, kwargs: dict[str, Any]
) -> None:
    """A function has the values of the function of PyTorch."""
    x = np.stack([VALUES, VALUES[::-1], 0.1 * VALUES])
    compare_function(target, kwargs, x)


def test_log_sigmoid(compare_function: Callable[..., None]) -> None:
    """`log_sigmoid` of the YAML is `logsigmoid` of PyTorch."""
    compare_function("log_sigmoid", {}, VALUES, torch_name="logsigmoid")
    compare_function("logsigmoid", {}, VALUES)


@pytest.mark.parametrize("target", ["tanh", "sigmoid", "relu", "flatten"])
def test_method(compare_function: Callable[..., None], target: str) -> None:
    """A method of a tensor is the function of the same name."""
    compare_function(target, {}, np.stack([VALUES, VALUES]), op="call_method")


def test_every_function_is_tested() -> None:
    """The functions which are registered are the ones which are compared."""
    tested = {target for target, _ in FUNCTION_CASES}
    tested |= {"log_sigmoid", "logsigmoid", "cat", "concat", "concatenate"}
    assert set(FUNCTIONS) == tested


def test_the_functions_support_both_backends() -> None:
    """Every function is written with the methods of the backend."""
    assert all(f.backends == ALL_BACKENDS for f in FUNCTIONS.values())


def test_the_values_are_finite(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """No function overflows on a large input."""
    for target in ("sigmoid", "softplus", "log_sigmoid", "mish", "silu", "elu"):
        (y,) = forward(function_model(target, {}), {}, VALUES)
        assert np.all(np.isfinite(y)), target


def _two_inputs(node: Node) -> NNModel:
    """Build a network of one node with the inputs `a` and `b`."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0"), Input(input_id="input1")],
        layers=[],
        forward=[
            Node(name="a", op="placeholder", target="a", args=[], kwargs={}),
            Node(name="b", op="placeholder", target="b", args=[], kwargs={}),
            node,
            Node(
                name="output", op="output", target="output", args=[node.name], kwargs={}
            ),
        ],
    )


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cat(forward: Callable[..., tuple[np.ndarray, ...]], dim: int) -> None:
    """`cat` joins the arrays of a list along an axis."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed=1)
    a, b = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
    node = Node(
        name="cat",
        op="call_function",
        target="cat",
        args=[["a", "b"]],
        kwargs={"dim": dim},
    )
    (observed,) = forward(_two_inputs(node), {}, a, b)
    expected = torch.cat([torch.from_numpy(a), torch.from_numpy(b)], dim=dim).numpy()
    np.testing.assert_array_equal(observed, expected)


def test_cat_of_arrays_which_do_not_fit(
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """The error of numpy names the network and the node."""
    node = Node(
        name="cat", op="call_function", target="cat", args=[["a", "b"]], kwargs={}
    )
    with pytest.raises(ValueError, match=r"Network 'net1', node 'cat'"):
        forward(_two_inputs(node), {}, np.zeros((2, 3)), np.zeros((2, 4)))


def test_a_keyword_argument_is_a_literal(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """`approximate="tanh"` is not the node which is named `tanh`."""
    model = function_model("gelu", {"approximate": "tanh"})
    model.forward[0].name = "tanh"
    model.forward[1].args = ["tanh"]
    (y,) = forward(model, {}, VALUES)
    (expected,) = forward(function_model("gelu", {"approximate": "tanh"}), {}, VALUES)
    np.testing.assert_array_equal(y, expected)


def test_a_function_without_an_implementation(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """An unknown function names the network, the node and the function."""
    with pytest.raises(UnsupportedLayerError, match=r"'net1'.*'f'.*'rrelu'") as info:
        forward(function_model("rrelu", {}), {}, VALUES)
    assert (info.value.network, info.value.node, info.value.target) == (
        "net1",
        "f",
        "rrelu",
    )


@pytest.mark.parametrize(
    ("target", "kwargs"),
    [
        ("relu", {"threshold": 1.0}),
        ("softmax", {}),
        ("gelu", {"approximate": "sigmoid"}),
    ],
)
def test_arguments_which_do_not_fit(
    function_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    target: str,
    kwargs: dict[str, Any],
) -> None:
    """An argument the function does not have is an error, it is not dropped."""
    with pytest.raises((UnsupportedLayerError, ValueError), match=r"node 'f'"):
        forward(function_model(target, kwargs), {}, VALUES)
````

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -x tests/sciml/test_functions.py`
Expected: FAIL with `UnsupportedLayerError: Network 'net1', node 'f': 'tanh' is not supported, the function is not implemented`.

- [ ] **Step 3: Write the functions**

Create `src/sbmlsim/sciml/layers/functions.py`:

````python
"""The activation functions and tensor operations, which both backends support.

Every function has the keyword arguments and the defaults of the function of
`torch.nn.functional` (or `torch` for the tensor operations) it is named
after. The functions with a condition are written with `Backend.select`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from sbmlsim.sciml.backend import Backend
from sbmlsim.sciml.layers.core import flatten_array
from sbmlsim.sciml.layers.registry import function

#: the constants of `selu`
SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


def clip(backend: Backend, x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Limit the values to the interval `[low, high]`.

    Args:
        backend: the backend.
        x: the values.
        low: the lower limit.
        high: the upper limit.

    Returns:
        `low` where `x <= low`, `high` where `x > high` and `x` between.
    """
    return backend.select(x, low, low, backend.select(x, high, x, high))


@function("tanh")
def tanh(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `tanh(x)`."""
    return backend.tanh(x)


@function("sigmoid")
def sigmoid(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `1 / (1 + exp(-x))`."""
    return 1.0 / (1.0 + backend.exp(-x))


@function("relu")
def relu(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `max(x, 0)`."""
    return backend.select(x, 0.0, 0.0, x)


@function("relu6")
def relu6(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `min(max(x, 0), 6)`."""
    return clip(backend, x, 0.0, 6.0)


@function("hardtanh")
def hardtanh(
    backend: Backend, x: np.ndarray, min_val: float = -1.0, max_val: float = 1.0
) -> np.ndarray:
    """Evaluate `min(max(x, min_val), max_val)`."""
    return clip(backend, x, min_val, max_val)


@function("hardsigmoid")
def hardsigmoid(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `min(max(x / 6 + 1 / 2, 0), 1)`."""
    return clip(backend, x / 6.0 + 0.5, 0.0, 1.0)


@function("hardswish")
def hardswish(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x * min(max(x + 3, 0), 6) / 6`."""
    return x * clip(backend, x + 3.0, 0.0, 6.0) / 6.0


@function("leaky_relu")
def leaky_relu(
    backend: Backend, x: np.ndarray, negative_slope: float = 0.01
) -> np.ndarray:
    """Evaluate `x` for `x > 0` and `negative_slope * x` otherwise."""
    return backend.select(x, 0.0, negative_slope * x, x)


@function("elu")
def elu(backend: Backend, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Evaluate `x` for `x > 0` and `alpha * (exp(x) - 1)` otherwise."""
    return backend.select(x, 0.0, alpha * (backend.exp(x) - 1.0), x)


@function("celu")
def celu(backend: Backend, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Evaluate `x` for `x > 0` and `alpha * (exp(x / alpha) - 1)` otherwise."""
    return backend.select(x, 0.0, alpha * (backend.exp(x / alpha) - 1.0), x)


@function("selu")
def selu(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `scale * elu(x, alpha)` with the constants of `selu`."""
    return SELU_SCALE * elu(backend, x, alpha=SELU_ALPHA)


@function("gelu")
def gelu(backend: Backend, x: np.ndarray, approximate: str = "none") -> np.ndarray:
    """Evaluate `x * Phi(x)` with the distribution function of the normal.

    Args:
        backend: the backend.
        x: the input.
        approximate: `none` for the error function, `tanh` for the
            approximation `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`.

    Returns:
        The output.

    Raises:
        ValueError: if `approximate` is neither `none` nor `tanh`.
    """
    if approximate == "none":
        return 0.5 * x * (1.0 + backend.erf(x / math.sqrt(2.0)))
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + backend.tanh(inner))
    raise ValueError(f"gelu: approximate '{approximate}' is not 'none' or 'tanh'")


@function("softplus")
def softplus(
    backend: Backend, x: np.ndarray, beta: float = 1.0, threshold: float = 20.0
) -> np.ndarray:
    """Evaluate `log(1 + exp(beta * x)) / beta`, `x` for `beta * x > threshold`."""
    scaled = beta * x
    return backend.select(
        scaled, threshold, backend.log(1.0 + backend.exp(scaled)) / beta, x
    )


@function("log_sigmoid", "logsigmoid")
def log_sigmoid(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `log(1 / (1 + exp(-x)))`.

    The two branches are the same function, written so that the exponential of
    the branch which is chosen does not overflow.
    """
    return backend.select(
        x,
        0.0,
        x - backend.log(1.0 + backend.exp(x)),
        -backend.log(1.0 + backend.exp(-x)),
    )


@function("mish")
def mish(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x * tanh(softplus(x))`."""
    return x * backend.tanh(softplus(backend, x))


@function("silu")
def silu(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x * sigmoid(x)`."""
    return x * sigmoid(backend, x)


@function("softsign")
def softsign(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x / (1 + |x|)`."""
    return x / (1.0 + backend.absolute(x))


@function("tanhshrink")
def tanhshrink(backend: Backend, x: np.ndarray) -> np.ndarray:
    """Evaluate `x - tanh(x)`."""
    return x - backend.tanh(x)


@function("softmax")
def softmax(backend: Backend, x: np.ndarray, dim: int) -> np.ndarray:
    """Evaluate `exp(x) / sum(exp(x))` along the axis `dim`."""
    exponential = backend.exp(x - backend.stabilizer(x, dim))
    return exponential / exponential.sum(axis=dim, keepdims=True)


@function("log_softmax")
def log_softmax(backend: Backend, x: np.ndarray, dim: int) -> np.ndarray:
    """Evaluate `x - log(sum(exp(x)))` along the axis `dim`."""
    shifted = x - backend.stabilizer(x, dim)
    return shifted - backend.log(backend.exp(shifted).sum(axis=dim, keepdims=True))


@function("flatten")
def flatten(
    backend: Backend, x: np.ndarray, start_dim: int = 0, end_dim: int = -1
) -> np.ndarray:
    """Evaluate `torch.flatten`, in row major order."""
    return flatten_array(x, start_dim, end_dim)


@function("cat", "concat", "concatenate")
def cat(backend: Backend, tensors: Sequence[np.ndarray], dim: int = 0) -> np.ndarray:
    """Evaluate `torch.cat`, i.e. join the arrays along the axis `dim`."""
    return np.concatenate(list(tensors), axis=dim)
````

- [ ] **Step 4: Register the functions**

Replace `src/sbmlsim/sciml/layers/__init__.py` with:

````python
"""The layers and functions of the forward pass of a network.

Every layer and every function of the NN YAML is implemented once, against a
backend, and registered under its PyTorch name in `LAYERS` or `FUNCTIONS`
with the backends it supports. Importing this package registers all of them.

| module | content | backends |
| --- | --- | --- |
| `core` | `Linear`, `Bilinear`, `Flatten`, the dropout layers | numpy, sympy |
| `functions` | the activation functions, `flatten`, `cat` | numpy, sympy |
| `convolution` | `Conv1-3d`, `ConvTranspose1-3d` | numpy |
| `pooling` | `MaxPool`, `AvgPool`, `LPPool` and the adaptive pools, `1-3d` | numpy |
| `normalization` | `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm` | numpy |
"""

from sbmlsim.sciml.layers import (
    core,
    functions,
)
from sbmlsim.sciml.layers.registry import (
    FUNCTIONS,
    LAYERS,
    ArraySpec,
    FunctionType,
    LayerType,
)

__all__ = [
    "FUNCTIONS",
    "LAYERS",
    "ArraySpec",
    "FunctionType",
    "LayerType",
    "core",
    "functions",
]
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `97 passed`.

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 7: Commit**

```bash
git add src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the activation functions and the tensor operations"
```

---

### Task 4: `Network`, its files and the ids of its elements

**Files:**
- Create: `src/sbmlsim/sciml/network.py`
- Modify: `src/sbmlsim/sciml/__init__.py` (export `Network`, `NetworkParameters`)
- Test: `tests/sciml/test_network.py`

**Interfaces:**
- Consumes: `evaluate`, `CALL_MODULE`, `CALL_FUNCTION`, `CALL_METHOD` of `sbmlsim.sciml.interpreter`; `LAYERS`, `FUNCTIONS`, `ArraySpec` of `sbmlsim.sciml.layers`; `NumpyBackend`, `BackendKind`, `ALL_BACKENDS`; `NetworkImportError`, `UnsupportedLayerError`; `NNModelStandard.load_data(filename: str) -> BaseModel`, `NNModelStandard.save_data(data=, filename=)`, `ArrayDataStandard.load_data(filename: str) -> BaseModel`, `ArrayData` (`.metadata.pytorch_format: bool`, `.parameters: dict[str, dict[str, dict[str, np.ndarray]]]`, network id -> layer id -> array name) of `petab_sciml`.
- Produces: `sbmlsim.sciml.Network` (dataclass `sid`, `model`, `parameters`) with `from_files(yaml_path: Path, array_path: Path | None = None, sid: str | None = None) -> Network`, `read_arrays(array_path: Path) -> NetworkParameters`, `array_specs() -> dict[str, dict[str, ArraySpec]]`, `used_layers() -> list[str]`, `backends() -> frozenset[BackendKind]`, `check_arrays(parameters, complete: bool = True) -> None`, `forward(*inputs: np.ndarray, parameters: NetworkParameters | None = None) -> tuple[np.ndarray, ...]`, `parameter_ids() -> dict[str, tuple[str, str, tuple[int, ...]]]`, `with_values(values: Mapping[str, float]) -> NetworkParameters`; `sbmlsim.sciml.network.NetworkParameters`, `element_id(network, layer, array, index) -> str`, `copy_parameters(parameters) -> NetworkParameters`, `load_array_data(path: Path) -> ArrayData`.

The rules of this task:

- The HDF5 file of the arrays has the datasets `parameters/<net>/<layer>/<array>` and `metadata/pytorch_format`. `ArrayDataStandard.load_data` returns the arrays in double precision.
- `sid` replaces the `nn_model_id` of the YAML, because a problem names its networks in its own YAML. The arrays are read from the group of the `sid`.
- An array of the size 0 is an array the file does not provide.
- `from_files` checks the arrays it reads (unknown layer, unknown array, shape, finite values) and does not demand that they are complete, because the parameter table of a problem may set the rest (task 8). `forward` demands the required arrays of the layers the forward pass calls. A layer which is defined and never called needs no arrays (cases 040 and 041 of the suite).
- The id of an element is `<net>__<layer>__<array>__<index>` with `_` between the axes of the index. Every character which is not in `[A-Za-z0-9_]` is replaced by `_`. `parameter_ids` lists the arrays with `trainable=True` of all layers, from the architecture and not from the values.
- `with_values` and `forward(parameters=...)` never change `Network.parameters`.

- [ ] **Step 1: Write the failing test**

Create `tests/sciml/test_network.py`:

````python
"""Tests of a network: its files, its ids and its forward pass."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, NNModelStandard, Node

from sbmlsim.sciml import (
    BackendKind,
    Network,
    NetworkImportError,
    UnsupportedLayerError,
)
from sbmlsim.sciml.network import element_id


def _node(name: str, op: str, target: str, args: list) -> Node:
    return Node(name=name, op=op, target=target, args=args, kwargs={})


def _model(layer_type: str = "Linear") -> NNModel:
    """Build `layer2(tanh(layer1(x)))` with 2 inputs, 3 hidden units, 1 output."""
    return NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type=layer_type,
                args={"in_features": 2, "out_features": 3, "bias": True},
            ),
            Layer(
                layer_id="layer2",
                layer_type="Linear",
                args={"in_features": 3, "out_features": 1, "bias": False},
            ),
        ],
        forward=[
            _node("net_input", "placeholder", "net_input", []),
            _node("layer1", "call_module", "layer1", ["net_input"]),
            _node("tanh", "call_method", "tanh", ["layer1"]),
            _node("layer2", "call_module", "layer2", ["tanh"]),
            _node("output", "output", "output", ["layer2"]),
        ],
    )


def _parameters() -> dict[str, dict[str, np.ndarray]]:
    return {
        "layer1": {
            "weight": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "bias": np.array([0.1, 0.2, 0.3]),
        },
        "layer2": {"weight": np.array([[1.0, -1.0, 0.5]])},
    }


def _write(path: Path, parameters: dict, pytorch_format: bool = True) -> Path:
    """Write the YAML and the array file of the network into a directory."""
    NNModelStandard.save_data(data=_model(), filename=str(path / "net1.yaml"))
    with h5py.File(path / "net1_ps.hdf5", "w") as f:
        f.create_group("metadata")["pytorch_format"] = pytorch_format
        for layer, arrays in parameters.items():
            for name, array in arrays.items():
                f[f"parameters/net1/{layer}/{name}"] = array
    return path


def test_the_forward_pass() -> None:
    """The network is evaluated by hand."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    x = np.array([0.5, -0.25])
    hidden = np.tanh(_parameters()["layer1"]["weight"] @ x + [0.1, 0.2, 0.3])
    (y,) = network.forward(x)
    np.testing.assert_allclose(y, [hidden[0] - hidden[1] + 0.5 * hidden[2]])
    assert y.dtype == float


def test_a_network_is_read_from_its_files(tmp_path: Path) -> None:
    """The YAML and the HDF5 file give the network."""
    _write(tmp_path, _parameters())
    network = Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")

    assert network.sid == "net1"
    assert list(network.parameters) == ["layer1", "layer2"]
    np.testing.assert_array_equal(
        network.parameters["layer1"]["weight"], _parameters()["layer1"]["weight"]
    )
    expected = Network(sid="net1", model=_model(), parameters=_parameters())
    x = np.array([[0.5, -0.25], [1.0, 2.0]])
    np.testing.assert_allclose(network.forward(x)[0], expected.forward(x)[0])


def test_the_id_of_a_problem_replaces_the_id_of_the_yaml(tmp_path: Path) -> None:
    """A problem names its networks, the arrays are read under that name."""
    NNModelStandard.save_data(data=_model(), filename=str(tmp_path / "net.yaml"))
    with h5py.File(tmp_path / "ps.hdf5", "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        f["parameters/net7/layer2/weight"] = np.ones((1, 3))
    network = Network.from_files(
        tmp_path / "net.yaml", tmp_path / "ps.hdf5", sid="net7"
    )
    assert network.sid == network.model.nn_model_id == "net7"
    assert next(iter(network.parameter_ids())) == "net7__layer1__weight__0_0"
    assert list(network.parameters) == ["layer2"]


def test_the_column_major_layout_is_permuted(tmp_path: Path) -> None:
    """Arrays which are not in the PyTorch layout have their axes reversed."""
    stored = {
        layer: {name: array.T for name, array in arrays.items()}
        for layer, arrays in _parameters().items()
    }
    _write(tmp_path, stored, pytorch_format=False)
    network = Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")
    np.testing.assert_array_equal(
        network.parameters["layer1"]["weight"], _parameters()["layer1"]["weight"]
    )
    assert network.parameters["layer2"]["weight"].shape == (1, 3)


def test_an_empty_array_is_an_array_without_values(tmp_path: Path) -> None:
    """The file of a problem leaves out the arrays the problem sets."""
    parameters = _parameters()
    parameters["layer1"]["weight"] = np.array([])
    _write(tmp_path, parameters)
    network = Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")
    assert list(network.parameters["layer1"]) == ["bias"]
    with pytest.raises(NetworkImportError, match=r"'layer1'.*'weight' has no values"):
        network.forward(np.zeros(2))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"layer1": {"weight": np.ones((2, 3))}}, r"'weight' has the shape \(2, 3\)"),
        ({"layer1": {"gain": np.ones(3)}}, "'gain' is not an array of the layer"),
        ({"layer9": {"weight": np.ones(3)}}, "the layer 'layer9'"),
        ({"layer1": {"bias": np.array([0.0, np.nan, 0.0])}}, "not finite"),
        ({"layer1": {"bias": np.array([0.0, np.inf, 0.0])}}, "not finite"),
    ],
)
def test_arrays_which_do_not_fit(tmp_path: Path, change: dict, message: str) -> None:
    """An array of a file which does not fit the architecture is an error."""
    parameters = _parameters()
    for layer, arrays in change.items():
        parameters.setdefault(layer, {}).update(arrays)
    _write(tmp_path, parameters)
    with pytest.raises(NetworkImportError, match=message):
        Network.from_files(tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5")


def test_files_which_do_not_exist(tmp_path: Path) -> None:
    """A missing file is an error of the import which names the file."""
    with pytest.raises(NetworkImportError, match=r"missing\.yaml"):
        Network.from_files(tmp_path / "missing.yaml")
    _write(tmp_path, _parameters())
    with pytest.raises(NetworkImportError, match=r"missing\.hdf5"):
        Network.from_files(tmp_path / "net1.yaml", tmp_path / "missing.hdf5")


def test_an_array_file_of_another_network(tmp_path: Path) -> None:
    """The file must hold the arrays of the network."""
    _write(tmp_path, _parameters())
    with pytest.raises(NetworkImportError, match=r"no parameters of the network"):
        Network.from_files(
            tmp_path / "net1.yaml", tmp_path / "net1_ps.hdf5", sid="net2"
        )


def test_the_ids_of_the_elements() -> None:
    """Every element has an id with its PyTorch index."""
    network = Network(sid="net1", model=_model())
    ids = network.parameter_ids()

    assert len(ids) == 6 + 3 + 3
    assert list(ids)[:3] == [
        "net1__layer1__weight__0_0",
        "net1__layer1__weight__0_1",
        "net1__layer1__weight__1_0",
    ]
    assert ids["net1__layer1__weight__2_1"] == ("layer1", "weight", (2, 1))
    assert ids["net1__layer1__bias__2"] == ("layer1", "bias", (2,))
    assert ids["net1__layer2__weight__0_2"] == ("layer2", "weight", (0, 2))
    assert "net1__layer2__bias__0" not in ids


def test_an_id_is_an_sid() -> None:
    """The dot of a nested layer is not part of an id."""
    assert (
        element_id("net1", "block.0", "weight", (1, 2)) == "net1__block_0__weight__1_2"
    )


def test_two_elements_with_one_id() -> None:
    """Layers whose ids differ only in a replaced character are an error."""
    model = _model()
    model.layers[0].layer_id = "block.0"
    model.layers[1] = model.layers[0].model_copy(update={"layer_id": "block_0"})
    with pytest.raises(NetworkImportError, match=r"have the id 'net1__block_0__"):
        Network(sid="net1", model=model).parameter_ids()


def test_values_replace_elements() -> None:
    """`with_values` returns a copy, the network keeps its nominal values."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    parameters = network.with_values(
        {"net1__layer1__weight__2_1": 60.0, "net1__layer2__weight__0_0": -7.0}
    )

    assert parameters["layer1"]["weight"][2, 1] == 60.0
    assert parameters["layer2"]["weight"][0, 0] == -7.0
    assert network.parameters["layer1"]["weight"][2, 1] == 6.0
    assert parameters["layer1"]["bias"] is not network.parameters["layer1"]["bias"]

    x = np.array([0.5, -0.25])
    assert network.forward(x, parameters=parameters)[0] != network.forward(x)[0]
    np.testing.assert_array_equal(
        network.forward(x, parameters=network.with_values({}))[0], network.forward(x)[0]
    )


def test_a_value_of_an_unknown_element() -> None:
    """An id which is not an element of the network is an error."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    with pytest.raises(KeyError, match=r"net1__layer1__weight__3_0"):
        network.with_values({"net1__layer1__weight__3_0": 1.0})


def test_a_value_of_an_array_without_values() -> None:
    """An element of an array without nominal values cannot be set."""
    network = Network(sid="net1", model=_model())
    with pytest.raises(NetworkImportError, match=r"'weight' has no values"):
        network.with_values({"net1__layer1__weight__0_0": 1.0})


def test_a_layer_which_is_not_called_needs_no_arrays() -> None:
    """Only the layers of the forward pass are evaluated."""
    model = _model()
    model.forward = [
        _node("net_input", "placeholder", "net_input", []),
        _node("tanh", "call_method", "tanh", ["net_input"]),
        _node("output", "output", "output", ["tanh"]),
    ]
    network = Network(sid="net1", model=model)
    assert network.used_layers() == []
    np.testing.assert_allclose(network.forward(np.array([0.5]))[0], np.tanh([0.5]))


def test_the_backends_of_a_network() -> None:
    """`Linear` and `tanh` are evaluated by both backends."""
    assert Network(sid="net1", model=_model()).backends() == frozenset(BackendKind)


def test_a_layer_without_an_implementation() -> None:
    """An unknown layer names the network, the node and the type."""
    network = Network(sid="net1", model=_model(layer_type="LSTM"))
    with pytest.raises(UnsupportedLayerError, match=r"'net1'.*'layer1'.*'LSTM'"):
        network.forward(np.zeros(2))
    with pytest.raises(UnsupportedLayerError):
        network.parameter_ids()


def test_the_number_of_inputs() -> None:
    """The inputs are the placeholders of the forward pass."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    with pytest.raises(ValueError, match=r"2 inputs were given for the 1 inputs"):
        network.forward(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match=r"0 inputs were given"):
        network.forward()


def test_an_input_of_the_wrong_size() -> None:
    """The error of numpy names the network and the node."""
    network = Network(sid="net1", model=_model(), parameters=_parameters())
    with pytest.raises(ValueError, match=r"Network 'net1', node 'layer1'"):
        network.forward(np.zeros(3))


def test_several_outputs() -> None:
    """An output node with a list returns one array per entry."""
    model = _model()
    model.forward[-1] = _node("output", "output", "output", [["layer1", "layer2"]])
    network = Network(sid="net1", model=model, parameters=_parameters())
    hidden, y = network.forward(np.array([0.5, -0.25]))
    assert hidden.shape == (3,)
    assert y.shape == (1,)
````

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -x tests/sciml/test_network.py`
Expected: FAIL at collection with `ImportError: cannot import name 'Network' from 'sbmlsim.sciml'`.

- [ ] **Step 3: Write the network**

Create `src/sbmlsim/sciml/network.py`:

````python
"""The architecture and the arrays of a neural network.

`Network` holds the architecture of a network as the `NNModel` of
`petab_sciml`, i.e. the content of the NN YAML, and the arrays of its layers
in the PyTorch layout. `Network.forward` evaluates it with numpy.

Every element of an array has an id, `<net>__<layer>__<array>__<index>` with
the PyTorch index of the element and `_` between the axes, e.g.
`net1__layer1__weight__0_1`. The id is a valid SBML `SId`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from petab_sciml import ArrayData, ArrayDataStandard, NNModel, NNModelStandard

from sbmlsim.sciml.backend import ALL_BACKENDS, BackendKind, NumpyBackend
from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError
from sbmlsim.sciml.interpreter import (
    CALL_FUNCTION,
    CALL_METHOD,
    CALL_MODULE,
    evaluate,
)
from sbmlsim.sciml.layers import FUNCTIONS, LAYERS, ArraySpec

logger = logging.getLogger(__name__)

#: the arrays of a network: layer id -> array name -> values
NetworkParameters = dict[str, dict[str, np.ndarray]]

#: separator of the parts of an id
ID_SEPARATOR = "__"

#: separator of the axes of the index of an id
INDEX_SEPARATOR = "_"


#: the characters of an id which are not part of an SBML `SId`
_NOT_SID = re.compile(r"[^A-Za-z0-9_]")


def element_id(network: str, layer: str, array: str, index: tuple[int, ...]) -> str:
    """Get the id of an element of an array.

    Args:
        network: id of the network.
        layer: id of the layer.
        array: name of the array, e.g. `weight`.
        index: the PyTorch index of the element.

    Returns:
        The id, e.g. `net1__layer1__weight__0_1`. A character which is not
        part of an SBML `SId` is replaced by `_`, e.g. the dot in the id of a
        layer of a nested module.
    """
    sid = ID_SEPARATOR.join(
        [network, layer, array, INDEX_SEPARATOR.join(str(i) for i in index)]
    )
    return _NOT_SID.sub("_", sid)


def copy_parameters(
    parameters: Mapping[str, Mapping[str, np.ndarray]],
) -> NetworkParameters:
    """Copy the arrays of a network.

    Args:
        parameters: the arrays, layer id -> array name -> values.

    Returns:
        A copy which shares no array with the original.
    """
    return {
        layer: {name: np.array(array, dtype=float) for name, array in arrays.items()}
        for layer, arrays in parameters.items()
    }


def load_array_data(path: Path) -> ArrayData:
    """Read an array file.

    Args:
        path: the HDF5 file.

    Returns:
        The arrays of the file.

    Raises:
        NetworkImportError: if the file does not exist or is not an array
            file.
    """
    if not path.is_file():
        raise NetworkImportError(f"The array file '{path}' does not exist")
    data = ArrayDataStandard.load_data(str(path))
    if not isinstance(data, ArrayData):
        raise NetworkImportError(f"'{path}' is not an array file")
    return data


@dataclass
class Network:
    """The architecture and the arrays of one network.

    Attributes:
        sid: id of the network.
        model: the architecture, i.e. the content of the NN YAML.
        parameters: the nominal values of the arrays in the PyTorch layout,
            layer id -> array name -> values.
    """

    sid: str
    model: NNModel
    parameters: NetworkParameters = field(default_factory=dict)

    @classmethod
    def from_files(
        cls, yaml_path: Path, array_path: Path | None = None, sid: str | None = None
    ) -> Network:
        """Read a network from its NN YAML and its array file.

        The arrays of a file with `metadata/pytorch_format` false are stored
        in the column major layout, i.e. with the axes in the reverse order,
        and are permuted into the PyTorch layout.

        Args:
            yaml_path: the NN YAML.
            array_path: the HDF5 file with the arrays of the network. Without
                it the network has no values, which a problem sets.
            sid: id of the network, the `nn_model_id` of the YAML when `None`.
                The arrays are read from the group of this id.

        Returns:
            The network.

        Raises:
            NetworkImportError: if a file does not exist, if the array file
                has no arrays for the network, or if an array does not belong
                to a layer or does not have the shape of the layer.
            UnsupportedLayerError: if the network has a layer without an
                implementation.
        """
        if not yaml_path.is_file():
            raise NetworkImportError(f"The NN YAML '{yaml_path}' does not exist")
        model = NNModelStandard.load_data(str(yaml_path))
        if not isinstance(model, NNModel):
            raise NetworkImportError(f"'{yaml_path}' is not a NN YAML")
        if sid is not None:
            model = model.model_copy(update={"nn_model_id": sid})
        network = cls(sid=model.nn_model_id, model=model)
        if array_path is not None:
            network.parameters = network.read_arrays(array_path)
        network.check_arrays(network.parameters, complete=False)
        return network

    def read_arrays(self, array_path: Path) -> NetworkParameters:
        """Read the arrays of the network from an array file.

        An empty array is an array the file does not provide, which is how a
        file leaves out the arrays a problem sets.

        Args:
            array_path: the HDF5 file.

        Returns:
            The arrays in the PyTorch layout.

        Raises:
            NetworkImportError: if the file does not exist or has no arrays
                for the network.
        """
        data = load_array_data(array_path)
        if self.sid not in data.parameters:
            raise NetworkImportError(
                f"Network '{self.sid}': the array file '{array_path}' has no "
                f"parameters of the network, it has {sorted(data.parameters)}"
            )
        pytorch_format = data.metadata.pytorch_format
        if not pytorch_format:
            logger.info(
                "Network '%s': the arrays of '%s' are permuted into the PyTorch layout",
                self.sid,
                array_path,
            )
        parameters: NetworkParameters = {}
        for layer, arrays in data.parameters[self.sid].items():
            for name, values in arrays.items():
                array = np.asarray(values, dtype=float)
                if array.size == 0:
                    continue
                if not pytorch_format:
                    array = np.ascontiguousarray(array.T)
                parameters.setdefault(layer, {})[name] = array
        return parameters

    def array_specs(self) -> dict[str, dict[str, ArraySpec]]:
        """Get the arrays of every layer of the network.

        Returns:
            The arrays of the layers, layer id -> array name -> shape and
            kind, in the order of the layers.

        Raises:
            UnsupportedLayerError: if a layer has no implementation.
        """
        specs: dict[str, dict[str, ArraySpec]] = {}
        for layer in self.model.layers:
            layer_type = LAYERS.get(layer.layer_type)
            if layer_type is None:
                raise UnsupportedLayerError(
                    self.sid,
                    layer.layer_id,
                    layer.layer_type,
                    "the layer is not implemented",
                )
            specs[layer.layer_id] = layer_type.arrays(layer.args or {})
        return specs

    def used_layers(self) -> list[str]:
        """Get the ids of the layers the forward pass calls, in its order."""
        used: list[str] = []
        for node in self.model.forward:
            if node.op == CALL_MODULE and node.target not in used:
                used.append(node.target)
        return used

    def backends(self) -> frozenset[BackendKind]:
        """Get the backends which evaluate every node of the forward pass.

        Returns:
            The backends all layers and functions of the forward pass support.

        Raises:
            UnsupportedLayerError: if a node has no implementation.
        """
        layers = {layer.layer_id: layer for layer in self.model.layers}
        backends = ALL_BACKENDS
        for node in self.model.forward:
            if node.op == CALL_MODULE:
                name = layers[node.target].layer_type
                supported = LAYERS.get(name)
            elif node.op in (CALL_FUNCTION, CALL_METHOD):
                name = node.target
                supported = FUNCTIONS.get(name)
            else:
                continue
            if supported is None:
                raise UnsupportedLayerError(
                    self.sid, node.name, name, "it is not implemented"
                )
            backends = backends & supported.backends
        return backends

    def check_arrays(
        self, parameters: Mapping[str, Mapping[str, np.ndarray]], complete: bool = True
    ) -> None:
        """Check arrays against the architecture.

        Args:
            parameters: the arrays, layer id -> array name -> values.
            complete: whether every required array of a layer of the forward
                pass must be there.

        Raises:
            NetworkImportError: if an array does not belong to a layer of the
                network, does not have the shape of the layer, holds a value
                which is not finite or, with `complete`, is missing.
        """
        specs = self.array_specs()
        for layer, arrays in parameters.items():
            if layer not in specs:
                raise NetworkImportError(
                    f"Network '{self.sid}': arrays are given for the layer "
                    f"'{layer}', the layers are {sorted(specs)}"
                )
            for name, array in arrays.items():
                if name not in specs[layer]:
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' is not an array of the layer, the arrays "
                        f"are {sorted(specs[layer])}"
                    )
                shape = specs[layer][name].shape
                if np.shape(array) != shape:
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' has the shape {np.shape(array)}, the "
                        f"layer needs {shape}"
                    )
                if not np.all(np.isfinite(array)):
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' holds values which are not finite"
                    )
        if not complete:
            return
        for layer in self.used_layers():
            for name, spec in specs[layer].items():
                if spec.required and name not in parameters.get(layer, {}):
                    raise NetworkImportError(
                        f"Network '{self.sid}', layer '{layer}': the array "
                        f"'{name}' has no values. A network is not "
                        f"initialized with random values, the array file or "
                        f"the problem must provide them"
                    )

    def forward(
        self, *inputs: np.ndarray, parameters: NetworkParameters | None = None
    ) -> tuple[np.ndarray, ...]:
        """Evaluate the network with numpy, in evaluation mode.

        Args:
            *inputs: the inputs in the PyTorch layout, one per input of the
                forward pass.
            parameters: the arrays the network is evaluated with, the nominal
                values when `None`.

        Returns:
            The outputs of the network.

        Raises:
            NetworkImportError: if an array of a layer is missing or has the
                wrong shape.
            UnsupportedLayerError: if a layer or function has no
                implementation.
            ValueError: if the inputs do not fit the network.
        """
        arrays = self.parameters if parameters is None else parameters
        self.check_arrays(arrays)
        return evaluate(self.model, arrays, inputs, NumpyBackend())

    def parameter_ids(self) -> dict[str, tuple[str, str, tuple[int, ...]]]:
        """Get the ids of the elements of the arrays which are parameters.

        The ids follow from the architecture, so they exist for an array
        without values as well. The running statistics of a normalization
        layer are not parameters.

        Returns:
            id of the element -> layer id, array name and PyTorch index, in
            the order of the layers, the arrays and the row major order of the
            elements.

        Raises:
            NetworkImportError: if two elements have the same id, which the
                ids of two layers such as `block.0` and `block_0` cause.
        """
        ids: dict[str, tuple[str, str, tuple[int, ...]]] = {}
        for layer, specs in self.array_specs().items():
            for name, spec in specs.items():
                if not spec.trainable:
                    continue
                for index in np.ndindex(spec.shape):
                    sid = element_id(self.sid, layer, name, index)
                    if sid in ids:
                        raise NetworkImportError(
                            f"Network '{self.sid}': the elements {ids[sid]} "
                            f"and {(layer, name, index)} have the id '{sid}'"
                        )
                    ids[sid] = (layer, name, index)
        return ids

    def with_values(self, values: Mapping[str, float]) -> NetworkParameters:
        """Get the arrays with the values of some elements replaced.

        Args:
            values: id of the element -> value.

        Returns:
            A copy of the nominal values with the elements replaced. The
            network is not changed.

        Raises:
            KeyError: if an id is not the id of an element of the network.
            NetworkImportError: if an element of an array without nominal
                values is set.
        """
        ids = self.parameter_ids()
        parameters = copy_parameters(self.parameters)
        for sid, value in values.items():
            if sid not in ids:
                raise KeyError(
                    f"Network '{self.sid}': '{sid}' is not the id of an element"
                )
            layer, name, index = ids[sid]
            if name not in parameters.get(layer, {}):
                raise NetworkImportError(
                    f"Network '{self.sid}', layer '{layer}': the array "
                    f"'{name}' has no values, '{sid}' cannot be set"
                )
            parameters[layer][name][index] = value
        return parameters
````

- [ ] **Step 4: Export the network**

Replace `src/sbmlsim/sciml/__init__.py` with:

````python
"""Neural networks of hybrid problems.

A hybrid problem combines a mechanistic model in SBML with neural networks,
which is what [PEtab SciML](https://github.com/PEtab-dev/petab_sciml)
describes. This package is the native half of the support: `Network` is the
architecture and the arrays of one network and `Network.forward` evaluates it
with numpy. The package knows nothing of PEtab, `sbmlsim.fit.petab_v2`
translates.

The architecture is read and written with `petab_sciml`, which is not a
dependency of `sbmlsim` but the `sciml` extra:

```bash
pip install sbmlsim[sciml]
```
"""

try:
    import petab_sciml  # noqa: F401
except ModuleNotFoundError as err:
    raise ImportError(
        "sbmlsim.sciml requires the package 'petab_sciml', which is installed "
        "with the extra 'sciml': pip install sbmlsim[sciml]"
    ) from err

from sbmlsim.sciml.backend import Backend, BackendKind, NumpyBackend
from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError
from sbmlsim.sciml.network import Network, NetworkParameters

__all__ = [
    "Backend",
    "BackendKind",
    "Network",
    "NetworkImportError",
    "NetworkParameters",
    "NumpyBackend",
    "UnsupportedLayerError",
]
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `121 passed`.

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 7: Commit**

```bash
git add src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the network with its files, its forward pass and the ids of its elements"
```

---

### Task 5: The convolution and the transposed convolution

**Files:**
- Create: `src/sbmlsim/sciml/layers/windows.py`
- Create: `src/sbmlsim/sciml/layers/convolution.py`
- Modify: `src/sbmlsim/sciml/layers/__init__.py` (import `convolution`)
- Test: `tests/sciml/test_layers_convolution.py`

**Interfaces:**
- Consumes: `layer_nd(template, arrays, backends)`, `ArraySpec`, `as_tuple(value, n)` of `sbmlsim.sciml.layers.registry`; `NUMPY_ONLY`, `Backend`; `Network`, `NetworkImportError`, `BackendKind` of `sbmlsim.sciml`; the fixtures `compare_layer`, `layer_model`, `forward`.
- Produces: the entries `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` of `LAYERS`, numpy only. `sbmlsim.sciml.layers.windows`: `add_batch(x, n, name) -> tuple[np.ndarray, bool]`, `pad_spatial(x, before, after, mode="constant", value=0.0) -> np.ndarray` (a negative padding removes values), `windows(x, kernel_size, stride, dilation) -> np.ndarray` of shape `(N, C, *output, *kernel_size)`, `output_size(size, kernel_size, stride, padding, dilation, ceil_mode) -> int`. Task 6 uses all four.

The semantics, with the argument names and defaults of PyTorch. `n` is the number of spatial dimensions, an argument per dimension is an integer (the same for every dimension) or a list of `n` integers. An input is `(N, C_in, *spatial)` or, without a batch, `(C_in, *spatial)`, and the output has a batch axis exactly when the input has one.

| layer | arguments | arrays | output size per spatial axis |
| --- | --- | --- | --- |
| `Conv{n}d` | `in_channels`, `out_channels`, `kernel_size`, `stride=1`, `padding=0` (an integer, a list, `"valid"` or `"same"`), `dilation=1`, `groups=1`, `bias=True`, `padding_mode="zeros"` (`"reflect"`, `"replicate"`, `"circular"`) | `weight (out_channels, in_channels / groups, *kernel_size)`, `bias (out_channels,)` | `floor((L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1` |
| `ConvTranspose{n}d` | `in_channels`, `out_channels`, `kernel_size`, `stride=1`, `padding=0`, `output_padding=0`, `groups=1`, `bias=True`, `dilation=1`, `padding_mode="zeros"` (nothing else) | `weight (in_channels, out_channels / groups, *kernel_size)`, `bias (out_channels,)` | `(L - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1` |

The algorithms:

- **Convolution.** It is a cross correlation, the kernel is not flipped. Pad the spatial axes (`numpy.pad` with the mode `constant`, `reflect`, `edge` for `replicate`, `wrap` for `circular`). `"same"` needs a stride of 1 and pads `total = dilation * (kernel_size - 1)` per axis, `total // 2` in front and the rest behind. Take the windows of the extent `dilation * (kernel_size - 1) + 1` with `sliding_window_view`, keep every `stride`-th window and every `dilation`-th element of a window. For every group `g` the output is `tensordot` of the windows of the input channels of the group with the weights of the output channels of the group, over the channel axis and the kernel axes. Add the bias per output channel.
- **Transposed convolution.** Write the input into an array of zeros of the size `(L - 1) * stride + 1` at every `stride`-th position. Pad it with `dilation * (kernel_size - 1) - padding` in front and the same plus `output_padding` behind, a negative padding removes values. The kernel of the convolution is the weight with the two channel axes swapped per group and the spatial axes flipped. Then it is the convolution above with the stride 1 and the dilation of the layer.
- The YAML of `petab_sciml` writes `bias` only for `Conv2d`, so a missing `bias` is `True`.

- [ ] **Step 1: Write the failing test**

Create `tests/sciml/test_layers_convolution.py`:

````python
"""The convolution and transposed convolution layers against PyTorch."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml import BackendKind, Network, NetworkImportError
from sbmlsim.sciml.backend import NUMPY_ONLY
from sbmlsim.sciml.layers import LAYERS

CONV_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    # the arguments as `petab_sciml` writes them, without a batch
    (
        "Conv1d",
        {
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": [5],
            "stride": [1],
            "padding": [0],
            "dilation": [1],
            "groups": 1,
            "padding_mode": "zeros",
        },
        (1, 20),
    ),
    ("Conv1d", {"in_channels": 2, "out_channels": 3, "kernel_size": 3}, (4, 2, 11)),
    (
        "Conv1d",
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 2,
            "padding": 2,
            "dilation": 2,
            "groups": 2,
            "bias": False,
        },
        (3, 4, 17),
    ),
    (
        "Conv2d",
        {"in_channels": 2, "out_channels": 3, "kernel_size": [5, 2], "bias": True},
        (2, 9, 8),
    ),
    (
        "Conv2d",
        {
            "in_channels": 4,
            "out_channels": 4,
            "kernel_size": [3, 2],
            "stride": [2, 1],
            "padding": [1, 2],
            "dilation": [1, 2],
            "groups": 4,
        },
        (2, 4, 9, 8),
    ),
    (
        "Conv2d",
        {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "padding": "same"},
        (1, 2, 6, 7),
    ),
    (
        "Conv2d",
        {
            "in_channels": 2,
            "out_channels": 3,
            "kernel_size": [4, 3],
            "padding": "same",
            "dilation": [1, 2],
        },
        (1, 2, 6, 7),
    ),
    (
        "Conv2d",
        {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "padding": "valid"},
        (1, 2, 6, 7),
    ),
    (
        "Conv3d",
        {"in_channels": 2, "out_channels": 1, "kernel_size": [5, 4, 3]},
        (2, 7, 6, 5),
    ),
    (
        "Conv3d",
        {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 2,
            "stride": [1, 2, 3],
            "padding": 1,
            "groups": 2,
        },
        (2, 2, 5, 6, 7),
    ),
]

PADDING_MODE_CASES = [
    (mode, padding)
    for mode in ("zeros", "reflect", "replicate", "circular")
    for padding in ([1, 2], 2)
]

CONV_TRANSPOSE_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    (
        "ConvTranspose1d",
        {
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": [5],
            "stride": [1],
            "padding": [0],
            "dilation": [1],
            "groups": 1,
            "padding_mode": "zeros",
            "output_padding": [0],
        },
        (1, 20),
    ),
    (
        "ConvTranspose1d",
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 3,
            "padding": 2,
            "output_padding": 2,
            "dilation": 2,
            "groups": 2,
            "bias": False,
        },
        (3, 4, 7),
    ),
    (
        "ConvTranspose2d",
        {"in_channels": 2, "out_channels": 1, "kernel_size": [5, 2]},
        (2, 6, 5),
    ),
    (
        "ConvTranspose2d",
        {
            "in_channels": 4,
            "out_channels": 2,
            "kernel_size": [3, 2],
            "stride": [2, 3],
            "padding": [1, 0],
            "output_padding": [1, 2],
            "dilation": [2, 1],
            "groups": 2,
        },
        (2, 4, 5, 4),
    ),
    (
        # the padding is larger than the extent of the kernel, the input is cropped
        "ConvTranspose2d",
        {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": 2,
            "stride": 2,
            "padding": 3,
        },
        (1, 1, 6, 6),
    ),
    (
        "ConvTranspose3d",
        {"in_channels": 2, "out_channels": 1, "kernel_size": [5, 4, 3]},
        (2, 4, 3, 2),
    ),
    (
        "ConvTranspose3d",
        {
            "in_channels": 2,
            "out_channels": 2,
            "kernel_size": 2,
            "stride": [1, 2, 3],
            "padding": [0, 1, 1],
            "output_padding": [0, 1, 2],
        },
        (2, 2, 3, 3, 3),
    ),
]


# PyTorch warns that `same` with an even kernel pads a copy of the input
@pytest.mark.filterwarnings("ignore:Using padding='same':UserWarning")
@pytest.mark.parametrize(("layer_type", "args", "shape"), CONV_CASES)
def test_conv(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A convolution has the values of PyTorch."""
    compare_layer(layer_type, args, shape)


@pytest.mark.parametrize(("padding_mode", "padding"), PADDING_MODE_CASES)
def test_conv_padding_mode(
    compare_layer: Callable[..., None], padding_mode: str, padding: Any
) -> None:
    """The padding modes are the modes of `numpy.pad`."""
    args = {
        "in_channels": 2,
        "out_channels": 3,
        "kernel_size": 3,
        "padding": padding,
        "padding_mode": padding_mode,
    }
    compare_layer("Conv2d", args, (2, 2, 6, 7))


@pytest.mark.parametrize(("layer_type", "args", "shape"), CONV_TRANSPOSE_CASES)
def test_conv_transpose(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A transposed convolution has the values of PyTorch."""
    compare_layer(layer_type, args, shape)


def test_the_shape_of_the_weight() -> None:
    """The weight of a transposed convolution has the input channels first."""
    args = {"in_channels": 4, "out_channels": 6, "kernel_size": [3, 2], "groups": 2}
    assert LAYERS["Conv2d"].arrays(args)["weight"].shape == (6, 2, 3, 2)
    assert LAYERS["ConvTranspose2d"].arrays(args)["weight"].shape == (4, 3, 3, 2)
    assert LAYERS["Conv2d"].arrays(args)["bias"].shape == (6,)
    assert "bias" not in LAYERS["Conv2d"].arrays({**args, "bias": False})


def test_the_convolutions_are_numpy_only() -> None:
    """A convolution is not part of a compiled network."""
    for n in (1, 2, 3):
        assert LAYERS[f"Conv{n}d"].backends == NUMPY_ONLY
        assert LAYERS[f"ConvTranspose{n}d"].backends == NUMPY_ONLY


def test_a_network_with_a_convolution_is_numpy_only(
    layer_model: Callable[..., NNModel],
) -> None:
    """The backends of a network are the backends of all its nodes."""
    model = layer_model(
        "Conv1d", {"in_channels": 1, "out_channels": 1, "kernel_size": 2}
    )
    assert Network(sid="net1", model=model).backends() == {BackendKind.NUMPY}


CONV2D = {"in_channels": 1, "out_channels": 1, "kernel_size": 3}
ARRAYS = {"layer1": {"weight": np.ones((1, 1, 3, 3)), "bias": np.ones(1)}}


def test_a_kernel_larger_than_the_input(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """An input which is smaller than the kernel is an error, not an empty array."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*larger than the input"):
        forward(layer_model("Conv2d", CONV2D), ARRAYS, np.ones((1, 1, 2, 5)))


def test_an_input_with_the_wrong_number_of_axes(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """The input of `Conv2d` has three or four axes."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*2 axes"):
        forward(layer_model("Conv2d", CONV2D), ARRAYS, np.ones((5, 5)))


def test_a_weight_of_the_wrong_shape(layer_model: Callable[..., NNModel]) -> None:
    """An array which does not fit the layer is an error of the import."""
    network = Network(sid="net1", model=layer_model("Conv2d", CONV2D))
    network.parameters = {"layer1": {"weight": np.ones((1, 1, 3, 2))}}
    with pytest.raises(NetworkImportError, match=r"'weight'.*\(1, 1, 3, 2\)"):
        network.forward(np.ones((1, 1, 5, 5)))


def test_an_unknown_padding_mode(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """A padding mode which does not exist is an error which names it."""
    model = layer_model("Conv2d", {**CONV2D, "padding_mode": "mirror"})
    with pytest.raises(ValueError, match=r"node 'layer1'.*'mirror'"):
        forward(model, ARRAYS, np.ones((1, 1, 5, 5)))
````

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -x tests/sciml/test_layers_convolution.py`
Expected: FAIL with `UnsupportedLayerError: Network 'net1', node 'layer1': 'Conv1d' is not supported, the layer is not implemented`.

- [ ] **Step 3: Write the windows**

Create `src/sbmlsim/sciml/layers/windows.py`:

````python
"""Sliding windows over the spatial axes of an array.

The convolution and the pooling layers are reductions over the windows of
their kernel. An input has the shape `(N, C, *spatial)` or, without a batch,
`(C, *spatial)`; the layers add the batch axis, work on the windows and remove
it again.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def add_batch(x: np.ndarray, n: int, name: str) -> tuple[np.ndarray, bool]:
    """Add the batch axis to an input without one.

    Args:
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.
        n: the number of spatial dimensions.
        name: the name of the layer, for the message of the error.

    Returns:
        The input of shape `(N, C, *spatial)` and whether the axis was added.

    Raises:
        ValueError: if the input has neither `n + 1` nor `n + 2` axes.
    """
    if x.ndim == n + 1:
        return x[np.newaxis], True
    if x.ndim == n + 2:
        return x, False
    raise ValueError(
        f"{name}: the input has {x.ndim} axes, expected {n + 1} (C and {n} "
        f"spatial axes) or {n + 2} (N, C and {n} spatial axes)"
    )


def pad_spatial(
    x: np.ndarray,
    before: Sequence[int],
    after: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
) -> np.ndarray:
    """Pad the spatial axes of an input, a negative padding removes values.

    Args:
        x: input of shape `(N, C, *spatial)`.
        before: padding in front of every spatial axis.
        after: padding behind every spatial axis.
        mode: mode of `numpy.pad`.
        value: the value of the padding for the mode `constant`.

    Returns:
        The padded input.
    """
    crop = [slice(None), slice(None)]
    width = [(0, 0), (0, 0)]
    for size, b, a in zip(x.shape[2:], before, after, strict=True):
        crop.append(slice(max(-b, 0), size - max(-a, 0)))
        width.append((max(b, 0), max(a, 0)))
    x = x[tuple(crop)]
    if mode == "constant":
        return np.pad(x, width, mode="constant", constant_values=value)
    return np.pad(x, width, mode=mode)  # ty: ignore[no-matching-overload]


def windows(
    x: np.ndarray,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> np.ndarray:
    """Get the windows of a kernel over the spatial axes.

    Args:
        x: input of shape `(N, C, *spatial)`, already padded.
        kernel_size: the size of the kernel per spatial axis.
        stride: the step between two windows per spatial axis.
        dilation: the step between two elements of a window per spatial axis.

    Returns:
        A view of shape `(N, C, *output, *kernel_size)` with
        `output = (spatial - dilation * (kernel_size - 1) - 1) // stride + 1`.

    Raises:
        ValueError: if the kernel is larger than the input.
    """
    n = len(kernel_size)
    extent = tuple(d * (k - 1) + 1 for k, d in zip(kernel_size, dilation, strict=True))
    if any(e > s for e, s in zip(extent, x.shape[2:], strict=True)):
        raise ValueError(
            f"the kernel with the extent {extent} is larger than the input "
            f"with the spatial shape {x.shape[2:]}"
        )
    view = sliding_window_view(x, extent, axis=tuple(range(2, 2 + n)))
    index = (
        slice(None),
        slice(None),
        *(slice(None, None, s) for s in stride),
        *(slice(None, None, d) for d in dilation),
    )
    return view[index]


def output_size(
    size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool,
) -> int:
    """Get the number of windows of a pooling layer along one axis.

    Args:
        size: the size of the input along the axis.
        kernel_size: the size of the kernel.
        stride: the step between two windows.
        padding: the padding on both sides.
        dilation: the step between two elements of a window.
        ceil_mode: whether the last window may reach over the end of the
            input. A window which starts behind the input and its left padding
            is not counted.

    Returns:
        The number of windows.
    """
    numerator = size + 2 * padding - dilation * (kernel_size - 1) - 1
    if ceil_mode:
        n = -(-numerator // stride) + 1
        if (n - 1) * stride >= size + padding:
            n -= 1
        return n
    return numerator // stride + 1
````

- [ ] **Step 4: Write the convolutions**

Create `src/sbmlsim/sciml/layers/convolution.py`:

````python
"""The convolution and the transposed convolution layers, in numpy only.

A convolution is the tensor product of the windows of the kernel with the
weight. A transposed convolution is a convolution of the input with zeros
between its elements and the kernel flipped, which is what the gradient of a
convolution is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import NUMPY_ONLY, Backend
from sbmlsim.sciml.layers.registry import ArraySpec, as_tuple, layer_nd
from sbmlsim.sciml.layers.windows import add_batch, pad_spatial, windows

#: the modes of `numpy.pad` for the `padding_mode` of a convolution
PADDING_MODES: dict[str, str] = {
    "zeros": "constant",
    "reflect": "reflect",
    "replicate": "edge",
    "circular": "wrap",
}


def conv_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a convolution layer with `n` spatial dimensions."""
    kernel_size = as_tuple(args["kernel_size"], n)
    groups = args.get("groups", 1)
    arrays = {
        "weight": ArraySpec(
            (args["out_channels"], args["in_channels"] // groups, *kernel_size)
        )
    }
    if args.get("bias", True):
        arrays["bias"] = ArraySpec((args["out_channels"],))
    return arrays


def conv_transpose_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a transposed convolution with `n` spatial dimensions."""
    kernel_size = as_tuple(args["kernel_size"], n)
    groups = args.get("groups", 1)
    arrays = {
        "weight": ArraySpec(
            (args["in_channels"], args["out_channels"] // groups, *kernel_size)
        )
    }
    if args.get("bias", True):
        arrays["bias"] = ArraySpec((args["out_channels"],))
    return arrays


def correlate(
    x: np.ndarray,
    weight: np.ndarray,
    stride: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> np.ndarray:
    """Calculate the cross correlation of a padded input with a kernel.

    Args:
        x: input of shape `(N, C_in, *spatial)`, already padded.
        weight: kernel of shape `(C_out, C_in / groups, *kernel_size)`.
        stride: the step between two windows per spatial axis.
        dilation: the step between two elements of the kernel per spatial axis.
        groups: the number of groups the channels are split into.

    Returns:
        The output of shape `(N, C_out, *output)`.
    """
    n = len(stride)
    c_out = weight.shape[0]
    c_in_group = weight.shape[1]
    c_out_group = c_out // groups
    view = windows(x, weight.shape[2:], stride, dilation)
    kernel_axes = list(range(2 + n, 2 + 2 * n))
    outputs = []
    for g in range(groups):
        view_g = view[:, g * c_in_group : (g + 1) * c_in_group]
        weight_g = weight[g * c_out_group : (g + 1) * c_out_group]
        # (N, C_in, *output, *kernel) . (C_out, C_in, *kernel) -> (N, *output, C_out)
        y = np.tensordot(
            view_g, weight_g, axes=([1, *kernel_axes], list(range(1, 2 + n)))
        )
        outputs.append(np.moveaxis(y, -1, 1))
    return np.concatenate(outputs, axis=1)


@layer_nd("Conv{n}d", arrays=conv_arrays, backends=NUMPY_ONLY)
def conv(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `Conv1d`, `Conv2d` and `Conv3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `in_channels`, `out_channels`, `kernel_size`, `stride` (default
            `1`), `padding` (default `0`, an integer, one per axis, `valid` or
            `same`), `dilation` (default `1`), `groups` (default `1`), `bias`
            (default `True`), `padding_mode` (default `zeros`).
        arrays: `weight` of shape `(out_channels, in_channels / groups,
            *kernel_size)` and `bias` of shape `(out_channels,)`.
        x: input of shape `(N, C_in, *spatial)` or `(C_in, *spatial)`.

    Returns:
        The output of shape `(N, C_out, *output)` or `(C_out, *output)` with
        `output = (spatial + 2 * padding - dilation * (kernel_size - 1) - 1)
        // stride + 1`.

    Raises:
        ValueError: if the padding mode is not known or `same` is combined
            with a stride.
    """
    weight = arrays["weight"]
    x, unbatched = add_batch(x, n, "Conv")
    stride = as_tuple(args.get("stride", 1), n)
    dilation = as_tuple(args.get("dilation", 1), n)
    padding = args.get("padding", 0)
    if padding == "valid":
        before = after = (0,) * n
    elif padding == "same":
        if any(s != 1 for s in stride):
            raise ValueError("Conv: padding 'same' requires a stride of 1")
        total = [d * (k - 1) for k, d in zip(weight.shape[2:], dilation, strict=True)]
        before = tuple(t // 2 for t in total)
        after = tuple(t - t // 2 for t in total)
    else:
        before = after = as_tuple(padding, n)
    padding_mode = args.get("padding_mode", "zeros")
    if padding_mode not in PADDING_MODES:
        raise ValueError(f"Conv: padding_mode '{padding_mode}' is not known")
    x = pad_spatial(x, before, after, mode=PADDING_MODES[padding_mode])

    y = correlate(x, weight, stride, dilation, args.get("groups", 1))
    if "bias" in arrays:
        y = y + arrays["bias"].reshape((1, -1) + (1,) * n)
    return y[0] if unbatched else y


@layer_nd("ConvTranspose{n}d", arrays=conv_transpose_arrays, backends=NUMPY_ONLY)
def conv_transpose(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `ConvTranspose1d`, `ConvTranspose2d` and `ConvTranspose3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `in_channels`, `out_channels`, `kernel_size`, `stride` (default
            `1`), `padding` (default `0`), `output_padding` (default `0`),
            `groups` (default `1`), `bias` (default `True`), `dilation`
            (default `1`), `padding_mode` (only `zeros`).
        arrays: `weight` of shape `(in_channels, out_channels / groups,
            *kernel_size)` and `bias` of shape `(out_channels,)`.
        x: input of shape `(N, C_in, *spatial)` or `(C_in, *spatial)`.

    Returns:
        The output of shape `(N, C_out, *output)` or `(C_out, *output)` with
        `output = (spatial - 1) * stride - 2 * padding + dilation *
        (kernel_size - 1) + output_padding + 1`.

    Raises:
        ValueError: if the padding mode is not `zeros`.
    """
    weight = arrays["weight"]
    x, unbatched = add_batch(x, n, "ConvTranspose")
    if args.get("padding_mode", "zeros") != "zeros":
        raise ValueError("ConvTranspose: only the padding_mode 'zeros' exists")
    stride = as_tuple(args.get("stride", 1), n)
    dilation = as_tuple(args.get("dilation", 1), n)
    padding = as_tuple(args.get("padding", 0), n)
    output_padding = as_tuple(args.get("output_padding", 0), n)
    groups = args.get("groups", 1)

    # the input with `stride - 1` zeros between its elements
    shape = (
        *x.shape[:2],
        *((s - 1) * st + 1 for s, st in zip(x.shape[2:], stride, strict=True)),
    )
    spread = np.zeros(shape, dtype=x.dtype)
    spread[(slice(None), slice(None), *(slice(None, None, st) for st in stride))] = x

    extent = [d * (k - 1) for k, d in zip(weight.shape[2:], dilation, strict=True)]
    before = [e - p for e, p in zip(extent, padding, strict=True)]
    after = [e - p + o for e, p, o in zip(extent, padding, output_padding, strict=True)]
    spread = pad_spatial(spread, before, after)

    # the kernel of the convolution: the channels of every group swapped and
    # the spatial axes flipped
    c_in_group = weight.shape[0] // groups
    flip = (slice(None), slice(None), *(slice(None, None, -1) for _ in range(n)))
    kernels = [
        np.swapaxes(weight[g * c_in_group : (g + 1) * c_in_group], 0, 1)[flip]
        for g in range(groups)
    ]
    kernel = np.concatenate(kernels, axis=0)

    y = correlate(spread, kernel, (1,) * n, dilation, groups)
    if "bias" in arrays:
        y = y + arrays["bias"].reshape((1, -1) + (1,) * n)
    return y[0] if unbatched else y
````

- [ ] **Step 5: Register the convolutions**

Replace `src/sbmlsim/sciml/layers/__init__.py` with:

````python
"""The layers and functions of the forward pass of a network.

Every layer and every function of the NN YAML is implemented once, against a
backend, and registered under its PyTorch name in `LAYERS` or `FUNCTIONS`
with the backends it supports. Importing this package registers all of them.

| module | content | backends |
| --- | --- | --- |
| `core` | `Linear`, `Bilinear`, `Flatten`, the dropout layers | numpy, sympy |
| `functions` | the activation functions, `flatten`, `cat` | numpy, sympy |
| `convolution` | `Conv1-3d`, `ConvTranspose1-3d` | numpy |
| `pooling` | `MaxPool`, `AvgPool`, `LPPool` and the adaptive pools, `1-3d` | numpy |
| `normalization` | `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm` | numpy |
"""

from sbmlsim.sciml.layers import (
    convolution,
    core,
    functions,
)
from sbmlsim.sciml.layers.registry import (
    FUNCTIONS,
    LAYERS,
    ArraySpec,
    FunctionType,
    LayerType,
)

__all__ = [
    "FUNCTIONS",
    "LAYERS",
    "ArraySpec",
    "FunctionType",
    "LayerType",
    "convolution",
    "core",
    "functions",
]
````

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `153 passed`.

- [ ] **Step 7: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 8: Commit**

```bash
git add src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the convolution and the transposed convolution layers"
```

---

### Task 6: The pooling layers

**Files:**
- Create: `src/sbmlsim/sciml/layers/pooling.py`
- Modify: `src/sbmlsim/sciml/layers/__init__.py` (import `pooling`)
- Test: `tests/sciml/test_layers_pooling.py`

**Interfaces:**
- Consumes: `add_batch(x, n, name)`, `pad_spatial(x, before, after, mode="constant", value=0.0)`, `windows(x, kernel_size, stride, dilation)`, `output_size(size, kernel_size, stride, padding, dilation, ceil_mode)` of `sbmlsim.sciml.layers.windows`; `layer_nd`, `as_tuple` of `sbmlsim.sciml.layers.registry`; `NUMPY_ONLY`, `Backend`; the fixtures `compare_layer`, `layer_model`, `forward`.
- Produces: the entries `MaxPool{n}d`, `AvgPool{n}d`, `LPPool{n}d`, `AdaptiveMaxPool{n}d`, `AdaptiveAvgPool{n}d` for `n` in 1, 2, 3 of `LAYERS`, numpy only, without arrays.

The semantics, with the argument names and defaults of PyTorch. An input is `(N, C, *spatial)` or `(C, *spatial)`.

| layer | arguments | value of a window |
| --- | --- | --- |
| `MaxPool{n}d` | `kernel_size`, `stride=None` (the kernel size), `padding=0`, `dilation=1`, `return_indices=False` (`True` is an error), `ceil_mode=False` | the maximum, the padding is `-inf` |
| `AvgPool{n}d` | `kernel_size`, `stride=None` (the kernel size), `padding=0`, `ceil_mode=False`, `count_include_pad=True`, `divisor_override=None` | the sum over the divisor, the padding is `0` |
| `LPPool{n}d` | `norm_type`, `kernel_size`, `stride=None` (the kernel size), `ceil_mode=False` | `(sum(x ** norm_type)) ** (1 / norm_type)` |
| `AdaptiveMaxPool{n}d` | `output_size` (an integer, a list, an entry `None` keeps the size of the input), `return_indices=False` (`True` is an error) | the maximum |
| `AdaptiveAvgPool{n}d` | `output_size` | the mean |

The output size per spatial axis of `MaxPool`, `AvgPool` (dilation 1) and `LPPool` (padding 0, dilation 1):

```
numerator = L + 2 * padding - dilation * (kernel_size - 1) - 1
ceil_mode False:  floor(numerator / stride) + 1
ceil_mode True:   ceil(numerator / stride) + 1, minus 1 if (out - 1) * stride >= L + padding
```

The algorithms:

- **Windows in `ceil_mode`.** Pad `padding` in front and `max(needed - L - padding, padding)` behind, with `needed = (out - 1) * stride + dilation * (kernel_size - 1) + 1`, so the last window is complete. Keep the first `out` windows of every axis.
- **The divisor of `AvgPool`.** `divisor_override` when it is given. Otherwise the number of elements of the window which count: pool an array of ones with the shape of the input, which is padded with ones where `count_include_pad` is true and with zeros where it is false. The part of a window which reaches over the padding in `ceil_mode` is padded with zeros in both cases and never counts.
- **`LPPool`.** PyTorch calculates the sum of a window as the mean of `AvgPool` (padding 0, `count_include_pad=True`) times the product of the kernel sizes. Follow it exactly: `(average(x ** norm_type) * prod(kernel_size)) ** (1 / norm_type)`. In `ceil_mode` a window over the end is therefore scaled by the kernel and not by its elements.
- **Adaptive pooling.** The window `i` of an axis of the size `L` with `O` outputs is `[floor(i * L / O), ceil((i + 1) * L / O))`. The windows of all axes are boxes, so the maximum and the mean are reduced one axis after the other: for every axis stack the reduction of `x.take(range(start, end), axis)` over the `O` windows.

- [ ] **Step 1: Write the failing test**

Create `tests/sciml/test_layers_pooling.py`:

````python
"""The pooling layers against PyTorch."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml.backend import NUMPY_ONLY
from sbmlsim.sciml.layers import LAYERS

POOL_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    # the arguments as `petab_sciml` writes them, without a batch
    (
        "MaxPool3d",
        {
            "kernel_size": [3, 2, 1],
            "stride": [3, 2, 1],
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        },
        (2, 7, 6, 5),
    ),
    ("MaxPool1d", {"kernel_size": 3}, (2, 3, 11)),
    ("MaxPool1d", {"kernel_size": 3, "stride": 2, "padding": 1}, (3, 11)),
    (
        "MaxPool1d",
        {"kernel_size": 3, "stride": 2, "padding": 1, "dilation": 2, "ceil_mode": True},
        (2, 3, 12),
    ),
    ("MaxPool2d", {"kernel_size": [2, 2], "stride": [2, 2]}, (1, 6, 8, 8)),
    (
        "MaxPool2d",
        {"kernel_size": [3, 2], "stride": [2, 3], "padding": [1, 1], "ceil_mode": True},
        (2, 3, 10, 9),
    ),
    # the last window of the ceil mode starts in the padding and is dropped
    (
        "MaxPool2d",
        {"kernel_size": 2, "stride": 2, "padding": 1, "ceil_mode": True},
        (1, 1, 5, 5),
    ),
    (
        "MaxPool3d",
        {"kernel_size": 2, "stride": [1, 2, 3], "padding": 1},
        (2, 2, 5, 6, 7),
    ),
    (
        "AvgPool3d",
        {
            "kernel_size": [3, 2, 1],
            "stride": [3, 2, 1],
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        },
        (2, 7, 6, 5),
    ),
    ("AvgPool1d", {"kernel_size": 3}, (2, 3, 11)),
    ("AvgPool1d", {"kernel_size": 3, "stride": 2, "padding": 1}, (3, 11)),
    (
        "AvgPool1d",
        {"kernel_size": 3, "stride": 2, "padding": 1, "count_include_pad": False},
        (2, 3, 11),
    ),
    (
        "AvgPool1d",
        {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True},
        (2, 3, 12),
    ),
    (
        "AvgPool2d",
        {
            "kernel_size": [3, 2],
            "stride": [2, 3],
            "padding": [1, 1],
            "ceil_mode": True,
            "count_include_pad": False,
        },
        (2, 3, 10, 9),
    ),
    (
        "AvgPool2d",
        {"kernel_size": [3, 2], "stride": [2, 3], "padding": [1, 1], "ceil_mode": True},
        (2, 3, 10, 9),
    ),
    (
        "AvgPool2d",
        {"kernel_size": 3, "padding": 1, "divisor_override": 4},
        (2, 3, 9, 9),
    ),
    (
        "AvgPool3d",
        {"kernel_size": 2, "stride": [1, 2, 3], "padding": 1},
        (2, 2, 5, 6, 7),
    ),
    (
        "AdaptiveMaxPool3d",
        {"output_size": [3, 2, 1], "return_indices": False},
        (2, 7, 6, 5),
    ),
    ("AdaptiveMaxPool1d", {"output_size": 4}, (2, 3, 11)),
    ("AdaptiveMaxPool2d", {"output_size": [3, None]}, (2, 3, 10, 7)),
    ("AdaptiveMaxPool2d", {"output_size": 5}, (3, 10, 7)),
    ("AdaptiveAvgPool3d", {"output_size": [3, 2, 1]}, (2, 7, 6, 5)),
    ("AdaptiveAvgPool1d", {"output_size": 4}, (2, 3, 11)),
    ("AdaptiveAvgPool2d", {"output_size": [3, None]}, (2, 3, 10, 7)),
    ("AdaptiveAvgPool2d", {"output_size": 5}, (3, 10, 7)),
    # more outputs than inputs, the windows overlap
    ("AdaptiveAvgPool1d", {"output_size": 7}, (2, 3, 4)),
]

LP_POOL_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    (
        "LPPool3d",
        {"norm_type": 2, "kernel_size": [3, 2, 1], "stride": None, "ceil_mode": False},
        (2, 7, 6, 5),
    ),
    ("LPPool1d", {"norm_type": 1, "kernel_size": 3, "stride": 2}, (2, 3, 11)),
    ("LPPool1d", {"norm_type": 3, "kernel_size": 3, "ceil_mode": True}, (2, 3, 11)),
    ("LPPool2d", {"norm_type": 2, "kernel_size": [3, 2], "stride": [2, 1]}, (3, 9, 8)),
    (
        "LPPool2d",
        {"norm_type": 1.5, "kernel_size": 2, "stride": 3, "ceil_mode": True},
        (2, 3, 9, 8),
    ),
]


@pytest.mark.parametrize(("layer_type", "args", "shape"), POOL_CASES)
def test_pool(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A pooling layer has the values of PyTorch."""
    compare_layer(layer_type, args, shape)


@pytest.mark.parametrize(("layer_type", "args", "shape"), LP_POOL_CASES)
def test_lp_pool(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """The p-norm of the windows has the values of PyTorch.

    The input is positive, the power of a negative number with a `norm_type`
    which is not an integer is not defined.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed=3)
    x = rng.uniform(0.1, 2.0, size=shape)
    with torch.no_grad():
        expected = getattr(torch.nn, layer_type)(**args)(torch.from_numpy(x)).numpy()
    (observed,) = forward(layer_model(layer_type, args), {}, x)
    assert observed.shape == expected.shape
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_the_pooling_layers_are_numpy_only() -> None:
    """A pooling layer is not part of a compiled network."""
    names = ["MaxPool", "AvgPool", "LPPool", "AdaptiveMaxPool", "AdaptiveAvgPool"]
    for name in names:
        for n in (1, 2, 3):
            layer_type = LAYERS[f"{name}{n}d"]
            assert layer_type.backends == NUMPY_ONLY
            assert layer_type.arrays({"kernel_size": 2}) == {}


@pytest.mark.parametrize("layer_type", ["MaxPool2d", "AdaptiveMaxPool2d"])
def test_the_indices_are_not_returned(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    layer_type: str,
) -> None:
    """`return_indices` changes the output of a layer and is an error."""
    args = {"kernel_size": 2, "output_size": 2, "return_indices": True}
    with pytest.raises(ValueError, match=r"node 'layer1'.*return_indices"):
        forward(layer_model(layer_type, args), {}, np.ones((1, 4, 4)))


def test_a_window_larger_than_the_input(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """An input which is smaller than the kernel is an error, not an empty array."""
    with pytest.raises(ValueError, match=r"node 'layer1'.*larger than the input"):
        forward(layer_model("MaxPool2d", {"kernel_size": 3}), {}, np.ones((1, 2, 5)))
````

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -x tests/sciml/test_layers_pooling.py`
Expected: FAIL with `UnsupportedLayerError: Network 'net1', node 'layer1': 'MaxPool3d' is not supported, the layer is not implemented`.

- [ ] **Step 3: Write the pooling layers**

Create `src/sbmlsim/sciml/layers/pooling.py`:

````python
"""The pooling layers, in numpy only.

A pooling layer reduces the windows of its kernel to one value: the maximum,
the mean or the p-norm. The adaptive layers choose the windows from the size
of the output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import NUMPY_ONLY, Backend
from sbmlsim.sciml.layers.registry import as_tuple, layer_nd
from sbmlsim.sciml.layers.windows import add_batch, output_size, pad_spatial, windows


def pooled_windows(
    x: np.ndarray,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    ceil_mode: bool,
    value: float,
) -> np.ndarray:
    """Get the windows of a pooling layer.

    Args:
        x: input of shape `(N, C, *spatial)`.
        kernel_size: the size of the kernel per spatial axis.
        stride: the step between two windows per spatial axis.
        padding: the padding on both sides per spatial axis.
        dilation: the step between two elements of a window per spatial axis.
        ceil_mode: whether the last window may reach over the end.
        value: the value of the padding.

    Returns:
        The windows of shape `(N, C, *output, *kernel_size)`.
    """
    after = []
    for size, k, s, p, d in zip(
        x.shape[2:], kernel_size, stride, padding, dilation, strict=True
    ):
        n_out = output_size(size, k, s, p, d, ceil_mode)
        needed = (n_out - 1) * s + d * (k - 1) + 1
        after.append(max(needed - size - p, p))
    padded = pad_spatial(x, padding, after, value=value)
    view = windows(padded, kernel_size, stride, dilation)
    n_outs = [
        output_size(size, k, s, p, d, ceil_mode)
        for size, k, s, p, d in zip(
            x.shape[2:], kernel_size, stride, padding, dilation, strict=True
        )
    ]
    index = (slice(None), slice(None), *(slice(0, n_out) for n_out in n_outs))
    return view[index]


def pool_arguments(
    args: Mapping[str, Any], n: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Get the kernel size, the stride and the padding of a pooling layer.

    Args:
        args: the arguments of the layer.
        n: the number of spatial dimensions.

    Returns:
        `kernel_size`, `stride` (the kernel size when it is not given) and
        `padding` (default `0`), each per spatial axis.
    """
    kernel_size = as_tuple(args["kernel_size"], n)
    stride = args.get("stride")
    padding = as_tuple(args.get("padding", 0), n)
    return kernel_size, kernel_size if stride is None else as_tuple(stride, n), padding


def average(
    x: np.ndarray,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> np.ndarray:
    """Calculate the mean of the windows of an average pooling.

    Args:
        x: input of shape `(N, C, *spatial)`.
        kernel_size: the size of the kernel per spatial axis.
        stride: the step between two windows per spatial axis.
        padding: the zero padding on both sides per spatial axis.
        ceil_mode: whether the last window may reach over the end.
        count_include_pad: whether the padding counts as elements of a window.
            The part of a window which reaches over the padding in `ceil_mode`
            never counts.
        divisor_override: the divisor of every window, the number of elements
            when `None`.

    Returns:
        The output of shape `(N, C, *output)`.
    """
    n = len(kernel_size)
    ones = (1,) * n
    kernel_axes = tuple(range(2 + n, 2 + 2 * n))
    total = pooled_windows(
        x, kernel_size, stride, padding, ones, ceil_mode, value=0.0
    ).sum(axis=kernel_axes)
    if divisor_override is not None:
        return total / divisor_override

    counted = np.ones((1, 1, *x.shape[2:]))
    if count_include_pad:
        counted = pad_spatial(counted, padding, padding, value=1.0)
        padding = (0,) * n
    count = pooled_windows(
        counted, kernel_size, stride, padding, ones, ceil_mode, value=0.0
    ).sum(axis=kernel_axes)
    return total / count


@layer_nd("MaxPool{n}d", backends=NUMPY_ONLY)
def max_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `MaxPool1d`, `MaxPool2d` and `MaxPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `kernel_size`, `stride` (default `kernel_size`), `padding`
            (default `0`, padded with `-inf`), `dilation` (default `1`),
            `return_indices` (only `False`), `ceil_mode` (default `False`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output)` or `(C, *output)` with `output =
        (spatial + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
        + 1`, rounded up in `ceil_mode`.

    Raises:
        ValueError: if `return_indices` is set.
    """
    if args.get("return_indices", False):
        raise ValueError("MaxPool: return_indices is not supported")
    x, unbatched = add_batch(x, n, "MaxPool")
    kernel_size, stride, padding = pool_arguments(args, n)
    dilation = as_tuple(args.get("dilation", 1), n)
    view = pooled_windows(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        args.get("ceil_mode", False),
        value=-np.inf,
    )
    y = view.max(axis=tuple(range(2 + n, 2 + 2 * n)))
    return y[0] if unbatched else y


@layer_nd("AvgPool{n}d", backends=NUMPY_ONLY)
def avg_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `AvgPool1d`, `AvgPool2d` and `AvgPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `kernel_size`, `stride` (default `kernel_size`), `padding`
            (default `0`, padded with zeros), `ceil_mode` (default `False`),
            `count_include_pad` (default `True`), `divisor_override` (default
            `None`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output)` or `(C, *output)` with `output =
        (spatial + 2 * padding - kernel_size) // stride + 1`, rounded up in
        `ceil_mode`.
    """
    x, unbatched = add_batch(x, n, "AvgPool")
    kernel_size, stride, padding = pool_arguments(args, n)
    y = average(
        x,
        kernel_size,
        stride,
        padding,
        args.get("ceil_mode", False),
        args.get("count_include_pad", True),
        args.get("divisor_override"),
    )
    return y[0] if unbatched else y


@layer_nd("LPPool{n}d", backends=NUMPY_ONLY)
def lp_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `LPPool1d`, `LPPool2d` and `LPPool3d`.

    PyTorch calculates the sum of a window as its mean times the size of the
    kernel, which is followed here: a window which reaches over the end in
    `ceil_mode` is scaled by the kernel and not by its elements.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `norm_type`, `kernel_size`, `stride` (default `kernel_size`),
            `ceil_mode` (default `False`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output `(sum(x ** norm_type)) ** (1 / norm_type)` over every
        window, of shape `(N, C, *output)` or `(C, *output)` with `output =
        (spatial - kernel_size) // stride + 1`, rounded up in `ceil_mode`.
    """
    x, unbatched = add_batch(x, n, "LPPool")
    kernel_size, stride, _ = pool_arguments(args, n)
    norm_type = float(args["norm_type"])
    mean = average(
        x**norm_type,
        kernel_size,
        stride,
        (0,) * n,
        args.get("ceil_mode", False),
        count_include_pad=True,
        divisor_override=None,
    )
    y = (mean * float(np.prod(kernel_size))) ** (1.0 / norm_type)
    return y[0] if unbatched else y


def adaptive(
    n: int,
    x: np.ndarray,
    output: Any,
    reduce: Callable[..., np.ndarray],
    name: str,
) -> np.ndarray:
    """Reduce the windows of an adaptive pooling, one axis after the other.

    The window `i` of an axis of the size `L` with `O` outputs is
    `[floor(i * L / O), ceil((i + 1) * L / O))`. The windows of all axes are
    boxes, so the maximum and the mean of a box are the reduction along one
    axis after the other.

    Args:
        n: the number of spatial dimensions.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.
        output: `output_size` of the layer, an integer or one entry per axis,
            `None` keeps the size of the input.
        reduce: `numpy.max` or `numpy.mean`.
        name: the name of the layer, for the message of the error.

    Returns:
        The output of shape `(N, C, *output_size)` or `(C, *output_size)`.
    """
    sizes = list(output) if isinstance(output, (list, tuple)) else [output] * n
    x, unbatched = add_batch(x, n, name)
    for k, n_out in enumerate(sizes):
        axis = 2 + k
        size = x.shape[axis]
        if n_out is None:
            continue
        slices = []
        for i in range(n_out):
            start = (i * size) // n_out
            end = -(-((i + 1) * size) // n_out)
            slices.append(reduce(x.take(range(start, end), axis=axis), axis=axis))
        x = np.stack(slices, axis=axis)
    return x[0] if unbatched else x


@layer_nd("AdaptiveMaxPool{n}d", backends=NUMPY_ONLY)
def adaptive_max_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `output_size`, `return_indices` (only `False`).
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output_size)` or `(C, *output_size)`.

    Raises:
        ValueError: if `return_indices` is set.
    """
    if args.get("return_indices", False):
        raise ValueError("AdaptiveMaxPool: return_indices is not supported")
    return adaptive(n, x, args["output_size"], np.max, "AdaptiveMaxPool")


@layer_nd("AdaptiveAvgPool{n}d", backends=NUMPY_ONLY)
def adaptive_avg_pool(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `output_size`.
        arrays: no arrays.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of shape `(N, C, *output_size)` or `(C, *output_size)`.
    """
    return adaptive(n, x, args["output_size"], np.mean, "AdaptiveAvgPool")
````

- [ ] **Step 4: Register the pooling layers**

Replace `src/sbmlsim/sciml/layers/__init__.py` with:

````python
"""The layers and functions of the forward pass of a network.

Every layer and every function of the NN YAML is implemented once, against a
backend, and registered under its PyTorch name in `LAYERS` or `FUNCTIONS`
with the backends it supports. Importing this package registers all of them.

| module | content | backends |
| --- | --- | --- |
| `core` | `Linear`, `Bilinear`, `Flatten`, the dropout layers | numpy, sympy |
| `functions` | the activation functions, `flatten`, `cat` | numpy, sympy |
| `convolution` | `Conv1-3d`, `ConvTranspose1-3d` | numpy |
| `pooling` | `MaxPool`, `AvgPool`, `LPPool` and the adaptive pools, `1-3d` | numpy |
| `normalization` | `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm` | numpy |
"""

from sbmlsim.sciml.layers import (
    convolution,
    core,
    functions,
    pooling,
)
from sbmlsim.sciml.layers.registry import (
    FUNCTIONS,
    LAYERS,
    ArraySpec,
    FunctionType,
    LayerType,
)

__all__ = [
    "FUNCTIONS",
    "LAYERS",
    "ArraySpec",
    "FunctionType",
    "LayerType",
    "convolution",
    "core",
    "functions",
    "pooling",
]
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `188 passed`.

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 7: Commit**

```bash
git add src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the pooling layers"
```

---

### Task 7: The normalization layers

**Files:**
- Create: `src/sbmlsim/sciml/layers/normalization.py`
- Modify: `src/sbmlsim/sciml/layers/__init__.py` (import `normalization`)
- Test: `tests/sciml/test_layers_normalization.py`

**Interfaces:**
- Consumes: `layer`, `layer_nd`, `ArraySpec(shape, required=True, trainable=True)` of `sbmlsim.sciml.layers.registry`; `NUMPY_ONLY`, `Backend`; `Network(sid=, model=).parameter_ids()`; the fixtures `compare_layer` (it fills `running_mean` with normal and `running_var` with uniform values in `[0.5, 2]` when the PyTorch layer has them), `layer_model`, `forward`, `rng`.
- Produces: the entries `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d`, `LayerNorm` of `LAYERS`, numpy only.

The semantics, with the argument names and defaults of PyTorch. Every layer is `y = (x - mean) / sqrt(var + eps) * weight + bias`, the variance is the biased one (`numpy.var` with its default).

| layer | arguments | arrays | input | statistics |
| --- | --- | --- | --- | --- |
| `BatchNorm{n}d` | `num_features`, `eps=1e-5`, `affine=True`, `bias=True`; `momentum` and `track_running_stats` are not used | `weight (C,)`, `bias (C,)` when affine; `running_mean (C,)`, `running_var (C,)` optional, not parameters | `(N, C, *spatial)`, for `n = 1` also `(N, C)`. An input without a batch axis is an error | the stored ones when the arrays hold both, otherwise over the axis 0 and the spatial axes |
| `InstanceNorm{n}d` | `num_features`, `eps=1e-5`, `affine=False`, `bias=True`; `momentum` and `track_running_stats` are not used | as `BatchNorm`, but not affine by default | `(N, C, *spatial)` or `(C, *spatial)` | the stored ones when the arrays hold both, otherwise over the spatial axes of every sample and channel |
| `LayerNorm` | `normalized_shape` (an integer or a list), `eps=1e-5`, `elementwise_affine=True`, `bias=True` | `weight normalized_shape`, `bias normalized_shape` when affine | `(*, *normalized_shape)` | over the last `len(normalized_shape)` axes |

The decision of this task (finding 2): the spec says that the normalization layers use their stored statistics. The array files of the test suite store none and its reference values are calculated from the batch. A layer therefore uses `running_mean` and `running_var` when the arrays hold them and calculates the statistics from its input when they hold none. `track_running_stats` of the YAML does not decide this, the arrays do. The running statistics are arrays with `required=False, trainable=False`, so they are neither demanded by `Network.check_arrays` nor listed by `Network.parameter_ids`.

- [ ] **Step 1: Write the failing test**

Create `tests/sciml/test_layers_normalization.py`:

````python
"""The normalization layers against PyTorch, in evaluation mode."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from petab_sciml import NNModel

from sbmlsim.sciml import Network
from sbmlsim.sciml.backend import NUMPY_ONLY
from sbmlsim.sciml.layers import LAYERS

#: the arguments as `petab_sciml` writes them
WRITTEN = {"momentum": 0.1, "eps": 1e-05, "bias": True}

STORED_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    (
        "BatchNorm1d",
        {"num_features": 3, "track_running_stats": True, "affine": True, **WRITTEN},
        (4, 3),
    ),
    ("BatchNorm1d", {"num_features": 3}, (4, 3, 7)),
    ("BatchNorm1d", {"num_features": 3, "affine": False, "eps": 0.1}, (4, 3, 7)),
    ("BatchNorm2d", {"num_features": 3}, (4, 3, 5, 6)),
    ("BatchNorm3d", {"num_features": 3}, (4, 3, 5, 6, 2)),
    ("InstanceNorm1d", {"num_features": 3, "track_running_stats": True}, (4, 3, 7)),
    (
        "InstanceNorm2d",
        {"num_features": 3, "track_running_stats": True, "affine": True},
        (4, 3, 5, 6),
    ),
    (
        "InstanceNorm3d",
        {"num_features": 3, "track_running_stats": True, "affine": True},
        (4, 3, 5, 6, 2),
    ),
]

CALCULATED_CASES: list[tuple[str, dict[str, Any], tuple[int, ...]]] = [
    ("BatchNorm1d", {"num_features": 3, "track_running_stats": False}, (4, 3)),
    ("BatchNorm1d", {"num_features": 3, "track_running_stats": False}, (4, 3, 7)),
    (
        "BatchNorm2d",
        {"num_features": 3, "track_running_stats": False, "affine": False},
        (4, 3, 5, 6),
    ),
    (
        "BatchNorm3d",
        {"num_features": 3, "track_running_stats": False, "eps": 0.1},
        (4, 3, 5, 6, 2),
    ),
    (
        "InstanceNorm1d",
        {"num_features": 3, "track_running_stats": False, "affine": True, **WRITTEN},
        (4, 3, 7),
    ),
    ("InstanceNorm1d", {"num_features": 3}, (3, 7)),
    ("InstanceNorm2d", {"num_features": 3, "affine": True}, (4, 3, 5, 6)),
    ("InstanceNorm2d", {"num_features": 3}, (3, 5, 6)),
    ("InstanceNorm3d", {"num_features": 3, "eps": 0.1}, (4, 3, 5, 6, 2)),
    (
        "LayerNorm",
        {
            "normalized_shape": [4, 10, 11, 12],
            "eps": 1e-05,
            "elementwise_affine": True,
            "bias": True,
        },
        (2, 4, 10, 11, 12),
    ),
    ("LayerNorm", {"normalized_shape": [20]}, (3, 20)),
    ("LayerNorm", {"normalized_shape": 20}, (20,)),
    (
        "LayerNorm",
        {"normalized_shape": [5, 4], "elementwise_affine": False},
        (3, 2, 5, 4),
    ),
    ("LayerNorm", {"normalized_shape": [5, 4], "bias": False, "eps": 0.1}, (3, 5, 4)),
]


@pytest.mark.parametrize(("layer_type", "args", "shape"), STORED_CASES)
def test_the_stored_statistics_are_used(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A layer with running statistics normalizes with them."""
    compare_layer(layer_type, args, shape)


@pytest.mark.parametrize(("layer_type", "args", "shape"), CALCULATED_CASES)
def test_the_statistics_of_the_input_are_used(
    compare_layer: Callable[..., None],
    layer_type: str,
    args: dict[str, Any],
    shape: tuple[int, ...],
) -> None:
    """A layer without running statistics calculates them from its input."""
    compare_layer(layer_type, args, shape)


def test_a_batch_norm_without_stored_statistics(
    layer_model: Callable[..., NNModel],
    forward: Callable[..., tuple[np.ndarray, ...]],
    rng: np.random.Generator,
) -> None:
    """The statistics of the batch are used when the arrays have none.

    This is the layer of the test suite: `track_running_stats` is true and
    the array file holds the weight and the bias only.
    """
    torch = pytest.importorskip("torch")
    args = {"num_features": 3, "track_running_stats": True, "affine": True, **WRITTEN}
    weight, bias = rng.normal(size=3), rng.normal(size=3)
    x = rng.normal(size=(4, 3, 5, 6))

    arrays = {"layer1": {"weight": weight, "bias": bias}}
    (observed,) = forward(layer_model("BatchNorm2d", args), arrays, x)

    module = torch.nn.BatchNorm2d(3, track_running_stats=False).double()
    module.load_state_dict(
        {"weight": torch.from_numpy(weight), "bias": torch.from_numpy(bias)}
    )
    module.eval()
    with torch.no_grad():
        expected = module(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_the_arrays_of_a_normalization_layer(
    layer_model: Callable[..., NNModel],
) -> None:
    """The running statistics are arrays, but not required and not parameters."""
    arrays = LAYERS["BatchNorm2d"].arrays({"num_features": 3})
    assert list(arrays) == ["weight", "bias", "running_mean", "running_var"]
    assert arrays["weight"].required and arrays["weight"].trainable
    assert not arrays["running_mean"].required
    assert not arrays["running_var"].trainable

    assert list(LAYERS["InstanceNorm2d"].arrays({"num_features": 3})) == [
        "running_mean",
        "running_var",
    ]
    assert list(LAYERS["LayerNorm"].arrays({"normalized_shape": [2, 3]})) == [
        "weight",
        "bias",
    ]
    assert LAYERS["LayerNorm"].arrays({"normalized_shape": 4})["weight"].shape == (4,)

    network = Network(sid="net1", model=layer_model("BatchNorm2d", {"num_features": 2}))
    assert list(network.parameter_ids()) == [
        "net1__layer1__weight__0",
        "net1__layer1__weight__1",
        "net1__layer1__bias__0",
        "net1__layer1__bias__1",
    ]


def test_the_normalization_layers_are_numpy_only() -> None:
    """A normalization layer is not part of a compiled network."""
    for name in ("BatchNorm1d", "InstanceNorm2d", "LayerNorm"):
        assert LAYERS[name].backends == NUMPY_ONLY


def test_a_batch_norm_needs_a_batch(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """`BatchNorm2d` has no input without a batch axis."""
    model = layer_model("BatchNorm2d", {"num_features": 3, "affine": False})
    with pytest.raises(ValueError, match=r"node 'layer1'.*3 axes"):
        forward(model, {}, np.ones((3, 4, 4)))


def test_a_layer_norm_on_the_wrong_shape(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """The last axes of the input are the normalized shape."""
    model = layer_model(
        "LayerNorm", {"normalized_shape": [4], "elementwise_affine": False}
    )
    with pytest.raises(ValueError, match=r"node 'layer1'.*normalized_shape"):
        forward(model, {}, np.ones((2, 5)))


def test_a_constant_input_is_normalized_to_zero(
    layer_model: Callable[..., NNModel], forward: Callable[..., tuple[np.ndarray, ...]]
) -> None:
    """`eps` keeps the division by a variance of zero finite."""
    model = layer_model("InstanceNorm1d", {"num_features": 2})
    (y,) = forward(model, {}, np.full((1, 2, 5), 3.0))
    np.testing.assert_array_equal(y, np.zeros((1, 2, 5)))
````

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -x tests/sciml/test_layers_normalization.py`
Expected: FAIL with `UnsupportedLayerError: Network 'net1', node 'layer1': 'BatchNorm1d' is not supported, the layer is not implemented`.

- [ ] **Step 3: Write the normalization layers**

Create `src/sbmlsim/sciml/layers/normalization.py`:

````python
"""The normalization layers, in numpy only and in evaluation mode.

A normalization layer is `y = (x - mean) / sqrt(var + eps) * weight + bias`.
`BatchNorm` and `InstanceNorm` use the stored statistics, i.e. the arrays
`running_mean` and `running_var` of the layer. A layer without stored
statistics calculates them from its input, which is what PyTorch does for a
layer with `track_running_stats=False` and for a layer whose statistics are
`None`: `BatchNorm` over the batch and the spatial axes, `InstanceNorm` over
the spatial axes of every sample. `LayerNorm` has no stored statistics. The
variance is the biased one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from sbmlsim.sciml.backend import NUMPY_ONLY, Backend
from sbmlsim.sciml.layers.registry import ArraySpec, layer, layer_nd

logger = logging.getLogger(__name__)


def norm_arrays(n: int, args: Mapping[str, Any], affine: bool) -> dict[str, ArraySpec]:
    """Get the arrays of a `BatchNorm` or `InstanceNorm` layer.

    Args:
        n: the number of spatial dimensions.
        args: the arguments of the layer.
        affine: the default of `affine`, which differs between the two.

    Returns:
        `weight` and `bias` for an affine layer and the running statistics,
        which are neither required nor parameters of a fit.
    """
    shape = (args["num_features"],)
    arrays: dict[str, ArraySpec] = {}
    if args.get("affine", affine):
        arrays["weight"] = ArraySpec(shape)
        if args.get("bias", True):
            arrays["bias"] = ArraySpec(shape)
    arrays["running_mean"] = ArraySpec(shape, required=False, trainable=False)
    arrays["running_var"] = ArraySpec(shape, required=False, trainable=False)
    return arrays


def batch_norm_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `BatchNorm` layer, which is affine by default."""
    return norm_arrays(n, args, affine=True)


def instance_norm_arrays(n: int, args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of an `InstanceNorm` layer, not affine by default."""
    return norm_arrays(n, args, affine=False)


def normalize(
    x: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    eps: float,
    axes: tuple[int, ...],
    channel_axis: int,
) -> np.ndarray:
    """Normalize an input per channel.

    Args:
        x: the input.
        arrays: the arrays of the layer, all of them of shape `(C,)`.
        eps: the value added to the variance.
        axes: the axes the statistics are calculated over when the layer
            stores none.
        channel_axis: the axis of the channels.

    Returns:
        The normalized input.
    """
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    if "running_mean" in arrays and "running_var" in arrays:
        mean = arrays["running_mean"].reshape(shape)
        var = arrays["running_var"].reshape(shape)
    else:
        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if "weight" in arrays:
        y = y * arrays["weight"].reshape(shape)
    if "bias" in arrays:
        y = y + arrays["bias"].reshape(shape)
    return y


@layer_nd("BatchNorm{n}d", arrays=batch_norm_arrays, backends=NUMPY_ONLY)
def batch_norm(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `BatchNorm1d`, `BatchNorm2d` and `BatchNorm3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `num_features`, `eps` (default `1e-5`), `affine` (default
            `True`), `bias` (default `True`); `momentum` and
            `track_running_stats` belong to the training and are not used.
        arrays: `weight` and `bias` of shape `(num_features,)` for an affine
            layer, `running_mean` and `running_var` when they are stored.
        x: input of shape `(N, C, *spatial)`, for one dimension also `(N, C)`.

    Returns:
        The output of the shape of the input.

    Raises:
        ValueError: if the input has no batch axis.
    """
    if x.ndim != n + 2 and not (n == 1 and x.ndim == 2):
        raise ValueError(
            f"BatchNorm{n}d: the input has {x.ndim} axes, expected {n + 2} "
            f"(N, C and {n} spatial axes)"
        )
    axes = (0, *range(2, x.ndim))
    return normalize(x, arrays, args.get("eps", 1e-5), axes, channel_axis=1)


@layer_nd("InstanceNorm{n}d", arrays=instance_norm_arrays, backends=NUMPY_ONLY)
def instance_norm(
    n: int,
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `InstanceNorm1d`, `InstanceNorm2d` and `InstanceNorm3d`.

    Args:
        n: the number of spatial dimensions.
        backend: the backend.
        args: `num_features`, `eps` (default `1e-5`), `affine` (default
            `False`), `bias` (default `True`); `momentum` and
            `track_running_stats` belong to the training and are not used.
        arrays: `weight` and `bias` of shape `(num_features,)` for an affine
            layer, `running_mean` and `running_var` when they are stored.
        x: input of shape `(N, C, *spatial)` or `(C, *spatial)`.

    Returns:
        The output of the shape of the input.

    Raises:
        ValueError: if the input has neither `n + 1` nor `n + 2` axes.
    """
    if x.ndim not in (n + 1, n + 2):
        raise ValueError(
            f"InstanceNorm{n}d: the input has {x.ndim} axes, expected {n + 1} "
            f"or {n + 2}"
        )
    axes = tuple(range(x.ndim - n, x.ndim))
    return normalize(
        x, arrays, args.get("eps", 1e-5), axes, channel_axis=x.ndim - n - 1
    )


def layer_norm_arrays(args: Mapping[str, Any]) -> dict[str, ArraySpec]:
    """Get the arrays of a `LayerNorm` layer."""
    normalized_shape = args["normalized_shape"]
    shape = (
        (int(normalized_shape),)
        if isinstance(normalized_shape, int)
        else tuple(int(s) for s in normalized_shape)
    )
    arrays: dict[str, ArraySpec] = {}
    if args.get("elementwise_affine", True):
        arrays["weight"] = ArraySpec(shape)
        if args.get("bias", True):
            arrays["bias"] = ArraySpec(shape)
    return arrays


@layer("LayerNorm", arrays=layer_norm_arrays, backends=NUMPY_ONLY)
def layer_norm(
    backend: Backend,
    args: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate `LayerNorm`.

    Args:
        backend: the backend.
        args: `normalized_shape`, `eps` (default `1e-5`),
            `elementwise_affine` (default `True`), `bias` (default `True`).
        arrays: `weight` and `bias` of the shape `normalized_shape` for an
            affine layer.
        x: input of shape `(*, *normalized_shape)`.

    Returns:
        The output of the shape of the input, normalized over the last axes
        which `normalized_shape` covers.

    Raises:
        ValueError: if the last axes of the input are not `normalized_shape`.
    """
    shape = layer_norm_arrays({**args, "elementwise_affine": True})["weight"].shape
    if x.shape[x.ndim - len(shape) :] != shape:
        raise ValueError(
            f"LayerNorm: the input of shape {x.shape} does not end with the "
            f"normalized_shape {shape}"
        )
    axes = tuple(range(x.ndim - len(shape), x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    y = (x - mean) / np.sqrt(var + args.get("eps", 1e-5))
    if "weight" in arrays:
        y = y * arrays["weight"]
    if "bias" in arrays:
        y = y + arrays["bias"]
    return y
````

- [ ] **Step 4: Register the normalization layers**

Replace `src/sbmlsim/sciml/layers/__init__.py` with:

````python
"""The layers and functions of the forward pass of a network.

Every layer and every function of the NN YAML is implemented once, against a
backend, and registered under its PyTorch name in `LAYERS` or `FUNCTIONS`
with the backends it supports. Importing this package registers all of them.

| module | content | backends |
| --- | --- | --- |
| `core` | `Linear`, `Bilinear`, `Flatten`, the dropout layers | numpy, sympy |
| `functions` | the activation functions, `flatten`, `cat` | numpy, sympy |
| `convolution` | `Conv1-3d`, `ConvTranspose1-3d` | numpy |
| `pooling` | `MaxPool`, `AvgPool`, `LPPool` and the adaptive pools, `1-3d` | numpy |
| `normalization` | `BatchNorm1-3d`, `InstanceNorm1-3d`, `LayerNorm` | numpy |
"""

from sbmlsim.sciml.layers import (
    convolution,
    core,
    functions,
    normalization,
    pooling,
)
from sbmlsim.sciml.layers.registry import (
    FUNCTIONS,
    LAYERS,
    ArraySpec,
    FunctionType,
    LayerType,
)

__all__ = [
    "FUNCTIONS",
    "LAYERS",
    "ArraySpec",
    "FunctionType",
    "LayerType",
    "convolution",
    "core",
    "functions",
    "normalization",
    "pooling",
]
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `216 passed`.

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 7: Commit**

```bash
git add src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the normalization layers"
```

---

### Task 8: The nominal values and the fit parameters of a network

**Files:**
- Create: `src/sbmlsim/sciml/parameters.py`
- Test: `tests/sciml/test_parameters.py`

**Interfaces:**
- Consumes: `Network` (`sid`, `parameters`, `array_specs()`, `parameter_ids()`, `check_arrays(parameters, complete=True)`, `with_values(values)`), `NetworkParameters`, `copy_parameters`, `element_id` of `sbmlsim.sciml.network`; `NetworkImportError`; `sbmlsim.fit.objects.FitParameter(pid, start_value=None, lower_bound=-inf, upper_bound=inf, unit=None, target=None, mappings=None)`, which raises `ValueError` with "is outside of the bounds" for a start value outside of its bounds.
- Produces: `sbmlsim.sciml.parameters.covered_arrays(network, key) -> tuple[int, list[tuple[str, str]]]`, `resolve_entries(network, entries) -> dict[tuple[str, str], T]`, `nominal_parameters(network, values=None) -> NetworkParameters`, `network_fit_parameters(network, estimate, bounds, values=None) -> list[FitParameter]`, `ELEMENT_UNIT = "dimensionless"`. Task 10 uses `nominal_parameters`.

The rules of this task:

- The key of an entry is the network (`net1`), a layer (`net1.layer1`) or an array (`net1.layer1.weight`). The precedence is network < layer < array, independent of the order of the entries.
- The id of a layer may hold a dot (`block.0`), so the rest of a key behind the network is tried as a layer first and as `<layer>.<array>` second.
- A key which names nothing of the network is a `KeyError`, it is not ignored. The running statistics of a normalization layer are not covered by any key.
- `nominal_parameters` replaces all elements of an array an entry covers by the value of the entry and creates the array when the array file does not hold it. It returns a copy and demands afterwards that every required array of a layer of the forward pass has values.
- `network_fit_parameters` returns one `FitParameter` per estimated element: `pid` is the id of the element, `start_value` its nominal value, the unit `dimensionless`, no `target` and no `mappings`. An element no entry of `estimate` covers is not estimated, an estimated element no entry of `bounds` covers has the bounds `(-inf, inf)`. The field `scale` of `FitParameter` and the target `sciml:<id>` are phase 3.

- [ ] **Step 1: Write the failing test**

Create `tests/sciml/test_parameters.py`:

````python
"""Tests of the nominal values and the fit parameters of a network."""

from itertools import pairwise

import numpy as np
import pytest
from petab_sciml import Input, Layer, NNModel, Node

from sbmlsim.sciml import Network, NetworkImportError
from sbmlsim.sciml.parameters import (
    covered_arrays,
    network_fit_parameters,
    nominal_parameters,
    resolve_entries,
)


def _network(with_values: bool = True) -> Network:
    """Build `layer2(layer1(x))`, `layer1` is a layer of a nested module."""
    layers = [
        Layer(
            layer_id="block.layer1",
            layer_type="Linear",
            args={"in_features": 2, "out_features": 2, "bias": True},
        ),
        Layer(
            layer_id="norm",
            layer_type="BatchNorm1d",
            args={"num_features": 2},
        ),
        Layer(
            layer_id="layer2",
            layer_type="Linear",
            args={"in_features": 2, "out_features": 1, "bias": True},
        ),
    ]
    names = ["net_input", "block.layer1", "norm", "layer2"]
    forward = [
        Node(name="net_input", op="placeholder", target="net_input", args=[], kwargs={})
    ]
    for previous, name in pairwise(names):
        forward.append(
            Node(name=name, op="call_module", target=name, args=[previous], kwargs={})
        )
    forward.append(
        Node(name="output", op="output", target="output", args=["layer2"], kwargs={})
    )
    model = NNModel(
        nn_model_id="net1",
        inputs=[Input(input_id="input0")],
        layers=layers,
        forward=forward,
    )
    parameters = {
        "block.layer1": {
            "weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": np.array([5.0, 6.0]),
        },
        "norm": {
            "weight": np.array([1.5, 2.5]),
            "bias": np.array([0.5, 0.25]),
            "running_mean": np.array([0.1, 0.2]),
            "running_var": np.array([1.0, 2.0]),
        },
        "layer2": {"weight": np.array([[7.0, 8.0]]), "bias": np.array([9.0])},
    }
    return Network(
        sid="net1", model=model, parameters=parameters if with_values else {}
    )


def test_the_arrays_an_entry_covers() -> None:
    """An entry is the network, a layer or an array."""
    network = _network()
    assert covered_arrays(network, "net1") == (
        0,
        [
            ("block.layer1", "weight"),
            ("block.layer1", "bias"),
            ("norm", "weight"),
            ("norm", "bias"),
            ("layer2", "weight"),
            ("layer2", "bias"),
        ],
    )
    assert covered_arrays(network, "net1.layer2") == (
        1,
        [("layer2", "weight"), ("layer2", "bias")],
    )
    assert covered_arrays(network, "net1.layer2.bias") == (2, [("layer2", "bias")])
    # the id of a layer of a nested module holds the separator
    assert covered_arrays(network, "net1.block.layer1") == (
        1,
        [("block.layer1", "weight"), ("block.layer1", "bias")],
    )
    assert covered_arrays(network, "net1.block.layer1.weight") == (
        2,
        [("block.layer1", "weight")],
    )


@pytest.mark.parametrize(
    "key",
    ["net2", "net1.layer3", "net1.layer2.gain", "net1.", "", "net1.norm.running_mean"],
)
def test_an_entry_which_covers_nothing(key: str) -> None:
    """A key which is not part of the network is an error, not an empty entry."""
    with pytest.raises(KeyError, match=r"is not the network, a layer or an array"):
        covered_arrays(_network(), key)


def test_the_more_specific_entry_wins() -> None:
    """The order of the entries does not matter, the array beats the layer."""
    network = _network()
    entries = {"net1.layer2.bias": 3.0, "net1.layer2": 2.0, "net1": 1.0}
    resolved = resolve_entries(network, entries)
    assert resolved[("block.layer1", "weight")] == 1.0
    assert resolved[("norm", "bias")] == 1.0
    assert resolved[("layer2", "weight")] == 2.0
    assert resolved[("layer2", "bias")] == 3.0
    assert resolved == resolve_entries(network, dict(reversed(entries.items())))


def test_the_nominal_values_of_the_array_file_are_kept() -> None:
    """Without entries the nominal values are the ones of the network."""
    network = _network()
    parameters = nominal_parameters(network)
    np.testing.assert_array_equal(
        parameters["layer2"]["weight"], network.parameters["layer2"]["weight"]
    )
    assert parameters["layer2"]["weight"] is not network.parameters["layer2"]["weight"]
    np.testing.assert_array_equal(parameters["norm"]["running_var"], [1.0, 2.0])


def test_a_value_replaces_the_elements_it_covers() -> None:
    """A layer is set to zero and the other layers keep their values."""
    network = _network()
    parameters = nominal_parameters(network, {"net1.block.layer1": 0.0})
    np.testing.assert_array_equal(
        parameters["block.layer1"]["weight"], np.zeros((2, 2))
    )
    np.testing.assert_array_equal(parameters["block.layer1"]["bias"], np.zeros(2))
    np.testing.assert_array_equal(parameters["layer2"]["weight"], [[7.0, 8.0]])
    # the network keeps its values
    assert network.parameters["block.layer1"]["weight"][0, 0] == 1.0

    parameters = nominal_parameters(
        network, {"net1": 1.0, "net1.layer2": 2.0, "net1.layer2.bias": 3.0}
    )
    np.testing.assert_array_equal(parameters["block.layer1"]["bias"], [1.0, 1.0])
    np.testing.assert_array_equal(parameters["layer2"]["weight"], [[2.0, 2.0]])
    np.testing.assert_array_equal(parameters["layer2"]["bias"], [3.0])
    # the running statistics are not parameters, a value does not reach them
    np.testing.assert_array_equal(parameters["norm"]["running_mean"], [0.1, 0.2])


def test_values_for_a_network_without_an_array_file() -> None:
    """The entries of a problem may be all the values a network has."""
    parameters = nominal_parameters(_network(with_values=False), {"net1": 0.5})
    assert parameters["block.layer1"]["weight"].shape == (2, 2)
    assert np.all(parameters["layer2"]["weight"] == 0.5)
    assert "running_mean" not in parameters["norm"]


def test_an_array_without_values_is_an_error() -> None:
    """A network is not initialized with random values."""
    with pytest.raises(NetworkImportError, match=r"'layer2'.*has no values"):
        nominal_parameters(
            _network(with_values=False), {"net1.block.layer1": 0.0, "net1.norm": 1.0}
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_value_which_is_not_finite(value: float) -> None:
    """A nominal value is a number."""
    with pytest.raises(NetworkImportError, match=r"not finite"):
        nominal_parameters(_network(), {"net1.layer2": value})


def test_the_fit_parameters_of_the_estimated_elements() -> None:
    """One parameter per estimated element, the other elements are frozen."""
    fit_parameters = network_fit_parameters(
        _network(),
        estimate={"net1": True, "net1.block.layer1": False, "net1.norm": False},
        bounds={"net1": (-10.0, 10.0), "net1.layer2.bias": (0.0, 20.0)},
        values={"net1.layer2.weight": 0.5},
    )
    assert [p.pid for p in fit_parameters] == [
        "net1__layer2__weight__0_0",
        "net1__layer2__weight__0_1",
        "net1__layer2__bias__0",
    ]
    weight, _, bias = fit_parameters
    assert (weight.start_value, weight.lower_bound, weight.upper_bound) == (
        0.5,
        -10.0,
        10.0,
    )
    assert (bias.start_value, bias.lower_bound, bias.upper_bound) == (9.0, 0.0, 20.0)
    assert bias.unit == "dimensionless"
    assert bias.target_id == bias.pid


def test_an_element_is_frozen_and_unbounded_by_default() -> None:
    """No entry means not estimated, no bounds means not bounded."""
    assert network_fit_parameters(_network(), estimate={}, bounds={}) == []

    fit_parameters = network_fit_parameters(
        _network(), estimate={"net1.block.layer1.bias": True}, bounds={}
    )
    assert [p.pid for p in fit_parameters] == [
        "net1__block_layer1__bias__0",
        "net1__block_layer1__bias__1",
    ]
    assert fit_parameters[0].lower_bound == -np.inf
    assert fit_parameters[0].upper_bound == np.inf


def test_a_nominal_value_outside_of_its_bounds() -> None:
    """The start value of a fit parameter is inside its bounds."""
    with pytest.raises(ValueError, match=r"outside of the bounds"):
        network_fit_parameters(
            _network(), estimate={"net1.layer2": True}, bounds={"net1": (0.0, 1.0)}
        )


def test_the_ids_are_the_ids_of_the_network() -> None:
    """The fit parameters can be written back with `with_values`."""
    network = _network()
    fit_parameters = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    assert [p.pid for p in fit_parameters] == list(network.parameter_ids())
    values = {p.pid: 0.0 for p in fit_parameters}
    parameters = network.with_values(values)
    assert all(
        np.all(parameters[layer][name] == 0.0)
        for layer, name, _ in network.parameter_ids().values()
    )
````

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -x tests/sciml/test_parameters.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'sbmlsim.sciml.parameters'`.

- [ ] **Step 3: Write the module**

Create `src/sbmlsim/sciml/parameters.py`:

````python
"""The nominal values and the fit parameters of a network.

A problem describes the elements of a network in groups: an entry is given
for the network (`net1`), for a layer (`net1.layer1`) or for an array
(`net1.layer1.weight`), and the more specific entry wins. This is how a
problem sets the elements of one layer to `0.0` while the other layers keep
the values of the array file, and how it estimates one layer and freezes the
others.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from sbmlsim.fit.objects import FitParameter
from sbmlsim.sciml.errors import NetworkImportError
from sbmlsim.sciml.network import (
    Network,
    NetworkParameters,
    copy_parameters,
    element_id,
)

logger = logging.getLogger(__name__)

#: separator of the network, the layer and the array in the key of an entry
KEY_SEPARATOR = "."

#: the unit of the elements of a network
ELEMENT_UNIT = "dimensionless"


def covered_arrays(network: Network, key: str) -> tuple[int, list[tuple[str, str]]]:
    """Get the arrays an entry covers.

    Args:
        network: the network.
        key: the network (`net1`), a layer (`net1.layer1`) or an array
            (`net1.layer1.weight`).

    Returns:
        How specific the entry is (0 for the network, 1 for a layer, 2 for an
        array) and the arrays as pairs of layer id and array name. Only the
        arrays which are parameters are covered, i.e. not the running
        statistics of a normalization layer.

    Raises:
        KeyError: if the key does not name the network, one of its layers or
            one of their arrays.
    """
    specs = {
        layer: [name for name, spec in arrays.items() if spec.trainable]
        for layer, arrays in network.array_specs().items()
    }
    if key == network.sid:
        return 0, [(layer, name) for layer, names in specs.items() for name in names]

    prefix = network.sid + KEY_SEPARATOR
    if key.startswith(prefix):
        rest = key[len(prefix) :]
        # the id of a layer may hold the separator, so the layer is tried first
        if rest in specs:
            return 1, [(rest, name) for name in specs[rest]]
        layer, _, name = rest.rpartition(KEY_SEPARATOR)
        if layer in specs and name in specs[layer]:
            return 2, [(layer, name)]
    raise KeyError(
        f"Network '{network.sid}': '{key}' is not the network, a layer or an "
        f"array of it. The layers and their arrays are {specs}"
    )


def resolve_entries[T](
    network: Network, entries: Mapping[str, T]
) -> dict[tuple[str, str], T]:
    """Resolve the entries of a problem to the arrays of a network.

    Args:
        network: the network.
        entries: key of the entry -> value, see `covered_arrays`.

    Returns:
        layer id and array name -> the value of the most specific entry which
        covers the array. An array no entry covers is not part of it.

    Raises:
        KeyError: if a key does not name the network, a layer or an array.
    """
    covered = {key: covered_arrays(network, key) for key in entries}
    resolved: dict[tuple[str, str], T] = {}
    # the network first and the arrays last, so the more specific entry wins
    for key in sorted(entries, key=lambda key: covered[key][0]):
        for array in covered[key][1]:
            resolved[array] = entries[key]
    return resolved


def nominal_parameters(
    network: Network, values: Mapping[str, float] | None = None
) -> NetworkParameters:
    """Get the nominal values of the arrays of a network.

    Args:
        network: the network with the values of its array file.
        values: key of the entry -> value of every element the entry covers,
            see `covered_arrays`. An array no entry covers keeps the values of
            the array file.

    Returns:
        The arrays in the PyTorch layout. The network is not changed.

    Raises:
        KeyError: if a key does not name the network, a layer or an array.
        NetworkImportError: if a value is not finite, or if an array of a
            layer of the forward pass has values neither in the array file
            nor in `values`.
    """
    parameters = copy_parameters(network.parameters)
    specs = network.array_specs()
    for (layer, name), value in resolve_entries(network, values or {}).items():
        if not np.isfinite(value):
            raise NetworkImportError(
                f"Network '{network.sid}', layer '{layer}': the value "
                f"'{value}' of the array '{name}' is not finite"
            )
        parameters.setdefault(layer, {})[name] = np.full(
            specs[layer][name].shape, float(value)
        )
    network.check_arrays(parameters, complete=True)
    return parameters


def network_fit_parameters(
    network: Network,
    estimate: Mapping[str, bool],
    bounds: Mapping[str, tuple[float, float]],
    values: Mapping[str, float] | None = None,
) -> list[FitParameter]:
    """Create the fit parameters of the estimated elements of a network.

    `estimate`, `bounds` and `values` are given for the network, for a layer
    or for an array, and the more specific entry wins, see `covered_arrays`.

    Args:
        network: the network with the values of its array file.
        estimate: key of the entry -> whether the elements are estimated. An
            element no entry covers is not estimated.
        bounds: key of the entry -> lower and upper bound of the elements. An
            estimated element no entry covers is not bounded.
        values: key of the entry -> nominal value of the elements, which
            replaces the values of the array file.

    Returns:
        One parameter per estimated element, named by the id of the element,
        with the nominal value as start value and without a unit, in the
        order of `Network.parameter_ids`.

    Raises:
        KeyError: if a key does not name the network, a layer or an array.
        NetworkImportError: if an estimated element has no nominal value.
        ValueError: if a nominal value is outside of its bounds.
    """
    parameters = nominal_parameters(network, values)
    estimated = resolve_entries(network, estimate)
    bounded = resolve_entries(network, bounds)

    fit_parameters: list[FitParameter] = []
    for layer, name, index in network.parameter_ids().values():
        if not estimated.get((layer, name), False):
            continue
        if name not in parameters.get(layer, {}):
            raise NetworkImportError(
                f"Network '{network.sid}', layer '{layer}': the array "
                f"'{name}' is estimated and has no nominal values"
            )
        lower, upper = bounded.get((layer, name), (-np.inf, np.inf))
        fit_parameters.append(
            FitParameter(
                pid=element_id(network.sid, layer, name, index),
                start_value=float(parameters[layer][name][index]),
                lower_bound=lower,
                upper_bound=upper,
                unit=ELEMENT_UNIT,
            )
        )
    logger.info(
        "Network '%s': %d of %d elements are estimated",
        network.sid,
        len(fit_parameters),
        len(network.parameter_ids()),
    )
    return fit_parameters
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/sciml`
Expected: `235 passed`.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 6: Commit**

```bash
git add src/sbmlsim/sciml tests/sciml
git commit -m "sciml: the nominal values and the fit parameters of a network"
```

---

### Task 9: The download and the cache of a test suite

**Files:**
- Create: `src/sbmlsim/testsuite/cache.py`
- Modify: `src/sbmlsim/testsuite/cases.py` (the imports, a new constant, `SemanticSuite.cache_path`, `SemanticSuite.load`)
- Test: `tests/testsuite/test_cache.py`

**Interfaces:**
- Consumes: `SemanticSuite` of `src/sbmlsim/testsuite/cases.py` with `cache_path(version) -> Path`, `cached(version)`, `load(version)`, `latest_version()`, `_cases_dir(staging) -> Path`, and the constants `SUITE_VERSION`, `SUITE_URL`.
- Produces: `sbmlsim.testsuite.cache.cache_root() -> Path` (`$XDG_CACHE_HOME/sbmlsim` or `~/.cache/sbmlsim`), `cache_path(variable: str, *parts: str) -> Path` (the value of the environment variable when it is set), `fetch(url: str, path: Path, select: Callable[[Path], Path]) -> Path`; `sbmlsim.testsuite.cases.SUITE_PATH_VARIABLE = "SBMLSIM_TEST_SUITE_PATH"`. Task 10 uses `cache_path` and `fetch`.

`SemanticSuite` keeps its interface: the signatures, the paths (`<cache>/sbmlsim/test-suite/<version>/semantic`), the environment variable and the staging directory `.<name>.incomplete` next to the target do not change. `tests/testsuite/test_cases.py` is not edited and must pass unchanged. `sbmlsim/testsuite/__init__.py` is not edited.

- [ ] **Step 1: Write the failing test**

Create `tests/testsuite/test_cache.py`:

````python
"""Tests of the download and the cache of a test suite."""

import zipfile
from pathlib import Path

import pytest

from sbmlsim.testsuite import cache
from sbmlsim.testsuite.cases import SemanticSuite


def _archive(path: Path, files: dict[str, str]) -> str:
    """Write a zip archive and return its URL."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path.as_uri()


def test_the_cache_is_in_the_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache follows `XDG_CACHE_HOME`, a variable points elsewhere."""
    monkeypatch.delenv("SBMLSIM_SOME_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    assert cache.cache_root() == Path("/tmp/cache/sbmlsim")
    assert cache.cache_path("SBMLSIM_SOME_SUITE_PATH", "suite", "1.0") == Path(
        "/tmp/cache/sbmlsim/suite/1.0"
    )

    monkeypatch.setenv("SBMLSIM_SOME_SUITE_PATH", "/elsewhere")
    assert cache.cache_path("SBMLSIM_SOME_SUITE_PATH", "suite", "1.0") == Path(
        "/elsewhere"
    )


def test_the_cache_without_xdg_is_in_the_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """`~/.cache` is the user cache when `XDG_CACHE_HOME` is not set or empty."""
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert cache.cache_root() == Path.home() / ".cache" / "sbmlsim"


def test_an_archive_is_unpacked_into_place(tmp_path: Path) -> None:
    """The selected directory of the archive becomes the target."""
    url = _archive(tmp_path / "suite.zip", {"suite-1.0/cases/001/a.txt": "a"})
    target = tmp_path / "cache" / "suite" / "1.0"

    path = cache.fetch(url, target, select=lambda staging: staging / "suite-1.0/cases")

    assert path == target
    assert (target / "001" / "a.txt").read_text() == "a"
    assert [p.name for p in target.parent.iterdir()] == ["1.0"]


def test_a_failed_download_leaves_nothing(tmp_path: Path) -> None:
    """Neither the target nor the staging directory exist after a failure."""
    target = tmp_path / "cache" / "suite" / "1.0"
    with pytest.raises(OSError):
        cache.fetch((tmp_path / "missing.zip").as_uri(), target, select=lambda s: s)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_a_file_which_is_not_an_archive(tmp_path: Path) -> None:
    """A download which is not a zip archive is an error and leaves nothing."""
    (tmp_path / "page.zip").write_text("<html>not found</html>")
    target = tmp_path / "cache" / "suite" / "1.0"
    with pytest.raises(zipfile.BadZipFile):
        cache.fetch((tmp_path / "page.zip").as_uri(), target, select=lambda s: s)
    assert list(target.parent.iterdir()) == []


def test_an_archive_without_cases(tmp_path: Path) -> None:
    """The error of the selection is raised and the staging is removed."""
    url = _archive(tmp_path / "suite.zip", {"readme.txt": "no cases"})
    target = tmp_path / "cache" / "semantic"
    with pytest.raises(OSError, match="No case directories"):
        cache.fetch(url, target, select=SemanticSuite._cases_dir)
    assert list(target.parent.iterdir()) == []


def test_the_semantic_suite_is_loaded_through_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SemanticSuite.load` downloads into the path of its release."""
    url = _archive(
        tmp_path / "semantic.zip", {"semantic/00001/00001-settings.txt": "start: 0"}
    )
    monkeypatch.delenv("SBMLSIM_TEST_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("sbmlsim.testsuite.cases.SUITE_URL", url)

    suite = SemanticSuite.load("9.9.9")

    assert suite.version == "9.9.9"
    assert suite.path == tmp_path / "cache/sbmlsim/test-suite/9.9.9/semantic"
    assert (suite.path / "00001" / "00001-settings.txt").is_file()
    assert SemanticSuite.cached("9.9.9") == suite
````

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -x tests/testsuite/test_cache.py`
Expected: FAIL at collection with `ImportError: cannot import name 'cache' from 'sbmlsim.testsuite'`.

- [ ] **Step 3: Write the cache**

Create `src/sbmlsim/testsuite/cache.py`:

````python
"""The download and the cache of a test suite.

A test suite is an archive which is downloaded once and unpacked into the
user cache, i.e. `XDG_CACHE_HOME` or `~/.cache`, under `sbmlsim/`. An
environment variable points at the cases when they live elsewhere, e.g. on a
machine without a network. The archive is unpacked next to its target and
moved into place, so an interrupted download does not leave a directory which
looks like a cached suite.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_root() -> Path:
    """Get the directory the test suites are cached in.

    Returns:
        `sbmlsim` in the user cache, i.e. in `XDG_CACHE_HOME` or `~/.cache`.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "sbmlsim"


def cache_path(variable: str, *parts: str) -> Path:
    """Get the directory a test suite is unpacked into.

    Args:
        variable: the environment variable which overrides the directory.
        *parts: the directories below the cache of `sbmlsim`, e.g. the name
            of the suite and its version.

    Returns:
        The value of the environment variable when it is set, the directory
        in the user cache otherwise.
    """
    override = os.environ.get(variable)
    if override:
        return Path(override)
    return cache_root().joinpath(*parts)


def fetch(url: str, path: Path, select: Callable[[Path], Path]) -> Path:
    """Download an archive and move a directory of it into place.

    Args:
        url: the zip archive.
        path: the directory which holds the cases afterwards. It must not
            exist.
        select: gets the directory the archive was unpacked into and returns
            the directory of it which becomes `path`.

    Returns:
        `path`.

    Raises:
        OSError: if the archive cannot be downloaded or unpacked, or if
            `select` does not find the cases.
    """
    staging = path.parent / f".{path.name}.incomplete"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "archive.zip"
    try:
        logger.info("Downloading '%s'", url)
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
        archive.unlink()
        selected = select(staging)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.replace(path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path
````

- [ ] **Step 4: Use the cache in `SemanticSuite`**

In `src/sbmlsim/testsuite/cases.py` replace the imports

```python
import logging
import os
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
```

with

```python
import logging
import re
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sbmlsim.testsuite import cache
```

(`urllib.request` stays, `latest_version` uses it.) In front of the comment `#: the newest release of the suite` add

```python
#: the environment variable which points at the cases when they are not in
#: the cache
SUITE_PATH_VARIABLE = "SBMLSIM_TEST_SUITE_PATH"

```

In `SemanticSuite.cache_path` replace the body behind the docstring

```python
        override = os.environ.get("SBMLSIM_TEST_SUITE_PATH")
        if override:
            return Path(override)
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        return cache / "sbmlsim" / "test-suite" / version / "semantic"
```

with

```python
        return cache.cache_path(SUITE_PATH_VARIABLE, "test-suite", version, "semantic")
```

In `SemanticSuite.load` replace

```python
        logger.info("Downloading the SBML Test Suite '%s' from '%s'", version, url)

        # unpack next to the target and move it into place, so an interrupted
        # download does not leave a directory which looks like a cached suite
        staging = path.parent / f".{path.name}.incomplete"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        archive = staging / "semantic.zip"
        try:
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(staging)
            archive.unlink()
            cases = cls._cases_dir(staging)
            path.parent.mkdir(parents=True, exist_ok=True)
            cases.replace(path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return cls(path=path, version=version)
```

with

```python
        logger.info("Downloading the SBML Test Suite '%s' from '%s'", version, url)
        cache.fetch(url, path, select=cls._cases_dir)
        return cls(path=path, version=version)
```

`_cases_dir` and everything else of the file stay as they are.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -x tests/testsuite`
Expected: all pass, `tests/testsuite/test_cache.py` contributes 7 tests and `tests/testsuite/test_cases.py` passes unchanged.

Run: `uv run python scripts/testsuite.py download`
Expected: the path of the cached suite is printed, with or without a download. This is the end to end check that the command line of the SBML Test Suite still works.

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

- [ ] **Step 7: Commit**

```bash
git add src/sbmlsim/testsuite tests/testsuite/test_cache.py
git commit -m "testsuite: the download and the cache of a suite as a module of its own"
```

---

### Task 10: The PEtab SciML test suite, its baseline and its tox environment

**Files:**
- Create: `src/sbmlsim/sciml/testsuite.py`
- Create: `scripts/sciml_testsuite.py`
- Create: `tests/data/sciml_baseline.json`
- Modify: `tox.ini` (new section `[testenv:sciml]`)
- Modify: `.github/workflows/ci-cd.yml` (job `testsuite`)
- Modify: `CLAUDE.md` (project, commands, architecture, conventions)
- Test: `tests/sciml/test_suite_cases.py` (runs in every session, writes its cases itself)
- Test: `tests/sciml/test_testsuite.py` (marker `sciml_testsuite`, one test per case of the downloaded suite)

**Interfaces:**
- Consumes: `cache.cache_path(variable, *parts)`, `cache.fetch(url, path, select)` of `sbmlsim.testsuite`; `Network.from_files(yaml_path, array_path=None, sid=None)`, `Network.read_arrays(array_path)`, `Network.forward(*inputs)`, `load_array_data(path) -> ArrayData`, `NetworkParameters`; `nominal_parameters(network, values)`; `NetworkImportError`, `UnsupportedLayerError`; `petab.v2.core.ProblemConfig(**raw, base_path=)` with `.extensions: dict`, `.parameter_files`, `.mapping_files`; `petab.v2.core.ParameterTable.from_tsv(path).elements` with `.id`, `.nominal_value` (a float, `None` or the string `array`); `petab.v2.core.MappingTable.from_tsv(path).elements` with `.petab_id`, `.model_id`; `petab.v2.extensions.sciml.SciMLConfig` with `.array_files`, `.neural_networks: dict[str, NeuralNetConfig]` (`.location`, `.format`, `.pre_initialization`); `petab_sciml.constants.ARRAY = "array"`.
- Produces: `sbmlsim.sciml.testsuite.SCIML_SUITE_COMMIT`, `SCIML_SUITE_URL`, `SCIML_SUITE_PATH_VARIABLE = "SBMLSIM_SCIML_SUITE_PATH"`, `MODEL_IMPORT`, `INITIALIZATION`, `PROBLEM_IMPORT`, `MODEL_IMPORT_TOLERANCE = 1e-3`, `DROPOUT_TOLERANCE = 1e-2`, `CaseStatus` (`PASS`, `TOLERANCE`, `SHAPE`, `UNSUPPORTED`, `ERROR`), `CaseResult` (`group`, `cid`, `status`, `message`, `max_difference`, `passed`, `key`), `ModelImportCase`, `InitializationCase`, `ProblemImportCase` (each with `from_directory(path)`, the first two with `run() -> CaseResult`), `SciMLSuite` (`path`, `commit`, `cache_path`, `cached`, `load`, `case_ids(group)`, `model_import_cases()`, `initialization_cases()`, `problem_import_cases()`, `run()`), `parameter_key(model_entity_id) -> str | None`, `compare_arrays`, `read_array`, `read_solutions`. Phase 3 adds the comparison of `ProblemImportCase` and moves `parameter_key` into the reader.

The layout of the suite (`test_cases/` of the repository, which becomes the cache directory `~/.cache/sbmlsim/petab-sciml-testsuite/<commit>/`):

```
ml_model_import/NNN/     net.yaml, net_input_i.hdf5, net_ps_i.hdf5, net_output_i.hdf5 (i = 1, 2, 3), solutions.yaml
initialization/NNN/      petab/ (problem.yaml, parameters.tsv, mapping.tsv, net1.yaml, net1_ps.hdf5, ...), net1_ref.hdf5, solutions.yaml
sciml_problem_import/NNN/ petab/, simulations.tsv, grad_mech.tsv, grad_net1.hdf5, solutions.yaml
```

The rules of this task:

- `solutions.yaml` of `ml_model_import` has the keys `net_file`, `net_input` (or `net_input_arg0`, `net_input_arg1` for a network with two inputs), `net_ps` (missing for a network without arrays), `net_output`, `input_order_py`, `output_order_py`, `dropout` (only the cases with dropout) and the `_jl` orders, which are not read. The combination `i` is the input `i`, the arrays `i` and the output `i`.
- An input file holds one dataset below `inputs/` (`inputs/input0/data`, also for the second input of a network), an output file one below `outputs/` (`outputs/output0/data`). `ArrayData` of `petab_sciml` has no outputs, so both are read with `h5py`.
- The arrays of the suite are in the PyTorch layout, which is the layout of `sbmlsim`. `input_order_py` and `output_order_py` name the axes of the arrays as they are stored, nothing is permuted. A case checks that the number of axes is the length of the order.
- The tolerance is absolute: `1e-3`, and `1e-2` for a case with `dropout` (finding 1).
- `run()` of a case never raises: a layer without an implementation is `UNSUPPORTED`, any other exception is `ERROR` with the message of the exception.
- An `initialization` problem is read without `petab.v2.Problem`, which needs `torch` (finding 4): `ProblemConfig` from the YAML, the mapping table resolves the ids of the parameter table to the keys of `nominal_parameters` (`net1.parameters` to `net1`, `net1.parameters[layer1]` to `net1.layer1`, `net1.parameters[layer1].weight` to `net1.layer1.weight`), a row with the `nominalValue` `array` keeps the values of the array file.
- The baseline lists a case which does not pass as `"<group>/<cid>": {"status": ..., "reason": ...}`. A test fails on a regression and on a case which passes and is still listed.
- The tests of `test_testsuite.py` do not download: they are parametrized over the cached suite and skip when it is not on the machine. `tox r -e sciml` downloads first.

- [ ] **Step 1: Write the failing tests**

Create `tests/sciml/test_suite_cases.py`:

````python
"""Tests of reading and running a case of the PEtab SciML test suite.

The cases are written by the tests, nothing is downloaded.
"""

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from petab_sciml import Input, Layer, NNModel, NNModelStandard, Node

from sbmlsim.sciml.testsuite import (
    SCIML_SUITE_COMMIT,
    CaseStatus,
    InitializationCase,
    ModelImportCase,
    ProblemImportCase,
    SciMLSuite,
    compare_arrays,
    parameter_key,
)

WEIGHT = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
BIAS = np.array([0.1, 0.2, 0.3])


def _model(sid: str = "net0", layer_type: str = "Linear") -> NNModel:
    """Build `tanh(layer1(x))` with 2 inputs and 3 outputs."""
    return NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type=layer_type,
                args={"in_features": 2, "out_features": 3, "bias": True},
            )
        ],
        forward=[
            Node(
                name="net_input",
                op="placeholder",
                target="net_input",
                args=[],
                kwargs={},
            ),
            Node(
                name="layer1",
                op="call_module",
                target="layer1",
                args=["net_input"],
                kwargs={},
            ),
            Node(
                name="tanh", op="call_method", target="tanh", args=["layer1"], kwargs={}
            ),
            Node(name="output", op="output", target="output", args=["tanh"], kwargs={}),
        ],
    )


def _h5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an HDF5 file of the suite, the keys are the paths of the arrays."""
    with h5py.File(path, "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        for key, array in arrays.items():
            f[key] = array


def _model_import_case(
    path: Path, shift: float = 0.0, layer_type: str = "Linear", **solutions: object
) -> Path:
    """Write a case of the group `ml_model_import` with two combinations."""
    directory = path / "ml_model_import" / "001"
    directory.mkdir(parents=True)
    NNModelStandard.save_data(
        data=_model(layer_type=layer_type), filename=str(directory / "net.yaml")
    )
    for i in (1, 2):
        x = np.array([0.5 * i, -0.25])
        _h5(directory / f"net_input_{i}.hdf5", {"inputs/input0/data": x.astype("f4")})
        _h5(
            directory / f"net_ps_{i}.hdf5",
            {
                "parameters/net0/layer1/weight": (i * WEIGHT).astype("f4"),
                "parameters/net0/layer1/bias": BIAS.astype("f4"),
            },
        )
        y = np.tanh(i * WEIGHT @ x + BIAS) + shift
        _h5(
            directory / f"net_output_{i}.hdf5", {"outputs/output0/data": y.astype("f4")}
        )
    content = {
        "net_file": "net.yaml",
        "net_input": ["net_input_1.hdf5", "net_input_2.hdf5"],
        "net_ps": ["net_ps_1.hdf5", "net_ps_2.hdf5"],
        "net_output": ["net_output_1.hdf5", "net_output_2.hdf5"],
        "input_order_py": ["W"],
        "output_order_py": ["W"],
        **solutions,
    }
    (directory / "solutions.yaml").write_text(yaml.safe_dump(content))
    return directory


def _initialization_case(
    path: Path, entity: str, reference: dict[str, np.ndarray]
) -> Path:
    """Write a case of the group `initialization`.

    The array file holds `WEIGHT` and `BIAS`, the parameter table sets the
    entity of the mapping table to zero.
    """
    directory = path / "initialization" / "001"
    petab = directory / "petab"
    petab.mkdir(parents=True)
    NNModelStandard.save_data(
        data=_model(sid="other"), filename=str(petab / "net1.yaml")
    )
    _h5(
        petab / "net1_ps.hdf5",
        {
            "parameters/net1/layer1/weight": WEIGHT,
            "parameters/net1/layer1/bias": BIAS,
        },
    )
    (petab / "problem.yaml").write_text(
        yaml.safe_dump(
            {
                "format_version": "2.0.0",
                "model_files": {"lv": {"location": "lv.xml", "language": "sbml"}},
                "parameter_files": ["parameters.tsv"],
                "mapping_files": ["mapping.tsv"],
                "measurement_files": [],
                "observable_files": [],
                "extensions": {
                    "sciml": {
                        "version": "0.1.0",
                        "required": True,
                        "array_files": ["net1_ps.hdf5"],
                        "hybridization_files": [],
                        "neural_networks": {
                            "net1": {
                                "location": "net1.yaml",
                                "pre_initialization": False,
                                "format": "YAML",
                            }
                        },
                    }
                },
            }
        )
    )
    (petab / "parameters.tsv").write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "alpha\tlin\t0.0\t15.0\t1.3\ttrue\n"
        "net1_part\tlin\t-inf\tinf\t0.0\ttrue\n"
        "net1_ps\tlin\t-inf\tinf\tarray\ttrue\n"
    )
    (petab / "mapping.tsv").write_text(
        "petabEntityId\tmodelEntityId\n"
        "net1_input1\tnet1.inputs[0][0]\n"
        "net1_output1\tnet1.outputs[0][0]\n"
        "net1_ps\tnet1.parameters\n"
        f"net1_part\t{entity}\n"
    )
    _h5(
        directory / "net1_ref.hdf5",
        {f"parameters/net1/layer1/{name}": array for name, array in reference.items()},
    )
    (directory / "solutions.yaml").write_text(
        yaml.safe_dump({"tol": 0.001, "parameter_files": {"net1": "net1_ref.hdf5"}})
    )
    return directory


# ---------------------------------------------------------------------------
# ml_model_import
# ---------------------------------------------------------------------------
def test_a_model_import_case_is_read(tmp_path: Path) -> None:
    """The combinations, the axis orders and the tolerance of a case."""
    case = ModelImportCase.from_directory(_model_import_case(tmp_path))

    assert case.cid == "001"
    assert case.net_file.name == "net.yaml"
    assert [[p.name for p in row] for row in case.inputs] == [
        ["net_input_1.hdf5"],
        ["net_input_2.hdf5"],
    ]
    assert [p.name for p in case.parameters] == ["net_ps_1.hdf5", "net_ps_2.hdf5"]
    assert [p.name for p in case.outputs] == ["net_output_1.hdf5", "net_output_2.hdf5"]
    assert case.input_order == ["W"]
    assert case.output_order == ["W"]
    assert case.dropout is None
    assert case.tolerance == 1e-3


def test_a_case_with_several_inputs(tmp_path: Path) -> None:
    """The inputs of a network with two inputs are listed per argument."""
    directory = _model_import_case(tmp_path)
    solutions = yaml.safe_load((directory / "solutions.yaml").read_text())
    del solutions["net_input"]
    solutions["net_input_arg1"] = ["b_1.hdf5", "b_2.hdf5"]
    solutions["net_input_arg0"] = ["a_1.hdf5", "a_2.hdf5"]
    (directory / "solutions.yaml").write_text(yaml.safe_dump(solutions))

    case = ModelImportCase.from_directory(directory)
    assert [[p.name for p in row] for row in case.inputs] == [
        ["a_1.hdf5", "b_1.hdf5"],
        ["a_2.hdf5", "b_2.hdf5"],
    ]


def test_a_case_with_dropout_has_a_wider_tolerance(tmp_path: Path) -> None:
    """The reference values of a dropout case are a mean of random passes."""
    case = ModelImportCase.from_directory(_model_import_case(tmp_path, dropout=40000))
    assert case.dropout == 40000
    assert case.tolerance == 1e-2


def test_a_model_import_case_passes(tmp_path: Path) -> None:
    """The forward pass reproduces the outputs of every combination."""
    result = ModelImportCase.from_directory(_model_import_case(tmp_path)).run()
    assert result.passed
    assert result.key == "ml_model_import/001"
    assert result.max_difference is not None
    assert result.max_difference < 1e-6


def test_an_output_outside_of_the_tolerance(tmp_path: Path) -> None:
    """A difference above the tolerance fails and is reported."""
    case = ModelImportCase.from_directory(_model_import_case(tmp_path, shift=0.002))
    result = case.run()
    assert result.status == CaseStatus.TOLERANCE
    assert result.max_difference == pytest.approx(0.002, rel=1e-3)
    assert "above the tolerance 0.001" in result.message


def test_a_case_with_a_layer_which_is_not_implemented(tmp_path: Path) -> None:
    """A case does not raise, the outcome names the layer."""
    case = ModelImportCase.from_directory(
        _model_import_case(tmp_path, layer_type="LSTM")
    )
    result = case.run()
    assert result.status == CaseStatus.UNSUPPORTED
    assert "'LSTM'" in result.message


def test_a_case_with_a_missing_file(tmp_path: Path) -> None:
    """A file which does not exist is an error of the case, not of the run."""
    directory = _model_import_case(tmp_path)
    (directory / "net_output_2.hdf5").unlink()
    result = ModelImportCase.from_directory(directory).run()
    assert result.status == CaseStatus.ERROR
    assert "net_output_2.hdf5" in result.message


def test_an_axis_order_which_does_not_fit(tmp_path: Path) -> None:
    """The axis order names the axes of the array."""
    directory = _model_import_case(tmp_path, input_order_py=["C", "W"])
    result = ModelImportCase.from_directory(directory).run()
    assert result.status == CaseStatus.ERROR
    assert "['C', 'W']" in result.message


def test_a_directory_which_is_not_a_case(tmp_path: Path) -> None:
    """A directory without `solutions.yaml` is not read."""
    with pytest.raises(ValueError, match=r"no 'solutions.yaml'"):
        ModelImportCase.from_directory(tmp_path)
    (tmp_path / "solutions.yaml").write_text("net_file: net.yaml\n")
    with pytest.raises(ValueError, match=r"no inputs or no outputs"):
        ModelImportCase.from_directory(tmp_path)


def test_the_comparison_of_arrays() -> None:
    """The shape, the values and the undefined values are compared."""
    a = np.array([1.0, 2.0])
    assert compare_arrays(a, a + 5e-4, 1e-3)[0] == CaseStatus.PASS
    assert compare_arrays(a, a + 2e-3, 1e-3)[0] == CaseStatus.TOLERANCE
    assert compare_arrays(a, a.reshape(1, 2), 1e-3)[0] == CaseStatus.SHAPE
    assert compare_arrays(np.array([1.0, np.nan]), a, 1e-3)[0] == CaseStatus.TOLERANCE
    assert compare_arrays(np.array([]), np.array([]), 1e-3)[0] == CaseStatus.PASS


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("entity", "key"),
    [
        ("net1.parameters", "net1"),
        ("net1.parameters[layer1]", "net1.layer1"),
        ("net1.parameters[layer1].weight", "net1.layer1.weight"),
        ("net1.parameters[block.0].bias", "net1.block.0.bias"),
        ("net1.inputs[0][1]", None),
        ("net1.outputs[0][0]", None),
        ("net1.parametersX", None),
        ("net1.parameters[layer1", None),
        ("net1.parameters[].weight", None),
        ("net1.parameters[layer1].", None),
        (".parameters", None),
        ("alpha", None),
        ("", None),
    ],
)
def test_the_key_of_an_entity(entity: str, key: str | None) -> None:
    """The entities of the mapping table which are parameters of a network."""
    assert parameter_key(entity) == key


@pytest.mark.parametrize(
    ("entity", "reference"),
    [
        ("net1.parameters[layer1]", {"weight": 0 * WEIGHT, "bias": 0 * BIAS}),
        ("net1.parameters[layer1].weight", {"weight": 0 * WEIGHT, "bias": BIAS}),
        ("net1.parameters[layer1].bias", {"weight": WEIGHT, "bias": 0 * BIAS}),
    ],
)
def test_an_initialization_case_passes(
    tmp_path: Path, entity: str, reference: dict[str, np.ndarray]
) -> None:
    """The row of the parameter table replaces the values of the array file."""
    case = InitializationCase.from_directory(
        _initialization_case(tmp_path, entity, reference)
    )
    assert case.tolerance == 0.001
    assert list(case.parameter_files) == ["net1"]

    nominal = case.nominal()
    np.testing.assert_array_equal(
        nominal["net1"]["layer1"]["weight"], reference["weight"]
    )
    np.testing.assert_array_equal(nominal["net1"]["layer1"]["bias"], reference["bias"])
    result = case.run()
    assert result.passed, result.message
    assert result.key == "initialization/001"


def test_an_initialization_which_differs(tmp_path: Path) -> None:
    """A nominal value which is not the reference value fails and is named."""
    directory = _initialization_case(
        tmp_path, "net1.parameters[layer1].bias", {"weight": WEIGHT, "bias": BIAS}
    )
    result = InitializationCase.from_directory(directory).run()
    assert result.status == CaseStatus.TOLERANCE
    assert "'net1.layer1.bias'" in result.message
    assert result.max_difference == pytest.approx(0.3)


def test_a_network_in_another_format(tmp_path: Path) -> None:
    """Only the format `YAML` is read."""
    directory = _initialization_case(
        tmp_path, "net1.parameters[layer1]", {"weight": WEIGHT, "bias": BIAS}
    )
    problem = directory / "petab" / "problem.yaml"
    problem.write_text(problem.read_text().replace("format: YAML", "format: equinox"))
    result = InitializationCase.from_directory(directory).run()
    assert result.status == CaseStatus.ERROR
    assert "'equinox' is not supported" in result.message


# ---------------------------------------------------------------------------
# sciml_problem_import
# ---------------------------------------------------------------------------
def test_a_problem_import_case_is_read(tmp_path: Path) -> None:
    """The reference values and the tolerances of a case are read."""
    directory = tmp_path / "sciml_problem_import" / "001"
    directory.mkdir(parents=True)
    (directory / "solutions.yaml").write_text(
        yaml.safe_dump(
            {
                "llh": 33.5,
                "tol_llh": 0.001,
                "tol_simulations": 0.002,
                "tol_grad": 0.1,
                "simulation_files": ["simulations.tsv"],
                "grad_files": {"mech": "grad_mech.tsv", "net1": "grad_net1.hdf5"},
            }
        )
    )
    case = ProblemImportCase.from_directory(directory)
    assert (case.llh, case.log_posterior) == (33.5, None)
    assert (case.tol_llh, case.tol_simulations, case.tol_grad) == (0.001, 0.002, 0.1)
    assert case.problem_path == directory / "petab" / "problem.yaml"
    assert [p.name for p in case.simulation_files] == ["simulations.tsv"]
    assert {k: p.name for k, p in case.gradient_files.items()} == {
        "mech": "grad_mech.tsv",
        "net1": "grad_net1.hdf5",
    }


def test_a_problem_import_case_with_priors(tmp_path: Path) -> None:
    """A case with priors states the log-posterior and its tolerance."""
    (tmp_path / "solutions.yaml").write_text(
        yaml.safe_dump(
            {
                "log_posterior": -12.5,
                "tol_log_posterior": 0.01,
                "tol_simulations": 0.002,
                "tol_grad": 0.1,
                "simulation_files": [],
                "grad_files": {},
            }
        )
    )
    case = ProblemImportCase.from_directory(tmp_path)
    assert (case.llh, case.log_posterior, case.tol_llh) == (None, -12.5, 0.01)


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------
def test_the_suite_iterates_its_groups(tmp_path: Path) -> None:
    """A suite yields the cases of its groups and runs the compared ones."""
    _model_import_case(tmp_path)
    _initialization_case(
        tmp_path, "net1.parameters[layer1]", {"weight": 0 * WEIGHT, "bias": 0 * BIAS}
    )
    (tmp_path / "ml_model_import" / "README.md").write_text("not a case")
    suite = SciMLSuite(path=tmp_path, commit="0" * 40)

    assert suite.case_ids("ml_model_import") == ["001"]
    assert suite.case_ids("sciml_problem_import") == []
    assert [case.cid for case in suite.model_import_cases()] == ["001"]
    assert [case.cid for case in suite.initialization_cases()] == ["001"]
    assert [r.key for r in suite.run()] == ["ml_model_import/001", "initialization/001"]
    assert all(r.passed for r in suite.run())


def test_the_cache_is_per_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A commit is cached under its hash, and can be pointed elsewhere."""
    monkeypatch.delenv("SBMLSIM_SCIML_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    assert SciMLSuite.cache_path() == Path(
        f"/tmp/cache/sbmlsim/petab-sciml-testsuite/{SCIML_SUITE_COMMIT}"
    )
    assert len(SCIML_SUITE_COMMIT) == 40

    monkeypatch.setenv("SBMLSIM_SCIML_SUITE_PATH", "/elsewhere/test_cases")
    assert SciMLSuite.cache_path() == Path("/elsewhere/test_cases")


def test_the_suite_is_loaded_from_the_archive_of_a_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory `test_cases` of the archive becomes the cache."""
    archive = tmp_path / "suite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("petab_sciml_testsuite-abc/README.md", "suite")
        zf.writestr(
            "petab_sciml_testsuite-abc/test_cases/ml_model_import/001/solutions.yaml",
            "net_file: net.yaml\n",
        )
        zf.writestr(
            "petab_sciml_testsuite-abc/test_cases/initialization/001/solutions.yaml",
            "tol: 0.001\n",
        )
    monkeypatch.delenv("SBMLSIM_SCIML_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("sbmlsim.sciml.testsuite.SCIML_SUITE_URL", archive.as_uri())

    assert SciMLSuite.cached("abc") is None
    suite = SciMLSuite.load("abc")

    assert suite.path == tmp_path / "cache/sbmlsim/petab-sciml-testsuite/abc"
    assert suite.case_ids("ml_model_import") == ["001"]
    assert suite.case_ids("initialization") == ["001"]
    assert SciMLSuite.cached("abc") == suite


def test_an_archive_which_is_not_the_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive without the groups is an error and is not cached."""
    archive = tmp_path / "suite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("something/README.md", "not the suite")
    monkeypatch.delenv("SBMLSIM_SCIML_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("sbmlsim.sciml.testsuite.SCIML_SUITE_URL", archive.as_uri())

    with pytest.raises(OSError, match=r"No directory 'ml_model_import'"):
        SciMLSuite.load("abc")
    assert SciMLSuite.cached("abc") is None
````

Create `tests/sciml/test_testsuite.py`:

````python
"""The cases of the PEtab SciML test suite.

Every case is a test, so a failure names the case. They are marked
`sciml_testsuite` and deselected by default, because they need the download
of the suite: `pytest -m sciml_testsuite` runs them and `tox r -e sciml`
downloads the cases first. A case is compared with
`tests/data/sciml_baseline.json`, which records the cases that do not pass,
each with its reason.

The baseline fails in both directions. A case which passed and now fails is a
regression. A case which is in the baseline and passes is a baseline which is
out of date.
"""

import json
from pathlib import Path

import pytest

from sbmlsim.sciml.testsuite import (
    INITIALIZATION,
    MODEL_IMPORT,
    SCIML_SUITE_COMMIT,
    CaseResult,
    InitializationCase,
    ModelImportCase,
    SciMLSuite,
)

pytestmark = pytest.mark.sciml_testsuite

#: the expected outcome of the cases which do not pass
BASELINE_PATH = Path(__file__).parent.parent / "data" / "sciml_baseline.json"

SKIP_REASON = (
    f"the PEtab SciML test suite '{SCIML_SUITE_COMMIT}' is not on this machine, "
    f"run `uv run python scripts/sciml_testsuite.py download` to fetch it"
)


def _case_ids(group: str) -> list[str]:
    """Get the cases of a group of the cached suite, for the parametrization."""
    suite = SciMLSuite.cached()
    return [] if suite is None else suite.case_ids(group)


MODEL_IMPORT_IDS = _case_ids(MODEL_IMPORT)
INITIALIZATION_IDS = _case_ids(INITIALIZATION)


@pytest.fixture(scope="session")
def baseline() -> dict:
    """Get the expected outcome of the cases."""
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def suite() -> SciMLSuite:
    """Get the cached suite, skipping the tests when it was not downloaded."""
    cached = SciMLSuite.cached()
    if cached is None:
        pytest.skip(SKIP_REASON)
    return cached


def _check(result: CaseResult, baseline: dict) -> None:
    """Compare the outcome of a case with the baseline."""
    expected = baseline["expected_failures"].get(result.key)
    if expected is None:
        assert result.passed, (
            f"'{result.key}' is a regression, it passed before: "
            f"{result.status.value} {result.message}"
        )
    else:
        assert not result.passed, (
            f"'{result.key}' passes now, it is expected to fail with "
            f"'{expected['status']}' ({expected['reason']}). Refresh the "
            f"baseline: uv run python scripts/sciml_testsuite.py baseline"
        )
        assert result.status.value == expected["status"], (
            f"'{result.key}' fails differently than recorded: "
            f"'{expected['status']}' -> '{result.status.value}' ({result.message})"
        )


@pytest.mark.parametrize("cid", MODEL_IMPORT_IDS)
def test_model_import(cid: str, suite: SciMLSuite, baseline: dict) -> None:
    """The forward pass of the case has the outcome the baseline records."""
    case = ModelImportCase.from_directory(suite.path / MODEL_IMPORT / cid)
    _check(case.run(), baseline)


@pytest.mark.parametrize("cid", INITIALIZATION_IDS)
def test_initialization(cid: str, suite: SciMLSuite, baseline: dict) -> None:
    """The nominal values of the case have the outcome the baseline records."""
    case = InitializationCase.from_directory(suite.path / INITIALIZATION / cid)
    _check(case.run(), baseline)


def test_the_baseline_matches_the_suite(suite: SciMLSuite, baseline: dict) -> None:
    """The baseline was recorded for the commit which is pinned and cached."""
    assert baseline["suite_commit"] == SCIML_SUITE_COMMIT == suite.commit
    assert [f"{i:03d}" for i in range(1, 55)] == MODEL_IMPORT_IDS
    assert INITIALIZATION_IDS == ["001", "002", "003"]
    assert baseline["n_cases"] == len(MODEL_IMPORT_IDS) + len(INITIALIZATION_IDS)
    assert baseline["n_passed"] == baseline["n_cases"] - len(
        baseline["expected_failures"]
    )
    keys = {f"{MODEL_IMPORT}/{cid}" for cid in MODEL_IMPORT_IDS}
    keys |= {f"{INITIALIZATION}/{cid}" for cid in INITIALIZATION_IDS}
    assert set(baseline["expected_failures"]) <= keys


def test_every_failure_has_a_reason(baseline: dict) -> None:
    """A case is not listed without saying why it does not pass."""
    for key, expected in baseline["expected_failures"].items():
        assert expected["status"] != "pass", key
        assert len(expected["reason"]) > 20, key
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -x tests/sciml/test_suite_cases.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'sbmlsim.sciml.testsuite'`.

- [ ] **Step 3: Write the cases and the suite**

Create `src/sbmlsim/sciml/testsuite.py`:

````python
"""The cases of the PEtab SciML test suite on disk.

The [test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) has
three groups of cases, every case is a directory named by its number with a
`solutions.yaml`:

| group | case | compared |
| --- | --- | --- |
| `ml_model_import` | `ModelImportCase` | the outputs of the forward pass |
| `initialization` | `InitializationCase` | the nominal values of the arrays |
| `sciml_problem_import` | `ProblemImportCase` | likelihood, simulations, gradient |

The suite has no releases, so it is pinned by a commit. `SciMLSuite` is the
directory of the groups with the download and the cache of the commit.

The arrays of the suite are in the PyTorch layout, which is the layout of
`sbmlsim`: the axis orders `input_order_py` and `output_order_py` of a case
name the axes of the arrays as they are stored, nothing is permuted.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from petab.v2.core import MappingTable, ParameterTable, ProblemConfig
from petab.v2.extensions.sciml import SciMLConfig
from petab_sciml.constants import ARRAY

from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError
from sbmlsim.sciml.network import Network, NetworkParameters, load_array_data
from sbmlsim.sciml.parameters import nominal_parameters
from sbmlsim.testsuite import cache

logger = logging.getLogger(__name__)

#: commit of the test suite the tests run against
SCIML_SUITE_COMMIT = "0622bbfc5e12eb9b482659eabd1756ca0e87dfc8"

#: archive of a commit of the suite
SCIML_SUITE_URL = (
    "https://github.com/PEtab-dev/petab_sciml_testsuite/archive/{commit}.zip"
)

#: the environment variable which points at the cases when they are not in
#: the cache, i.e. at the directory `test_cases` of a checkout
SCIML_SUITE_PATH_VARIABLE = "SBMLSIM_SCIML_SUITE_PATH"

#: the groups of cases, which are the directories of the suite
MODEL_IMPORT = "ml_model_import"
INITIALIZATION = "initialization"
PROBLEM_IMPORT = "sciml_problem_import"

#: absolute tolerance of the outputs of a network. The cases of the group
#: `ml_model_import` do not state one, these are the tolerances the suite
#: checks its own reference values with (`pysrc/ml_import_helper.py`)
MODEL_IMPORT_TOLERANCE = 1e-3

#: absolute tolerance of a case with dropout, whose reference values are the
#: mean of forward passes in training mode
DROPOUT_TOLERANCE = 1e-2


class CaseStatus(StrEnum):
    """The outcome of a case."""

    PASS = "pass"
    TOLERANCE = "tolerance"
    SHAPE = "shape"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


@dataclass(frozen=True)
class CaseResult:
    """The outcome of a case.

    Attributes:
        group: the group of the case.
        cid: the number of the case, e.g. `001`.
        status: the outcome.
        message: what failed, empty for a case which passes.
        max_difference: the largest absolute difference to the reference
            values, `None` when nothing was compared.
    """

    group: str
    cid: str
    status: CaseStatus
    message: str = ""
    max_difference: float | None = None

    @property
    def passed(self) -> bool:
        """Check whether the case passes."""
        return self.status == CaseStatus.PASS

    @property
    def key(self) -> str:
        """Get the key of the case in the baseline, e.g. `ml_model_import/001`."""
        return f"{self.group}/{self.cid}"


def read_solutions(path: Path) -> dict[str, Any]:
    """Read the `solutions.yaml` of a case.

    Args:
        path: the directory of the case.

    Returns:
        The content of the file.

    Raises:
        ValueError: if the directory has no `solutions.yaml` or the file is
            not a mapping.
    """
    solutions_path = path / "solutions.yaml"
    if not solutions_path.is_file():
        raise ValueError(f"The case '{path}' has no 'solutions.yaml'")
    solutions = yaml.safe_load(solutions_path.read_text(encoding="utf-8"))
    if not isinstance(solutions, dict):
        raise ValueError(f"'{solutions_path}' is not a mapping")
    return solutions


def read_array(path: Path, group: str) -> np.ndarray:
    """Read the single array of an input or output file of a case.

    Args:
        path: the HDF5 file.
        group: `inputs` or `outputs`.

    Returns:
        The array in the PyTorch layout, in double precision.

    Raises:
        ValueError: if the file does not hold exactly one array in the group
            or is not in the PyTorch layout.
    """
    arrays: list[np.ndarray] = []
    pytorch_format: list[bool] = []

    def collect(name: str, item: object) -> None:
        if not isinstance(item, h5py.Dataset):
            return
        if name == "metadata/pytorch_format":
            pytorch_format.append(bool(item[()]))
        elif name.startswith(f"{group}/"):
            arrays.append(np.asarray(item[()], dtype=float))

    with h5py.File(path, "r") as f:
        f.visititems(collect)
    if pytorch_format != [True]:
        raise ValueError(f"'{path}' is not in the PyTorch layout")
    if len(arrays) != 1:
        raise ValueError(f"'{path}' holds {len(arrays)} arrays in '{group}', not one")
    return arrays[0]


def compare_arrays(
    observed: np.ndarray, expected: np.ndarray, tolerance: float
) -> tuple[CaseStatus, str, float | None]:
    """Compare an array with its reference values.

    Args:
        observed: the values of `sbmlsim`.
        expected: the reference values.
        tolerance: the absolute tolerance.

    Returns:
        The outcome, the message and the largest absolute difference.
    """
    if observed.shape != expected.shape:
        return (
            CaseStatus.SHAPE,
            f"the shape is {observed.shape}, expected {expected.shape}",
            None,
        )
    if observed.size == 0:
        return CaseStatus.PASS, "", 0.0
    difference = float(np.max(np.abs(observed - expected)))
    if not np.all(np.isfinite(observed)) or difference > tolerance:
        return (
            CaseStatus.TOLERANCE,
            f"the largest difference {difference:.3g} is above the tolerance "
            f"{tolerance:.3g}",
            difference,
        )
    return CaseStatus.PASS, "", difference


def _worst(
    group: str, cid: str, outcomes: list[tuple[CaseStatus, str, float | None]]
) -> CaseResult:
    """Get the result of a case from the outcomes of its comparisons.

    Args:
        group: the group of the case.
        cid: the number of the case.
        outcomes: outcome, message and largest difference of every comparison.

    Returns:
        The first comparison which fails, or the pass with the largest
        difference of all comparisons.
    """
    for status, message, difference in outcomes:
        if status != CaseStatus.PASS:
            return CaseResult(group, cid, status, message, difference)
    differences = [d for _, _, d in outcomes if d is not None]
    return CaseResult(
        group, cid, CaseStatus.PASS, "", max(differences) if differences else None
    )


@dataclass(frozen=True)
class ModelImportCase:
    """A case of the group `ml_model_import`.

    A case is a network with combinations of inputs, arrays and the outputs
    the network has for them. The combination `i` is the input `i`, the
    arrays `i` and the output `i`.

    Attributes:
        cid: the number of the case, e.g. `001`.
        path: the directory of the case.
        net_file: the NN YAML.
        inputs: the input files of every combination, one per input of the
            network.
        parameters: the array file of every combination, empty for a network
            without arrays.
        outputs: the output file of every combination.
        input_order: the axes of the inputs, e.g. `["C", "H", "W"]`.
        output_order: the axes of the outputs.
        dropout: the number of forward passes in training mode the reference
            values are the mean of, `None` for a case without dropout.
    """

    cid: str
    path: Path
    net_file: Path
    inputs: list[list[Path]]
    parameters: list[Path]
    outputs: list[Path]
    input_order: list[str] = field(default_factory=list)
    output_order: list[str] = field(default_factory=list)
    dropout: int | None = None

    @classmethod
    def from_directory(cls, path: Path) -> ModelImportCase:
        """Read a case from its directory.

        Args:
            path: the directory of the case, named by its number.

        Returns:
            The case.

        Raises:
            ValueError: if the directory has no `solutions.yaml` or the file
                does not list inputs and outputs.
        """
        solutions = read_solutions(path)
        if "net_input" in solutions:
            columns = [solutions["net_input"]]
        else:
            keys = sorted(
                (key for key in solutions if key.startswith("net_input_arg")),
                key=lambda key: int(key.removeprefix("net_input_arg")),
            )
            columns = [solutions[key] for key in keys]
        if not columns or "net_output" not in solutions:
            raise ValueError(f"The case '{path}' lists no inputs or no outputs")
        return cls(
            cid=path.name,
            path=path,
            net_file=path / solutions.get("net_file", "net.yaml"),
            inputs=[
                [path / name for name in row] for row in zip(*columns, strict=True)
            ],
            parameters=[path / name for name in solutions.get("net_ps", [])],
            outputs=[path / name for name in solutions["net_output"]],
            input_order=list(solutions.get("input_order_py", [])),
            output_order=list(solutions.get("output_order_py", [])),
            dropout=solutions.get("dropout"),
        )

    @property
    def tolerance(self) -> float:
        """Get the absolute tolerance of the outputs."""
        return MODEL_IMPORT_TOLERANCE if self.dropout is None else DROPOUT_TOLERANCE

    def network(self, i: int) -> Network:
        """Get the network with the arrays of a combination.

        Args:
            i: the index of the combination.

        Returns:
            The network.
        """
        array_path = self.parameters[i] if self.parameters else None
        return Network.from_files(self.net_file, array_path)

    def run(self) -> CaseResult:
        """Evaluate the network for every combination and compare the outputs.

        Returns:
            The outcome of the case. It does not raise: a layer without an
            implementation is `UNSUPPORTED` and any other error is `ERROR`.
        """
        outcomes: list[tuple[CaseStatus, str, float | None]] = []
        try:
            for i, output_path in enumerate(self.outputs):
                inputs = [read_array(path, "inputs") for path in self.inputs[i]]
                expected = read_array(output_path, "outputs")
                for order, array in (
                    (self.input_order, inputs[0]),
                    (self.output_order, expected),
                ):
                    if order and len(order) != array.ndim:
                        raise ValueError(
                            f"an array with {array.ndim} axes does not have "
                            f"the axes {order}"
                        )
                (observed,) = self.network(i).forward(*inputs)
                outcomes.append(compare_arrays(observed, expected, self.tolerance))
        except UnsupportedLayerError as err:
            return CaseResult(MODEL_IMPORT, self.cid, CaseStatus.UNSUPPORTED, str(err))
        except Exception as err:
            return CaseResult(MODEL_IMPORT, self.cid, CaseStatus.ERROR, str(err))
        return _worst(MODEL_IMPORT, self.cid, outcomes)


def parameter_key(model_entity_id: str) -> str | None:
    """Get the key of an entry from a `modelEntityId` of the mapping table.

    Args:
        model_entity_id: the entity, e.g. `net1.parameters[layer1].weight`.

    Returns:
        The key of `sbmlsim.sciml.parameters`, i.e. `net1` for
        `net1.parameters`, `net1.layer1` for `net1.parameters[layer1]` and
        `net1.layer1.weight` for `net1.parameters[layer1].weight`. `None` for
        an entity which is not the parameters of a network.
    """
    network, separator, rest = model_entity_id.partition(".parameters")
    if not separator or not network:
        return None
    if not rest:
        return network
    if not rest.startswith("["):
        return None
    layer, closing, array = rest[1:].partition("]")
    if not closing or not layer:
        return None
    if not array:
        return f"{network}.{layer}"
    if not array.startswith(".") or len(array) == 1:
        return None
    return f"{network}.{layer}{array}"


@dataclass(frozen=True)
class InitializationCase:
    """A case of the group `initialization`.

    A case is a PEtab SciML problem whose parameter table sets the nominal
    values of a part of a network, and the arrays the network has after the
    import.

    Attributes:
        cid: the number of the case, e.g. `001`.
        path: the directory of the case.
        problem_path: the YAML of the problem.
        tolerance: the absolute tolerance of the arrays.
        parameter_files: id of the network -> the file with its reference
            values.
    """

    cid: str
    path: Path
    problem_path: Path
    tolerance: float
    parameter_files: dict[str, Path]

    @classmethod
    def from_directory(cls, path: Path) -> InitializationCase:
        """Read a case from its directory.

        Args:
            path: the directory of the case, named by its number.

        Returns:
            The case.

        Raises:
            ValueError: if the directory has no `solutions.yaml`.
        """
        solutions = read_solutions(path)
        return cls(
            cid=path.name,
            path=path,
            problem_path=path / "petab" / "problem.yaml",
            tolerance=float(solutions["tol"]),
            parameter_files={
                network: path / name
                for network, name in solutions["parameter_files"].items()
            },
        )

    def nominal(self) -> dict[str, NetworkParameters]:
        """Import the networks of the problem with their nominal values.

        The problem is read with the classes of `petab` which need no
        `torch`: the configuration, the parameter table and the mapping
        table. The rows of the parameter table which the mapping table
        resolves to the parameters of a network are the entries of
        `nominal_parameters`; the `nominalValue` `array` keeps the values of
        the array file.

        Returns:
            id of the network -> the arrays of the network.

        Raises:
            NetworkImportError: if the problem is not a PEtab SciML problem,
                a network is not in the format `YAML` or an array has no
                values.
        """
        base = self.problem_path.parent
        raw = yaml.safe_load(self.problem_path.read_text(encoding="utf-8"))
        config = ProblemConfig(**raw, base_path=base)
        sciml = (config.extensions or {}).get("sciml")
        if not isinstance(sciml, SciMLConfig):
            raise NetworkImportError(f"'{self.problem_path}' has no 'sciml' extension")

        keys: dict[str, str] = {}
        for mapping_file in config.mapping_files:
            table = MappingTable.from_tsv(base / str(mapping_file))
            for mapping in table.elements:
                key = parameter_key(mapping.model_id or "")
                if key is not None:
                    keys[mapping.petab_id] = key
        values: dict[str, float] = {}
        for parameter_file in config.parameter_files:
            for parameter in ParameterTable.from_tsv(
                base / str(parameter_file)
            ).elements:
                value = parameter.nominal_value
                if parameter.id in keys and value is not None and value != ARRAY:
                    values[keys[parameter.id]] = float(value)

        nominal: dict[str, NetworkParameters] = {}
        for sid, network_config in (sciml.neural_networks or {}).items():
            if network_config.format.lower() != "yaml":
                raise NetworkImportError(
                    f"Network '{sid}': the format '{network_config.format}' "
                    f"is not supported, only 'YAML'"
                )
            network = Network.from_files(base / str(network_config.location), sid=sid)
            for array_file in sciml.array_files:
                array_path = base / str(array_file)
                if sid in load_array_data(array_path).parameters:
                    network.parameters = network.read_arrays(array_path)
            nominal[sid] = nominal_parameters(
                network,
                {
                    key: value
                    for key, value in values.items()
                    if key == sid or key.startswith(f"{sid}.")
                },
            )
        return nominal

    def expected(self) -> dict[str, NetworkParameters]:
        """Read the reference values of the arrays.

        Returns:
            id of the network -> the arrays of the network.
        """
        expected: dict[str, NetworkParameters] = {}
        for sid, path in self.parameter_files.items():
            data = load_array_data(path)
            expected[sid] = {
                layer: {
                    name: np.asarray(array, dtype=float)
                    for name, array in arrays.items()
                }
                for layer, arrays in data.parameters[sid].items()
            }
        return expected

    def run(self) -> CaseResult:
        """Import the networks and compare their nominal values.

        Returns:
            The outcome of the case. It does not raise: a layer without an
            implementation is `UNSUPPORTED` and any other error is `ERROR`.
        """
        outcomes: list[tuple[CaseStatus, str, float | None]] = []
        try:
            nominal = self.nominal()
            for sid, layers in self.expected().items():
                for layer, arrays in layers.items():
                    for name, expected in arrays.items():
                        observed = nominal.get(sid, {}).get(layer, {}).get(name)
                        if observed is None:
                            outcomes.append(
                                (
                                    CaseStatus.ERROR,
                                    f"the array '{sid}.{layer}.{name}' was "
                                    f"not imported",
                                    None,
                                )
                            )
                            continue
                        status, message, difference = compare_arrays(
                            observed, expected, self.tolerance
                        )
                        if message:
                            message = f"'{sid}.{layer}.{name}': {message}"
                        outcomes.append((status, message, difference))
        except UnsupportedLayerError as err:
            return CaseResult(
                INITIALIZATION, self.cid, CaseStatus.UNSUPPORTED, str(err)
            )
        except Exception as err:
            return CaseResult(INITIALIZATION, self.cid, CaseStatus.ERROR, str(err))
        return _worst(INITIALIZATION, self.cid, outcomes)


@dataclass(frozen=True)
class ProblemImportCase:
    """A case of the group `sciml_problem_import`.

    The case reads its files. Its comparison needs the log-likelihood, the
    simulation and the gradient of a hybrid problem and is not part of this
    class yet.

    Attributes:
        cid: the number of the case, e.g. `001`.
        path: the directory of the case.
        problem_path: the YAML of the problem.
        llh: the log-likelihood at the nominal values, `None` for a case with
            priors, which states `log_posterior`.
        log_posterior: the log-posterior at the nominal values, `None` for a
            case without priors.
        simulation_files: the simulations at the measurement points.
        gradient_files: `mech` or the id of a network -> the gradient of the
            mechanistic parameters (TSV) or of the network (HDF5).
        tol_llh: tolerance of the log-likelihood or the log-posterior.
        tol_simulations: tolerance of the simulations.
        tol_grad: tolerance of the gradient.
    """

    cid: str
    path: Path
    problem_path: Path
    llh: float | None
    log_posterior: float | None
    simulation_files: list[Path]
    gradient_files: dict[str, Path]
    tol_llh: float
    tol_simulations: float
    tol_grad: float

    @classmethod
    def from_directory(cls, path: Path) -> ProblemImportCase:
        """Read a case from its directory.

        Args:
            path: the directory of the case, named by its number.

        Returns:
            The case.

        Raises:
            ValueError: if the directory has no `solutions.yaml`.
        """
        solutions = read_solutions(path)
        llh = solutions.get("llh")
        log_posterior = solutions.get("log_posterior")
        return cls(
            cid=path.name,
            path=path,
            problem_path=path / "petab" / "problem.yaml",
            llh=None if llh is None else float(llh),
            log_posterior=None if log_posterior is None else float(log_posterior),
            simulation_files=[
                path / name for name in solutions.get("simulation_files", [])
            ],
            gradient_files={
                key: path / name
                for key, name in solutions.get("grad_files", {}).items()
            },
            tol_llh=float(solutions.get("tol_llh", solutions.get("tol_log_posterior"))),
            tol_simulations=float(solutions["tol_simulations"]),
            tol_grad=float(solutions["tol_grad"]),
        )


@dataclass(frozen=True)
class SciMLSuite:
    """The cases of a commit of the PEtab SciML test suite.

    Attributes:
        path: the directory which holds the directories of the groups, i.e.
            `test_cases` of the suite.
        commit: the commit of the suite.
    """

    path: Path
    commit: str

    @staticmethod
    def cache_path(commit: str = SCIML_SUITE_COMMIT) -> Path:
        """Get the directory a commit of the suite is unpacked into.

        `SBMLSIM_SCIML_SUITE_PATH` overrides it. Otherwise it is
        `sbmlsim/petab-sciml-testsuite/<commit>` in the user cache.

        Args:
            commit: the commit of the suite.

        Returns:
            The directory the groups of the commit live in.
        """
        return cache.cache_path(
            SCIML_SUITE_PATH_VARIABLE, "petab-sciml-testsuite", commit
        )

    @classmethod
    def cached(cls, commit: str = SCIML_SUITE_COMMIT) -> SciMLSuite | None:
        """Get a commit of the suite if it is already on this machine.

        Args:
            commit: the commit of the suite.

        Returns:
            The suite, or `None` if it was not downloaded yet.
        """
        path = cls.cache_path(commit)
        return cls(path=path, commit=commit) if path.is_dir() else None

    @classmethod
    def load(cls, commit: str = SCIML_SUITE_COMMIT) -> SciMLSuite:
        """Get a commit of the suite, downloading it if it is not cached.

        Args:
            commit: the commit of the suite.

        Returns:
            The suite with its cases unpacked in the cache.

        Raises:
            OSError: if the commit cannot be downloaded.
        """
        suite = cls.cached(commit)
        if suite is not None:
            return suite
        path = cls.cache_path(commit)
        url = SCIML_SUITE_URL.format(commit=commit)
        logger.info("Downloading the PEtab SciML test suite '%s'", commit)
        cache.fetch(url, path, select=cls._cases_dir)
        return cls(path=path, commit=commit)

    @staticmethod
    def _cases_dir(staging: Path) -> Path:
        """Get the directory of the groups of an unpacked archive.

        Args:
            staging: directory the archive was unpacked into.

        Returns:
            The directory which holds the directory `ml_model_import`.

        Raises:
            OSError: if the archive holds no such directory.
        """
        for candidate in sorted(staging.glob(f"**/{MODEL_IMPORT}")):
            if candidate.is_dir():
                return candidate.parent
        raise OSError(
            f"No directory '{MODEL_IMPORT}' in the unpacked suite '{staging}'"
        )

    def case_ids(self, group: str) -> list[str]:
        """Get the numbers of the cases of a group.

        Args:
            group: the group, i.e. the name of its directory.

        Returns:
            The names of the case directories, sorted, empty for a group the
            suite does not have.
        """
        directory = self.path / group
        if not directory.is_dir():
            return []
        return sorted(
            p.name for p in directory.iterdir() if p.is_dir() and p.name.isdigit()
        )

    def model_import_cases(self) -> Iterator[ModelImportCase]:
        """Iterate the cases of the group `ml_model_import`."""
        for cid in self.case_ids(MODEL_IMPORT):
            yield ModelImportCase.from_directory(self.path / MODEL_IMPORT / cid)

    def initialization_cases(self) -> Iterator[InitializationCase]:
        """Iterate the cases of the group `initialization`."""
        for cid in self.case_ids(INITIALIZATION):
            yield InitializationCase.from_directory(self.path / INITIALIZATION / cid)

    def problem_import_cases(self) -> Iterator[ProblemImportCase]:
        """Iterate the cases of the group `sciml_problem_import`."""
        for cid in self.case_ids(PROBLEM_IMPORT):
            yield ProblemImportCase.from_directory(self.path / PROBLEM_IMPORT / cid)

    def run(self) -> list[CaseResult]:
        """Run the cases which are compared.

        Returns:
            The results of the groups `ml_model_import` and `initialization`,
            in this order.
        """
        results = [case.run() for case in self.model_import_cases()]
        results.extend(case.run() for case in self.initialization_cases())
        return results
````

- [ ] **Step 4: Run the tests which need no download**

Run: `uv run pytest -q -x tests/sciml`
Expected: `269 passed`, the tests of `test_testsuite.py` are deselected.

- [ ] **Step 5: Write the command line**

Create `scripts/sciml_testsuite.py`:

````python
"""Download and run the PEtab SciML test suite.

The cases of the [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite)
say which networks and which hybrid problems `sbmlsim` supports. This script
is the command line around `sbmlsim.sciml.testsuite`:

```bash
# fetch the pinned commit into the cache, which the tests need
uv run python scripts/sciml_testsuite.py download

# run the cases and report the outcome
uv run python scripts/sciml_testsuite.py run

# refresh the expected outcomes after a change which fixes or breaks cases
uv run python scripts/sciml_testsuite.py baseline
```
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sbmlsim import log
from sbmlsim.console import console
from sbmlsim.sciml.testsuite import (
    SCIML_SUITE_COMMIT,
    CaseResult,
    CaseStatus,
    SciMLSuite,
)

#: the expected outcomes the tests compare a run with
BASELINE_PATH = Path(__file__).parent.parent / "tests" / "data" / "sciml_baseline.json"

#: the reason of a case which is new in the baseline. The tests reject it, so
#: a case is not listed until somebody wrote down why it does not pass
MISSING_REASON = "reason missing"


def run(suite: SciMLSuite) -> list[CaseResult]:
    """Run the cases of the suite and report the outcome."""
    console.print(
        f"Running the PEtab SciML test suite '{suite.commit}' from {suite.path}"
    )
    results = suite.run()
    counts = Counter(r.status for r in results)
    console.print(f"[bold]{counts[CaseStatus.PASS]}/{len(results)} cases pass[/bold]")
    for result in results:
        if not result.passed:
            console.print(f"  {result.key}  {result.status.value}  {result.message}")
    return results


def write_baseline(results: list[CaseResult], suite: SciMLSuite, path: Path) -> None:
    """Write the cases which do not pass, keeping the reasons which are recorded.

    Args:
        results: the results of a run.
        suite: the suite which was run.
        path: the baseline.
    """
    reasons: dict[str, str] = {}
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        reasons = {
            key: expected["reason"]
            for key, expected in recorded["expected_failures"].items()
        }
    failures = {
        r.key: {"status": r.status.value, "reason": reasons.get(r.key, MISSING_REASON)}
        for r in results
        if not r.passed
    }
    baseline = {
        "suite_commit": suite.commit,
        "n_cases": len(results),
        "n_passed": len(results) - len(failures),
        "expected_failures": failures,
    }
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    console.print(f"Baseline: {path}")
    for key, expected in failures.items():
        if expected["reason"] == MISSING_REASON:
            console.print(f"  [red]{key}: write the reason into the baseline[/red]")


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "run", "baseline"])
    parser.add_argument("--commit", default=SCIML_SUITE_COMMIT)
    args = parser.parse_args(argv)

    log.enable_rich_logging()
    suite = SciMLSuite.load(args.commit)
    if args.command == "download":
        console.print(f"PEtab SciML test suite '{suite.commit}': {suite.path}")
        return 0

    results = run(suite)
    if args.command == "baseline":
        write_baseline(results, suite, BASELINE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

- [ ] **Step 6: Write the baseline**

Create `tests/data/sciml_baseline.json`:

```json
{
  "suite_commit": "0622bbfc5e12eb9b482659eabd1756ca0e87dfc8",
  "n_cases": 57,
  "n_passed": 56,
  "expected_failures": {
    "ml_model_import/020": {
      "status": "tolerance",
      "reason": "AlphaDropout: the reference values are the mean of 40000 forward passes in training mode, which for AlphaDropout is not the input. sbmlsim evaluates dropout in evaluation mode, where it is the identity (gap sciml-training-mode)"
    }
  }
}
```

- [ ] **Step 7: Download the suite and run its cases**

Run: `uv run python scripts/sciml_testsuite.py download`
Expected: the path `~/.cache/sbmlsim/petab-sciml-testsuite/0622bbfc5e12eb9b482659eabd1756ca0e87dfc8` is printed. On a machine without a network set `SBMLSIM_SCIML_SUITE_PATH` to the directory `test_cases` of a checkout of the commit.

Run: `uv run python scripts/sciml_testsuite.py run`
Expected: `56/57 cases pass` and the one line `ml_model_import/020  tolerance  the largest difference 0.515 is above the tolerance 0.01`.

Run: `uv run pytest -q -x -m sciml_testsuite tests/sciml`
Expected: `59 passed` (54 + 3 cases and the two tests of the baseline).

Run: `uv run python scripts/sciml_testsuite.py baseline && git diff --stat tests/data/sciml_baseline.json`
Expected: no difference, the baseline which is written is the one of step 6. If a case fails which the baseline does not list, this is a defect of a layer of the tasks 2 to 8: find it with the test of the layer against PyTorch, do not list the case.

- [ ] **Step 8: Add the tox environment**

In `tox.ini` add behind the section `[testenv:testsuite]`:

```ini
[testenv:sciml]
# the cases of the PEtab SciML test suite, which a normal run deselects. The
# cases are fetched first, without them the tests would only skip
commands =
    python {toxinidir}/scripts/sciml_testsuite.py download
    pytest -m sciml_testsuite tests/sciml
```

Run: `uvx --with tox-uv tox r -e sciml`
Expected: `59 passed` and `sciml: OK`.

- [ ] **Step 9: Run the cases before a release**

In `.github/workflows/ci-cd.yml`, in the job `testsuite`, add behind the step `Run the semantic cases against the pinned release`:

```yaml
    - name: Run the cases of the PEtab SciML test suite against the pinned commit
      # against the pinned commit, which is the one the baseline records
      run: uvx --with tox-uv tox -e sciml
```

- [ ] **Step 10: Describe the package in `CLAUDE.md`**

Four edits, every paragraph stays a single line.

In the paragraph below `## Project` replace the sentence

```
`amici`, `basico`/COPASI, `h5py`, `pypesto` and `juliacall` are not dependencies; the comparison scripts and examples which use them import them optionally.
```

with

```
`petab-sciml` (which brings `h5py`) is the extra `sciml`, which `sbmlsim.sciml` needs, and `torch` is only in the `dev` extra as the reference of the tests. `amici`, `basico`/COPASI, `pypesto` and `juliacall` are not dependencies; the comparison scripts and examples which use them import them optionally.
```

In the code block below `## Commands` add behind the line `tox r -e testsuite ...`:

```
pytest -m sciml_testsuite                          # the cases of the PEtab SciML test suite, deselected by default
tox r -e sciml                                     # the same, with the cases downloaded first
```

Behind the paragraph which starts with `**`testsuite/`` add a new paragraph (one line, with an empty line in front of it and behind it):

```
**`sciml/` - neural networks of hybrid problems.** The package needs the extra `sciml` (`petab-sciml`), its `__init__.py` names the extra when it is missing, and no other module of `sbmlsim` imports it at import time. It knows nothing of PEtab. `Network` (`sciml/network.py`) is the architecture of a network as the `NNModel` of `petab_sciml` and its arrays in the PyTorch layout, `layer id -> array name -> values`; `from_files` reads the NN YAML and the HDF5 array file, `forward` evaluates in evaluation mode with numpy and without torch, `parameter_ids` gives every element of an array its id `<net>__<layer>__<array>__<index>`, a valid SBML `SId`, and `with_values` returns a copy of the arrays with elements replaced. `evaluate` (`sciml/interpreter.py`) is the one interpreter of the nodes of the forward pass (`placeholder`, `call_module`, `call_function`, `call_method`, `output`), and every layer and function is implemented once against a `Backend` (`sciml/backend.py`) and registered under its PyTorch name in `LAYERS` or `FUNCTIONS` (`sciml/layers/registry.py`) with the backends it supports: `layers/core.py` (`Linear`, `Bilinear`, `Flatten`, dropout as the identity) and `layers/functions.py` (the activations, `flatten`, `cat`) support both backends, `layers/convolution.py`, `layers/pooling.py` and `layers/normalization.py` are numpy only. `Backend.select` is the only function with a condition, a backend on sympy expressions implements it as a `Piecewise`. A normalization layer uses its stored statistics (`running_mean`, `running_var`) when the arrays hold them and the statistics of its input otherwise. `sciml/parameters.py` resolves the entries a problem gives for the network, a layer or an array (`net1`, `net1.layer1`, `net1.layer1.weight`, the more specific entry wins) into the nominal values (`nominal_parameters`) and into one `FitParameter` per estimated element (`network_fit_parameters`). `sciml/testsuite.py` is the PEtab SciML test suite, pinned by `SCIML_SUITE_COMMIT` and cached under `~/.cache/sbmlsim/petab-sciml-testsuite/<commit>/` (`SBMLSIM_SCIML_SUITE_PATH` overrides it) through `testsuite/cache.py`, which `SemanticSuite` shares: `ModelImportCase` compares the outputs of the forward pass, `InitializationCase` the nominal values after the import, `ProblemImportCase` reads its files and is compared in a later phase. `scripts/sciml_testsuite.py` is the command line (`download`, `run`, `baseline`), `tests/sciml/test_testsuite.py` is one test per case against `tests/data/sciml_baseline.json` with the marker `sciml_testsuite`, and the tests against PyTorch in `tests/sciml/` need no download and run in every session.
```

Below `## Conventions` replace

```
- An optional dependency which is never installed (amici, basico, pypesto, juliacall, h5py) is imported with `# ty: ignore[unresolved-import]`.
```

with

```
- An optional dependency which is never installed (amici, basico, pypesto, juliacall) is imported with `# ty: ignore[unresolved-import]`. `torch` is installed with the `dev` extra but not in the tox environment of ty, the tests import it with `pytest.importorskip("torch")`.
```

Run: `git diff CLAUDE.md | grep '^+' | grep -c $'\u2014'`
Expected: `0`, the added lines hold no em dash. The existing headings of the architecture section keep theirs, they are not edited.

- [ ] **Step 11: Run everything**

Run: `uv run pytest -q -x`
Expected: all tests pass, the cases of both test suites are deselected.

Run: `uv run ruff check && uv run ruff format --check && uvx ty check`
Expected: zero diagnostics.

Run: `uvx --with tox-uv tox r -e ty`
Expected: `ty: OK`. This is the environment of the required check `ty`, it has `petab-sciml` through the extra and no torch.

Run: `uvx --with tox-uv tox r -e py3.14`
Expected: `py3.14: OK`, the log shows the installation of the CPU build of torch in `commands_pre`.

Run: `uv run pytest -q -rs tests/sciml`
Expected: `269 passed` and no skipped test, i.e. the comparisons with PyTorch ran.

- [ ] **Step 12: Commit**

```bash
git add src/sbmlsim/sciml/testsuite.py scripts/sciml_testsuite.py tests/data/sciml_baseline.json tests/sciml tox.ini .github/workflows/ci-cd.yml CLAUDE.md
git commit -m "sciml: the PEtab SciML test suite with its baseline and its tox environment"
```

---

## Done when

- `uv run pytest -q` passes, `uv run ruff check`, `uv run ruff format --check` and `uvx ty check` are at zero diagnostics.
- `uv run pytest -q -m sciml_testsuite tests/sciml` reports `59 passed`: `ml_model_import` 001 to 054 and `initialization` 001 to 003 pass, except `ml_model_import/020`, which `tests/data/sciml_baseline.json` lists with its reason.
- `uvx --with tox-uv tox r -e sciml` passes.
- `python -c "import sbmlsim, sbmlsim.fit"` works in an environment without the extra, and `import sbmlsim.sciml` names the extra there.
