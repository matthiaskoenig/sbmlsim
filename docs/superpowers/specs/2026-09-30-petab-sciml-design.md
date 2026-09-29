# PEtab SciML: hybrid problems of an SBML model and neural networks

Support for [PEtab SciML](https://github.com/PEtab-dev/petab_sciml), the extension of PEtab v2 for problems in which a mechanistic model in SBML is hybridized with neural networks and the parameters of both are estimated together. This is the design for issue #207.

## The goal

A problem of the [PEtab SciML test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) is read into an `OptimizationProblem`, simulated with roadrunner, evaluated and fitted, and the log-likelihood, the simulations and the gradient match the reference values of the test suite within its tolerances. A problem which is read is written back without a difference. A hybrid problem can also be defined in python without PEtab, the PEtab layer only translates.

## The starting point

| | |
| --- | --- |
| reading the files | `petab` 0.9.0 ships `petab.v2.extensions.sciml` (`SciMLExt`, `SciMLConfig`, `HybridizationTable`), which reads the extension block, the hybridization table, the NN YAML and the HDF5 arrays with `petab_sciml`. `Problem.from_yaml` raises `ModuleNotFoundError` on a SciML problem today, because `petab_sciml` is not installed |
| mapping table | `PetabReader` does not read `petab_problem.mappings` |
| conditions | a `targetValue` which is neither a number nor the id of an estimated parameter raises (`reader.py:448`) |
| foreign extensions | `extension_of` reads only `extensions["sbmlsim"]`, any other block is ignored, also when it says `required: true` |
| likelihood | there is none, a numeric noise value becomes the standard deviation of the data (`reader.py:587`) and enters the weights |
| parameters of a fit | `_simulate_groups` merges `ParameterMapping.changes_for` into the changes of the first timecourse (`optimization.py:1324`), there is no place where a change is calculated from the parameters |
| scale | `FitSettings.parameter_scale` is one scale for all parameters, `LOG10` by default, which a weight with a negative value cannot use |
| observables | `petab_v2/observables.py` writes a formula observable into the model as a parameter with an assignment rule and stores the model as `<stem>_observables.xml` |
| roadrunner | an assignment rule follows a change of a parameter in every case. An initial assignment follows it only if the change is set as `init(w)`, a plain `r["w"] = 2.0` leaves it at the old value. `SimulatorSerial` sets the plain form |
| test suite | `SemanticSuite` (`testsuite/cases.py`) holds the download and the cache of the SBML Test Suite and is written for that suite alone |

## Decisions

| | |
| --- | --- |
| scope | one design, four phases, every phase with its own implementation plan, pull request and verification |
| where the networks live | a native package `sbmlsim.sciml`, which knows nothing of PEtab. `OptimizationProblem` takes the hybridizations, `fit/petab_v2` translates in both directions |
| reading and writing the files | `petab_sciml` and `petab.v2.extensions.sciml` wherever they have the function, own code only for what they do not have: the forward pass without torch and the compilation into SBML |
| math | sympy expressions, converted to MathML with `sbmlmath` |
| network in the right hand side | compiled into the SBML model as parameters with assignment rules |
| network in an observable | compiled into the SBML model in the same way |
| network before the simulation | evaluated in numpy, its outputs are changes of the simulation |
| log-likelihood | calculated for the evaluation of a problem, the optimizer stays a weighted least squares fit |
| gradient | central finite differences of the log-likelihood |
| training mode | not supported, dropout and the normalization layers are evaluated in evaluation mode |
| model formats | only `format: YAML`. `pytorch`, `equinox` and `lux.jl` are gaps |
| priors on network parameters | not part of this design, they wait for #190. The cases 032 to 034 are listed in the baseline with this reason |
| estimated noise parameters | stay the `noise-parameters` gap, the error model is its own design |

## The three patterns

A network sits in one of three places. The place decides how `sbmlsim` runs it.

| pattern | in PEtab | inputs | outputs | executed by |
| --- | --- | --- | --- | --- |
| `PRE_INITIALIZATION` | `pre_initialization: true` | constants: parameters of the parameter table, of a condition, estimated parameters, arrays | parameters and initial values, set before the simulation | numpy, once per simulation |
| `RHS` | `pre_initialization: false`, the outputs are targets of the hybridization table | formulas of species, parameters and time | parameters of the rate equations | roadrunner, as assignment rules |
| `OBSERVABLE` | `pre_initialization: false`, the outputs appear in an `observableFormula` | formulas of species, parameters and time | terms of the observable | roadrunner, as assignment rules |

The right hand side and the observable are one mechanism: an assignment rule is evaluated by the integrator wherever it is used. A network before the simulation is not compiled into initial assignments for two reasons. roadrunner evaluates an initial assignment again only for a change set as `init(key)`, which the simulator does not do, and the networks of this pattern are the ones which take arrays as input and use convolution layers, which MathML does not express.

A problem may hold several networks and mix the patterns.

## Dependencies

- A new extra `sciml = ["petab-sciml>=0.0.3"]`. It brings `h5py`, `mkstd` and `ruamel.yaml`. It is part of the `dev` extra.
- `torch` is only in the `dev` extra. The tests compare the forward pass of `sbmlsim` against `NNModel.to_pytorch_module()`. The package never imports it.
- `sbmlmath` becomes a declared dependency. It is installed today as a dependency of a dependency.
- `sbmlsim.sciml` imports `petab_sciml` at the top of its modules. `sbmlsim/sciml/__init__.py` catches the `ModuleNotFoundError` and raises an `ImportError` which names the extra: `pip install sbmlsim[sciml]`. The reader does the same before it hands a problem with a `sciml` block to `petab`. No other module of the package imports `sbmlsim.sciml` at import time.

## `sbmlsim.sciml`

### `network.py`

`Network` is the architecture and the arrays of one network.

```python
@dataclass
class Network:
    sid: str
    model: NNModel                                  # petab_sciml
    parameters: dict[str, dict[str, np.ndarray]]    # layer id -> array name -> values

    @classmethod
    def from_files(cls, yaml_path: Path, array_path: Path | None = None) -> Network: ...
    def forward(self, *inputs: np.ndarray, parameters: NetworkParameters | None = None) -> tuple[np.ndarray, ...]: ...
    def parameter_ids(self) -> dict[str, tuple[str, str, tuple[int, ...]]]: ...
    def with_values(self, values: Mapping[str, float]) -> NetworkParameters: ...
```

`parameters` holds the nominal values in the PyTorch layout, row major, which is what the HDF5 files store when `metadata/pytorch_format` is true. An array stored in the other layout is permuted when it is read, so that the objects of `sbmlsim` have one layout. An element which the array file does not provide and no row of the parameter table sets is an error of the import, `sbmlsim` does not initialize a network with random values.

`parameter_ids()` maps the id of every array element to its layer, array and index. `with_values` returns a copy of the parameters with the given elements replaced, which is what a fit needs.

### Identifiers

Every element of an array has one id, which is a valid SBML `SId`, the name of the `FitParameter` and the name in the parameter sets and the report.

| entity | id | example |
| --- | --- | --- |
| element of an array | `<net>__<layer>__<array>__<index>` | `net1__layer1__weight__0_1` |
| unit of a node of the forward pass | `<net>__<node>__<index>` | `net1__tanh_1__3` |
| input | `<net>__input<k>__<index>` | `net1__input0__1` |
| output | `<net>__output<k>__<index>` | `net1__output0__0` |

The index is the PyTorch index of the element with `_` between the axes. The double underscore separates the parts, because the ids of networks, layers and nodes contain single underscores. An id of the model which collides with one of these ids is an error of the compilation.

### `backend.py` and `layers.py`

The forward pass of the NN YAML is a list of nodes (`placeholder`, `call_module`, `call_method`, `call_function`, `output`). One interpreter walks the list and one implementation per layer and per function exists. Both are written against a backend, which provides the elementwise functions and the array type:

- `NumpyBackend` works on `float` arrays and is the forward pass.
- `SympyBackend` works on `object` arrays of sympy expressions and is the first half of the compiler.

Both use the numpy array operations (`@`, `reshape`, `sum`), which work on `object` arrays as well. They differ only in the elementwise functions (`np.tanh` and `sympy.tanh`) and in the functions with a condition (`relu` is `np.maximum` and a `Piecewise`). A layer is therefore implemented once and the compiled model cannot describe a different network than the forward pass.

Every layer declares the backends it supports. `Linear`, `Bilinear`, `Flatten`, the elementwise activations, `softmax` and `log_softmax` and the tensor operations of `petab_sciml.constants` support both. The convolution, the transposed convolution, the pooling, the normalization and the dropout layers support numpy only. Dropout is the identity and the normalization layers use their stored statistics.

A layer or a function without an implementation raises `UnsupportedLayerError` with the network, the node and the type.

### `hybridization.py`

`Hybridization` says where a network sits.

```python
class NetworkPattern(StrEnum):
    PRE_INITIALIZATION = "pre_initialization"
    RHS = "rhs"
    OBSERVABLE = "observable"

@dataclass
class NetworkInput:
    formula: str | None = None                      # "prey", "alpha * prey", "0.5"
    arrays: dict[str, np.ndarray] | None = None     # condition id -> array, "0" for all

@dataclass
class Hybridization:
    network: Network
    pattern: NetworkPattern
    model: str                                      # id of the model in the experiment
    inputs: dict[str, NetworkInput]                 # "<net>__input<k>__<index>" or "<net>__input<k>" for an array
    outputs: dict[str, str]                         # "<net>__output<k>__<index>" -> target
    frozen: set[str]                                # ids of the elements which are not estimated
```

The target of an output is an entity of the model for `RHS` and `PRE_INITIALIZATION`. For `OBSERVABLE` it is the symbol the observable formula uses, which the compiler adds to the model as a parameter.

`validate()` checks the shapes of the inputs and outputs against the network, that every target exists in the model, that a `RHS` or `OBSERVABLE` network has only formulas as inputs and layers of the sympy backend, and that a `PRE_INITIALIZATION` network has no input which depends on time or on a species.

### `compiler.py`

```python
def compile_network(sbml_path: Path, hybridizations: list[Hybridization], output_path: Path) -> Path
```

The compiler adds all `RHS` and `OBSERVABLE` networks of one model in one pass and writes one model.

1. Every element of the arrays becomes a parameter of the model with its nominal value, `constant="true"`, dimensionless.
2. Every input becomes a parameter with an assignment rule of its formula.
3. The interpreter runs with the `SympyBackend`. After every node the expressions of its units are replaced by symbols, and every unit becomes a parameter with an assignment rule of its expression. The expressions stay one layer deep, so the size of the model grows with the number of units and not with the depth of the network.
4. The output becomes a parameter with an assignment rule. The target gets the assignment rule `target = output`. A target which is a constant parameter is set to `constant="false"`. A target which already has a rule, or which an event or a reaction changes, is an error.
5. sympy expressions are converted to MathML with `sbmlmath`. The model is checked for consistency with libsbml and written to `output_path`.

The compiled model is a file, because the workers of a parallel fit load a model from its path. It is written into the directory the models with observables are written into (`derived_dir`), as `<stem>_sciml.xml`. `observables.py` runs on the compiled model afterwards when the problem also has formula observables.

Errors are `NetworkCompilationError`. The message names the network, the node and the reason.

### `parameters.py`

```python
def network_fit_parameters(network: Network, estimate: Mapping[str, bool], bounds: Mapping[str, tuple[float, float]], values: Mapping[str, float] | None = None) -> list[FitParameter]
```

`estimate`, `bounds` and `values` are given for the network (`net1`), for a layer (`net1.layer1`) or for an array (`net1.layer1.weight`). The more specific entry wins. A value replaces the nominal values of all elements the entry covers, which is how a problem sets a layer to `0.0` while the other layers keep the values of the array file (the `initialization` cases). The function returns one `FitParameter` per estimated element with the nominal value as start value, the linear scale and no unit, and the hybridization keeps the other elements as `frozen`.

## The fit

The changes to `sbmlsim.fit` are generic. None of them names a network.

### Derived changes

`OptimizationProblem` takes `hybridizations: list[Hybridization] | None`. They are part of the definition and are pickled with it.

`_simulate_groups` calls the derived changes after `ParameterMapping.changes_for` and before the simulation. For every `PRE_INITIALIZATION` hybridization of the model of the group:

1. The inputs are resolved. A formula is evaluated with `mathml.evaluate` on the values of the fit parameters, the changes of the simulation and the nominal values of the model, in this order of precedence. An array is selected by the condition of the simulation, `"0"` is the array for all conditions.
2. The parameters of the network are the nominal values with the current values of the estimated elements.
3. `forward` runs and its outputs are added to the changes of the first timecourse, with the unit of the target.

This covers the inputs which are estimated parameters (cases 005 and 006) without a special case: the evaluation is part of every call of `residuals`.

### Parameters which are not entities of the model

The elements of a `PRE_INITIALIZATION` network are not in the model. Their `FitParameter`s have the target `sciml:<id>`. `ParameterMapping` writes no change for a target with this prefix, and the derived changes read it. The elements of a compiled network are entities of the model and need nothing.

### Scale and bounds per parameter

- `FitParameter` gains `scale: ParameterScaleType | None = None`. `None` is the scale of `FitSettings`, so every existing definition means what it means today. `problem.to_scale` and `from_scale` convert every parameter with its own scale. The jacobian of `fisher.py` and the profiles of `identifiability.py` work in the space of the optimizer and use the same two functions.
- A parameter on the linear scale may have infinite bounds. scipy `least_squares` takes them. `_validate_parameters` raises for such a parameter when the optimizer is differential evolution, which needs a finite box.
- The start value of a parameter with an infinite bound is its nominal value in every repeat. `fit/sampling.py` does not sample it and no longer replaces the bound with `max_bound`. The repeats of a fit then differ in the start values of the bounded parameters.

### Report and console

The parameter table of the report and of the console lists the parameters of a fit. A network adds hundreds of rows, so the elements of a network are shown as one row per array (number of elements, number of estimated elements, minimum, maximum and norm of the values), and the TSV of the parameter sets keeps every element. The overview of a report names the networks with their pattern, their layers and their targets.

## The PEtab v2 layer

### `likelihood.py`

```python
def log_likelihood(problem: OptimizationProblem, parameters: ParameterSet | None = None) -> float
def gradient(problem: OptimizationProblem, parameters: ParameterSet | None = None, step: float = 1e-6) -> pd.Series
```

`log_likelihood` simulates the problem at the parameters, the nominal values by default, and sums the log density of every measurement over the training data. The distributions are the ones of PEtab v2: `normal`, `log-normal`, `laplace` and `log-laplace`. The noise formula is a number, a parameter of the parameter table or a formula of them, and is evaluated with sympy. A noise formula which holds an estimated parameter is evaluated at the value of the parameter set, it is not estimated.

`gradient` is the central finite difference of `log_likelihood` on the linear scale with the step `step * max(|x|, 1)`, which is the rule `fisher.py` uses. It returns one value per estimated parameter. The test suite gives the gradient of the mechanistic parameters as a table and the gradient of a network as arrays, so `Network.parameter_ids()` maps between the two.

The reader keeps the noise formula and the distribution of every observable in the resolved problem, in addition to the standard deviation it derives from them today.

### `reader.py`

- A foreign extension block with `required: true` which `sbmlsim` does not know raises. A block with `required: false` is ignored with a log message. This is a fix of the present behaviour and independent of SciML.
- The mapping table is read. Its rows resolve the ids of the hybridization table, of the parameter table and of the observable formulas to the inputs, outputs and parameters of the networks (`net1.inputs[0][1]`, `net1.outputs[0][0]`, `net1.parameters`, `net1.parameters[layer1]`, `net1.parameters[layer1].weight`).
- Every network of `problem.extensions.sciml` becomes a `Network` and a `Hybridization`. The pattern is `PRE_INITIALIZATION` when the block says so, `OBSERVABLE` when an output is used in an observable formula, and `RHS` otherwise. A network with outputs of both kinds is split into two hybridizations of one network.
- The rows of the parameter table which the mapping table resolves to the parameters of a network go through `network_fit_parameters`. Their `nominalValue` is `array`, i.e. the values of the array file, or a number, which is the value of every element the row covers.
- The models with `RHS` or `OBSERVABLE` networks are compiled before the observables are written.
- A problem which is read runs serially as today, because its experiment class is built at runtime.

### `export.py`

The exporter writes the `sciml` block, the hybridization table, the rows of the mapping table and of the parameter table, the NN YAML and the HDF5 arrays through `SciMLExt` and `petab_sciml`. It exports the model the problem was defined with and not the compiled one. The `sbmlsim` block keeps the scale of the parameters, the settings and the kinds as today.

One row of the parameter table stands for the elements it covers. The exporter writes the most general rows which describe the `estimate` and the bounds of all elements, i.e. one row for the network when all its elements agree, and rows for the layers or arrays which differ.

### `gaps.py`

| id | kind | meaning |
| --- | --- | --- |
| `sciml-model-format` | `UNSUPPORTED` | a network in the format `pytorch`, `equinox` or `lux.jl` |
| `sciml-layer-sbml` | `UNSUPPORTED` | a layer in a `RHS` or `OBSERVABLE` network which MathML does not express |
| `sciml-training-mode` | `LOSSY` | dropout and normalization layers are evaluated in evaluation mode |
| `sciml-priors` | `UNSUPPORTED` | priors on the parameters of a network, until #190 |
| `sciml-parameter-scale` | `EXTENSION` | the scale of a single parameter, which PEtab v2 does not have |

## The test suite

The test suite has no releases, so it is pinned by a commit.

- `testsuite/cache.py` takes the download, the staging directory with the atomic replace and the resolution of the cache path out of `SemanticSuite`. `SemanticSuite` uses it and keeps its interface.
- `sciml/testsuite.py` holds `SCIML_SUITE_COMMIT`, `SciMLSuite` and the three kinds of cases (`ModelImportCase`, `ProblemImportCase`, `InitializationCase`), which read their `solutions.yaml`. The cache is `~/.cache/sbmlsim/petab-sciml-testsuite/<commit>/`, `SBMLSIM_SCIML_SUITE_PATH` overrides it.
- `tests/sciml/test_testsuite.py` is one test per case with the marker `sciml_testsuite`, which `addopts` deselects like `testsuite`. `tox r -e sciml` downloads the cases and runs them. The `testsuite` job of `ci-cd.yml` runs them on a tag.
- `tests/data/sciml_baseline.json` lists the cases which do not pass, each with its reason. A test fails on a regression and on a case which passes and is still listed.

| group | compared | tolerance |
| --- | --- | --- |
| `ml_model_import` | the outputs of `forward` for every combination of input and parameters, with the axis order `input_order_py` and `output_order_py` | `tol` |
| `initialization` | the nominal parameters of the network after the import | `tol` |
| `sciml_problem_import` | `log_likelihood`, the simulations at the measurement points, `gradient` for the mechanistic parameters and for every network | `tol_llh`, `tol_simulations`, `tol_grad` |

The tests which need no download run in every test session:

- the forward pass against torch for every layer and function, with random parameters,
- the compiled model against the forward pass: the outputs of the assignment rules read from roadrunner equal `forward` at the same inputs,
- the derived changes, the scale per parameter and the unbounded parameters on a small problem,
- `log_likelihood` against values calculated by hand for every distribution,
- the round trip of a problem with a network of every pattern.

## Phases

Every phase is one implementation plan and one pull request against `develop`, and leaves the tree with passing tests, `ruff` and `ty` at zero diagnostics.

| phase | content | done when |
| --- | --- | --- |
| 1 | the extra, `sciml/network.py`, `backend.py`, `layers.py` with the numpy backend, `testsuite/cache.py`, `sciml/testsuite.py` | `ml_model_import` 001 to 054 and `initialization` 001 to 003 pass or are in the baseline with a reason |
| 2 | `petab_v2/likelihood.py`, the noise model in the resolved problem, the error on a required foreign extension | the tests of the likelihood pass, the hctz problem has a log-likelihood |
| 3 | the sympy backend, `compiler.py`, `hybridization.py`, `parameters.py`, the changes to the fit, the reader, the gaps | `sciml_problem_import` 001 to 039 pass, except 032 to 034 and the cases in the baseline with a reason |
| 4 | the exporter, the report and the console, `examples/sciml/`, the documentation | the round trip of every case which is read is exact, the examples run in `tests/examples` |

Inside phase 1 `Linear`, `Flatten` and the activations come first, because phase 3 needs nothing else. Phases 1 and 2 are independent of each other.

Phase 4 writes `examples/sciml/` with the Lotka-Volterra problem of case 001 and the neural ODE of the getting started tutorial, both fitted and reported. It adds the section on SciML to `docs/petab.md` with the table of the layers (the table of https://petab-sciml.readthedocs.io/latest/layers.html with a column for `sbmlsim` and the backends of every layer), the gaps, the API page `docs/api/sciml.md` and the references in `docs/references.md`. The column for `sbmlsim` in the documentation of PEtab SciML is requested upstream after the release.

## Risks

| risk | handling |
| --- | --- |
| the finite difference gradient does not reach `tol_grad` | the comparison runs with tight tolerances of the integrator. A case which still fails is listed in the baseline with the reason and the design of analytic sensitivities is a follow-up |
| `petab_sciml` is at 0.0.3 and its API moves | only `network.py`, the reader and the exporter import it. The extra pins the lower bound and the tests of the test suite show a break |
| the model with a compiled network is large | the size grows with the number of units. The networks of the test suite have 5 units per layer. The time of the simulation against the size of the network is measured in phase 3 and documented |
| a species or parameter of the model named like an id of a network | the compiler raises and names the id |
| the number of fit parameters | scipy `least_squares` builds the jacobian by finite differences, i.e. one simulation per parameter and step. This is the cost of keeping the optimizer, and the documentation says for which size of network it is usable |

## Out of scope

- Priors on the parameters of a network (#190).
- The negative log-likelihood as the objective of a fit and the estimation of noise parameters.
- Gradients by sensitivities or automatic differentiation, and optimizers for large networks (Adam, mini batches, multiple shooting, curriculum learning).
- The training mode of dropout and the normalization layers.
- Networks in the formats `pytorch`, `equinox` and `lux.jl`.
- Convolution, pooling and normalization layers in the right hand side or in an observable.
