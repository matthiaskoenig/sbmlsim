# PEtab SciML

[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) extends [PEtab v2](petab.md) to hybrid problems, in which a model is combined with neural networks. `sbmlsim` reads, simulates, fits and writes such problems. You need the `sciml` extra, which installs `petab-sciml`:

```bash
pip install sbmlsim[sciml]
```

## Try it

Two examples show the whole workflow. Both write into `results/` of the working directory.

```bash
python -m examples.sciml.lotka_volterra_fit
python -m examples.sciml.neural_ode.fitting --runs=2 --cores=2
```

`lotka_volterra_fit` reads case 001 of the [test suite](sciml_testsuite.md), a Lotka-Volterra model with a network in place of the interaction term of the predator. It fits the model parameters and the network together, starting from the values of the problem (`SamplingType.START`, one run), writes the report, writes the fitted problem as PEtab SciML and reads it back.

`neural_ode` defines a neural ODE in python: a network 2-5-5-2 with 57 elements gives the rates of a Lotka-Volterra system. It is trained on the first four seconds of data and validated on the two seconds after them. It fits once from the values of the network and once as a multistart from random values within the bounds of the elements on several cores, and writes the better fit as PEtab SciML.

## Networks and where they sit

`sbmlsim.sciml` has the building blocks. A `Network` holds the architecture and the arrays of a network, and `Network.forward` evaluates it with numpy. A `Hybridization` says where the network sits in the problem, in one of three patterns (`NetworkPattern`):

| pattern | the network is evaluated | inputs | outputs |
| --- | --- | --- | --- |
| `pre_initialization` | once per simulation, with numpy, before the simulation | constants: formulas of parameters, arrays | changes of the simulation, i.e. parameters or initial values |
| `rhs` | by roadrunner at every step, compiled into the model | formulas of species, parameters and time, arrays | parameters of the rate equations |
| `observable` | by roadrunner at every step, compiled into the model | formulas of species, parameters and time, arrays | symbols of an observable |

For `rhs` and `observable`, `compile_network` writes the network into the model as parameters with assignment rules, one rule per unit. The output of an `rhs` network is a parameter of the model, which becomes variable; a rule or event that also sets it is refused. A layer that has no MathML (convolution, pooling, normalization, and `gelu` with the error function) cannot be compiled, so a network with such a layer has to run before the simulation.

A compiled network costs time at load and at every step. As measured: a `Linear`-`tanh`-`Linear` network with 5 units per layer (51 elements) loads in 0.1 s and simulates 101 points in 1.4 ms, against 0.6 ms without it. With 20 units per layer (501 elements) it loads in 2 s and simulates in 9 ms, with 50 units per layer (2751 elements) it loads in 44 s and simulates in 32 ms. Load time grows faster than the number of units, so a large network belongs before the simulation. A `softmax` is compiled to `n²` terms for `n` units, and a `log_softmax` needs a maximum, which before SBML L3V2 is a piecewise with `n²` conditions: 8 units load in 25 s at L3V1 and in 1.2 s at L3V2, and 16 units are unusable before L3V2.

## Fitting a hybrid problem

The simplest way is to read a PEtab SciML problem with `from_petab`, which gives a problem you simulate, evaluate and fit like any other (see [Reading a PEtab problem](petab.md#reading-a-petab-problem)). The reader writes the model with the compiled networks as `<stem>_sciml.xml` next to the model; `derived_dir` of `from_petab` or `PetabReader.from_yaml` puts it elsewhere.

How the reader translates a problem:

- Every element of a network that is not frozen is estimated and becomes a parameter of the fit, on the linear scale.
- A network before the simulation needs every element to come from a parameter of the fit. A parameter of the fit must not write a frozen element or a network output.
- The inputs of a network are those of the first period of an experiment. The reader refuses a network input that a later period sets, which includes the main period after a pre-equilibration.
- The scale of a parameter is read from the `sbmlsim` block of the problem. PEtab SciML problems still carry the `parameterScale` column of PEtab v1, which is used when the block has no scale.
- Networks are read from the NN YAML and the array files without `torch`.

You can also define a hybrid fit in python with the same objects. `Hybridization.fit_parameters(estimate, bounds)` returns one `FitParameter` per estimated element, starting from the value the network carries, and the hybridization with every other element frozen. Pass the hybridizations to `OptimizationProblem(..., hybridizations=[...])`. `network_fit_parameters(network, estimate, bounds, external=...)` is the same for a network without a hybridization, and freezes nothing. `nominal_parameters` sets the values for a whole network, a layer or an array, and `dataclasses.replace(network, parameters=...)` gives them to the network.

## Writing a hybrid problem

`to_petab` writes a problem with hybridizations as PEtab SciML: the model the problem was defined with, the `sciml` block of the YAML, the hybridization table, the mapping table rows for the inputs, outputs and arrays of every network, the parameter table rows that say which arrays are estimated with which bounds, the NN YAML `<net>.yaml` and the array file `<net>_arrays.hdf5`. The values of the network go into the array file. With `to_petab(..., parameter_set=...)` the values of the parameter set are written instead.

Details of the export:

- A model with compiled networks or formula observables records what was added to it (`sbmlsim.model.provenance`), and the exporter writes the original model. A derived model that was changed by hand afterwards is refused.
- An input whose formula holds for every condition becomes a row of the hybridization table. An input with a formula per condition becomes a change of the conditions.
- A condition and an input before the simulation may only use parameters of the parameter table. A model parameter the fit does not estimate is written as a parameter that is not estimated; any other model entity, e.g. a compartment, is refused (gap `sciml-input-formula`).
- PEtab does not allow an id in both the parameter table and the conditions, so a simulation that changes a parameter of the fit is refused, with the parameter, experiment and simulation named.
- A row of PEtab SciML describes a whole array. A fit that estimates only some elements of an array, or bounds them differently, is refused with the name of the array (gap `sciml-partial-array`). Every estimated element must start from the value of its network, on the linear scale.

The round trip is exact: every case of the test suite that is read, and every problem defined in python in the tests, reads back to the same parameters, hybridizations, data and log-likelihood.

## The size of a network

The optimizer is a least squares fit with a jacobian from finite differences, so one iteration costs one simulation per parameter: a network with `n` elements costs `n + 1` simulations per iteration. Measured on the neural ODE example with a fixed grid and tolerances of `1e-10`:

- 57 elements: a simulation takes 2 ms and an iteration 0.15 s. A fit of the first four seconds from a random start converges in 80 evaluations, about 15 s.
- 162 elements: a simulation takes 4 ms and an iteration 0.7 s. A fit over three oscillations does not converge from a random start within 150 iterations, which is the usual difficulty of a neural ODE over a long window.
- 2751 elements in the right hand side: a simulation takes 32 ms, about 90 s per iteration.

So a network of a few hundred elements is what this kind of fit is made for. Networks with thousands of elements need gradients from sensitivities or automatic differentiation and the optimizers of machine learning frameworks, which `sbmlsim` does not provide.

Two settings matter for networks:

- The default finite difference step of `sbmlsim.fit.cli` (`diff_step=0.05`, relative, made for parameters on a log scale) is too coarse for elements near zero. The examples pass `diff_step=1e-4` and `x_scale="jac"` to `run_optimization` and `run_fit`.
- An element without bounds starts from the value of the network in every run of a multistart. Give the elements bounds when the runs should start from different values, and use `SamplingType.START` to fit from the values of the network.

## The report of a hybrid fit

The console output and the report do not list the elements of a network one by one. The overview names every network with its pattern, layers and targets, and shows one row per array: the number of elements, how many are estimated, the bounds when the elements share them, and the minimum, maximum and norm of the values of every parameter set. The parameter table keeps the model parameters, and `report.txt` has one line per array and set. The stored parameter sets (`parameters.json`) and runs (`optimization_result.tsv`, `optimization_result.json`) keep every element.

Warnings and analyses summarize arrays too: a warning about parameters at a bound counts the elements of an array (`3 of the 25 elements of 'net1.layer2.weight' within 5% of a bound`), the Fisher information shows an array as one row with the norm of its values and the range of the standard errors of its elements, and the correlation matrix leaves the elements out. `profile_likelihood` and `identifiability_cli` profile only the parameters that are not network elements, unless `pids` or `--parameter` name an element: one scan per element is impractical for hundreds of elements, while the Fisher information covers them with one jacobian (see [Identifiability](identifiability.md)).

## The layers

The layers and functions of PEtab SciML (see the [table of PEtab SciML](https://petab-sciml.readthedocs.io/latest/layers.html), `-` where it does not list a layer for a tool) and the backends of `sbmlsim`. A layer with only the `numpy` backend runs before the simulation. A layer with both backends can also be compiled into the model, i.e. sit in the right hand side or in an observable.

Every network is evaluated in evaluation mode (gap `sciml-training-mode`): dropout is the identity, and the normalization layers use their stored statistics, or the statistics of their input when the arrays hold none. `gelu` with the error function (`approximate="none"`, the default) is evaluated but not compiled, because the error function has no MathML (gap `sciml-layer-sbml`); `approximate="tanh"` is compiled.

| layer | PEtab.jl | AMICI | sbmlsim |
| --- | --- | --- | --- |
| `Linear` | yes | yes | numpy, sympy |
| `Bilinear` | yes | - | numpy, sympy |
| `Flatten` | yes | yes | numpy, sympy |
| `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`, `AlphaDropout` | yes | - | numpy, sympy (the identity) |
| `FeatureAlphaDropout` | - | - | numpy, sympy (the identity) |
| `Conv1d`, `Conv2d`, `Conv3d` | yes | yes | numpy |
| `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` | yes | yes | numpy |
| `MaxPool1d`, `MaxPool2d`, `MaxPool3d` | yes | yes | numpy |
| `AvgPool1d`, `AvgPool2d`, `AvgPool3d` | yes | yes | numpy |
| `LPPool1d`, `LPPool2d`, `LPPool3d` | yes | yes | numpy |
| `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d` | yes | yes | numpy |
| `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d` | yes | yes | numpy |
| `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d` | - | - | numpy (evaluation mode) |
| `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d` | - | - | numpy (evaluation mode) |
| `LayerNorm` | - | - | numpy |

| function | PEtab.jl | AMICI | sbmlsim |
| --- | --- | --- | --- |
| `relu`, `relu6`, `hardtanh`, `hardswish`, `hardsigmoid`, `leaky_relu` | yes | yes | numpy, sympy |
| `selu`, `elu`, `celu`, `softplus`, `softsign`, `tanhshrink`, `mish`, `silu` | yes | yes | numpy, sympy |
| `tanh`, `sigmoid` | yes | yes | numpy, sympy |
| `log_sigmoid` (`logsigmoid`) | - | - | numpy, sympy |
| `gelu` | yes | yes | numpy, sympy (compiled with `approximate="tanh"` only) |
| `softmax`, `log_softmax` | yes | yes | numpy, sympy (see the cost of `log_softmax` above) |
| `flatten`, `cat` (`concat`, `concatenate`) | - | - | numpy, sympy |

## What is not supported

The differences between `sbmlsim` and PEtab SciML are listed in the catalogue of gaps, `sbmlsim.fit.petab_v2.gaps`. The cases of the [PEtab SciML Test Suite](sciml_testsuite.md) that do not pass are listed with the gap they run into.

- `priors`: the prior of a model parameter is its `FitParameter.prior`, which `log_prior` evaluates and the optimizer does not use (issue #190).
- `sciml-priors`: priors on the parameters of a network are not supported; the reader raises. The cases with such priors state a log-posterior, which the log-likelihood is not (issue #190).
- `sciml-model-format`: networks in the formats `pytorch`, `equinox` or `lux.jl` are not read, only the NN YAML.
- `sciml-layer-sbml`: a layer without MathML cannot sit in the right hand side or in an observable.
- `sciml-training-mode`: dropout and the normalization layers are evaluated in evaluation mode, while the reference values of the suite for dropout are built in training mode.
- `sciml-parameter-scale`: the scale of a single parameter has no place in PEtab v2 and goes to the `sbmlsim` extension.
- `sciml-partial-array`: a fit that estimates only some elements of an array, or bounds them differently, cannot be written; the export raises and names the array.
- `sciml-input-formula`: an input of a network before the simulation, or with a formula per condition, may only use parameters of the parameter table. Any other model entity, e.g. a compartment, makes the export raise and name the network, the input and the entity.
