# PEtab SciML

[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) is the extension of [PEtab v2](petab.md) for hybrid problems, in which the model is combined with neural networks. The extra `sciml` (`pip install sbmlsim[sciml]`, which brings `petab-sciml`, `h5py` and `pyyaml`) is required to read one. `sbmlsim.sciml` is the native half: a `Network` is the architecture and the arrays of a network, which `Network.forward` evaluates with numpy, and a `Hybridization` says where it sits, in one of three patterns:

| pattern | the network is evaluated | inputs | outputs |
| --- | --- | --- | --- |
| `pre_initialization` | once per simulation, with numpy, before the simulation | constants: formulas of parameters, arrays | changes of the simulation, i.e. parameters or initial values |
| `rhs` | by roadrunner at every step, compiled into the model | formulas of species, parameters and time, arrays | parameters of the rate equations |
| `observable` | by roadrunner at every step, compiled into the model | formulas of species, parameters and time, arrays | symbols of an observable |

`compile_network` compiles a network of the last two patterns into the model as parameters with assignment rules, one rule per unit and one layer deep, so the size of the model grows with the number of units and not with the depth of the network. The target of an output in the right hand side is a parameter of the model, which becomes variable; a rule or an event which sets it is refused. A layer without MathML (convolution, pooling, normalization, `gelu` with the error function) cannot be compiled, such a network runs before the simulation. A softmax is compiled without a maximum, which roadrunner inlines to `n**2` terms for `n` units, and a `log_softmax` needs the maximum in every unit, which before SBML L3V2 is a piecewise with `n**2` conditions: measured, 8 units load in 25 s at L3V1 and in 1.2 s at L3V2, and 16 units are unusable before L3V2.

A hybrid fit is defined in python with the same objects: `Hybridization.fit_parameters(estimate, bounds)` gives one parameter of the fit per estimated element, with the value the network carries as its start value, and the hybridization with every other element frozen, and `OptimizationProblem(..., hybridizations=[...])` runs the networks. `network_fit_parameters(network, estimate, bounds, external=...)` is the primitive it and the reader share, for networks without a hybridization; it freezes nothing. The nominal values are the values of the network, which `nominal_parameters` sets for the network, a layer or an array and `dataclasses.replace(network, parameters=...)` gives to the network, so the frozen elements run with the values the fit parameters start from.

The reader translates a problem with the `sciml` extension into these objects: the model with the networks is written as `<stem>_sciml.xml`, next to the model by default, `derived_dir` of the reader redirects it, the elements of the networks are parameters of the fit on the linear scale, and `from_petab` gives a problem which is simulated, evaluated and fitted like any other. An element of a network which is not frozen is estimated and therefore a parameter of the fit, a network before the simulation refuses an element which no parameter of the fit provides, and a parameter of the fit must not write a frozen element or an output. The inputs of a network are the ones of the first period of an experiment: the reader refuses a network input which a condition of a later period sets, which includes a problem whose main period after a pre-equilibration sets the inputs of a network. The scale of a parameter (`FitParameter.scale`) is written as `scale` into the `sbmlsim` block and read from it, and the problems of PEtab SciML carry the column `parameterScale` of PEtab v1, which becomes the scale of the parameter when the block does not give one. The networks are read from the NN YAML and the array files without `torch`, which `petab` needs for them.

The cost of a compiled network is the cost of its rules: a `Linear`-`tanh`-`Linear` network of 5 units per layer (51 elements) loads in `0.1 s` and simulates 101 points in `1.4 ms` against `0.6 ms` for the model without it, 20 units per layer (501 elements) load in `2 s` and simulate in `9 ms`, and 50 units per layer (2751 elements) load in `44 s` and simulate in `32 ms`. roadrunner compiles every rule when it loads the model, so the time to load grows faster than the number of units, and a large network belongs before the simulation.

## Writing a hybrid problem

`to_petab` writes a problem with hybridizations as PEtab SciML: the model the problem was defined with, the `sciml` block of the YAML, the hybridization table, the rows of the mapping table which name the inputs, the outputs and the arrays of every network, the rows of the parameter table which say which arrays are estimated with which bounds (one row for the network with the description most of its arrays share, rows for the layers or the arrays which differ), the NN YAML `<net>.yaml` and the array file `<net>_arrays.hdf5` with the arrays of the network and of its inputs, keyed by the condition of the experiment. A model which carries compiled networks or formula observables records what was added to it (`sbmlsim.model.provenance`), and the exporter writes the model it was derived from; a derived model which was changed by hand after the derivation is refused. The values of the network are its nominal values, so the exported problem carries them in the array file and no numeric rows, and `to_petab(..., parameter_set=...)` writes the values of the set into the array file instead; the scale of a parameter goes to the `sbmlsim` block.

An input of a network whose formula holds for every condition is a row of the hybridization table, which PEtab SciML evaluates along the trajectory, and an input with a formula per condition is a change of the conditions of the experiments. A parameter which only the rows of the hybridization table use, e.g. a constant of a hybridization, gets a row of the mapping table without a `modelEntityId`, so that the linter of `petab` counts it as used. A condition and an input before the simulation use only the parameters of the parameter table: a parameter of the model which the fit does not estimate is written as a row which is not estimated, any other entity of the model is the gap `sciml-input-formula`. PEtab does not allow an id in both tables, so the export refuses a simulation whose changes set a parameter of the parameter table, i.e. a parameter of the fit, and names the parameter, the experiment and the simulation.

The round trip is exact: every case of the test suite which is read (`tox r -e sciml` runs `test_problem_round_trip`) and every python defined problem of `tests/sciml/test_export.py` reads back to the same parameters, hybridizations, data and log-likelihood. What cannot be written is refused with the name of the array: a fit which estimates some elements of an array, or bounds them differently, is the gap `sciml-partial-array`, because a row of PEtab SciML describes an array as a whole, and an element which is estimated must start from the value of its network, on the linear scale; `network_fit_parameters` and `Hybridization.fit_parameters(estimate, bounds)` describe the elements per network, layer and array, which is what PEtab SciML expresses. A hook which is no network is refused as well.

`examples/sciml/lotka_volterra_fit.py` reads the case 001 of the test suite, fits it in one process from the values of the problem (`SamplingType.START`, one run, because a problem which is read builds its experiment at runtime, which the workers of a parallel fit cannot import), reports it, writes it with the fitted parameter set and reads it back. `examples/sciml/neural_ode/` defines a neural ODE in python, a network 2-5-5-2 of 57 elements in the right hand side of the model, trained on the first four seconds of a Lotka-Volterra system and validated on the two seconds after them; it compiles the network into the `base_path` of the problem, so that the workers of a parallel fit load the same file, fits it once from the values of the network (`SamplingType.START`) and as a multistart from random values in the bounds of the elements on several cores, and writes the better of the two fits as PEtab SciML (`python -m examples.sciml.neural_ode.fitting --runs=2 --cores=2`). Both write into `results/` of the working directory.

## The report of a hybrid fit

The console and the report do not list the elements of a network one by one: the overview names every network with its pattern, its layers and its targets, and shows one row per array with the number of its elements, the number of them the fit estimates, the bounds when the elements agree on them and the minimum, the maximum and the norm of the values of every parameter set; the parameter table keeps the parameters of the model, and `report.txt` has one line per array and set. The parameter sets (`parameters.json`) and the runs of the optimization (`optimization_result.tsv`, `optimization_result.json`) keep every element. A warning about a parameter at a bound counts the elements of an array (`3 of the 25 elements of 'net1.layer2.weight' within 5% of a bound`), the table of the Fisher information shows an array as one row with the norm of its values and the range of the standard errors of its elements, and the correlation matrix leaves the elements out. `profile_likelihood` and `identifiability_cli` profile the parameters which are no elements of a network, unless `pids` or `--parameter` name an element: a profile is one optimization scan per parameter, which a network of hundreds of elements makes impractical, while the Fisher information is one jacobian and covers them.

## The size of a network

The optimizer stays a least squares fit whose jacobian is built by finite differences: an iteration costs one simulation per parameter, so a network with `n` elements costs `n + 1` simulations per iteration. Measured on the neural ODE of the example with a fixed grid and the tolerances `1e-10`: a network of 57 elements simulates in 2 ms, an iteration takes 0.15 s and a fit of the first four seconds of the Lotka-Volterra system from a random initialization converges in 80 evaluations, i.e. 15 s; a network of 162 elements simulates in 4 ms, an iteration takes 0.7 s, and the same fit over three oscillations does not converge from a random initialization within 150 iterations (2 minutes, the cost falls from 91 to 70), which is the well known difficulty of a neural ODE over a long window and not a matter of the jacobian. A network of a few hundred elements is the size a fit with the finite difference jacobian is made for; a network of thousands of elements (2751 elements in the right hand side simulate in 32 ms, see above, i.e. 90 s per iteration) needs the gradients by sensitivities or automatic differentiation and the optimizers of the machine learning frameworks, which are out of scope. The step of the finite differences matters: the default of `sbmlsim.fit.cli` (`diff_step=0.05`, relative, made for parameters on a logarithmic scale) is too coarse for elements around zero, the examples pass `diff_step=1e-4` and `x_scale="jac"` to `run_optimization` and `run_fit`. An element without bounds starts from the value of the network in every run of a multistart; give the elements bounds when the runs should start from different values, and use `SamplingType.START` for a fit from the values of the network.

## The layers

The layers and the functions of PEtab SciML (the [table of PEtab SciML](https://petab-sciml.readthedocs.io/latest/layers.html), `-` where it does not list a layer for a tool) with the backends of `sbmlsim`: a layer of the `numpy` backend runs before the simulation, a layer of both backends is also compiled into the model, i.e. can sit in the right hand side or in an observable. `gelu` with the error function (`approximate="none"`, the default) is evaluated on expressions but not compiled, the error function has no MathML (gap `sciml-layer-sbml`); its approximation `approximate="tanh"` is compiled. Dropout is the identity and the normalization layers use their stored statistics, or the statistics of their input when the arrays hold none, i.e. every network is evaluated in evaluation mode (gap `sciml-training-mode`).

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
| `softmax`, `log_softmax` | yes | yes | numpy, sympy (see the size of `log_softmax` above) |
| `flatten`, `cat` (`concat`, `concatenate`) | - | - | numpy, sympy |

## What is not supported

What `sbmlsim` does not express of a hybrid problem is in the catalogue of the gaps, `sbmlsim.fit.petab_v2.gaps`, and the cases of the [PEtab SciML Test Suite](sciml_testsuite.md) which do not pass are listed with the gap they run into:

- `priors`: a prior on a parameter of the model is dropped with a warning which names the parameter, the objective of `sbmlsim` has none (issue #190)
- `sciml-priors`: the cases with priors on the parameters of a network state a log-posterior, which the log-likelihood is not, and wait for issue #190
- `sciml-model-format`: a network in the format `pytorch`, `equinox` or `lux.jl`, which is not read
- `sciml-layer-sbml`: a layer without MathML in the right hand side or in an observable
- `sciml-training-mode`: dropout and the normalization layers are evaluated in evaluation mode, the reference values of the suite are built in training mode for dropout
- `sciml-parameter-scale`: the scale of one parameter, which goes to the extension
- `sciml-partial-array`: a fit which estimates some elements of an array or bounds them differently, which a row of the parameter table of PEtab SciML cannot say; the export raises and names the array
- `sciml-input-formula`: an input of a network before the simulation, or with a formula per condition, which uses an entity of the model other than a parameter the fit does not estimate, e.g. a compartment; a condition of PEtab uses only the parameters of the parameter table, the export raises and names the network, the input and the entity
