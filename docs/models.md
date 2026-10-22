# Models

`sbmlsim` simulates models in the [Systems Biology Markup Language](https://sbml.org) (SBML) with [libroadrunner](https://libroadrunner.org). This guide shows how a model is loaded, what the package reads from it and how the model is changed before a simulation.

## Loading a model

The simulator loads a model from a path or an SBML string, with the integrator settings of the simulator:

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = simulator.load(REPRESSILATOR_SBML)
print(model)
```

`sbmlsim.resources` provides the three models used throughout the documentation and the tests: `REPRESSILATOR_SBML`, the repressilator of Elowitz and Leibler, `DEMO_SBML`, a small demo model with compartments, and `MIDAZOLAM_SBML`, a whole body pharmacokinetics model of midazolam.

The model is a `RoadrunnerSBMLModel`, which owns the roadrunner instance `r`, the units of the model and the selections, i.e., the variables recorded in a simulation:

```python
from sbmlsim.model import RoadrunnerSBMLModel

model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
print(model.r)  # the roadrunner.RoadRunner instance
print(model.selections)
```

The model can be created explicitly and passed to the simulator, which is useful when several simulators or experiments share a model.

## Compiling a model

roadrunner compiles the code of a model when it is loaded, with LLJIT at an optimization level of LLVM, `JitOptimization`: `O0` compiles fastest, the higher levels optimize the code for the simulation. `O2` is the default and simulates as fast as roadrunner's default JIT, MCJIT, which compiles slower; `O0` gives the results of MCJIT bitwise and pays for a run which compiles more than it simulates, e.g. the initialization of the experiments of a model. On the dulaglutide model of pkdb_models `O0` compiles in 1.1 s and `O2` in 4.1 s, while a scan of 300 doses takes 4.5 s at `O0` and 3.9 s at `O2`.

A model takes its level as `jit_optimization`, an `ExperimentRunner` and an `OptimizationProblem` compile all their models at theirs, and the environment variable `SBMLSIM_JIT_OPTIMIZATION` sets the level of everything which is not given one:

```python
from sbmlsim.model import JitOptimization, RoadrunnerSBMLModel

model = RoadrunnerSBMLModel(
    source=REPRESSILATOR_SBML, jit_optimization=JitOptimization.O0
)
```

```bash
SBMLSIM_JIT_OPTIMIZATION=O0 pytest
```

The workers of a scan and of a fit compile a model at the level of the parent, whatever their environment says.

## Units of a model

The units of every parameter, species and compartment are read from the SBML and stored as a `UnitsInformation`, a mapping from identifier to unit string. All changes and results of a simulation carry these units, see [Units](units.md):

```python
uinfo = model.uinfo
print(uinfo["X"])  # unit of the species X
print(uinfo["time"])
```

## Changes and selections

A `RoadrunnerSBMLModel` accepts `changes`, which every simulation of the model applies before the initialization unless the simulation sets the target itself, and `selections`, the variables recorded in a simulation. Changes are quantities with units or plain floats in the units of the model:

```python
model = RoadrunnerSBMLModel(
    source=REPRESSILATOR_SBML,
    changes={"X": 5.0, "Y": 10.0},
    selections=["time", "X", "Y", "Z"],
)
print(model.changes)
print(model.selections)
```

The roadrunner integrator is configured with `settings`, e.g., `settings={"absolute_tolerance": 1e-10}`, or later with `set_integrator_settings`; a `Simulator` sets its own integrator settings on every model it runs, and a setting of the model which the simulator does not set, e.g. `maximum_num_steps`, stays, also in the workers of a pool. The absolute tolerance is one value per state from the kind of the state, a concentration species gets the tolerance of a concentration times the initial volume of its compartment, see [Simulations](simulation.md#selections-and-integrator-settings). The vector depends on the initial volumes of the model as loaded, not on an earlier simulation; `RoadrunnerSBMLModel.tolerances()` lists it.

## Abstract models

A `SimulationExperiment` (see [Simulation experiments](experiments.md)) describes its models as an `AbstractModel`: the source and the changes without loading the model. The `ExperimentRunner` resolves the abstract models into roadrunner models when the experiment is run:

```python
from sbmlsim.model import AbstractModel

abstract_model = AbstractModel(
    source=REPRESSILATOR_SBML,
    changes={"X": 5.0},
)
print(abstract_model)

model = RoadrunnerSBMLModel.from_abstract_model(abstract_model)
print(model.r)
```

The source of a model is resolved by `sbmlsim.model.model_resources`: a path relative to the `base_path` of the experiment, an absolute path, or the SBML itself. A model is never downloaded, a source which is neither a path nor SBML raises an `OSError`.

## Inspecting a model

The parameters and species of the model with their current values and units are available as data frames:

```python
print(RoadrunnerSBMLModel.parameter_df(model.r).head())
print(RoadrunnerSBMLModel.species_df(model.r).head())
```

The ids of the constant parameters and of the species of the SBML which roadrunner simulates (with the parameters added to the model) are read once per instance of roadrunner and kept, so that a design over all parameters (`sampling.parameters_of`) does not parse the SBML again for every simulation:

```python
ids = model.sbml_ids()
print(ids.constant_parameters[:5], ids.species[:5])
```
