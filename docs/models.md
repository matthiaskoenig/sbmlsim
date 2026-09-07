# Models

`sbmlsim` simulates models in the [Systems Biology Markup Language](https://sbml.org) (SBML) with [libroadrunner](https://libroadrunner.org). This guide shows how a model is loaded, what the package reads from it and how the model is changed before a simulation.

## Loading a model

The simulator loads a model from a path, a URL or an SBML string:

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulator import SimulatorSerial

simulator = SimulatorSerial(model=REPRESSILATOR_SBML)
print(simulator.model)
```

`sbmlsim.resources` provides the three models used throughout the documentation and the tests: `REPRESSILATOR_SBML`, the repressilator of Elowitz and Leibler, `DEMO_SBML`, a small demo model with compartments, and `MIDAZOLAM_SBML`, a whole body pharmacokinetics model of midazolam.

Behind the simulator is a `RoadrunnerSBMLModel`, which owns the roadrunner instance `r`, the units of the model and the selections, i.e., the variables recorded in a simulation:

```python
from sbmlsim.model import RoadrunnerSBMLModel

model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
print(model.r)  # the roadrunner.RoadRunner instance
print(model.selections)
```

The model can be created explicitly and passed to the simulator, which is useful when several simulators or experiments share a model.

## Units of a model

The units of every parameter, species and compartment are read from the SBML and stored as a `UnitsInformation`, a mapping from identifier to unit string. All changes and results of a simulation carry these units, see [Units](units.md):

```python
uinfo = model.uinfo
print(uinfo["X"])  # unit of the species X
print(uinfo["time"])
```

## Changes and selections

A `RoadrunnerSBMLModel` accepts `changes`, which are applied to the model whenever it is reset, and `selections`, the variables recorded in a simulation. Changes are quantities with units or plain floats in the units of the model:

```python
model = RoadrunnerSBMLModel(
    source=REPRESSILATOR_SBML,
    changes={"X": 5.0, "Y": 10.0},
    selections=["time", "X", "Y", "Z"],
)
print(model.changes)
print(model.selections)
```

The roadrunner integrator is configured with `settings`, e.g., `settings={"absolute_tolerance": 1e-10}`; the defaults are set by `RoadrunnerSBMLModel.set_default_settings`.

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

The source of a model is resolved by `sbmlsim.model.model_resources`: a path relative to the `base_path` of the experiment, an absolute path, a URL, or a `urn:miriam:biomodels.db:` URN which downloads the model from BioModels.

## Clamping species

`ModelChange` implements structural changes of the model, currently clamping a species to a fixed value or formula. Clamping is a boundary condition set during a simulation and is part of a `Timecourse`, see the `model_manipulations` of [Timecourse simulations](simulation.md#clamping-species):

```python
from sbmlsim.model import ModelChange

r = model.r
ModelChange.clamp_species(r, "X", "10.0")  # clamp X to 10.0
ModelChange.clamp_species(r, "X", False)  # release the clamp
```

## Inspecting a model

The parameters and species of the model with their current values and units are available as data frames:

```python
print(RoadrunnerSBMLModel.parameter_df(model.r).head())
print(RoadrunnerSBMLModel.species_df(model.r).head())
```
