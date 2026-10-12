# Simulations

A `Simulation` describes what is simulated: the interval, the changes applied to the model before it is initialized, the changes at times, an optional pre-equilibration and the output. It is the one way to set up a simulation in `sbmlsim`: simulation experiments, scans, parameter fits and the problems read from PEtab use it. The semantics are the ones of [PEtab v2](https://petab.readthedocs.io), see [the order below](#the-semantics).

## A simulation

`Simulation(start, end)` integrates the model from `start` to `end`. Without further settings the output are the steps of the integrator, i.e. the time points the integrator chose, which end in the state at the end, have a change after the start as two time points at the time of the change, the states before and after it, and an event of the model as two time points at the time of the event, the states before and after it, also at the end (see [The semantics](#the-semantics)); `Simulator.run` returns a `ScanResult`:

```python
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)

res = simulator.run(model, Simulation(end=100))
print(res["time"].values[:5])
print(res["[X]"].values[:5])
```

The variables are accessed with the selection ids of roadrunner: `X` for the amount of a species and `[X]` for its concentration. The output is set with `times`, exact output times, or with `steps`, an equidistant grid of `steps + 1` points:

```python
res = simulator.run(model, Simulation(end=100, steps=100))
res = simulator.run(model, Simulation(end=100, times=[0, 1, 5, 10, 50, 100]))
print(res["time"].values)
```

An output time which is the time of a change after the start appears twice, the state before the change and the state after it, so a figure draws the jump of the change vertically and a time before the change is interpolated toward the state before it; a change between two output times has no time point:

```python
from sbmlsim.simulation import Change

res = simulator.run(
    model,
    Simulation(end=100, times=[0, 50, 100], changes=[Change(50, {"[X]": 0.0})]),
)
print(res["time"].values)  # [  0.  50.  50. 100.]
```

Output times which differ by less than the resolution of the time are one output time, so digitized times such as `3.99999999999999` and `4.0` are simulated once, at the smaller one. The tolerance is `1e-9` times the larger of the absolute start, the absolute end and the length of the interval, and a chain of times whose consecutive distances are within it is merged, into its smallest time inside the simulation. A cluster is not anchored to the time of a change: a cluster around a change time is its smallest member, i.e. before the change. A fit merges the times of its data the same way and compares every data point with the merged output time.

## Units and times

A value is a number in the unit of its target in the model or a quantity, which is converted into the unit of the model; `Q` is the quantity of the unit registry of `sbmlsim`, see [Units](units.md). The times of a simulation are numbers in its `time_unit`, the time unit of the model without one, or quantities. A simulation can start at a negative time:

```python
from sbmlsim import Q

sim = Simulation(time_unit="hr", start=-24, end=Q(30, "min"), steps=50)
```

## Changes before the initialization

`preinit_changes` are applied to the model before it is initialized: the initial assignments of the model are evaluated with them, so an entity whose initial assignment reads a changed parameter follows the change. A target which is set replaces its own initial assignment:

```python
res = simulator.run(
    model, Simulation(end=100, steps=100, preinit_changes={"X": 10, "Y": 200})
)
print(res["X"].values[0], res["Y"].values[0])
```

## Changes at times and multiple dosing

A `Change` sets values at one time or at a vector of times. A change at several times is the same change at each of them, which is a multiple dosing:

```python
from sbmlsim.simulation import Change

sim = Simulation(
    end=150,
    changes=[
        Change(50, {"X": 10, "Y": 20}),
        Change([0, 50, 100], {"[Z]": 5.0}),
    ],
    steps=150,
)
res = simulator.run(model, sim)
```

A value of a change is a number, a quantity or a formula. A formula is a string of the math of PEtab over the symbols of the model, in the units of the model, and is evaluated with the state at the time of the change; `S` is the amount of a species, `[S]` its concentration and `time` the time of the change:

```python
sim = Simulation(
    end=100,
    changes=[Change([20, 40, 60], {"[X]": "[X] + 5"})],
    steps=100,
)
res = simulator.run(model, sim)
```

A dosing protocol whose data is reported from the last dose starts at a negative time, e.g. eleven doses every twelve hours:

```python
dose_times = [-120 + 12 * k for k in range(11)]
sim = Simulation(
    time_unit="hr",
    start=-120,
    end=60,
    changes=[Change(dose_times, {"[X]": "[X] + 10"})],
)
```

## Pre-equilibration

`presimulation=SteadyState()` integrates the model until the rates of change vanish, `|dx/dt| <= absolute_tolerance + relative_tolerance * |x|`, before the simulation starts. The model is initialized once, with the `preinit_changes` of the simulation and of the steady state, and what differs after the steady state is a change at the start:

```python
from sbmlsim.simulation import SteadyState

sim = Simulation(
    end=100,
    preinit_changes={"n": 1.0},
    presimulation=SteadyState(preinit_changes={"ps_a": 0.5}),
    changes=[Change(0, {"ps_a": 0.4})],
    times=[0, 50, 100],
)
res = simulator.run(model, sim)
```

A model which does not reach a steady state by `SteadyState(max_time=...)` raises a `SteadyStateError`.

## The semantics

| step | what happens |
| --- | --- |
| pre-initialization | the `preinit_changes` are set on the model before its initialization; a compartment keeps the concentration of a species whose initial value is a concentration or which is set as a concentration and the amount of every other species, a species of an assignment rule follows its rule |
| initialization | the initial assignments are evaluated, except the ones of the targets of the pre-initialization |
| steady state | with a `SteadyState`, the model is integrated until the rates of change vanish |
| a change at a time | every value is evaluated with the state at that time, then all values are set at once; a compartment keeps the concentration of the concentration species in it and the amount of the amount species, a species of an assignment rule follows its rule |
| events | the events of the model whose trigger became true fire after the change; a number with a unit in the math of an event, as sbmlutils writes it, is its value, so are the constants `pi`, `exponentiale` and `avogadro` (the value of roadrunner), and a logarithm to a base keeps its base; an event whose trigger or assignments are no formulas the simulator evaluates, e.g. with a function definition, is reported with a warning which names it and does not fire at the change (a trigger once per model instance, i.e. once per worker of a pool) |
| integration | the interval is split at the times of the changes, and every segment ends in its state at its end, which the change at that time is applied to, also with the steps of the integrator as the output; an event of the model fires at its time, except that with the steps of the integrator roadrunner fires an event with a delay at the end of the step after its time |
| output | an output time at a change after the start, the change at the end included, has two time points, the state before the change and the state after it, in that order; the steps of the integrator have every change. A time of a grid (`time=`, `ScanResult.interpolate`) sees: (1) before a change, the value interpolated toward the state before it; (2) at a change, the state after it; (3) the first of two equal times of the grid, the state before it. Jumps closer than the resolution of the time, e.g. a change and an event of the model, are one chain, and a time at it or inside it sees the state after the last jump. A fit compares its data at a change with the state before it, see [Parameter fitting](fitting.md#data-at-the-time-of-a-change). An event of the model at the time of a change fires after the change, also at the end |
| events in the output | with the steps of the integrator an event of the model is two time points at the time of the event, the states before and after it, also an event at the end of the simulation, which an output time at the end sees after it. A time of a grid at an event sees the state after it, and so does a time at most the resolution of the time before it, since the integrator locates an event a little after its time. sbmlsim gives both time points the time of the event with a tolerance of libroadrunner which the first integration of a process fixes, see [`model.roottol`](api/model.roottol.md): load a model with sbmlsim before integrating with roadrunner directly, otherwise sbmlsim warns once, and a run lacks the steps of the integrator right after an event which a run with workers has |

## Selections and integrator settings

The variables recorded in a simulation are the selections of the model. By default all species (amounts and concentrations), parameters, reactions and compartments are recorded, except a compartment without a size (`NaN`), e.g. a membrane whose area the model does not use; a smaller selection speeds up the simulation:

```python
model.set_selections(["time", "[X]", "[Y]", "[Z]"])
res = simulator.run(model, Simulation(end=10, steps=10))
print(res.variables)
```

The integrator settings of roadrunner are passed to the simulator or set afterwards. Every setting of the integrator is passed on, a name the integrator does not have is an error, and the settings apply to every model the simulator runs, e.g. the models of the tasks of an experiment, also in the workers of a pool:

```python
simulator = Simulator(absolute_tolerance=1e-10, relative_tolerance=1e-10)
simulator.set_integrator_settings(stiff=True)
```

The absolute tolerance of CVODE is one value per state, which sbmlsim sets from the kind of the state. The tolerances are plain numbers in the units of the model, so they work for a model without units. A species with `hasOnlySubstanceUnits=true` is an `amount`, its tolerance is the one of the amounts. Any other species is a `concentration`: CVODE integrates its amount, so its tolerance is the one of the concentrations times the reference volume of its compartment. The reference volume is the initial volume of the model as loaded, raised to `1e-6` times the largest initial volume of the model when it is smaller, not finite or not positive, which is logged once per model; a volume which the pre-initialization of a simulation changes, e.g. through a body weight, does not change the tolerances, so a simulation does not depend on the one which ran before it on the same model and a pool gives the serial result. Every other state, i.e. a parameter or a compartment with a rate rule, is `other`. A float is the same tolerance for every kind, an `AbsoluteTolerance` gives one per kind and overrides single states by their id; an override is the tolerance of the integrated value, i.e. the amount of a concentration species. The default is `1e-10` for every kind. `tolerances()` of the model lists the tolerance of every state:

```python
from sbmlsim.model.tolerances import AbsoluteTolerance

simulator.set_integrator_settings(
    absolute_tolerance=AbsoluteTolerance(
        amount=1e-10, concentration=1e-10, other=1e-10, ids={"PX": 1e-12}
    )
)
print(simulator.load(model).tolerances())
```

CVODE estimates its first step after the start and after every change. A model which does not read the time, i.e. no rule, kinetic law, initial assignment or event of it reads the csymbol `time` or `delay` and no event has a delay, is integrated in local time: every segment between two changes starts at the time 0 of roadrunner and its output is shifted back to the absolute time. The first step is then never smaller than the resolution of the time, the results agree with the ones of the absolute time within the tolerances. A model which reads the time is integrated in absolute time. At a late time a state which starts from 0, e.g. a dose after a reset, can then give a first step which is smaller than the resolution of the time, and CVODE warns "t + h = t on the next step". A small positive `initial_time_step` in the time unit of the model avoids the estimate; a step which is too large fails the error test of the integrator:

```python
simulator.set_integrator_settings(initial_time_step=1e-10)
```

## Results

The result of a simulation is a `ScanResult`, which wraps an `xarray.Dataset` and keeps the units of its variables, see [Parameter scans](scans.md#working-with-scan-results). With `times` or `steps` it has the dimension `time`; with the steps of the integrator it has the dimension `_point` and the time as a variable. `simulator.simulate(model, simulation)` gives the native solution of one simulation, a `TimecourseResult`. A result is converted to pandas for further processing and stored as netCDF with its units:

```python
from pathlib import Path

df = res.ds.to_dataframe()
print(df.head())

res.to_netcdf(Path("repressilator.nc"))
```

## Serialization

A `Simulation` is serialized to JSON with its units and read back, which is how a simulation experiment stores its simulations:

```python
json_str = sim.to_json()
sim2 = Simulation.from_json(json_str)
print(sim2)
```
