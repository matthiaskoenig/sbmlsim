# Simulations

A `Simulation` describes what is simulated: the interval, the changes applied to the model before it is initialized, the changes at times, an optional pre-equilibration and the output. It is the one way to set up a simulation in `sbmlsim`: simulation experiments, scans, parameter fits and the problems read from PEtab use it. The semantics are the ones of [PEtab v2](https://petab.readthedocs.io), see [the order below](#the-semantics).

## A simulation

`Simulation(start, end)` integrates the model from `start` to `end`. Without further settings the output are the steps of the integrator, i.e. the time points the integrator chose; the simulator returns an `XResult`:

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import SimulatorSerial

simulator = SimulatorSerial(model=REPRESSILATOR_SBML)

xres = simulator.run_simulation(Simulation(end=100))
print(xres["time"].values[:5])
print(xres["[X]"].values[:5])
```

The variables are accessed with the selection ids of roadrunner: `X` for the amount of a species and `[X]` for its concentration. The output is set with `times`, exact output times, or with `steps`, an equidistant grid of `steps + 1` points:

```python
xres = simulator.run_simulation(Simulation(end=100, steps=100))
xres = simulator.run_simulation(Simulation(end=100, times=[0, 1, 5, 10, 50, 100]))
print(xres["time"].values)
```

## Units and times

A value is a number in the unit of its target in the model or a quantity, which is converted into the unit of the model; `Q` is the quantity of the unit registry of `sbmlsim`, see [Units](units.md). The times of a simulation are numbers in its `time_unit`, the time unit of the model without one, or quantities. A simulation can start at a negative time:

```python
from sbmlsim import Q

sim = Simulation(time_unit="hr", start=-24, end=Q(30, "min"), steps=50)
```

## Changes before the initialization

`preinit_changes` are applied to the model before it is initialized: the initial assignments of the model are evaluated with them, so an entity whose initial assignment reads a changed parameter follows the change. A target which is set replaces its own initial assignment:

```python
xres = simulator.run_simulation(
    Simulation(end=100, steps=100, preinit_changes={"X": 10, "Y": 200})
)
print(xres["X"].values[0], xres["Y"].values[0])
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
xres = simulator.run_simulation(sim)
```

A value of a change is a number, a quantity or a formula. A formula is a string of the math of PEtab over the symbols of the model, in the units of the model, and is evaluated with the state at the time of the change; `S` is the amount of a species, `[S]` its concentration and `time` the time of the change:

```python
sim = Simulation(
    end=100,
    changes=[Change([20, 40, 60], {"[X]": "[X] + 5"})],
    steps=100,
)
xres = simulator.run_simulation(sim)
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
xres = simulator.run_simulation(sim)
```

A model which does not reach a steady state by `SteadyState(max_time=...)` raises a `SteadyStateError`.

## The semantics

| step | what happens |
| --- | --- |
| pre-initialization | the `preinit_changes` are set on the model before its initialization |
| initialization | the initial assignments are evaluated, except the ones of the targets of the pre-initialization |
| steady state | with a `SteadyState`, the model is integrated until the rates of change vanish |
| a change at a time | every value is evaluated with the state at that time, then all values are set at once; a compartment keeps the concentration of the concentration species in it and the amount of the amount species |
| events | the events of the model whose trigger became true fire after the change |
| output | an output time which is the time of a change is the state after the change |

## Selections and integrator settings

The variables recorded in a simulation are the selections of the simulator. By default all species (amounts and concentrations), parameters, reactions and compartments are recorded, except a compartment without a size (`NaN`), e.g. a membrane whose area the model does not use; a smaller selection speeds up the simulation:

```python
simulator.set_timecourse_selections(["time", "[X]", "[Y]", "[Z]"])
xres = simulator.run_simulation(Simulation(end=10, steps=10))
print(list(xres.xds.data_vars))
```

The integrator settings of roadrunner are passed to the simulator or set afterwards. Every setting of the integrator is passed on, a name the integrator does not have is an error, and the settings apply to every model the simulator runs, e.g. the models of the tasks of an experiment:

```python
simulator = SimulatorSerial(
    model=REPRESSILATOR_SBML, absolute_tolerance=1e-10, relative_tolerance=1e-10
)
simulator.set_integrator_settings(stiff=True)
```

The absolute tolerance of CVODE is one value per state, which sbmlsim sets from the kind of the state. The tolerances are plain numbers in the units of the model, so they work for a model without units. A species with `hasOnlySubstanceUnits=true` is an `amount`, its tolerance is the one of the amounts. Any other species is a `concentration`: CVODE integrates its amount, so its tolerance is the one of the concentrations times the reference volume of its compartment. The reference volume is the initial volume, raised to `1e-6` times the largest initial volume of the model when it is smaller, not finite or not positive, which is logged once per model. Every other state, i.e. a parameter or a compartment with a rate rule, is `other`. A float is the same tolerance for every kind, an `AbsoluteTolerance` gives one per kind and overrides single states by their id; an override is the tolerance of the integrated value, i.e. the amount of a concentration species. The default is `1e-10` for every kind. `tolerances()` of the model lists the tolerance of every state:

```python
from sbmlsim.model.tolerances import AbsoluteTolerance

simulator.set_integrator_settings(
    absolute_tolerance=AbsoluteTolerance(
        amount=1e-10, concentration=1e-10, other=1e-10, ids={"PX": 1e-12}
    )
)
print(simulator.model_loaded.tolerances())
```

CVODE estimates its first step after the start and after every change. A model which does not read the time, i.e. no rule, kinetic law, initial assignment or event of it reads the csymbol `time` or `delay`, is integrated in local time: every segment between two changes starts at the time 0 of roadrunner and its output is shifted back to the absolute time. The first step is then never smaller than the resolution of the time, the results agree with the ones of the absolute time within the tolerances. A model which reads the time is integrated in absolute time. At a late time a state which starts from 0, e.g. a dose after a reset, can then give a first step which is smaller than the resolution of the time, and CVODE warns "t + h = t on the next step". A small positive `initial_time_step` in the time unit of the model avoids the estimate; a step which is too large fails the error test of the integrator:

```python
simulator.set_integrator_settings(initial_time_step=1e-10)
```

## Results

The result of a simulation is a labeled array (an `xarray.Dataset` wrapped in an `XResult`) with the dimension `_point`, the output points of the simulation, and the time as a variable like every selection. The simulations of a [scan](scans.md) keep their own time points, `XResult.interpolate(times)` puts them on a common grid. An `XResult` is converted to pandas for further processing and stored as netCDF or TSV:

```python
from pathlib import Path

df = xres.to_dataframe()
print(df.head())

xres.to_netcdf(Path("repressilator.nc"))
xres.to_tsv(Path("repressilator.tsv"))
```

## Serialization

A `Simulation` is serialized to JSON with its units and read back, which is how a simulation experiment stores its simulations:

```python
json_str = sim.to_json()
sim2 = Simulation.from_json(json_str)
print(sim2)
```
