# Timecourse simulations

A timecourse simulation integrates the model over a period of time. In `sbmlsim` a period is a `Timecourse` with its changes, and a `TimecourseSim` concatenates timecourses into one simulation. This is how dosing protocols, perturbations and pre-simulations are described.

## A single timecourse

`Timecourse(start, end, steps)` integrates from `start` to `end` in `steps` intervals, i.e., `steps + 1` time points. The simulator runs the `TimecourseSim` and returns an `XResult`:

```python
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial

simulator = SimulatorSerial(model=REPRESSILATOR_SBML)

tcsim = TimecourseSim(Timecourse(start=0, end=100, steps=100))
xres = simulator.run_timecourse(tcsim)
print(xres)
```

The result is a labeled N-dimensional array (an `xarray.Dataset` wrapped in an `XResult`) with a `_time` dimension. The variables are accessed with the selection ids of roadrunner: `X` for the amount of a species and `[X]` for its concentration:

```python
print(xres["time"].values[:5])
print(xres["[X]"].values[:5])
```

## Changes

A timecourse applies `changes` before it starts: values of parameters, initial amounts (`X`) or initial concentrations (`[X]`) of species. Changes are plain floats in the units of the model or quantities with units, which are converted to the model units, see [Units](units.md):

```python
tcsim = TimecourseSim(
    Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 200})
)
xres = simulator.run_timecourse(tcsim)
print(xres["X"].values[0], xres["Y"].values[0])
```

## Concatenated timecourses

Several timecourses are simulated one after the other. Every timecourse continues from the end state of the previous one and applies its changes; the time of the result is continuous:

```python
tcsim = TimecourseSim(
    [
        Timecourse(start=0, end=100, steps=100),
        Timecourse(start=0, end=100, steps=100, changes={"X": 10, "Y": 20}),
        Timecourse(start=0, end=100, steps=100, changes={"X": 0.5}),
    ]
)
xres = simulator.run_timecourse(tcsim)
print(xres["time"].values[[0, 100, 101, 200, 201, -1]])
```

This is the pattern for a dosing protocol: every dose is a timecourse whose change sets the dose parameter. A `Timecourse` with `discard=True` is simulated but removed from the result, which is how a pre-simulation to a steady state is described.

By default a `TimecourseSim` resets the model to its initial state before the first timecourse (`reset=True`); `time_offset` shifts the time of the complete result.

## Clamping species

Structural changes of the model, e.g., clamping a species to a fixed value, are `model_manipulations` of a timecourse. Here `X` is clamped during the second period and released in the third:

```python
from sbmlsim.model import ModelChange

tcsim = TimecourseSim(
    [
        Timecourse(start=0, end=100, steps=100),
        Timecourse(
            start=0,
            end=100,
            steps=100,
            model_manipulations={ModelChange.CLAMP_SPECIES: {"X": True}},
        ),
        Timecourse(
            start=0,
            end=100,
            steps=100,
            model_manipulations={ModelChange.CLAMP_SPECIES: {"X": False}},
        ),
    ]
)
xres = simulator.run_timecourse(tcsim)
print(xres["[X]"].values[100:105])
```

## Selections and integrator settings

The variables recorded in a simulation are the selections of the model. By default all species (amounts and concentrations), parameters, reactions and compartments are recorded; a smaller selection speeds up the simulation:

```python
simulator.set_timecourse_selections(["time", "[X]", "[Y]", "[Z]"])
xres = simulator.run_timecourse(TimecourseSim(Timecourse(start=0, end=10, steps=10)))
print(list(xres.xds.data_vars))
```

The integrator settings of roadrunner are passed to the simulator or set afterwards:

```python
simulator = SimulatorSerial(
    model=REPRESSILATOR_SBML, absolute_tolerance=1e-10, relative_tolerance=1e-10
)
simulator.set_integrator_settings(variable_step_size=False)
```

## Results

An `XResult` is converted to pandas for further processing and stored as netCDF or TSV:

```python
from pathlib import Path

df = xres.to_dataframe()
print(df.head())

xres.to_netcdf(Path("repressilator.nc"))
xres.to_tsv(Path("repressilator.tsv"))
```

`XResult.dim_mean`, `dim_std`, `dim_min` and `dim_max` reduce the result over all dimensions except time and return quantities with the units of the variable, see [Parameter scans](scans.md).

## Serialization

A `TimecourseSim` is serialized to JSON and read back, which is how a simulation experiment stores its simulations:

```python
json_str = tcsim.to_json()
tcsim2 = TimecourseSim.from_json(json_str)
print(tcsim2)
```
