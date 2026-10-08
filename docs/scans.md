# Parameter scans

A parameter scan runs a simulation for every combination of parameter values. In `sbmlsim` a `ScanSim` combines a `Simulation` with one or more `Dimension` objects, each describing the changes along one axis of the scan. The result is an `XResult` with the output points of the simulations and one dimension per scan dimension.

## A one dimensional scan

A `Dimension` is a set of changes with vectors of values. The values of all changes of a dimension are applied together, element by element; the index of the dimension is the position in these vectors:

```python
import numpy as np

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Dimension, ScanSim, Simulation
from sbmlsim.simulator import SimulatorSerial

simulator = SimulatorSerial(model=REPRESSILATOR_SBML)

scan = ScanSim(
    simulation=Simulation(end=100, steps=100),
    dimensions=[
        Dimension("dim_n", values={"n": np.linspace(2, 4, num=5)}),
    ],
)
xres = simulator.run_scan(scan)
print(xres.xds.dims)
print(xres["PX"].shape)
```

The values of a dimension are quantities with units or floats in the units of the model. Values from a distribution are a scan as well:

```python
scan = ScanSim(
    simulation=Simulation(end=100, steps=100),
    dimensions=[
        Dimension("dim_n", values={"n": np.random.normal(loc=3.0, scale=0.2, size=20)}),
    ],
)
xres = simulator.run_scan(scan)
print(xres["PX"].sizes)
```

## Where the values of a scan go

The value of a scan replaces the value of its target wherever the simulation sets it, i.e. in the `preinit_changes` and in every `Change`; a target which the simulation does not set is a change before the initialization. Scanning the dose of a multiple dosing therefore scans every dose:

```python
dosing = Simulation(end=150, changes=[Change([0, 50, 100], {"X": 10.0})], steps=150)
scan = ScanSim(dosing, [Dimension("dim_dose", values={"X": np.array([5.0, 20.0])})])
_, simulations = scan.to_simulations()
print([s.changes[0].values for s in simulations])
```

A dimension with `at` applies its values as a `Change` at that time instead:

```python
scan = ScanSim(
    Simulation(end=200, steps=200, changes=[Change(100, {"X": 10.0})]),
    [Dimension("dim_X0", values={"X": np.array([1.0, 50.0])}, at=0)],
)
xres = simulator.run_scan(scan)
```

## Multi-dimensional scans

Several dimensions are combined: every combination of the indices is simulated. The result has a dimension for every `Dimension`:

```python
scan = ScanSim(
    simulation=Simulation(end=100, steps=100),
    dimensions=[
        Dimension("dim_n", values={"n": np.linspace(2, 4, num=3)}),
        Dimension("dim_X", values={"X": np.array([1.0, 10.0, 100.0, 1000.0])}),
    ],
)
xres = simulator.run_scan(scan)
print(xres["PX"].sizes)
```

The indices of the scan are available as `scan.indices()` and the individual simulations as `scan.to_simulations()`, which returns the indices and one `Simulation` per combination.

## Working with scan results

Every simulation of a scan keeps its own output points. With `times` or `steps` they agree, with the steps of the integrator they differ and a simulation with fewer points is padded with `NaN`. `XResult.interpolate(times)` puts the simulations on a common grid, and `XResult.dim_mean`, `dim_std`, `dim_min` and `dim_max` reduce over all scan dimensions on given times, or on the union of the time points of the simulations, and return quantities with units:

```python
da = xres["PX"]
print(da.isel(dim_n=0, dim_X=1).values[:3])  # a single simulation

grid = xres.interpolate(np.linspace(0, 100, num=11))
print(grid["PX"].mean(dim="dim_X").sizes)  # mean over one dimension

mean = xres.dim_mean("PX")  # mean over all scan dimensions, a quantity
print(mean.units, mean.magnitude[:3])
```

`XResult.to_mean_dataframe` reduces every variable to its mean over the scan dimensions and returns a data frame with one row per time point.

## Sensitivity scans

`ModelSensitivity` creates scans of all parameters of a model, either by relative differences or by sampling from distributions, see `sbmlsim.simulation.sensitivity`. The reference values are the ones of the model with the `preinit_changes` of the simulation:

```python
from sbmlsim.simulation.sensitivity import ModelSensitivity

model = simulator.model_loaded
simulation = Simulation(end=100, steps=100)

diff_scan = ModelSensitivity.difference_sensitivity_scan(
    model=model, simulation=simulation, difference=0.1
)
xres = simulator.run_scan(diff_scan)
print(xres["PX"].sizes)

distrib_scan = ModelSensitivity.distribution_sensitivity_scan(
    model=model, simulation=simulation, cv=0.05, size=10
)
xres = simulator.run_scan(distrib_scan)
print(xres["PX"].sizes)
```

The difference scan varies every constant parameter up and down by the relative `difference` (two simulations per parameter); the distribution scan samples `size` values of every parameter from a normal distribution with the coefficient of variation `cv`. The global sensitivity methods of `sbmlsim.sensitivity` build on scans like these, see [Sensitivity analysis](sensitivity.md).
