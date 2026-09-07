# Parameter scans

A parameter scan runs a simulation for every combination of parameter values. In `sbmlsim` a `ScanSim` combines a `TimecourseSim` with one or more `Dimension` objects, each describing the changes along one axis of the scan. The result is an `XResult` with one dimension per scan dimension and the time.

## A one dimensional scan

A `Dimension` is a set of changes with vectors of values. The values of all changes of a dimension are applied together, element by element; the index of the dimension is the position in these vectors:

```python
import numpy as np

from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, ScanSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial

simulator = SimulatorSerial(model=REPRESSILATOR_SBML)

scan = ScanSim(
    simulation=TimecourseSim(Timecourse(start=0, end=100, steps=100)),
    dimensions=[
        Dimension("dim_n", changes={"n": np.linspace(2, 4, num=5)}),
    ],
)
xres = simulator.run_scan(scan)
print(xres.xds.dims)
print(xres["PX"].shape)
```

The values of a dimension are quantities with units or floats in the units of the model. Values from a distribution are a scan as well:

```python
scan = ScanSim(
    simulation=TimecourseSim(Timecourse(start=0, end=100, steps=100)),
    dimensions=[
        Dimension(
            "dim_n", changes={"n": np.random.normal(loc=3.0, scale=0.2, size=20)}
        ),
    ],
)
xres = simulator.run_scan(scan)
print(xres["PX"].sizes)
```

## Multi-dimensional scans

Several dimensions are combined: every combination of the indices is simulated. The result has a dimension for every `Dimension`:

```python
scan = ScanSim(
    simulation=TimecourseSim(Timecourse(start=0, end=100, steps=100)),
    dimensions=[
        Dimension("dim_n", changes={"n": np.linspace(2, 4, num=3)}),
        Dimension("dim_X", changes={"X": np.array([1.0, 10.0, 100.0, 1000.0])}),
    ],
)
xres = simulator.run_scan(scan)
print(xres["PX"].sizes)
```

The indices of the scan are available as `scan.indices()` and the individual simulations as `scan.to_simulations()`, which returns the indices and one `TimecourseSim` per combination.

## Working with scan results

The result is an `xarray.Dataset`, so the usual selections and reductions apply. `XResult.dim_mean`, `dim_std`, `dim_min` and `dim_max` reduce over all scan dimensions and return quantities with units:

```python
da = xres["PX"]
print(da.isel(dim_n=0, dim_X=1).values[:3])  # single timecourse
print(da.mean(dim="dim_X").sizes)  # mean over one dimension

mean = xres.dim_mean("PX")  # mean over all scan dimensions, a quantity
print(mean.units, mean.magnitude[:3])
```

`XResult.to_mean_dataframe` reduces every variable to its mean over the scan dimensions and returns a data frame with one row per time point.

## Sensitivity scans

`ModelSensitivity` creates scans of all parameters of a model, either by relative differences or by sampling from distributions, see `sbmlsim.simulation.sensitivity`:

```python
from sbmlsim.simulation.sensitivity import ModelSensitivity

model = simulator.model_loaded
tcsim = TimecourseSim(Timecourse(start=0, end=100, steps=100))

diff_scan = ModelSensitivity.difference_sensitivity_scan(
    model=model, simulation=tcsim, difference=0.1
)
xres = simulator.run_scan(diff_scan)
print(xres["PX"].sizes)

distrib_scan = ModelSensitivity.distribution_sensitivity_scan(
    model=model, simulation=tcsim, cv=0.05, size=10
)
xres = simulator.run_scan(distrib_scan)
print(xres["PX"].sizes)
```

The difference scan varies every constant parameter up and down by the relative `difference` (two simulations per parameter, plus the reference); the distribution scan samples `size` values of every parameter from a normal distribution with the coefficient of variation `cv`. The global sensitivity methods of `sbmlsim.sensitivity` build on scans like these, see [Sensitivity analysis](sensitivity.md).
