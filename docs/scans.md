# Parameter scans

A scan runs a simulation for every point of its dimensions. A `Scan` combines a `Simulation` with `Dimension` objects; `Simulator.run` runs it serially or in a pool of processes and answers with a `ScanResult`, a labeled array whose coordinates are the labels of the dimensions and the values they change. A `Simulation` is a scan without dimensions, `simulator.run(model, simulation)`.

## A one dimensional scan

A `Dimension` maps targets to arrays of values. The arrays of a dimension have one length and are coupled: point `k` sets the `k`-th value of every target. The values are numbers in the unit of their target in the model or quantities.

```python
import numpy as np

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

simulator = Simulator()
model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
model.set_selections(["time", "PX", "PY", "PZ"])

scan = Scan(
    simulation=Simulation(end=100, steps=100),
    dimensions=[Dimension("dim_n", values={"n": np.linspace(2, 4, num=5)})],
)
res = simulator.run(model, scan)
print(res)
print(res["PX"].dims, res["PX"].shape)
print(res["n"].values)
```

The result has the dimensions of the scan first and the time last, `(dim_n, time)`. A changed target is a coordinate along its dimension, here `n`, unless it is also a selection of the model: then the result keeps its timecourse as a variable. The labels of a dimension of values are `0..n-1` by default and can be given with `labels=`.

## Where the values of a scan go

A value replaces its target wherever the simulation sets it, i.e. in the `preinit_changes` and in every `Change`; a target which the simulation does not set is a change before the initialization. Scanning the dose of a multiple dosing therefore scans every dose. A dimension with `at` sets its values as a `Change` at that time instead:

```python
dosing = Simulation(end=150, changes=[Change([0, 50, 100], {"X": 10.0})], steps=150)
res = simulator.run(
    model, Scan(dosing, [Dimension("dim_dose", values={"X": [5.0, 20.0]})])
)

scan = Scan(
    Simulation(end=200, steps=200, changes=[Change(100, {"X": 10.0})]),
    [Dimension("dim_X0", values={"X": [1.0, 50.0]}, at=0)],
)
res = simulator.run(model, scan)
print(res["PX"].sel(dim_X0=1).values[:3])
```

## Several dimensions, simulations and models

Several dimensions span their cartesian product; the points are enumerated in C order of the dimensions, the last one fastest. A sampled design, e.g. values from a distribution, is one dimension with coupled values:

```python
rng = np.random.default_rng(seed=1234)
scan = Scan(
    simulation=Simulation(end=100, steps=100),
    dimensions=[
        Dimension(
            "sample",
            values={
                "n": rng.normal(3.0, 0.2, size=20),
                "Y": rng.normal(10, 1, size=20),
            },
        ),
        Dimension("dim_X", values={"X": [1.0, 10.0, 100.0]}),
    ],
)
res = simulator.run(model, scan)
print(res["PX"].sizes)
```

A dimension of `simulations` gives every point its own simulation, a dimension of `models` its own model; the model of the run is then `None`:

```python
scan = Scan(
    Simulation(end=100, steps=100),
    [
        Dimension(
            "regimen",
            simulations={
                "single": Simulation(
                    end=100, steps=100, changes=[Change(0, {"X": 10.0})]
                ),
                "multiple": Simulation(
                    end=100, steps=100, changes=[Change([0, 50], {"X": 10.0})]
                ),
            },
        ),
    ],
)
res = simulator.run(model, scan)
print(res["PX"].sel(regimen="multiple").values[-1])
```

## The output and the run

With `times` or `steps` every simulation has the same output times and the result has the dimension `time`. With the steps of the integrator every simulation keeps its own time points: the result has the dimension `_point` and the variable `time`, padded with `NaN`. `time=` interpolates every timecourse onto a grid; the value at the time of a change is the value after it:

```python
res = simulator.run(
    model, Scan(Simulation(end=100), [Dimension("dim_n", values={"n": [2.0, 3.0]})])
)
print(res.ragged, res["time"].dims)
res = simulator.run(
    model,
    Scan(Simulation(end=100), [Dimension("dim_n", values={"n": [2.0, 3.0]})]),
    time=np.linspace(0, 100, 11),
)
print(res.ragged, res["time"].values)
```

`Simulator(n_workers=...)` sets the processes: `1` runs in the calling process, a number is the size of the pool, and `None` (the default) uses every CPU for a scan of `sbmlsim.parallel.POOL_THRESHOLD` points or more, i.e. 256, and runs a smaller scan in the calling process. The pool is kept for the process, so the start of the workers is paid by the first pooled run of a process and every worker loads a model once; a short script with a single scan of a cheap model can therefore be faster with `n_workers=1`. The result does not depend on the number of workers. A script which runs a pool must do it behind `if __name__ == "__main__":`, because the workers import the script again. `on_error="flag"` keeps the points which ran when a point fails in the integrator: its values are `NaN` and the variable `status` is `1`.

## Working with scan results

A `ScanResult` is an `xarray.Dataset` with units, `res.ds`; `res.quantity(key)` gives the values with their unit, `res.summary(dims)` the statistics over dimensions and `res.interpolate(times)` puts a ragged result on a grid. `res.ds.to_dataframe()` is the table, `res.to_netcdf(path)` and `ScanResult.from_netcdf(path)` store the result with its units:

```python
from pathlib import Path

from sbmlsim.result import ScanResult

res = simulator.run(
    model,
    Scan(
        Simulation(end=100, steps=100),
        [Dimension("dim_n", values={"n": np.linspace(2, 4, num=5)})],
    ),
)
print(res["PX"].isel(dim_n=0).values[:3])  # a single simulation

summary = res.summary("dim_n", quantiles=[0.05, 0.95])
print(summary["PX"].sel(statistic="mean").values[:3])

res.to_netcdf(Path("scan.nc"))
print(ScanResult.from_netcdf(Path("scan.nc")).quantity("time")[-1])
```

## Sensitivity scans

`ModelSensitivity` creates scans of all parameters of a model, either by relative differences or by sampling from distributions, see `sbmlsim.simulation.sensitivity`. The reference values are the ones of the model with the `preinit_changes` of the simulation:

```python
from sbmlsim.simulation.sensitivity import ModelSensitivity

simulation = Simulation(end=100, steps=100)
diff_scan = ModelSensitivity.difference_sensitivity_scan(
    model=model, simulation=simulation, difference=0.1
)
res = simulator.run(model, diff_scan)
print(res["PX"].sizes)

distrib_scan = ModelSensitivity.distribution_sensitivity_scan(
    model=model, simulation=simulation, cv=0.05, size=10
)
res = simulator.run(model, distrib_scan)
print(res["PX"].sizes)
```

The difference scan varies every constant parameter up and down by the relative `difference` (two simulations per parameter); the distribution scan samples `size` values of every parameter from a normal distribution with the coefficient of variation `cv`. The global sensitivity methods of `sbmlsim.sensitivity` build on scans like these, see [Sensitivity analysis](sensitivity.md).
