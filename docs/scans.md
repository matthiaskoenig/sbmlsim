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

The result has the dimensions of the scan first and the time last, `(dim_n, time)`. A changed target is a coordinate along its dimension, here `n`, unless it is also a selection of the model: then the result keeps its timecourse under the plain name and stores the values of the dimension as `<dimension>.<target>`, e.g. `dim_n.n` when `n` is selected; the `Data` of a task of a simulation experiment names them `<dimension>.<target>` in either case, see [Data](data.md#selecting-points). The labels of a dimension of values are `0..n-1` by default and can be given with `labels=`.

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

A dimension and a scan do not change after they were created, the values of a dimension are read-only copies. `dataclasses.replace(dimension, at=10)` creates another dimension with other fields, validated as a new one; it keeps the labels, `labels=None` takes the default labels of values of another length.

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

When every simulation has the same output times (`times` or `steps`), the result has the dimension `time`. With the steps of the integrator every simulation keeps its own time points: the result has the dimension `_point` and the variable `time`, padded with `NaN`. `time=` interpolates every timecourse onto a grid; the value at the time of a change is the value after it:

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

## Observables

A scan computes observables from every simulation, e.g. the maximum of a concentration or the parameters of a non-compartmental analysis, with `Simulator.run(model, scan, observables, keep=...)`; see [Observables](observables.md):

```python
from sbmlsim.simulation import Formula

res = simulator.run(
    model,
    Scan(
        Simulation(end=100, steps=100), [Dimension("dim_n", values={"n": [2.0, 3.0]})]
    ),
    [Formula("px_max", "max(PX)")],
)
print(res["px_max"].values)
```

## Working with scan results

A `ScanResult` wraps an `xarray.Dataset`, `res.ds`, and keeps the units of its variables and coordinates; `res.quantity(key)` gives the values with their unit, `res.summary(dims)` the statistics over dimensions and `res.interpolate(times)` puts a ragged result on a grid. `res.ds.to_dataframe()` is the table, `res.to_netcdf(path)` and `ScanResult.from_netcdf(path)` store the result with its units:

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

## Designs

A design is a dimension whose values follow a design: the sampler `sbmlsim.simulation.sampling` creates the local design of all parameters, random draws, Latin hypercubes, the designs of the global sensitivity analyses, draws of the parameters of a fit and virtual populations, see [Sampling and uncertainty](sampling.md):

```python
from sbmlsim.simulation import sampling

simulation = Simulation(end=100, steps=100)
parameters = sampling.parameters_of(model)
local = sampling.local(parameters, delta=0.1, model=model)
res = simulator.run(model, Scan(simulation, [local]))
print(res["PX"].sizes)

draws = sampling.lhs(
    {pid: sampling.LogNormal(cv=0.05) for pid in parameters}, 10, seed=1, model=model
)
res = simulator.run(model, Scan(simulation, [draws]))
print(res["PX"].sizes)
```

The local design varies every constant parameter alone up and down by the relative `delta` around its reference, the value the model gives it after the pre-initialization of the simulation; the Latin hypercube draws 10 points of lognormal distributions around the references. The record of a design is part of the provenance of the result.
