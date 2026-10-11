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

A value replaces its target wherever the simulation sets it, i.e. in the `preinit_changes` and in every `Change`; a target which the simulation does not set is a change before the initialization. Scanning the dose of a multiple dosing therefore scans every dose. Scanning the rate of an infusion which a later `Change` stops therefore also replaces the stop, so a scan logs a warning for a dimension without `at` whose target the simulation sets to different values (in `changes` at different times, in `preinit_changes` and a change, or in the presimulation), naming the values with their times; a dose repeated with one value does not warn. Use `at` to set the values at one time and keep the other changes of the simulation. A dimension with `at` sets its values as a `Change` at that time instead:

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

When every simulation has the same output times (`times` or `steps`), the result has the dimension `time`; an output time at a change after the start appears twice in it, the states before and after the change (see [Simulations](simulation.md#the-semantics)). Simulations whose output times differ only in the changes at them, e.g. a dimension of simulations with a dose at different times of the grid, have the times of the grid, a time twice where one of them has a change at it. With the steps of the integrator every simulation keeps its own time points: the result has the dimension `_point` and the variable `time`, padded with `NaN`.

`time=` interpolates every timecourse onto a grid, and `res.interpolate(times)` interpolates a result by the same rule, with the times of the changes of the simulations, which the result keeps in `attrs["changes"]`:

- a time before a change is interpolated toward the state before it;
- a time at a change sees the state after it;
- a time which appears twice in the grid sees the state before the change first, then the state after it.

An event of the model is a jump as well, whose states before and after it the steps of the integrator have at the time of the event: a time at it sees the state after it, and so does a time at most the resolution of the time before it, since the integrator locates an event a little after its time. Jumps closer than the resolution of the time, e.g. a change and an event, are one chain, and a time at it or inside it sees the state after the last jump, as the data of a fit does (see [Fitting](fitting.md#data-at-the-time-of-a-change)):

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

Since the time of a change appears twice in `res.ds`, the idioms of xarray which need a unique index raise there, e.g. `res.ds.interp(time=...)`, `res.ds.sel(time=..., method="nearest")` and `res.ds.reindex(time=...)`: `res.interpolate(times)` puts a result on other times, `res.sel(time=t)` at a change selects both rows and `res.isel(time=k)` one of them. The `sel` of a sensitivity result, `SensitivityResult.sel(time=t)`, selects the state after the change instead, as a time of a grid sees it (see [Sensitivity analysis](sensitivity.md#the-result)).

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

A `ScanResult` wraps an `xarray.Dataset`, `res.ds`, and keeps the units of its variables and coordinates; `res.quantity(key)` gives the values with their unit, `res.summary(dims)` the statistics over dimensions and `res.interpolate(times)` puts a ragged result on a grid. `res.ds.to_dataframe()` is the table, `res.to_netcdf(path)` and `ScanResult.from_netcdf(path)` store the result with its units. The time of a change after the start appears twice in `res.ds`, the states before and after the change, so `res.interpolate(times)` replaces `res.ds.interp`, see [The output and the run](#the-output-and-the-run):

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
