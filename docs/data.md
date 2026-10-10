# Data

Simulation experiments compare simulations with experimental data. `sbmlsim` represents experimental data as `DataSet` objects, data frames which know the units of their columns, and references both simulation results and datasets through `Data` objects, which plots, functions and fit mappings consume.

## Datasets

A `DataSet` is a `pandas.DataFrame` with units. It is created from a data frame whose columns declare their units, either with a `*_unit` column per column or with one `unit` column which applies to `mean`, `value`, `median` and their `sd`/`se` columns:

```python
import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.units import ureg

df = pd.DataFrame(
    {
        "time": [0.0, 10.0, 20.0, 30.0],
        "time_unit": ["min"] * 4,
        "mean": [0.0, 2.1, 1.6, 1.1],
        "mean_sd": [0.0, 0.3, 0.2, 0.2],
        "mean_unit": ["mg/l"] * 4,
    }
)
dset = DataSet.from_df(df, ureg=ureg)
print(dset.uinfo["time"], dset.uinfo["mean"])
print(dset)
```

The units of the dataset are a `UnitsInformation` like the units of a model, see [Units](units.md). A column is read as a quantity and converted:

```python
q = dset.get_quantity("mean")
print(q.to("mg/dl"))
```

In an experiment the datasets are returned by `datasets()`, keyed by identifier, and read from the `data_path` of the experiment. `load_pkdb_dataframe` reads the TSV files of a [PK-DB](https://pk-db.com) study, see `examples/glucose/experiments/dose_response.py`.

## Data references

A `Data` object is a promise for data: it names a variable of a task, a column of a dataset, or a function of other data, and is resolved when the experiment has run. A plot curve or a fit mapping is described with `Data` objects before any simulation exists:

```python
from sbmlsim.data import Data

x = Data("time", task="task_tc")
y = Data("[X]", task="task_tc")
y_data = Data("mean", dataset="dset1")
print(x.sid, x.dtype, y.selection)
print(y_data.sid, y_data.dtype)
```

A species in brackets is a concentration, without brackets an amount; `Data.selection` is the roadrunner selection recorded for it. In an experiment the `data()` method returns the `Data` objects, and every variable used in a figure or a fit must be declared there, since the selections of the simulation are reduced to them.

## Functions of data

Data can be a function of other data, written as a formula with the referenced data as variables. The function is evaluated on the resolved data with their units:

```python
f = Data(
    "[X]_ratio",
    function="x / y",
    variables={"x": Data("[X]", task="task_tc"), "y": Data("[Y]", task="task_tc")},
)
print(f.sid, f.dtype, f.function)
```

The function is a formula of the math of PEtab, the same as the formulas of changes and observables. One extension serves data: `max` and `min` of a single argument reduce it along the time of every simulation and ignore the `NaN` of padding, so `Y/max(Y)` normalizes every simulation of a scan to its own maximum; with two or more arguments they are the elementwise maximum and minimum. The reductions `mean` and `at` need the time points of a simulation and are part of the observables of a scan, see [Observables](observables.md).

The math of PEtab is not the L3 formula syntax of SBML which a function of data was written in before, and one difference changes a result without an error: `log(x)` is the natural logarithm, as in PEtab, where the L3 syntax read it as the logarithm to base 10. Write `log10(x)` for base 10 (`log2(x)` for base 2) and `log(x, b)` for the logarithm of `x` to base `b`. The other functions and constants of the L3 syntax which differ fail with a `ValueError` when the data is evaluated:

- `asin`, `acos`, `atan`, `asinh` and the other inverse functions are `arcsin`, `arccos`, `arctan`, `arcsinh` and so on.
- `root(n, x)` is `sqrt(x)` for `n = 2` and `x^(1/n)` otherwise.
- `pi` is an identifier like any other in PEtab, so its value is a parameter of the data, `Data(..., function="x * pi", parameters={"pi": math.pi})`.
- `ceiling`, `ceil`, `floor` and `factorial` are not functions of the math of PEtab and have no equivalent.

## Resolving data

`Data.get_data(experiment)` returns the values of the data in a run experiment as a labelled array, an `xarray.DataArray` with the dimensions of its source, their coordinates and the unit in `attrs["units"]`, optionally converted to other units; `sbmlsim.data.to_quantity(array, ureg)` gives the pint quantity:

```python
from pathlib import Path
from typing import override

from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task


class DataExperiment(SimulationExperiment):
    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": REPRESSILATOR_SBML}

    @override
    def simulations(self) -> dict[str, Simulation]:
        return {"tc": Simulation(end=100, steps=100)}

    @override
    def tasks(self) -> dict[str, Task]:
        return {"task_tc": Task(model="model", simulation="tc")}

    @override
    def data(self) -> dict[str, Data]:
        data = [Data(sid, task="task_tc") for sid in ["time", "[X]"]]
        return {d.sid: d for d in data}


runner = ExperimentRunner(
    [DataExperiment],
    simulator=Simulator(),
    base_path=Path.cwd(),
    data_path=Path.cwd(),
)
results = runner.run_experiments(output_path=Path.cwd() / "results")
experiment = results[0].experiment

time = Data("time", task="task_tc").get_data(experiment)
print(time.attrs["units"], time.values[:3])
x = Data("[X]", task="task_tc").get_data(experiment, to_units="dimensionless")
print(x.attrs["units"], x.values[:3])
```

## Selecting points

The data of a task has the dimensions of its scan: a timecourse is over `(*dims, time)`, or `(*dims, _point)` for a ragged result whose simulations keep their own time points padded with `NaN`, a value per simulation is over `(*dims)`, and the values a dimension sets are `Data("<dimension>.<target>")` over the dimension, while the plain name of a symbol is its timecourse, also when the scan changes it. `sel` selects labels: `Data("[X]", task="task_scan", sel={"dose": "high"})` keeps one point and drops the dimension, `sel={"dose": ["low", "high"]}` keeps the dimension with two labels. A dimension of the scan which the data has not, e.g. the dimension of a scan for the time on a common grid, is skipped, so the same `sel` serves the time and the values of a curve; a dimension or label which does not exist raises with the ones which do. The data of a dataset is a column over the dimension `row`, and `sel={"group": "b"}` keeps the rows whose column `group` has the value. A function broadcasts its data by the names of their dimensions, and `max` and `min` of a single argument reduce along the time of a timecourse, or along the rows of a dataset.

## Data of observables

An experiment declares observables with `observables()`, the `Formula`, `PK` and `Custom` of [Observables](observables.md), and a `Data` of a task reads one by its id, the parameter of a `PK` observable as `<id>.<parameter>`. A task computes the observables its data read, in one run with the selections it reads:

```python
import numpy as np

from sbmlsim.simulation import Dimension, Formula, Observable, Scan


class ObservableExperiment(DataExperiment):
    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        return {
            "tc": Simulation(end=100, steps=100),
            "scan": Scan(
                Simulation(end=100, steps=100),
                [Dimension("x0", values={"X": np.array([10.0, 20.0, 40.0])})],
            ),
        }

    @override
    def observables(self) -> dict[str, Observable]:
        return {"xmax": Formula("xmax", "max([X])")}

    @override
    def tasks(self) -> dict[str, Task]:
        return {
            "task_tc": Task(model="model", simulation="tc"),
            "task_scan": Task(model="model", simulation="scan"),
        }

    @override
    def data(self) -> dict[str, Data]:
        return {
            "data_xmax": Data("xmax", task="task_scan"),
            "x": Data("[X]", task="task_scan"),
        }


results = ExperimentRunner(
    [ObservableExperiment],
    simulator=Simulator(),
    base_path=Path.cwd(),
    data_path=Path.cwd(),
).run_experiments(output_path=Path.cwd() / "results")
experiment = results[0].experiment

xmax = Data("xmax", task="task_scan").get_data(experiment)
print(xmax.dims, xmax["X"].values, xmax.values.round(3))
x = Data("[X]", task="task_scan", sel={"x0": 2}).get_data(experiment)
print(x.dims, float(x.max()) == float(xmax.sel(x0=2)))
```
