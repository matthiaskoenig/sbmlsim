# Data

Simulation experiments compare simulations with experimental data. `sbmlsim` represents experimental data as `DataSet` objects, data frames which know the units of their columns, and references both simulation results and datasets through `Data` objects, which plots, functions and fit mappings consume.

## Datasets

A `DataSet` is a `pandas.DataFrame` with units. It is created from a data frame whose columns declare their units, either with a `*_unit` column per column or with one `unit` column which applies to `mean`, `value`, `median` and their `sd`/`se` columns:

```python
import pandas as pd

from sbmlsim.data import DataSet
from sbmlsim.units import UnitsInformation

df = pd.DataFrame(
    {
        "time": [0.0, 10.0, 20.0, 30.0],
        "time_unit": ["min"] * 4,
        "mean": [0.0, 2.1, 1.6, 1.1],
        "mean_sd": [0.0, 0.3, 0.2, 0.2],
        "mean_unit": ["mg/l"] * 4,
    }
)
ureg = UnitsInformation._default_ureg()
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

## Resolving data

`Data.get_data(experiment)` returns the quantity for the data in a run experiment, i.e., the values of the task result or the dataset column with their units, optionally converted to other units:

```python
from pathlib import Path

from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.task import Task


class DataExperiment(SimulationExperiment):
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": REPRESSILATOR_SBML}

    def simulations(self) -> dict[str, AbstractSim]:
        return {"tc": TimecourseSim(Timecourse(start=0, end=100, steps=100))}

    def tasks(self) -> dict[str, Task]:
        return {"task_tc": Task(model="model", simulation="tc")}

    def data(self) -> dict[str, Data]:
        data = [Data(sid, task="task_tc") for sid in ["time", "[X]"]]
        return {d.sid: d for d in data}


runner = ExperimentRunner(
    [DataExperiment],
    simulator=SimulatorSerial(),
    base_path=Path.cwd(),
    data_path=Path.cwd(),
)
results = runner.run_experiments(output_path=Path.cwd() / "results")
experiment = results[0].experiment

time = Data("time", task="task_tc").get_data(experiment)
print(time.units, time.magnitude[:3])
x = Data("[X]", task="task_tc").get_data(experiment, to_units="dimensionless")
print(x.units, x.magnitude[:3])
```

## Data generators

A `DataGenerator` (see `sbmlsim.combine.datagenerator`) post-processes results, e.g., `DataGeneratorIndexingFunction` reduces a scan result to a single time point, which turns a scan over doses into a dose response, see `examples/datagenerator.py`.
