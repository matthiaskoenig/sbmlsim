# Simulation experiments

A `SimulationExperiment` is the reproducible description of an experiment: the models, the datasets, the simulations, the tasks which apply a simulation to a model, the data derived from the results, and the figures and reports. The experiment is a python class; the `ExperimentRunner` executes it and writes results, figures and a JSON serialization.

## Defining an experiment

An experiment subclasses `SimulationExperiment` and overrides the methods for its parts. Every method returns a dictionary keyed by identifier, and the parts reference each other by these identifiers:

```python
from pathlib import Path

from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import AbstractSim, Timecourse, TimecourseSim
from sbmlsim.task import Task


class RepressilatorExperiment(SimulationExperiment):
    """Repressilator with a perturbation of X."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": REPRESSILATOR_SBML}

    def simulations(self) -> dict[str, AbstractSim]:
        return {
            "tc": TimecourseSim(
                [
                    Timecourse(start=0, end=100, steps=100),
                    Timecourse(start=0, end=100, steps=100, changes={"X": 10}),
                ]
            )
        }

    def tasks(self) -> dict[str, Task]:
        return {"task_tc": Task(model="model", simulation="tc")}

    def data(self) -> dict[str, Data]:
        data = [Data(sid, task="task_tc") for sid in ["time", "[X]", "[Y]", "[Z]"]]
        return {d.sid: d for d in data}

    def figures(self) -> dict[str, Figure]:
        fig = Figure(experiment=self, sid="fig1", name="Repressilator", num_rows=1)
        plots = fig.create_plots(
            xaxis=Axis("time", unit="second"),
            yaxis=Axis("concentration", unit="dimensionless"),
            legend=True,
        )
        for sid in ["[X]", "[Y]", "[Z]"]:
            plots[0].curve(
                x=Data("time", task="task_tc"), y=Data(sid, task="task_tc"), label=sid
            )
        return {"fig1": fig}
```

- **models** are paths, URLs or `AbstractModel` objects with changes, see [Models](models.md). They are resolved relative to the `base_path` of the experiment.
- **simulations** are `TimecourseSim` or `ScanSim` objects, see [Timecourse simulations](simulation.md) and [Parameter scans](scans.md).
- **tasks** apply a simulation to a model; the results of the experiment are keyed by task.
- **data** are `Data` objects referencing a task or a dataset, see [Data](data.md).
- **figures** are `Figure` objects with plots and curves, see [Plots and reports](plotting.md).
- **datasets** (not used here) are `DataSet` objects with experimental data, see [Data](data.md).

## Running an experiment

The `ExperimentRunner` creates the experiments, loads their models into a simulator and runs them. The results, the figures (`svg` by default) and the JSON serialization of every experiment are written below `output_path`:

```python
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.simulator import SimulatorSerial

runner = ExperimentRunner(
    [RepressilatorExperiment],
    simulator=SimulatorSerial(),
    base_path=Path.cwd(),
    data_path=Path.cwd(),
)
results = runner.run_experiments(output_path=Path.cwd() / "results")
print(results[0].experiment)
print(sorted(p.name for p in (Path.cwd() / "results").rglob("*") if p.is_file()))
```

`base_path` is the directory the model sources are resolved against, `data_path` the directory of the datasets. The `results` of an experiment are the `XResult` of every task:

```python
experiment = results[0].experiment
xres = experiment.results["task_tc"]
print(xres["[X]"].values[-3:])
```

`run_experiments(reduced_selections=True)` records only the variables the data of the experiment refer to, which speeds up large experiments; `reduced_selections=False` records everything.

## Reports

`ExperimentReport` renders the results of one or several experiments into an HTML (or markdown) report with the figures, the models and the simulations of every experiment:

```python
from sbmlsim.report.experiment_report import ExperimentReport

report = ExperimentReport(results)
report_path = report.create_report(output_path=Path.cwd() / "results")
print(report_path)
```

## Serialization

Every experiment is serialized to JSON when it is run, `<ExperimentId>.json` below the output path, with the models, simulations, tasks, data and figures:

```python
print(experiment.to_json()[:300])
```

## Where to look

The [examples](https://github.com/matthiaskoenig/sbmlsim/tree/develop/examples) contain complete experiments: `examples/initial_assignment` for a model with changes and two simulations, `examples/curve_types` for the curve types of the plots, `examples/glucose` for an experiment with datasets and a dose response scan, and `examples/hctz` for pharmacokinetics experiments with data from two studies and the parameter fitting problems built on them.
