# Simulation experiments

A `SimulationExperiment` is the reproducible description of an experiment: the models, the datasets, the simulations, the tasks which apply a simulation to a model, the data derived from the results, and the figures. The experiment is a python class; the `ExperimentRunner` executes it and writes results, figures and a JSON serialization.

## Defining an experiment

An experiment subclasses `SimulationExperiment` and overrides the methods for its parts. Every method returns a dictionary keyed by identifier, and the parts reference each other by these identifiers. The methods are marked with `typing.override`, so that a type checker reports a method which overrides nothing, e.g. a misspelled `simulation`, and the examples define them in the order of their dependencies: datasets, models, simulations, observables, tasks, data, fit mappings and figures.

```python
from pathlib import Path
from typing import override

from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Change, Simulation
from sbmlsim.task import Task


class RepressilatorExperiment(SimulationExperiment):
    """Repressilator with a perturbation of X."""

    @override
    def models(self) -> dict[str, AbstractModel | Path]:
        return {"model": REPRESSILATOR_SBML}

    @override
    def simulations(self) -> dict[str, Simulation]:
        return {"tc": Simulation(end=200, changes=[Change(100, {"X": 10})], steps=200)}

    @override
    def tasks(self) -> dict[str, Task]:
        return {"task_tc": Task(model="model", simulation="tc")}

    @override
    def data(self) -> dict[str, Data]:
        data = [Data(sid, task="task_tc") for sid in ["time", "[X]", "[Y]", "[Z]"]]
        return {d.sid: d for d in data}

    @override
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

- **models** are paths, SBML strings or `AbstractModel` objects with changes, see [Models](models.md). They are resolved relative to the `base_path` of the experiment.
- **simulations** are `Simulation` or `Scan` objects, see [Simulations](simulation.md) and [Parameter scans](scans.md); the changes of a model are changes before the initialization of every simulation of it, unless the simulation sets the target itself.
- **observables** (not used here) are the `Formula`, `PK` and `Custom` observables the data of the tasks read, see [Observables and scans](#observables-and-scans).
- **tasks** apply a simulation to a model; the results of the experiment are keyed by task.
- **data** are `Data` objects referencing a task or a dataset, see [Data](data.md).
- **figures** are `Figure` objects with plots and curves, see [Plots and reports](plotting.md).
- **datasets** (not used here) are `DataSet` objects with experimental data, see [Data](data.md).

## Observables and scans

`observables()` declares the observables of the experiment, the `Formula`, `PK` and `Custom` of [Observables](observables.md), by their id; it is defined after `simulations()` and before `tasks()`. A `Data` of a task reads an observable by its id (`Data("pk.cmax", task="task_doses")` for a parameter of a `PK` observable), and every task computes the observables its data read, in one run with the selections they read; the data of `data()`, of the fit mappings and of the figures count. The index of every task data is checked when the experiment is initialized: an index which is neither `time`, an observable, a coordinate of the scan of the task nor a selection of its model raises before anything is simulated, as do a `PK` observable read without a parameter and a dimension or label of `sel` which the scan of the task has not. The data of a task is a labelled array with the dimensions of its scan, see [Data](data.md); a curve draws one line, so the point of a scan it shows is selected with `Data(sel=...)`. `examples/experiment_scans.py` reads the PK parameters of midazolam over three doses.

## Running an experiment

The `ExperimentRunner` creates the experiments, loads their models into a simulator and runs them. The results, the figures (`svg` by default) and the JSON serialization of every experiment are written below `output_path`:

```python
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.simulator import Simulator

runner = ExperimentRunner(
    [RepressilatorExperiment],
    simulator=Simulator(),
    base_path=Path.cwd(),
    data_path=Path.cwd(),
)
results = runner.run_experiments(output_path=Path.cwd() / "results", keep_results=True)
print(results[0].experiment)
print(sorted(p.name for p in (Path.cwd() / "results").rglob("*") if p.is_file()))
```

`base_path` is the directory the model sources are resolved against, `data_path` the directory of the datasets. The `results` of an experiment are the `ScanResult` of every task, written as netCDF with `save_results=True`. The runner releases the results of an experiment once its outputs are written, so that a run of many experiments holds only the results of the one it runs, and `results` raises afterwards; `keep_results=True` keeps them, as above:

```python
experiment = results[0].experiment
res = experiment.results["task_tc"]
print(res["[X]"].values[-3:])
```

`run_experiments(reduced_selections=True)` records for every task only the variables its data refer to, which speeds up large experiments; `reduced_selections=False` records every variable of the model and the ones the data of the task refer to.

## Reports

`ExperimentReport` renders the results of one or several experiments into an HTML (or markdown) report with the figures, the models and the simulations of every experiment:

```python
from sbmlsim.report.experiment_report import ExperimentReport

report = ExperimentReport(results)
report_path = report.create_report(output_path=Path.cwd() / "results")
```

The report ends the output with its section, the number of experiments and the link to the report, which the terminal opens with a click (a hyperlink in VS Code, iTerm2, Windows Terminal, GNOME Terminal or kitty, the `file://` URI in any other terminal and in a log). `show_report=True` opens the report in the web browser. The sections, key/value blocks and links of `sbmlsim.display` are the output of scripts as well, e.g. `display.link("figures", figures_dir)`.

## Serialization

Every experiment is serialized to JSON when it is run, `<ExperimentId>.json` below the output path, with the models, simulations, tasks, data and figures:

```python
print(experiment.to_json()[:300])
```

## Where to look

The [examples](https://github.com/matthiaskoenig/sbmlsim/tree/develop/examples) contain complete experiments: `examples/initial_assignment` for a model with changes and two simulations, `examples/curve_types` for the curve types of the plots, `examples/glucose` for an experiment with datasets and a dose response scan, and `examples/hctz_fitting` for pharmacokinetics experiments with data from two studies and the parameter fitting problems built on them.
