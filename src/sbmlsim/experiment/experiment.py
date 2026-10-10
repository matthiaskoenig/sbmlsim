"""SimulationExperiments and helpers."""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib import pyplot as plt

from sbmlsim.data import Data, DataSet, check_sel
from sbmlsim.fit import FitMapping
from sbmlsim.fit.objects import FitDataInitialized
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot import Figure
from sbmlsim.plot.points import point_linestyles
from sbmlsim.plot.serialization_matplotlib import (
    FigureMPL,
    MatplotlibFigureSerializer,
)
from sbmlsim.result import ScanResult
from sbmlsim.result.scan import POINT, TIME
from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulation.observables import PK, Observable
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from sbmlsim.units import UnitRegistry
from sbmlsim.units import ureg as package_ureg
from sbmlsim.utils import timeit

logger = logging.getLogger(__name__)

#: format of the static images, which matplotlib draws
STATIC_FORMAT = "svg"

#: format of the interactive figures, which plotly draws. It is a value of
#: `figure_formats`, so an experiment asks for the pages the way it asks for
#: an image: `figure_formats=["svg", "html"]`
INTERACTIVE_FORMAT = "html"


class SimulationExperiment:
    """Generic simulation experiment.

    Consists of models, datasets, simulations, tasks, results, processing, figures
    """

    def __init__(
        self,
        sid: str | None = None,
        base_path: Path | None = None,
        data_path: Path | Iterable[Path] | None = None,
        ureg: UnitRegistry | None = None,
        **kwargs,
    ):
        """SimulationExperiement.

        :param sid:
        :param base_path:
        :param data_path:
        :param ureg:
        :param kwargs:
        """
        self.sid: str = sid if sid else self.__class__.__name__
        # the simulator is set by the ExperimentRunner
        self.simulator: Simulator | None = None

        if base_path:
            base_path = Path(base_path).resolve()
            if not base_path.exists():
                raise OSError(f"base_path '{base_path}' does not exist")
        else:
            logger.warning(
                "No 'base_path' provided, reading/writing of resources may fail."
            )
        self.base_path = base_path

        data_paths: list[Path] | None = None
        if data_path:
            if isinstance(data_path, (str, Path)):
                data_paths = [Path(data_path).resolve()]
            else:
                data_paths = [Path(p).resolve() for p in data_path]
            for p in data_paths:
                if not p.exists():
                    raise OSError(f"data_path '{p}' does not exist")
        else:
            logger.warning("No 'data_path' provided, reading of datasets may fail.")
        self.data_path: list[Path] | None = data_paths

        # single UnitRegistry per SimulationExperiment (can be shared)
        if not ureg:
            ureg = package_ureg
        self.ureg = ureg

        # settings
        self.settings = kwargs

        # init variables
        self._models: dict[str, RoadrunnerSBMLModel] = {}
        self._data: dict[str, Data] = {}
        self._datasets: dict[str, DataSet] = {}
        self._fit_mappings: dict[str, FitMapping] = {}
        self._simulations: dict[str, Simulation | Scan] = {}
        self._observables: dict[str, Observable] = {}
        self._tasks: dict[str, Task] = {}
        self._figures: dict[str, Figure] = {}
        self._results: dict[str, ScanResult] = {}
        # the keys of the matplotlib figures of the last run; the figures are
        # released once they are written (the report needs only the keys)
        self._mpl_figure_keys: list[str] = []
        # the results were released after the run wrote its outputs
        self._results_released: bool = False

    def initialize(self) -> None:
        """Initialize SimulationExperiment.

        Initialization must be separated from object construction due to
        the parallel execution of the problem later on.
        Certain objects cannot be serialized and must be initialized.
        :return:
        """
        try:
            # initialized from the outside
            # self._models: dict[str, AbstractModel] = self.models()
            self._datasets.update(self.datasets())
            self._simulations.update(self.simulations())
            self._observables.update(self.observables())
            self._tasks.update(self.tasks())
            self._data.update(self.data())
            self._figures.update(self.figures())
            self._fit_mappings.update(self.fit_mappings())

            # validation of information
            self._check_keys()
            self._check_types()
            self._check_task_data()
            self._check_figures()
        except Exception as err:
            logger.error("Problem initializing '%s'", self.__class__.__name__)
            raise err

    def __str__(self) -> str:
        """Get string representation."""
        info = [
            f"*** SimulationExperiment: {self.__class__.__name__} ***",
            f"{'data':20} {list(self._data.keys())}",
            f"{'datasets':20} {list(self._datasets.keys())}",
            f"{'fit_mappings':20} {list(self._fit_mappings.keys())}",
            f"{'simulations':20} {list(self._simulations.keys())}",
            f"{'observables':20} {list(self._observables.keys())}",
            f"{'tasks':20} {list(self._tasks.keys())}",
            f"{'results':20} {list(self._results.keys())}",
            f"{'figures':20} {list(self._figures.keys())}",
        ]
        return "\n".join(info)

    def models(self) -> dict[str, AbstractModel | Path]:
        """Define model definitions.

        The child classes fill out the information.
        """
        return {}

    def datasets(self) -> dict[str, DataSet]:
        """Define dataset definitions (experimental data).

        The child classes fill out the information.
        """
        return {}

    def simulations(self) -> Mapping[str, Simulation | Scan]:
        """Define simulation definitions.

        The child classes fill out the information.
        """
        return {}

    def observables(self) -> dict[str, Observable]:
        """Define the observables of the experiment, by their id.

        A `Formula`, `PK` or `Custom` of `sbmlsim.simulation.observables`. A
        `Data` of a task reads an observable by its id and a parameter of a
        `PK` observable as `<id>.<parameter>`; a task computes the observables
        its data read, see `_run_tasks`. The child classes fill out the
        information.
        """
        return {}

    def tasks(self) -> dict[str, Task]:
        """Define task definitions.

        The child classes fill out the information.
        """
        return {}

    def data(self) -> dict[str, Data]:
        """Define the data of the experiment, including functions of other data.

        This determines the selection in the model.

        The data of tasks, fit mappings and figures determines the selections
        of a simulation experiment; registering the data of a figure here is no
        longer needed. Registered data of a task which no figure or mapping
        reads still counts, e.g. via `add_selections_data`.
        """
        return {}

    def figures(self) -> dict[str, Figure]:
        """Figure definition.

        The data the curves and areas of the figures read counts for the
        selections of the tasks, it needs no registration in `data()`.

        Most figures do not require access to concrete data, but only abstract
        data concepts.
        """
        return {}

    def figures_mpl(self) -> dict[str, FigureMPL]:
        """Matplotlib figure definition.

        Selections accessed in figures and analyses must be registered beforehand
        via the data of the experiment.

        Most figures do not require access to concrete data, but only abstract
        data concepts.
        """
        return {}

    def fit_mappings(self) -> dict[str, FitMapping]:
        """Define fit mappings.

        Mapping reference data on observables.
        Used for the optimization of parameters.
        The child classes fill out the information.
        """
        return {}

    # --- DATA ------------------------------------------------------------------------
    def add_data(self, d: Data) -> None:
        """Add data to the tracked data."""
        self._data[d.sid] = d

    def add_selections_data(
        self,
        selections: Iterable[str],
        task_ids: Iterable[str] | None = None,
    ) -> None:
        """Add selections to given tasks.

        The data for the selections will be part of the results.

        Selections are necessary to access data from simulations.
        Here these selections are added to the tasks. If no tasks are given,
        the selections are added to all tasks.

        :param reset: drop and reset all selections.
        """
        # FIXME: handle reset
        # if reset is False:
        #     self._data = {}

        if task_ids is None:
            task_ids = self._tasks.keys()

        for task_id in task_ids:
            for selection in selections:
                self.add_data(Data(index=selection, task=task_id))

    # --- RESULTS ---------------------------------------------------------------------
    @property
    def results(self) -> dict[str, ScanResult]:
        """Access simulation results.

        Results are mapped on tasks based on the task_ids. E.g.
        to get the results for the task with id 'task_glciv' use
        ```
            simexp.results["task_glciv"]
            self.results["task_glciv"]
        ```
        """
        if self._results_released:
            raise RuntimeError(
                f"The results of '{self.sid}' were released once its outputs were "
                f"written; run it with `keep_results=True` to keep them."
            )
        if self._results is None:
            self._run_tasks(self.simulator)
        return self._results

    # --- VALIDATION ------------------------------------------------------------------
    def _check_keys(self):
        """Check keys in information dictionaries."""
        # string keys for main objects must be unique on SimulationExperiment
        all_keys = {}
        allowed_types = dict
        for field_key in [
            "_models",
            "_datasets",
            "_simulations",
            "_observables",
            "_tasks",
            "_data",
            "_figures",
            "_fit_mappings",
        ]:
            field = getattr(self, field_key)

            if not isinstance(field, allowed_types):
                raise ValueError(
                    f"SimulationExperiment '{self.sid}': '{field_key} must be a "
                    f"'{allowed_types}', but '{field}' is type '{type(field)}'. "
                    f"Check that the respective definition returns an object of type "
                    f"'{allowed_types}. Often simply the return statement is missing "
                    f"(returning NoneType)."
                )

            pattern_sid = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
            for key in getattr(self, field_key):
                if not isinstance(key, str):
                    raise ValueError(
                        f"'{field_key} keys must be str: '{key} -> {type(key)}'"
                    )
                # Check that valid Sid
                try:
                    if not pattern_sid.fullmatch(key):
                        raise ValueError(
                            f"{field_key} key is not a valid SId "
                            f"({pattern_sid.pattern}): '{key}'"
                        )
                except TypeError as err:
                    raise ValueError(
                        f"{field_key} key is not a valid SId. "
                        f"Incorrect type: '{key}', {type(key)}"
                    ) from err

                if key in all_keys:
                    raise ValueError(
                        f"Duplicate key '{key}' for '{field_key}' and '{all_keys[key]}'"
                    )
                all_keys[key] = field_key

    def _check_types(self):
        """Check for correctness of types."""
        for key, dset in self._datasets.items():
            if not isinstance(dset, DataSet):
                # FIXME: relaxing for now (re-enable) !!!
                logger.error(
                    "datasets must be of type DataSet, but dataset '%s' has type: '%s'",
                    key,
                    type(dset),
                )
                # raise ValueError(
                #     f"datasets must be of type DataSet, but "
                #     f"dataset '{key}' has type: '{type(dset)}'"
                # )

        for key, model in self._models.items():
            if not isinstance(model, AbstractModel):
                raise ValueError(
                    f"model must be of type AbstractModel, but "
                    f"model '{key}' has type: '{type(model)}'"
                )

        for key, sim in self._simulations.items():
            if not isinstance(sim, Simulation | Scan):
                raise ValueError(
                    f"simulations must be of type Simulation or Scan, but "
                    f"simulation '{key}' has type: '{type(sim)}'"
                )

        for key, observable in self._observables.items():
            if not isinstance(observable, Observable):
                raise ValueError(
                    f"observables must be of type Formula, PK or Custom, but "
                    f"observable '{key}' has type: '{type(observable)}'"
                )
            if observable.id != key:
                raise ValueError(
                    f"The observable of the key '{key}' has the id "
                    f"'{observable.id}': the key of an observable is its id."
                )

        for key, task in self._tasks.items():
            if not isinstance(task, Task):
                raise ValueError(
                    f"tasks must be of type Task, but "
                    f"task '{key}' has type: '{type(task)}'"
                )
            if task.simulation_id not in self._simulations:
                raise ValueError(
                    f"The task '{key}' of the experiment '{self.sid}' runs the "
                    f"simulation '{task.simulation_id}', which is no simulation of "
                    f"the experiment: {sorted(self._simulations)}."
                )
            if task.model_id not in self._models:
                raise ValueError(
                    f"The task '{key}' of the experiment '{self.sid}' runs the "
                    f"model '{task.model_id}', which is no model of the "
                    f"experiment: {sorted(self._models)}."
                )

        for key, data in self._data.items():
            if not isinstance(data, Data):
                raise ValueError(
                    f"data must be of type Data, but "
                    f"task '{key}' has type: '{type(data)}'"
                )

        for key, figure in self._figures.items():
            if not isinstance(figure, Figure):
                raise ValueError(
                    f"figure must be of type Figure, but "
                    f"task '{key}' has type: '{type(figure)}'"
                )

        for key, mapping in self._fit_mappings.items():
            if not isinstance(mapping, FitMapping):
                raise ValueError(
                    f"fit_mappings must be of type FitMapping, but "
                    f"mapping '{key}' has type: '{type(mapping)}'"
                )

    # --- EXECUTE ---------------------------------------------------------------------

    @timeit
    def run(
        self,
        simulator: Simulator | None,
        output_path: Path | None = None,
        show_figures: bool = False,
        save_results: bool = False,
        figure_formats: list[str] | None = None,
        reduced_selections: bool = True,
        keep_results: bool = True,
        on_error: Literal["raise", "log"] = "raise",
    ) -> "ExperimentResult":
        """Execute given experiment and store results.

        Args:
            simulator: the simulator of the tasks.
            output_path: directory of the outputs (datasets, results, figures and
                the serialization), none are written without it.
            show_figures: show the matplotlib figures.
            save_results: write the results of the tasks into the output path.
            figure_formats: formats of the figures, `STATIC_FORMAT` by default.
            reduced_selections: simulate only the selections the experiment uses.
            keep_results: keep the results of the tasks after the outputs are
                written. Without them an experiment holds only what its report
                needs, and a run of many experiments only the results of the one
                it runs; `results` then raises.
            on_error: what a failing figure does. `"raise"` (the default) raises
                its error, so a user who runs one experiment sees it; `"log"`
                logs the error with its traceback, skips the figure, writes the
                others and records the figure in `ExperimentResult.failed_figures`,
                which is what the `ExperimentRunner` uses.

        Returns:
            The result of the experiment, which the report is created from.
        """
        failed_figures: dict[str, str] = {}
        # run simulations (sets self._results)
        self._results_released = False
        self._run_tasks(simulator, reduced_selections=reduced_selections)

        # evaluate mappings
        self.evaluate_fit_mappings()

        # create outputs
        if output_path is None:
            if save_results:
                logger.error("'output_path' required to save results.")

        else:
            if not Path.exists(output_path):
                Path.mkdir(output_path, parents=True)
                logger.debug("'output_path' created: '%s'", output_path)

            # save outputs
            self.save_datasets(output_path)

            # Saving takes often much longer then simulation
            if save_results:
                self.save_results(output_path)

        # the format decides which backend draws a figure: matplotlib draws
        # the static images, plotly the interactive pages
        formats = figure_formats if figure_formats is not None else [STATIC_FORMAT]
        static_formats = [f for f in formats if f != INTERACTIVE_FORMAT]
        interactive = INTERACTIVE_FORMAT in formats

        # create figures, but only when something looks at them: rendering
        # every figure is most of the time a run takes, and a run without an
        # output path which does not show them would close them again
        self._mpl_figure_keys = []
        if show_figures or (output_path and static_formats):
            mpl_figures = self.create_mpl_figures(
                on_error=on_error, failed=failed_figures
            )
            if show_figures:
                self.show_mpl_figures(mpl_figures=mpl_figures)
            if output_path and static_formats:
                self.save_mpl_figures(
                    output_path,
                    mpl_figures=mpl_figures,
                    figure_formats=static_formats,
                    on_error=on_error,
                    failed=failed_figures,
                )
            self.close_mpl_figures(mpl_figures=mpl_figures)
            # only the keys are kept: a figure keeps the pixel buffer of its last
            # rendering, so the figures of every experiment of a run would stay
            # in memory until its end
            self._mpl_figure_keys = list(mpl_figures)

        if output_path and interactive:
            self.save_interactive_figures(
                output_path, on_error=on_error, failed=failed_figures
            )

        # only perform serialization after data evaluation (to access units)
        if output_path:
            # serialization
            self.to_json(output_path / f"{self.sid}.json")

        if not keep_results:
            # every output is written, the report needs no results
            self._results = {}
            self._results_released = True

        return ExperimentResult(
            experiment=self, output_path=output_path, failed_figures=failed_figures
        )

    @timeit
    def _run_tasks(
        self, simulator: Simulator | None, reduced_selections: bool = True
    ) -> None:
        """Run the tasks of the experiment, the tasks of a model one after another.

        The selections of a task are the selections its data read (the data
        of `data()`, of the fit mappings and of the figures), without
        `reduced_selections` every selection of the model and the ones its
        data read; they are set on the model right before the task runs. A
        task whose data read observables runs with the observables they need
        and keeps them next to its selections in one run. The changes of a
        model are defaults of the pre-initialization changes of every
        simulation of it, see `Simulator.compile`.

        Raises:
            ValueError: without a simulator.
        """
        if simulator is None:
            raise ValueError(
                f"The experiment '{self.sid}' has no simulator: run it with a "
                f"Simulator or through an ExperimentRunner with one."
            )
        if self._results is None:
            self._results = {}
        model_tasks: dict[str, list[str]] = defaultdict(list)
        for task_key, task in self._tasks.items():
            model_tasks[task.model_id].append(task_key)
        for model_id, task_keys in model_tasks.items():
            model = self._models[model_id]
            if not reduced_selections:
                model.set_selections(None)
                every = [s for s in model.selections or [] if s != TIME]
            for task_key in task_keys:
                task = self._tasks[task_key]
                scan = self._simulations[task.simulation_id]
                observed, selections = self._task_outputs(task_key)
                if not reduced_selections:
                    selections = [*every, *(s for s in selections if s not in every)]
                if not observed:
                    # the selections of the data of this task, not the ones of
                    # the other tasks of the model
                    model.set_selections(sorted({TIME, *selections}))
                    self._results[task_key] = simulator.run(model, scan)
                    continue
                self._results[task_key] = simulator.run(
                    model,
                    scan,
                    self._needed_observables(observed),
                    keep=[*observed, *selections],
                )

    def _figure_data(self) -> Iterator[Data]:
        """Iterate the data the curves and areas of the figures read."""
        for figure in self._figures.values():
            for plot in figure.get_plots():
                for curve in plot.curves:
                    for d in (curve.x, curve.y, curve.xerr, curve.yerr):
                        if d is not None:
                            yield d
                for area in plot.areas:
                    yield from (area.x, area.yfrom, area.yto)
                for band in plot.bands:
                    yield from (band.x, band.y)

    def _task_data(self) -> Iterator[Data]:
        """Iterate the data of the experiment which comes from a task.

        The data of `data()`, of every fit mapping and of every figure, and
        the variables of a function among them, i.e. everything a run has to
        simulate. A `FitData` builds its `Data` when it is created and does
        not register it, so a fit mapping is asked for its data here rather
        than looked up in `self._data`.

        Yields:
            Every `Data` of the experiment which reads a task.
        """

        def walk(d: Data) -> Iterator[Data]:
            if d.is_task():
                yield d
            elif d.is_function():
                for variable in d.variables.values():
                    found = (
                        self._data.get(variable)
                        if isinstance(variable, str)
                        else variable
                    )
                    if isinstance(found, Data):
                        yield from walk(found)

        sources: list[Data] = list(self._data.values())
        for mapping in self._fit_mappings.values():
            for fit_data in (mapping.reference, mapping.observable):
                for key in FitDataInitialized.KEYS:
                    # the `y` of an observable model is the name of a formula
                    # of selections, not an index of the simulation
                    if (
                        key == "y"
                        and fit_data is mapping.observable
                        and mapping.observable_model is not None
                    ):
                        continue
                    d = getattr(fit_data, key, None)
                    if isinstance(d, Data):
                        sources.append(d)
        sources.extend(self._figure_data())
        for d in sources:
            yield from walk(d)

    def _scan_dims(self, d: Data) -> set[str] | None:
        """Get the dimensions of the scan of task data after its `sel`.

        A variable or an observable of a scan has every dimension of the scan,
        a coordinate its own dimension, the time none on a common grid and
        every dimension of a ragged scan; a dimension selected by one label is
        gone. Dataset and function data are not known before they are drawn.

        Returns:
            The dimensions, `None` for dataset and function data.
        """
        if not d.is_task():
            return None
        simulation = self._simulations[self._tasks[str(d.task_id)].simulation_id]
        if not isinstance(simulation, Scan):
            return set()
        dims = [dimension.id for dimension in simulation.dimensions]
        kind = self._index_kind(d)
        if kind == "coordinate":
            head = d.selection.partition(".")[0]
            found = (
                {head}
                if head in dims
                else {
                    dimension.id
                    for dimension in simulation.dimensions
                    if d.selection in dimension.coordinates
                }
            )
        elif kind == "time":
            found = set(dims) if _ragged(simulation) else set()
        else:
            found = set(dims)
        single = {
            dim
            for dim, label in d.sel.items()
            if not isinstance(label, list | tuple | np.ndarray)
        }
        return found - single

    def _check_figures(self) -> None:
        """Check the curves and bands of the figures before anything is simulated.

        Every dimension of a scan which the task data of a curve has must be
        its axis (the one dimension of x which is not in `over`), in `over`,
        or selected by one label; a band reduces `across` and needs a common
        grid. Dataset and function data are checked when they are drawn.

        Raises:
            ValueError: for a dimension which is not named, an `over` or
                `across` dimension the data has not, more points of a second
                `over` dimension than line styles, or a band of a ragged scan.
        """
        for key, figure in self._figures.items():
            for plot in figure.get_plots():
                for curve in plot.curves:
                    self._check_lines(
                        key,
                        curve.sid,
                        curve.over,
                        None,
                        curve.x,
                        [curve.y, curve.xerr, curve.yerr],
                    )
                for band in plot.bands:
                    self._check_lines(
                        key, band.sid, band.over, band.across, band.x, [band.y]
                    )

    def _check_lines(
        self,
        figure: str,
        sid: str | None,
        over: tuple[str, ...],
        across: str | None,
        x: Data,
        others: Sequence[Data | None],
    ) -> None:
        """Check the dimensions of a curve or band, see `_check_figures`."""
        what = "band" if across is not None else "curve"
        xs = self._scan_dims(x)
        ys = [self._scan_dims(d) for d in others if d is not None]
        if xs is None or any(dims is None for dims in ys):
            return
        found = set(xs).union(*[dims for dims in ys if dims is not None])
        missing = [dim for dim in over if dim not in found]
        if missing:
            raise ValueError(
                f"The {what} '{sid}' of the figure '{figure}' draws a line per point of "
                f"{missing}, which its data has not: {sorted(found)}."
            )
        if across is not None:
            y = others[0]
            simulation = (
                self._simulations[self._tasks[str(y.task_id)].simulation_id]
                if y is not None and y.is_task()
                else None
            )
            if isinstance(simulation, Scan) and _ragged(simulation):
                raise ValueError(
                    f"The band '{sid}' of the figure '{figure}' reduces '{across}' of a "
                    f"ragged scan, whose simulations keep their own time points; run "
                    f"the scan on a common grid (a simulation with steps or times)."
                )
            if across not in found:
                raise ValueError(
                    f"The band '{sid}' of the figure '{figure}' reduces the dimension "
                    f"'{across}', which its data has not: {sorted(found)}."
                )
        axis = xs - set(over) - {across}
        if len(axis) > 1:
            raise ValueError(
                f"x of the {what} '{sid}' of the figure '{figure}' has the dimensions "
                f"{sorted(axis)} of the scan; name them in over= or select a label "
                f"with Data(sel=...)."
            )
        for dim in sorted(found - set(over) - axis - {across}):
            raise ValueError(
                f"y of {what} '{sid}' of the figure '{figure}' has the dimension "
                f"'{dim}'; name it with over='{dim}' or select a label with "
                f"Data(sel=...)."
            )
        if len(over) == 2:
            task_data = next(d for d in [x, *others] if d is not None and d.is_task())
            labels = _labels(
                self._simulations[self._tasks[str(task_data.task_id)].simulation_id]
            )
            known = labels.get(over[1])
            if known is not None:
                # the points which remain after the `sel` of the data, of y
                # if x and y select differently
                selected = [
                    d.sel[over[1]]
                    for d in (others[0], x)
                    if d is not None and d.is_task() and over[1] in d.sel
                ]
                if selected and isinstance(selected[0], list | tuple | np.ndarray):
                    point_linestyles(len(selected[0]))
                else:
                    point_linestyles(len(known))

    def scan_dimension(self, task: str, dim: str) -> Dimension | None:
        """Get a dimension of the scan of a task by its id, `None` if it has none of it."""
        simulation = self._simulations[self._tasks[task].simulation_id]
        if not isinstance(simulation, Scan):
            return None
        return next((d for d in simulation.dimensions if d.id == dim), None)

    def model_units(self, task: str) -> Mapping[str, str]:
        """Get the units of the symbols of the model of a task, `{}` for a model which is not loaded."""
        model = self._models.get(self._tasks[task].model_id)
        uinfo = getattr(model, "uinfo", None)
        return dict(uinfo) if uinfo is not None else {}

    def _index_kind(self, d: Data) -> str:
        """Classify the index of task data.

        Returns:
            `"time"`, `"observable"` (an observable or a parameter of a `PK`
            observable), `"coordinate"` (a dimension of the scan of the task,
            `<dimension>.<target>` of a target it changes or a coordinate of
            a dimension) or `"selection"`.

        Raises:
            ValueError: if the data reads a task which does not exist, a `PK`
                observable without a parameter, or an index which is none of
                these.
        """
        task = self._tasks.get(str(d.task_id))
        if task is None:
            raise ValueError(
                f"{d} reads the task '{d.task_id}', which is no task of the "
                f"experiment '{self.sid}': {sorted(self._tasks)}."
            )
        index = d.selection
        if index == TIME:
            return "time"
        if index in self._observables:
            if isinstance(self._observables[index], PK):
                raise ValueError(
                    f"{d} of the experiment '{self.sid}' reads the PK observable "
                    f"'{index}', which has no value of its own: read a parameter "
                    f"of it as '{index}.<parameter>', e.g. '{index}.cmax'."
                )
            return "observable"
        head, _, parameter = index.partition(".")
        if parameter and isinstance(self._observables.get(head), PK):
            return "observable"
        if index in _coordinates(self._simulations[task.simulation_id]):
            return "coordinate"
        model = self._models.get(task.model_id)
        if (
            isinstance(model, RoadrunnerSBMLModel)
            and model.r is not None
            and not model.has_selection(index)
        ):
            raise ValueError(
                f"{d} of the experiment '{self.sid}' reads '{index}', which is "
                f"neither an observable of the experiment "
                f"{sorted(self._observables)}, a coordinate of the scan of the "
                f"task '{d.task_id}' nor a selection of the model "
                f"'{task.model_id}'."
            )
        return "selection"

    def _check_task_data(self) -> None:
        """Check every task data before anything is simulated.

        The index, see `_index_kind`, and the dimensions and labels of `sel`
        against the scan of the task, see `check_sel`; the time of a task is
        selected by its values, which are known when the task ran.

        Raises:
            ValueError: see `_index_kind` and `check_sel`.
        """
        for d in self._task_data():
            self._index_kind(d)
            if d.sel:
                simulation = self._simulations[
                    self._tasks[str(d.task_id)].simulation_id
                ]
                check_sel(d, d.sel, _labels(simulation))

    def _task_outputs(self, task_key: str) -> tuple[list[str], list[str]]:
        """Get the observable outputs and the selections the data of a task read.

        Returns:
            The observable ids and `<id>.<parameter>` of PK observables, and the
            selections, each in the order of the data.
        """
        observed: dict[str, None] = {}
        selections: dict[str, None] = {}
        for d in self._task_data():
            if d.task_id != task_key:
                continue
            kind = self._index_kind(d)
            if kind == "observable":
                observed[d.selection] = None
            elif kind == "selection":
                selections[d.selection] = None
        return list(observed), list(selections)

    def _needed_observables(self, outputs: Iterable[str]) -> list[Observable]:
        """Get the observables the outputs need, in the order of `observables()`.

        An output is an observable id or `<id>.<parameter>` of a PK observable;
        an observable needs the observables its formula or function reads.
        """
        needed: set[str] = set()
        stack = [o if o in self._observables else o.partition(".")[0] for o in outputs]
        while stack:
            name = stack.pop()
            if name in needed or name not in self._observables:
                continue
            needed.add(name)
            for symbol in self._observables[name].reads:
                stack.append(
                    symbol if symbol in self._observables else symbol.partition(".")[0]
                )
        return [o for name, o in self._observables.items() if name in needed]

    def evaluate_fit_mappings(self):
        """Evaluate fit mappings."""
        for _, mapping in self._fit_mappings.items():
            for fit_data in [mapping.reference, mapping.observable]:
                # Get actual data from the results
                fit_data.get_data()

    # --- SERIALIZATION -------------------------------------------------------
    @timeit
    def to_json(self, path: Path | None = None, indent: int = 2):
        """Convert experiment to JSON for exchange.

        :param path: path for file, if None JSON str is returned
        :return:
        """
        d = self.to_dict()
        if path is None:
            return json.dumps(d, cls=ObjectJSONEncoder, indent=indent)
        with open(path, "w", encoding="utf-8") as f_json:
            json.dump(d, fp=f_json, cls=ObjectJSONEncoder, indent=indent)
        return None

    def to_dict(self):
        """Convert to dictionary.

        This is the basis for the JSON serialization.
        """
        # FIXME: resolve paths relative to base_paths
        return {
            "experiment_id": self.sid,
            "base_path": str(self.base_path) if self.base_path else None,
            "data_path": [str(p) for p in self.data_path] if self.data_path else None,
            "models": {k: v.to_dict() for k, v in self._models.items()},
            "tasks": {
                k: {
                    **v.to_dict(),
                    "observables": [
                        o.id for o in self._needed_observables(self._task_outputs(k)[0])
                    ],
                }
                for k, v in self._tasks.items()
            },
            "observables": self._observables,
            "simulations": {k: v.to_dict() for k, v in self._simulations.items()},
            "data": self._data,
            "figures": self._figures,
        }

    @timeit
    def save_datasets(self, results_path: Path) -> None:
        """Save datasets."""
        if self._datasets is None:
            logger.warning("No datasets in SimulationExperiment: '%s'", self.sid)
        else:
            for dkey, dset in self._datasets.items():
                dset.to_csv(
                    results_path / f"{self.sid}_{dkey}.tsv", sep="\t", index=False
                )

    @timeit
    def save_results(self, results_path: Path) -> None:
        """Save the result of every task as netCDF, see `ScanResult.to_netcdf`.

        Args:
            results_path: directory of the files, `<sid>_<task>.nc`.
        """
        if self.results is None:
            logger.warning("No results in SimulationExperiment: '%s'", self.sid)
        else:
            for rkey, result in self.results.items():
                result.to_netcdf(results_path / f"{self.sid}_{rkey}.nc")

    @timeit
    def create_mpl_figures(
        self,
        on_error: Literal["raise", "log"] = "raise",
        failed: dict[str, str] | None = None,
    ) -> dict[str, FigureMPL | Figure]:
        """Create matplotlib figures.

        Args:
            on_error: `"raise"` raises the error of a figure, `"log"` logs it,
                skips the figure and goes on with the next.
            failed: collects the failed figures as `{key: error}`.

        Returns:
            The figures by their key, without the failed ones.
        """
        mpl_figures = {}
        for fig_key, fig in self._figures.items():
            try:
                mpl_figures[fig_key] = MatplotlibFigureSerializer.to_figure(self, fig)
            except Exception as err:
                self._figure_failed(fig_key, err, on_error, failed)

        # additional custom figures
        try:
            for fig_key, fig_mpl in self.figures_mpl().items():
                mpl_figures[fig_key] = fig_mpl
        except Exception as err:
            self._figure_failed("figures_mpl", err, on_error, failed)

        return mpl_figures

    def _figure_failed(
        self,
        fig_key: str,
        err: Exception,
        on_error: Literal["raise", "log"],
        failed: dict[str, str] | None,
    ) -> None:
        """Handle the error of a figure: raise it or log it and record the key."""
        if on_error == "raise":
            raise err
        logger.error(
            "The figure '%s' of '%s' failed and is skipped: %s",
            fig_key,
            self.sid,
            err,
            exc_info=err,
        )
        if failed is not None:
            failed[fig_key] = f"{type(err).__name__}: {err}"

    @timeit
    def show_mpl_figures(self, mpl_figures: dict[str, FigureMPL]) -> None:
        """Show matplotlib figures.

        The figures of the serializer are created without `pyplot`, so that
        rendering does not fill its global registry; a figure it does not
        manage cannot show itself. Showing is the one place which needs the
        state machine, so a figure gets a manager here and only here.

        Args:
            mpl_figures: the figures to show.
        """
        for _, fig_mpl in mpl_figures.items():
            if fig_mpl.canvas.manager is None:
                manager = plt.figure().canvas.manager
                if manager is not None:
                    manager.canvas.figure = fig_mpl
                    fig_mpl.set_canvas(manager.canvas)
            fig_mpl.show()

    @timeit
    def save_mpl_figures(
        self,
        results_path: Path,
        mpl_figures: dict[str, FigureMPL],
        figure_formats: list[str] | None = None,
        on_error: Literal["raise", "log"] = "raise",
        failed: dict[str, str] | None = None,
    ) -> dict[str, list[Path]]:
        """Save matplotlib figures.

        Args:
            results_path: directory of the files.
            mpl_figures: the figures to save.
            figure_formats: formats of the files, `svg` by default.
            on_error: `"raise"` raises the error of a figure, `"log"` logs it
                and goes on with the next figure.
            failed: collects the failed figures as `{key: error}`.
        """
        if figure_formats is None:
            # default to SVG output
            figure_formats = ["svg"]
        paths = defaultdict(list)
        for fkey, fig_mpl in mpl_figures.items():  # type
            for fig_format in figure_formats:
                fig_path = results_path / f"{self.sid}_{fkey}.{fig_format}"
                try:
                    fig_mpl.savefig(fig_path, bbox_inches="tight")
                except Exception as err:
                    self._figure_failed(fkey, err, on_error, failed)
                    break

                paths[fig_format].append(fig_path)

        return paths

    def save_interactive_figures(
        self,
        results_path: Path,
        on_error: Literal["raise", "log"] = "raise",
        failed: dict[str, str] | None = None,
    ) -> dict[str, Path]:
        """Write the figures as interactive pages.

        The pages are drawn by plotly, see
        `sbmlsim.plot.serialization_plotly`: matplotlib draws the static
        images a publication needs and plotly the pages a reader zooms and
        hovers over. Both read the same `Figure`, so the two cannot disagree
        about what they show.

        plotly is not a dependency of `sbmlsim`; a run which asks for the
        interactive figures without it says so and writes none.

        Args:
            results_path: directory of the pages.
            on_error: `"raise"` raises the error of a figure, `"log"` logs it
                and goes on with the next figure.
            failed: collects the failed figures as `{key: error}`.

        Returns:
            The path of the page of every figure, empty without plotly.
        """
        try:
            from sbmlsim.plot.serialization_plotly import figures_to_html
        except ImportError:
            logger.error(
                "The interactive figures of '%s' need plotly, which is not "
                "installed: `pip install plotly`. No interactive figure was "
                "written.",
                self.sid,
            )
            return {}

        paths: dict[str, Path] = {}
        for key, figure in self._figures.items():
            try:
                paths.update(figures_to_html(self, results_path, {key: figure}))
            except Exception as err:
                self._figure_failed(key, err, on_error, failed)
        return paths

    @classmethod
    def close_mpl_figures(cls, mpl_figures: dict[str, FigureMPL]) -> None:
        """Close matplotlib figures.

        A figure of the serializer is not in the registry of `pyplot` and is
        released when nothing refers to it any more, for which this is a no-op.
        A figure of `figures_mpl()` may well come from `pyplot`, and that one
        is closed here.

        Args:
            mpl_figures: the figures to close.
        """
        for _, fig_mpl in mpl_figures.items():
            plt.close(fig_mpl)


@dataclass
class ExperimentResult:
    """Result of a simulation experiment."""

    experiment: SimulationExperiment
    output_path: Path | None
    #: the error of an experiment which failed in a run of the `ExperimentRunner`
    error: str | None = None
    #: the figures which failed and were skipped, `{key: error}`
    failed_figures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Conversion to dictionary.

        Used in serialization and required for reports.
        """
        return {
            "output_path": self.output_path,
            "error": self.error,
            "failed_figures": self.failed_figures,
        }


def _ragged(simulation: Scan) -> bool:
    """Check whether the result of a scan keeps the time points of every simulation.

    A scan with dimensions whose simulations output the steps of the
    integrator (neither `steps` nor `times`) is ragged.
    """
    return bool(simulation.dimensions) and any(
        s.steps is None and s.times is None for s in simulation.simulations()
    )


def _coordinates(simulation: Simulation | Scan) -> set[str]:
    """Get the names the result of a scan has as coordinates of its dimensions.

    The dimension ids, `<dimension>.<target>` of the targets and
    `<dimension>.<coordinate>` of the coordinates a dimension has, and the
    plain names of its coordinates; a plain target is the timecourse of the
    symbol, so it is no coordinate, and a simulation has none.
    """
    if not isinstance(simulation, Scan):
        return set()
    names: set[str] = set()
    for dimension in simulation.dimensions:
        names.add(dimension.id)
        names.update(f"{dimension.id}.{target}" for target in dimension.values)
        names.update(f"{dimension.id}.{name}" for name in dimension.coordinates)
        names.update(dimension.coordinates)
    return names


def _labels(simulation: Simulation | Scan) -> dict[str, list[Any] | None]:
    """Get the labels of the dimensions the result of a simulation has.

    The labels of every dimension of a scan, and the time (`time`, or
    `_point` of a ragged result), whose labels are not checked, see
    `check_sel`.
    """
    labels: dict[str, list[Any] | None] = {}
    if isinstance(simulation, Scan):
        for dimension in simulation.dimensions:
            labels[dimension.id] = np.asarray(dimension.labels).tolist()
    labels[TIME] = None
    labels[POINT] = None
    return labels
