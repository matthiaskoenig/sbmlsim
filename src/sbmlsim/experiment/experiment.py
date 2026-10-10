"""SimulationExperiments and helpers."""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt

from sbmlsim.data import Data, DataSet
from sbmlsim.fit import FitMapping
from sbmlsim.fit.objects import FitDataInitialized
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.plot import Figure
from sbmlsim.plot.serialization_matplotlib import (
    FigureMPL,
    MatplotlibFigureSerializer,
)
from sbmlsim.result import ScanResult
from sbmlsim.result.scan import TIME
from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.simulation import Scan, Simulation
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
        if not sid:
            self.sid = self.__class__.__name__
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

            # \w matches any alphanumeric character; this is equivalent to [a-zA-Z0-9_]
            pattern_sid = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
            for key in getattr(self, field_key):
                if not isinstance(key, str):
                    raise ValueError(
                        f"'{field_key} keys must be str: '{key} -> {type(key)}'"
                    )
                # Check that valid Sid
                try:
                    if not re.match(pattern_sid, key):
                        raise ValueError(
                            f"{field_key} key is not a valid SId "
                            f"([a-zA-Z0-9][a-zA-Z0-9_]*): '{key}'"
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
    ) -> "ExperimentResult":
        """Execute given experiment and store results."""
        # run simulations (sets self._results)
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
        self._mpl_figures = {}
        if show_figures or (output_path and static_formats):
            self._mpl_figures = self.create_mpl_figures()
            if show_figures:
                self.show_mpl_figures(mpl_figures=self._mpl_figures)
            if output_path and static_formats:
                self.save_mpl_figures(
                    output_path,
                    mpl_figures=self._mpl_figures,
                    figure_formats=static_formats,
                )
            self.close_mpl_figures(mpl_figures=self._mpl_figures)

        if output_path and interactive:
            self.save_interactive_figures(output_path)

        # only perform serialization after data evaluation (to access units)
        if output_path:
            # serialization
            self.to_json(output_path / f"{self.sid}.json")

        return ExperimentResult(experiment=self, output_path=output_path)

    @timeit
    def _run_tasks(
        self, simulator: Simulator | None, reduced_selections: bool = True
    ) -> None:
        """Run the tasks of the experiment, the tasks of a model one after another.

        The selections of a task are the variables its data refers to, every
        variable of the model without `reduced_selections`, set on the model
        right before the task runs; the coordinates of its own scan are never
        among them. The changes of a
        model are defaults of the pre-initialization changes of every
        simulation of it, see `Simulator.compile`. A task whose data read
        observables runs with the observables they need and keeps them and the
        selections its data read, every selection of the model without
        `reduced_selections`; a task without runs with the selections of the
        model.

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
                coordinates = _coordinates(scan)
                observed, selections = self._task_outputs(task_key)
                if not reduced_selections:
                    selections = [s for s in every if s not in coordinates]
                if not observed:
                    # a changed target is a coordinate of the result of its own
                    # scan and never a timecourse, whatever other tasks read
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

    def _index_kind(self, d: Data) -> str:
        """Classify the index of task data.

        Returns:
            `"time"`, `"observable"` (an observable or a parameter of a `PK`
            observable), `"coordinate"` (a dimension of the scan of the task, a
            target it changes or a coordinate of a dimension) or `"selection"`.

        Raises:
            ValueError: if the data reads a task which does not exist, or an
                index which is none of these.
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
        """Check the index of every task data before anything is simulated.

        Raises:
            ValueError: see `_index_kind`.
        """
        for d in self._task_data():
            self._index_kind(d)

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

    def _selections_of_model(self, model_id: str) -> set[str]:
        """Get the selections a model has to be simulated with.

        Args:
            model_id: the model the tasks are run on.

        Returns:
            `time` and the selections the data of the tasks of the model read;
            observables and coordinates are no selections.
        """
        selections = {TIME}
        for task_key, task in self._tasks.items():
            if task.model_id == model_id:
                selections.update(self._task_outputs(task_key)[1])
        return selections

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
    def create_mpl_figures(self) -> dict[str, FigureMPL | Figure]:
        """Create matplotlib figures."""
        mpl_figures = {}
        for fig_key, fig in self._figures.items():
            fig_mpl = MatplotlibFigureSerializer.to_figure(self, fig)
            mpl_figures[fig_key] = fig_mpl

        # additional custom figures
        for fig_key, fig_mpl in self.figures_mpl().items():
            mpl_figures[fig_key] = fig_mpl

        return mpl_figures

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
    ) -> dict[str, list[Path]]:
        """Save matplotlib figures."""
        if figure_formats is None:
            # default to SVG output
            figure_formats = ["svg"]
        paths = defaultdict(list)
        for fkey, fig_mpl in mpl_figures.items():  # type
            for fig_format in figure_formats:
                fig_path = results_path / f"{self.sid}_{fkey}.{fig_format}"
                fig_mpl.savefig(fig_path, bbox_inches="tight")

                paths[fig_format].append(fig_path)

        return paths

    def save_interactive_figures(self, results_path: Path) -> dict[str, Path]:
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

        return figures_to_html(self, results_path)

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

    def to_dict(self) -> dict:
        """Conversion to dictionary.

        Used in serialization and required for reports.
        """
        return {
            "output_path": self.output_path,
        }


def _coordinates(simulation: Simulation | Scan) -> set[str]:
    """Get the names the result of a scan has as coordinates of its dimensions.

    The dimension ids, the targets the dimensions change and the coordinates
    of the dimensions; a simulation has none.
    """
    if not isinstance(simulation, Scan):
        return set()
    names: set[str] = set()
    for dimension in simulation.dimensions:
        names.add(dimension.id)
        names.update(dimension.values)
        names.update(dimension.coordinates)
    return names
