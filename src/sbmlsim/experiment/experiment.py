"""SimulationExperiments and helpers."""

import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

from sbmlsim.data import Data, DataSet
from sbmlsim.fit import FitMapping
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Figure
from sbmlsim.plot.plotting_matplotlib import FigureMPL, MatplotlibFigureSerializer, plt
from sbmlsim.result import XResult
from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.simulation import AbstractSim, ScanSim, TimecourseSim
from sbmlsim.task import Task
from sbmlsim.units import UnitRegistry, Units
from sbmlsim.utils import timeit


logger = logging.getLogger(__name__)


class ExperimentDict(dict):
    """Dictionary for experiments."""

    def __getitem__(self, k):
        """Get item k."""
        try:
            return super().__getitem__(k)
        except KeyError:
            raise KeyError(f"Key '{k}' not in {sorted(self.keys())}")


class SimulationExperiment(object):
    """Generic simulation experiment.

    Consists of models, datasets, simulations, tasks, results, processing, figures
    """

    def __init__(
        self,
        sid: str = None,
        base_path: Path = None,
        data_path: Path = None,
        ureg: UnitRegistry = None,
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

        if base_path:
            base_path = Path(base_path).resolve()
            if not base_path.exists():
                raise IOError(f"base_path '{base_path}' does not exist")
        else:
            logger.warning(
                "No 'base_path' provided, reading/writing of resources may fail."
            )
        self.base_path = base_path

        if data_path:
            if isinstance(data_path, (list, tuple, set)):
                data_path = [Path(p).resolve() for p in data_path]
            else:
                data_path = [Path(data_path).resolve()]
            for p in data_path:
                if not p.exists():
                    raise IOError(f"data_path '{p}' does not exist")
        else:
            logger.warning("No 'data_path' provided, reading of datasets may fail.")
        self.data_path = data_path

        # single UnitRegistry per SimulationExperiment (can be shared)
        if not ureg:
            ureg = Units.default_ureg()
        self.ureg = ureg
        self.Q_ = ureg.Quantity

        # settings
        self.settings = kwargs

        # init variables
        self._models = ExperimentDict()
        self._data = ExperimentDict()
        self._datasets = ExperimentDict()
        self._fit_mappings = ExperimentDict()
        self._simulations = ExperimentDict()
        self._tasks = ExperimentDict()
        self._results = ExperimentDict()
        self._figures = ExperimentDict()

    def initialize(self) -> None:
        """Initialize SimulationExperiment.

        Initialization must be separated from object construction due to
        the parallel execution of the problem later on.
        Certain objects cannot be serialized and must be initialized.
        :return:
        """
        # process all information necessary to run the simulations, i.e.,
        # all data required from the model
        self._datasets = self.datasets()  # storage of datasets
        self._simulations = self.simulations()  # storage of simulation definition
        self._tasks = self.tasks()
        self._fit_mappings = self.fit_mappings()  # type: Dict[str, FitMapping]
        self.datagenerators()  # definition of data accessed later on (sets self._data)

        # validation of information
        self._check_keys()
        self._check_types()

        # print(self)

    def __str__(self) -> str:
        """Get string representation."""
        info = [
            "-" * 80,
            f"SimulationExperiment: {self.__class__.__name__}: {self.sid}",
            "-" * 80,
            f"{'data':20} {list(self._data.keys())}",
            f"{'datasets':20} {list(self._datasets.keys())}",
            f"{'fit_mappings':20} {list(self._fit_mappings.keys())}",
            f"{'simulations':20} {list(self._simulations.keys())}",
            f"{'tasks':20} {list(self._tasks.keys())}",
            f"{'results':20} {list(self._results.keys())}",
            f"{'figures':20} {list(self._figures.keys())}",
            "-" * 80,
        ]
        return "\n".join(info)

    def models(self) -> Dict[str, AbstractModel]:
        """Define model definitions.

        The child classes fill out the information.
        """
        return ExperimentDict()

    def datasets(self) -> Dict[str, DataSet]:
        """Define dataset definitions (experimental data).

        The child classes fill out the information.
        """
        return ExperimentDict()

    def tasks(self) -> Dict[str, Task]:
        """Define task definitions.

        The child classes fill out the information.
        """
        return ExperimentDict()

    def simulations(self) -> Dict[str, AbstractSim]:
        """Define simulation definitions.

        The child classes fill out the information.
        """
        return ExperimentDict()

    def fit_mappings(self) -> Dict[str, FitMapping]:
        """Define fit mappings.

        Mapping reference data on observables.
        Used for the optimization of parameters.
        The child classes fill out the information.
        """
        return ExperimentDict()

    def datagenerators(self) -> None:
        """Define DataGenerators including functions.

        All data which is accessed in a simulation result must be defined in a
        data generator. The data generators are important for defining the
        selections of a simulation experiment.
        """
        return None

    def add_selections(self, selections: Iterable[str], task_ids: Iterable[str] = None):
        """Add selections to given tasks.

        Selections are necessary to access data from simulations.
        Here these selections are added to the tasks. If no tasks are given,
        the selections are added to all tasks.
        """
        if task_ids is None:
            task_ids = self._tasks.keys()

        for task_id in task_ids:
            for selection in selections:
                Data(self, index=selection, task=task_id)

    # --- RESULTS ---------------------------------------------------------------------
    @property
    def results(self) -> Dict[str, XResult]:
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

    def figures(self) -> Dict[str, Figure]:
        """Figure definition.

        Selections accessed in figures and analyses must be registered beforehand
        via datagenerators.

        Most figures do not require access to concrete data, but only abstract
        data concepts.
        """
        return {}

    # --- VALIDATION ------------------------------------------------------------------
    def _check_keys(self):
        """Check keys in information dictionaries."""
        # string keys for main objects must be unique on SimulationExperiment
        all_keys = dict()
        allowed_types = dict
        for field_key in [
            "_models",
            "_datasets",
            "_tasks",
            "_simulations",
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
            for key in getattr(self, field_key).keys():
                if not isinstance(key, str):
                    raise ValueError(
                        f"'{field_key} keys must be str: " f"'{key} -> {type(key)}'"
                    )
                if key in all_keys:
                    raise ValueError(
                        f"Duplicate key '{key}' for '{field_key}' and '{all_keys[key]}'"
                    )
                else:
                    all_keys[key] = field_key

    def _check_types(self):
        """Check for correctness of types."""
        for key, dset in self._datasets.items():
            if not isinstance(dset, DataSet):
                raise ValueError(
                    f"datasets must be of type DataSet, but "
                    f"dataset '{key}' has type: '{type(dset)}'"
                )

        for key, model in self._models.items():
            if not isinstance(model, AbstractModel):
                raise ValueError(
                    f"datasets must be of type AbstractModel, but "
                    f"model '{key}' has type: '{type(model)}'"
                )

        for key, task in self._tasks.items():
            if not isinstance(task, Task):
                raise ValueError(
                    f"tasks must be of type Task, but "
                    f"task '{key}' has type: '{type(task)}'"
                )

        for key, sim in self._simulations.items():
            if not isinstance(sim, AbstractSim):
                raise ValueError(
                    f"simulations must be of type AbstractSim, but "
                    f"simulation '{key}' has type: '{type(sim)}'"
                )

        for key, mapping in self._fit_mappings.items():
            if not isinstance(mapping, FitMapping):
                raise ValueError(
                    f"fit_mappings must be of type FitMappintg, but "
                    f"mapping '{key}' has type: '{type(mapping)}'"
                )

    # --- EXECUTE ---------------------------------------------------------------------

    @timeit
    def run(
        self,
        simulator,
        output_path: Path = None,
        show_figures: bool = True,
        save_results: bool = False,
        figure_formats: List[str] = None,
        reduced_selections: bool = True,
    ) -> "ExperimentResult":
        """Execute given experiment and store results."""

        # run simulations
        self._run_tasks(
            simulator, reduced_selections=reduced_selections
        )  # sets self._results

        # evaluate mappings
        self.evaluate_fit_mappings()

        # some of the figures require actual numerical results!
        self._figures = self.figures()

        # create outputs
        if output_path is None:
            if save_results:
                logger.error("'output_path' required to save results.")

        else:
            if not Path.exists(output_path):
                Path.mkdir(output_path, parents=True)
                logging.info(f"'output_path' created: '{output_path}'")

            # save outputs
            self.save_datasets(output_path)

            # Saving takes often much longer then simulation
            if save_results:
                self.save_results(output_path)

        # create figures
        mpl_figures = self.create_mpl_figures()
        if show_figures:
            self.show_mpl_figures(mpl_figures=mpl_figures)
        if output_path:
            self.save_mpl_figures(
                output_path, mpl_figures=mpl_figures, figure_formats=figure_formats
            )
        self.close_mpl_figures(mpl_figures=mpl_figures)

        # only perform serialization after data evaluation (to access units)
        if output_path:
            # serialization
            self.to_json(output_path / f"{self.sid}.json")

        return ExperimentResult(experiment=self, output_path=output_path)

    @timeit
    def _run_tasks(self, simulator, reduced_selections: bool = True):
        """Run simulations and scans.

        This should not be called directly, but the results of the simulations
        should be requested by the results property.
        This allows to hash executed simulations.
        """
        if self._results is None:
            self._results = ExperimentDict()

        # get all tasks for given model
        model_tasks = defaultdict(list)
        for task_key, task in self._tasks.items():
            model_tasks[task.model_id].append(task_key)

        # execute all tasks for given model
        for model_id, task_keys in model_tasks.items():

            # load model in simulator
            model = self._models[model_id]  # type: AbstractModel
            simulator.set_model(model=model)

            if reduced_selections:
                # set selections based on data
                selections = {"time"}
                for d in self._data.values():  # type: Data
                    if d.is_task():
                        # check if selection is for current model
                        task = self._tasks[d.task_id]
                        if task.model_id == model_id:
                            selections.add(d.index)
                selections = sorted(list(selections))
                # print(f"Selections for model '{model_id}': {selections}")
                simulator.set_timecourse_selections(selections=selections)
            else:
                # use the complete selection
                simulator.set_timecourse_selections(selections=None)

            # normalize model changes (these must be set in simulation!)
            model.normalize(udict=simulator.udict, ureg=simulator.ureg)

            for task_key in task_keys:  # type: str
                task = self._tasks[task_key]

                sim = self._simulations[
                    task.simulation_id
                ]  # type: Union[ScanSim, TimecourseSim]

                # normalization before running to ensure correct serialization
                sim.normalize(udict=simulator.udict, ureg=simulator.ureg)

                # inject model changes (copy to create independent)
                sim = deepcopy(sim)
                sim.add_model_changes(model.changes)

                if isinstance(sim, TimecourseSim):
                    self._results[task_key] = simulator.run_timecourse(sim)
                elif isinstance(sim, ScanSim):
                    self._results[task_key] = simulator.run_scan(sim)
                else:
                    raise ValueError(f"Unsupported simulation type: " f"{type(sim)}")

    def evaluate_fit_mappings(self):
        """Evaluate fit mappings."""
        for _, mapping in self._fit_mappings.items():
            for fit_data in [mapping.reference, mapping.observable]:
                # Get actual data from the results
                fit_data.get_data()

    # --- SERIALIZATION -------------------------------------------------------
    @timeit
    def to_json(self, path=None, indent=2):
        """Convert experiment to JSON for exchange.

        :param path: path for file, if None JSON str is returned
        :return:
        """
        d = self.to_dict()
        # from pprint import pprint
        # pprint(d)
        if path is None:
            return json.dumps(d, cls=ObjectJSONEncoder, indent=indent)
        else:
            with open(path, "w") as f_json:
                json.dump(d, fp=f_json, cls=ObjectJSONEncoder, indent=indent)

    def to_dict(self):
        """Convert to dictionary.

        This is the basis for the JSON serialization.
        """
        # FIXME: resolve paths relative to base_paths
        return {
            "experiment_id": self.sid,
            "base_path": str(self.base_path) if self.base_path else None,
            "data_path": str(self.data_path) if self.data_path else None,
            "models": {k: v.to_dict() for k, v in self._models.items()},
            "tasks": {k: v.to_dict() for k, v in self._tasks.items()},
            "simulations": {k: v.to_dict() for k, v in self._simulations.items()},
            "data": self._data,
            "figures": self._figures,
        }

    @classmethod
    def from_json(cls, json_info: Union[Path, str]) -> "SimulationExperiment":
        """Load experiment from json path or str."""
        # FIXME: update serialization
        if isinstance(json_info, Path):
            with open(json_info, "r") as f_json:
                d = json.load(f_json)
        elif isinstance(json_info, str):
            d = json.loads(json_info)
        else:
            raise ValueError("Unsupported json format.")

        return SimulationExperiment.from_dict(d)

    @timeit
    def save_datasets(self, results_path: Path) -> None:
        """Save datasets."""
        if self._datasets is None:
            logger.warning(f"No datasets in SimulationExperiment: '{self.sid}'")
        else:
            for dkey, dset in self._datasets.items():
                dset.to_csv(
                    results_path / f"{self.sid}_{dkey}.tsv", sep="\t", index=False
                )

    @timeit
    def save_results(self, results_path: Path) -> None:
        """Save results (mean timecourse).

        :param results_path:
        :return:
        """
        if self.results is None:
            logger.warning(f"No results in SimulationExperiment: '{self.sid}'")
        else:
            for rkey, result in self.results.items():
                result.to_netcdf(results_path / f"{self.sid}_{rkey}.nc")
                result.to_tsv(results_path / f"{self.sid}_{rkey}.tsv")

    @timeit
    def create_mpl_figures(self) -> Dict[str, Union[FigureMPL, Figure]]:
        """Create matplotlib figures."""
        mpl_figures = {}
        for fig_key, fig in self._figures.items():
            if isinstance(fig, Figure):
                fig_mpl = MatplotlibFigureSerializer.to_figure(fig)
            elif isinstance(fig, FigureMPL):
                fig_mpl = fig

            mpl_figures[fig_key] = fig_mpl

        return mpl_figures

    @timeit
    def show_mpl_figures(self, mpl_figures: Dict[str, FigureMPL]) -> None:
        """Show matplotlib figures."""
        for _, fig_mpl in mpl_figures.items():
            fig_mpl.show()

    @timeit
    def save_mpl_figures(
        self,
        results_path: Path,
        mpl_figures: Dict[str, FigureMPL],
        figure_formats: List[str] = None,
    ) -> Dict[str, List[Path]]:
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

    @classmethod
    def close_mpl_figures(cls, mpl_figures: Dict[str, FigureMPL]):
        """Close matplotlib figures."""
        for _, fig_mpl in mpl_figures.items():
            plt.close(fig_mpl)


@dataclass
class ExperimentResult:
    """Result of a simulation experiment."""

    experiment: SimulationExperiment
    output_path: Path

    def to_dict(self) -> Dict:
        """Conversion to dictionary.

        Used in serialization and required for reports.
        """
        d = {
            "output_path": self.output_path,
        }
        return d
