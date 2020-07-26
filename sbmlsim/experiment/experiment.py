import logging
from pathlib import Path
import json
from typing import Dict, List
from collections import defaultdict
import multiprocessing

from dataclasses import dataclass

from sbmlsim.task import Task
from sbmlsim.simulation import AbstractSim, TimecourseSim, ScanSim
from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.result import XResult
from sbmlsim.data import DataSet, Data
from sbmlsim.model import AbstractModel
from sbmlsim.utils import timeit
from sbmlsim.units import UnitRegistry, Units
from sbmlsim.fit import FitMapping, FitData

from sbmlsim.plot import Figure
from sbmlsim.plot.plotting_matplotlib import MatplotlibFigureSerializer, FigureMPL, plt

logger = logging.getLogger(__name__)


class SimulationExperiment(object):
    """Generic simulation experiment.

    Consists of models, datasets, simulations, tasks, results, processing, figures
    """

    def __init__(self,
                 sid: str = None, base_path: Path = None,
                 data_path: Path = None,
                 ureg: UnitRegistry = None, **kwargs):
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
                "No 'base_path' provided, reading/writing of resources may fail.")
        self.base_path = base_path

        if data_path:
            data_path = Path(data_path).resolve()
            if not data_path.exists():
                raise IOError(f"data_path '{data_path}' does not exist")
        else:
            logger.warning(
                "No 'data_path' provided, reading of datasets may fail.")
        self.data_path = data_path

        # single UnitRegistry per SimulationExperiment (can be shared)
        if not ureg:
            ureg = Units.default_ureg()
        self.ureg = ureg
        self.Q_ = ureg.Quantity

        # settings
        self.settings = kwargs
        self._models = None  # instances are loaded by runner !

    def initialize(self):
        """
        :return:
        """
        # load everything once, never call this function again
        self._data = {}
        self._datasets = self.datasets()
        self._simulations = self.simulations()
        self._tasks = self.tasks()
        self._fit_mappings = self.fit_mappings()  # type: Dict[str, FitMapping]
        self.datagenerators()  # definition of data accessed later on (sets self._data)

        # task results
        self._results = None
        # processing
        self._functions = None

        # figures
        self._figures = None
        self._figures_mpl = None

        # validation
        self._check_keys()
        self._check_types()

    # --- MODELS --------------------------------------------------------------
    def models(self) -> Dict[str, AbstractModel]:
        logger.debug(f"No models defined for '{self.sid}'.")
        return {}

    # --- DATASETS ------------------------------------------------------------
    def datasets(self) -> Dict[str, DataSet]:
        """Dataset definition (experimental data)"""
        return {}

    # --- TASKS ---------------------------------------------------------------
    def tasks(self) -> Dict[str, Task]:
        """Task definitions."""
        return {}

    # --- SIMULATIONS ---------------------------------------------------------
    def simulations(self) -> Dict[str, AbstractSim]:
        """Simulation definitions."""
        return {}

    # --- FITTING -------------------------------------------------------------
    def fit_mappings(self) -> Dict[str, FitMapping]:
        """ Fit mappings, mapping reference data on observables.

        Used for the optimization of parameters.
        """
        return {}

    # --- FUNCTIONS -----------------------------------------------------------
    def datagenerators(self) -> None:
        """DataGenerator definitions including functions."""
        return

    # --- RESULTS -------------------------------------------------------------
    @property
    def results(self) -> Dict[str, XResult]:
        if self._results is None:
            self._run_tasks(self.simulator)
        return self._results

    # --- FIGURES -------------------------------------------------------------
    def figures(self) -> Dict[str, Figure]:
        """ sbmlsim figures.

        These figures register their data automatically, whereas mpl figures
        have to manually register data via the datagenerators to ensure data
        is available.

        These figures do not have access to concrete data, but only abstract
        data concepts.
        """
        return {}

    def figures_mpl(self) -> Dict[str, FigureMPL]:
        """ matplotlib figures.

        Figures which require access to actual data, or manual manipulation
        of matplotlib figure properties.
        These are concrete instances of figures with data, but bound
        to a plotting framework, here matplotlib.
        """
        return {}

    # --- VALIDATION ----------------------------------------------------------
    def _check_keys(self):
        """Check that everything is okay with the experiment."""
        # string keys for main objects must be unique on SimulationExperiment
        all_keys = dict()
        allowed_types = dict
        for field_key in ["_models", "_datasets", "_tasks", "_simulations", "_fit_mappings"]:
            field = getattr(self, field_key)

            if not isinstance(field, allowed_types):
                raise ValueError(
                    f"SimulationExperiment '{self.sid}': '{field_key} must be a '{allowed_types}', but '{field}' is type '{type(field)}'. "
                    f"Check that the respective definition returns an object of type '{allowed_types}. "
                    f"Often simply the return statement is missing (returning NoneType).")
            for key in getattr(self, field_key).keys():
                if not isinstance(key, str):
                    raise ValueError(f"'{field_key} keys must be str: "
                                     f"'{key} -> {type(key)}'")
                if key in all_keys:
                    raise ValueError(
                        f"Duplicate key '{key}' for '{field_key}' and '{all_keys[key]}'")
                else:
                    all_keys[key] = field_key

    def _check_types(self):
        """Check that correct types"""
        for key, dset in self._datasets.items():
            if not isinstance(dset, DataSet):
                raise ValueError(f"datasets must be of type DataSet, but "
                                 f"dataset '{key}' has type: '{type(dset)}'")

        for key, model in self._models.items():
            if not isinstance(model, AbstractModel):
                raise ValueError(f"datasets must be of type AbstractModel, but "
                                 f"model '{key}' has type: '{type(model)}'")

        for key, task in self._tasks.items():
            if not isinstance(task, Task):
                raise ValueError(f"tasks must be of type Task, but "
                                 f"task '{key}' has type: '{type(task)}'")

        for key, sim in self._simulations.items():
            if not isinstance(sim, AbstractSim):
                raise ValueError(
                    f"simulations must be of type AbstractSim, but "
                    f"simulation '{key}' has type: '{type(sim)}'")

        for key, mapping in self._fit_mappings.items():
            if not isinstance(mapping, FitMapping):
                raise ValueError(
                    f"fit_mappings must be of type FitMappintg, but "
                    f"mapping '{key}' has type: '{type(mapping)}'")

    # --- EXECUTE -------------------------------------------------------------

    @timeit
    def run(self, simulator, output_path: Path = None, show_figures: bool = True,
            save_results: bool = False, reduced_selections: bool = True) -> 'ExperimentResult':
        """
        Executes given experiment and stores results.
        Returns info dictionary.
        """
        # some of the figures require actual numerical results!
        # this results in execution of the tasks! On the other hand we
        # want to register the data before running the tasks.
        # FIXME (executed 2 times)
        self._figures = self.figures()

        # run simulations
        self._run_tasks(simulator, reduced_selections=reduced_selections)  # sets self._results
        # FIXME
        # self._figures = self.figures()

        # evaluate mappings
        self.evaluate_mappings()

        # some of the figures require actual numerical results!
        # FIXME (executed 2 times)
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

            # serialization
            self.to_json(output_path / f"{self.sid}.json")

        # create figures
        mpl_figures = self.create_mpl_figures()
        if show_figures:
            self.show_figures(mpl_figures=mpl_figures)
        if output_path:
            self.save_figures(output_path, mpl_figures=mpl_figures)
            self.clear_figures(mpl_figures=mpl_figures)

        return ExperimentResult(experiment=self, output_path=output_path)

    @timeit
    def _run_tasks(self, simulator, reduced_selections:bool = True):
        """Run simulations & scans.

        This should not be called directly, but the results of the simulations
        should be requested by the results property.
        This allows to hash executed simulations.
        """
        if self._results is None:
            self._results = {}

        # get all tasks for given model
        model_tasks = defaultdict(list)
        for task_key, task in self._tasks.items():
            model_tasks[task.model_id].append(task_key)

        # execute all tasks for given model
        for model_id, task_keys in model_tasks.items():
            # load model in simulator
            model = self._models[model_id]
            simulator.set_model(model=model)

            if reduced_selections:
                # set selections based on data
                selections = set()
                for d in self._data.values():  # type: Data
                    if d.is_task():
                        selections.add(d.index)
                selections = sorted(list(selections))
                print(f"Setting selections: {selections}")
                simulator.set_timecourse_selections(selections=selections)
            else:
                # use the complete selection
                simulator.set_timecourse_selections(selections=None)

            for task_key in task_keys:  # type: str
                task = self._tasks[task_key]
                sim = self._simulations[task.simulation_id]
                if isinstance(sim, TimecourseSim):
                    self._results[task_key] = simulator.run_timecourse(sim)
                elif isinstance(sim, ScanSim):
                    self._results[task_key] = simulator.run_scan(sim)
                else:
                    raise ValueError(f"Unsupported simulation type: "
                                     f"{type(sim)}")

    def evaluate_mappings(self):
        """Evaluates the fit mappings."""
        for key, mapping in self._fit_mappings.items():
            for fit_data in [mapping.reference, mapping.observable]:
                # Get actual data from the results
                fit_data.get_data()

    # --- SERIALIZATION -------------------------------------------------------
    @timeit
    def to_json(self, path=None, indent=2):
        """ Convert experiment to JSON for exchange.

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
        # FIXME: ordered dict

        return {
            "experiment_id": self.sid,
            "base_path": str(self.base_path) if self.base_path else None,
            "data_path": str(self.data_path) if self.data_path else None,
            # "unit_registry": self.ureg,
            "models": {k: v.to_dict() for k, v in self._models.items()},
            "tasks": {k: v.to_dict() for k, v in self._tasks.items()},
            "simulations": {k: v.to_dict() for k, v in
                            self._simulations.items()},
            "data": self._data,
            "figures": self._figures,
        }

    @classmethod
    def from_json(cls, json_info) -> 'SimulationExperiment':
        """Load experiment from json path or str"""
        # FIXME: update serialization
        if isinstance(json_info, Path):
            with open(json_info, "r") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)

        return JSONExperiment.from_dict(d)

    @timeit
    def save_datasets(self, results_path):
        """ Save datasets

        :param results_path:
        :return:
        """
        if self._datasets is None:
            logger.warning(f"No datasets in SimulationExperiment: '{self.sid}'")
        else:
            for dkey, dset in self._datasets.items():
                dset.to_csv(results_path / f"{self.sid}_{dkey}.tsv",
                            sep="\t", index=False)

    @timeit
    def save_results(self, results_path):
        """ Save results (mean timecourse)

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
    def create_mpl_figures(self) -> Dict[str, FigureMPL]:
        """ Create matplotlib figures.

        :return:
        """
        mpl_figures = {}
        for fkey, fig in self._figures.items():
            if isinstance(fig, Figure):
                fig_mpl = MatplotlibFigureSerializer.to_figure(fig)
            else:
                fig_mpl = fig

            mpl_figures[fkey] = fig_mpl

        return mpl_figures

    @timeit
    def show_figures(self, mpl_figures: Dict[str, FigureMPL]):

        for fig_key, fig_mpl in mpl_figures.items():
            fig_mpl.show()

        # multiprocessing with matplotlib creates issues
        # pool = multiprocessing.Pool()
        # pool.map(self._show_figure, mpl_figures.values())
        # pool.map_async(self._show_figure, mpl_figures.values())

    @staticmethod
    def _show_figure(args):
        fig_mpl = args  # type: FigureMPL
        fig_mpl.show()

    @timeit
    def save_figures(self, results_path: Path, mpl_figures: Dict[str, FigureMPL]) -> List[Path]:
        """ Save matplotlib figures.

        :param results_path:
        :return:
        """
        paths = []
        input = []
        for fkey, fig_mpl in mpl_figures.items():  # type
            path_svg = results_path / f"{self.sid}_{fkey}.svg"
            fig_mpl.savefig(path_svg, bbox_inches="tight")
            # fig_mpl.savefig(path_png, bbox_inches="tight")

            input.append([path_svg, fig_mpl])
            paths.append(path_svg)

        # multiprocessing of figures (problems in travis)
        # pool = multiprocessing.Pool()
        # pool.map(self._save_figure, input)
        # pool.map_async(self._save_figure, input)

        return paths

    def clear_figures(self, mpl_figures: Dict[str, FigureMPL]):
        for fig_key, fig_mpl in mpl_figures.items():
            plt.close(fig_mpl)

    @staticmethod
    def _save_figure(args):
        path_svg, fig_mpl = args
        fig_mpl.savefig(path_svg, bbox_inches="tight")


# FIXME: deprecated, remove
class JSONExperiment(SimulationExperiment):
    """An experiment loaded from JSON serialization."""

    @property
    def simulations(self):
        return self._simulations

    @classmethod
    def from_dict(self, d) -> 'JSONExperiment':
        experiment = JSONExperiment(model_path=None,
                                    data_path=None)
        experiment.sid = d['experiment_id']
        # parse simulation definitions
        simulations = {}
        for key, data in d['simulations'].items():
            tcsim = TimecourseSim(**data)
            for tc in tcsim.timecourses:
                # parse the serialized magnitudes
                tc.changes = {k: v["_magnitude"] for k, v in tc.changes.items()}
            simulations[key] = tcsim
        experiment._simulations = simulations

        return experiment

@dataclass
class ExperimentResult:
    """Result of a simulation experiment"""
    experiment: SimulationExperiment
    output_path: Path