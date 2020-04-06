import logging

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict

from sbmlsim.task import Task
from sbmlsim.simulator import SimulatorSerial
from sbmlsim.simulator.simulation_ray import SimulatorParallel
from sbmlsim.simulation import AbstractSim, TimecourseSim, ScanSim
from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.result import XResult
from sbmlsim.data import DataSet
from sbmlsim.model import AbstractModel
from sbmlsim.utils import timeit
from sbmlsim.units import UnitRegistry, Units

from sbmlsim.plot import Figure
from sbmlsim.plot.plotting_matplotlib import plt, to_figure
from matplotlib.pyplot import Figure as FigureMPL

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a simulation experiment"""
    experiment: 'SimulationExperiment'
    output_path: Path


class SimulationExperiment(object):
    """Generic simulation experiment.

    Consists of models, datasets, simulations, tasks, results, processing, figures
    """

    def __init__(self, sid: str = None, base_path: Path = None,
                 data_path: Path = None,
                 ureg: UnitRegistry = None, **kwargs):
        """

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
                "No base_path provided, reading/writing of resources may fail.")
        self.base_path = base_path

        if data_path:
            data_path = Path(data_path).resolve()
            if not data_path.exists():
                raise IOError(f"data_path '{data_path}' does not exist")
        else:
            logger.warning(
                "No data_path provided, reading of datasets may fail.")
        self.data_path = data_path
        # if self.base_path:
        # resolve data_path relative to base_path
        # self.data_path = self.data_path.relative_to(self.base_path)

        # single UnitRegistry per SimulationExperiment (can be shared)
        if not ureg:
            ureg = Units.default_ureg()
        self.ureg = ureg
        self.Q_ = ureg.Quantity

        # load everything once, never call this function again
        self._models = self.models()
        self._datasets = self.datasets()

        # FIXME: multiple analysis possible, handle consistently
        self._simulations = self.simulations()
        self._tasks = self.tasks()

        # task results
        self._results = None

        # processing
        self._data = None
        self._functions = None

        # figures
        self._figures = None

        # settings
        self.settings = kwargs

    # --- MODELS --------------------------------------------------------------
    @timeit
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

    # --- FUNCTIONS -----------------------------------------------------------
    def datagenerators(self) -> None:
        """DataGenerator definitions."""
        return

    # --- RESULTS -------------------------------------------------------------
    @property
    def results(self) -> Dict[str, XResult]:
        if self._results is None:
            self._run_tasks()
        return self._results

    # --- PROCESSING ----------------------------------------------------------
    # TODO:

    # --- FIGURES -------------------------------------------------------------
    def figures(self) -> Dict[str, FigureMPL]:
        """ Figures."""
        return {}

    # --- VALIDATION ----------------------------------------------------------
    def _check_keys(self):
        """Check that everything is okay with the experiment."""
        # string keys for main objects must be unique on SimulationExperiment
        all_keys = dict()
        for field_key in ["_models", "_datasets", "_tasks", "_simulations"]:
            field = getattr(self, field_key)
            if not isinstance(field, dict):
                raise ValueError(
                    f"'{field_key} must be a dict, but '{field}' is type '{type(field)}'.")
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

    # --- EXECUTE -------------------------------------------------------------
    @timeit
    def run(self, output_path: Path, show_figures: bool = True,
            save_results: bool = False) -> ExperimentResult:
        """
        Executes given experiment and stores results.
        Returns info dictionary.
        """
        if not Path.exists(output_path):
            Path.mkdir(output_path, parents=True)
            logging.info(f"'output_path' created: '{output_path}'")

        # validation
        self._check_keys()
        self._check_types()

        # normalize the tasks
        for task_id, task in self._tasks.items():
            model = self._models[task.model_id]
            sim = self._simulations[task.simulation_id]

            # normalize simulations with respective model dictionary
            sim.normalize(udict=model.udict, ureg=model.ureg)

        # run simulations
        self._run_tasks()  # sets self._results

        # definition of data accessed later on
        self.datagenerators()

        # some of the figures require actual numerical results!
        self._figures = self.figures()

        # save outputs
        self.save_datasets(output_path)

        # Saving takes often much longer then simulation
        if save_results:
            self.save_results(output_path)

        # save figure
        self.save_figures(output_path)

        # serialization
        self.to_json(output_path / f"{self.sid}.json")

        # display figures
        if show_figures:
            plt.show()

        return ExperimentResult(experiment=self, output_path=output_path)

    @timeit
    def _run_tasks(self, Simulator=SimulatorSerial,
                   absolute_tolerance=1E-14,
                   relative_tolerance=1E-14):
        """Run simulations & scans.

        This should not be called directly, but the results of the simulations
        should be requested by the results property.
        This allows to hash executed simulations.
        """
        # FIXME: this can be parallized (or on level of the SimulationExperiment)
        # tasks for individual models can be run separately !
        for task_key, task in self._tasks.items():
            model = self._models[task.model_id]
            # FIXME: creates new simulator for every task! we can reuse all simulators for a given model
            simulator = Simulator(model=model,
                                  absolute_tolerance=absolute_tolerance,
                                  relative_tolerance=relative_tolerance)
            # run tasks
            if self._results is None:
                self._results = {}

            sim = self._simulations[task.simulation_id]

            if isinstance(sim, TimecourseSim):
                logger.info(f"Run timecourse task: '{task_key}'")
                self._results[task_key] = simulator.run_timecourse(sim)
            elif isinstance(sim, ScanSim):
                logger.info(f"Run scan task: '{task_key}'")
                self._results[task_key] = simulator.run_scan(sim)
            else:
                raise ValueError(f"Unsupported simulation type: "
                                 f"{type(sim)}")

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
                result.to_netcdf(results_path / f"{self.sid}_{rkey}.h5")

    @timeit
    def save_figures(self, results_path):
        """ Save figures.
        :param results_path:
        :return:
        """
        paths = []
        for fkey, fig in self._figures.items():
            path_svg = results_path / f"{self.sid}_{fkey}.svg"

            if isinstance(fig, Figure):
                fig_mpl = to_figure(fig)
            else:
                fig_mpl = fig

            fig_mpl.savefig(path_svg, dpi=150, bbox_inches="tight")

            paths.append(path_svg)
        return paths


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
