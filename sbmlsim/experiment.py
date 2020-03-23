import logging

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict
from pint import UnitRegistry
import pandas as pd

from sbmlsim.utils import timeit, deprecated
from sbmlsim.tasks import Task
from sbmlsim.simulation_serial import SimulatorSerial
from sbmlsim.timecourse import AbstractSim, TimecourseSim, TimecourseScan
from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.result import Result
from sbmlsim.data import Data, DataSet, load_dataframe
from sbmlsim.models import RoadrunnerSBMLModel, AbstractModel
from sbmlsim.plotting_matplotlib import plt, to_figure
from matplotlib.pyplot import Figure as FigureMPL
from sbmlsim.plotting import Figure as FigureSEDML
from sbmlsim.units import Units
import os

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

    def __init__(self, sid: str=None, base_path: Path=None,
                 data_path: Path=None,
                 ureg: UnitRegistry=None, **kwargs):
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
            logger.warning("No base_path provided, reading/writing of resources may fail.")
        self.base_path = base_path

        if data_path:
            data_path = Path(data_path).resolve()
            if not data_path.exists():
                raise IOError(f"data_path '{data_path}' does not exist")
        else:
            logger.warning("No data_path provided, reading of datasets may fail.")
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

        # Normalize the tasks
        for task_id, task in self._tasks.items():
            model = self._models[task.model_id]
            sim = self._simulations[task.simulation_id]

            # normalize simulations with respective model dictionary
            sim.normalize(udict=model.udict, ureg=model.ureg)

        # task results
        self._results = None

        # processing
        self._datagenerators = None

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

    def data(self) -> Dict[str, Data]:
        # different then datasets
        return self._data

    @deprecated
    def load_data(self, sid, **kwargs) -> pd.DataFrame:
        """ Loads data from given figure/table id.

        Use load_dataset instead. This function will be removed
        in future releases.
        """
        df = load_dataframe(sid=sid, data_path=self.data_path, **kwargs)
        return df

    def load_dataset(self, sid, udict=None, **kwargs) -> DataSet:
        """ Loads DataSet (with units) from given figure/table id.

        :param sid:
        :param ureg:
        :param udict: additional units from the outside
        :param kwargs:
        :return:
        """
        df = load_dataframe(sid=sid, data_path=self.data_path, **kwargs)
        return load_dataframe_from_df(df=df, udict=udict, **kwargs)

    def load_dataset_from_df(self, df: pd.DataFrame, udict=None, **kwargs) -> DataSet:
        """ Loads DataSet (with units) from given pandas dataframe.

        :param sid:
        :param ureg:
        :param udict: additional units from the outside
        :param kwargs:
        :return:
        """
        all_udict = {}
        for key in df.columns:
            if key.endswith("_unit"):
                # parse the item and unit in dict
                units = df[key].unique()
                if len(units) > 1:
                    logger.error(f"Column '{key}' in '{sid}' has multiple "
                                 f"units: '{units}'")
                item_key = key[0:-5]
                if item_key not in df.columns:
                    logger.error(f"Missing * column '{item_key}' for unit "
                                 f"column: '{key}'")
                all_udict[item_key] = units[0]

        # add external definitions
        if udict:
            all_udict.update(udict)
        # FIXME: move all logic on the Dataset class
        return DataSet.from_df(df, udict=all_udict, ureg=self.ureg)


    def load_units(self, sids, df=None, units_dict=None):
        """ Loads units from given dataframe."""
        if df is not None:
            all_udict = {key: df[f"{key}_unit"].unique()[0] for key in sids}
        elif units_dict is not None:
            all_udict = {}
            for sid in sids:
                all_udict[sid] = units_dict[sid]
        return all_udict

    @deprecated
    def load_data_pkdb(self, sid, **kwargs) -> Dict[str, pd.DataFrame]:
        """Load timecourse data with units."""
        df = self.load_data(sid=sid, **kwargs)
        dframes = {}
        for substance in df.substance.unique():
            dframes[substance] = df[df.substance == substance]
        return dframes

    # --- TASKS ---------------------------------------------------------------
    def tasks(self) -> Dict[str, Task]:
        """Task definitions."""
        return {}

    def simulations(self) -> Dict[str, AbstractSim]:
        """Simulation definitions."""
        return {}

    #def scans(self) -> Dict[str, TimecourseScan]:
    #    """Scan definitions."""
    #    return {}

    # --- RESULTS -------------------------------------------------------------
    @property
    def results(self) -> Dict[str, Result]:
        if self._results is None:
            self._run_tasks()
        return self._results

    # --- PROCESSING ----------------------------------------------------------
    # TODO:

    # --- FIGURES ----------------------------------------------------------
    def figures(self) -> Dict[str, FigureMPL]:
        """ Figures."""
        return {}

    # --- VALIDATION -------------------------------------------------------------
    def _check_keys(self):
        """Check that everything is okay with the experiment."""
        # string keys for main objects must be unique on SimulationExperiment
        all_keys = dict()
        for field_key in ["_models", "_datasets", "_tasks", "_simulations"]:
            field = getattr(self, field_key)
            if not isinstance(field, dict):
                raise ValueError(f"'{field_key} must be a dict, but '{field}' is type '{type(field)}'.")
            for key in getattr(self, field_key).keys():
                if not isinstance(key, str):
                    raise ValueError(f"'{field_key} keys must be str: "
                                     f"'{key} -> {type(key)}'")
                if key in all_keys:
                    raise ValueError(f"Duplicate key '{key}' for '{field_key}' and '{all_keys[key]}'")
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
                raise ValueError(f"simulations must be of type AbstractSim, but "
                                 f"simulation '{key}' has type: '{type(sim)}'")

    # --- EXECUTE -------------------------------------------------------------
    @timeit
    def run(self, output_path: Path, show_figures: bool = True) -> ExperimentResult:
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

        # run simulations
        self._run_tasks()  # sets self._results

        # some of the figures require actual numerical results!
        self._figures = self.figures()

        # save outputs
        self.save_datasets(output_path)
        self.save_results(output_path)
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
                self._results[task_key] = simulator.timecourses(sim)
            elif isinstance(sim, TimecourseScan):
                logger.info(f"Run scan task: '{task_key}'")
                self._results[task_key] = simulator.scan(sim)
            else:
                raise ValueError(f"Unsupported simulation type: "
                                 f"{type(sim)}")

    # --- SERIALIZATION -------------------------------------------------------
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
            "simulations": {k: v.to_dict() for k, v in self._simulations.items()},
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
                result.to_hdf5(results_path / f"{self.sid}_{rkey}.h5")

    @timeit
    def save_figures(self, results_path):
        """ Save figures.
        :param results_path:
        :return:
        """
        paths = []
        for fkey, fig in self._figures.items():
            path_svg = results_path / f"{self.sid}_{fkey}.svg"

            if isinstance(fig, FigureSEDML):
                fig_mpl = to_figure(fig)
            else:
                fig_mpl = fig

            fig_mpl.savefig(path_svg, dpi=150, bbox_inches="tight")

            paths.append(path_svg)
        return paths



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
