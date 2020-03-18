import logging

from pathlib import Path
import os
import warnings
import json
import pandas as pd
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple
import abc
from abc import ABC


from sbmlsim.utils import deprecated
from sbmlsim.simulation_serial import SimulatorSerial
from sbmlsim.timecourse import TimecourseSim, TimecourseScan
from sbmlsim.units import Units
from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.result import Result
from sbmlsim.data import Data
from sbmlsim.models import RoadrunnerSBMLModel, AbstractModel

from sbmlsim.plotting_matplotlib import plt, to_figure
from matplotlib.pyplot import Figure as FigureMPL
from sbmlsim.plotting import Figure as FigureSEDML

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a simulation experiment"""
    experiment: 'SimulationExperiment'
    output_path: Path
    model_path: Path
    data_path: Path
    results_path: Path


class SimulationExperiment(ABC):
    """Generic simulation experiment.

    Consists of models, datasets, simulations, tasks, results, processing, figures
    """

    def __init__(self, model_path=None, data_path=None, **kwargs):
        """ Constructor.

        When an instance of the SimulationExperiment is created all information
        defined in the static methods is collected.

        The execute function executes the model subsequently.

        :param model_path:
        :param data_path:
        :param kwargs:
        """

        self.name = self.__class__.__name__

        # models
        self.model_path = model_path

        # datasets
        self.data_path = data_path
        self._datasets = None

        # task results
        self._results = None
        self._scan_results = None

        # processing
        self._datagenerators = None

        # figures
        self._figures = None

        # settings
        self.settings = kwargs

    # --- MODELS --------------------------------------------------------------
    # FIXME: have multiple models which could be used and defined at the beginning.
    # These could be derived models based on applied model changes.

    @abc.abstractclassmethod
    def _models(cls, self) -> Dict:
        return

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, model_path):
        self._model_path = model_path
        if model_path:
            model = RoadrunnerSBMLModel(source=self._model_path)
            self.r = model._model
            self.udict, self.ureg = Units.get_units_from_sbml(self._model_path)
            self.Q_ = self.ureg.Quantity
        else:
            self.r = None
            self.udict = None
            self.ureg = None
            self.Q_ = None

    # --- DATASETS ------------------------------------------------------------
    @property
    def datasets(self) -> Dict[str, pd.DataFrame]:
        """ Datasets.

        This corresponds to datasets in SED-ML
        """
        logger.debug(f"No datasets defined for '{self.sid}'.")
        return {}

    @property
    def data(self) -> Dict[str, Data]:
        return self._data

    def load_data(self, sid, **kwargs):
        """ Loads data from given figure/table id."""
        df = load_data(sid=sid, data_path=self.data_path, **kwargs)
        return df

    # --- TASKS ---------------------------------------------------------------
    @property
    def simulations(self) -> Dict[str, TimecourseSim]:
        """ Simulation definitions. """
        logger.debug(f"No simulations defined for '{self.sid}'.")
        return {}

    @property
    def scans(self) -> Dict[str, TimecourseScan]:
        """ Scan definitions. """
        logger.debug(f"No scans defined for '{self.sid}'.")
        return {}

    # FIXME: make sure things are only executed once
    def results(self) -> Dict[str, Result]:
        if self._results is None:
            self._run_tasks()
        return self._results

    def scan_results(self) -> Dict[str, Result]:
        if self._scan_results is None:
            self._run_tasks()
        return self._scan_results

    # --- PROCESSING ----------------------------------------------------------
    # TODO:

    # --- FIGURES ----------------------------------------------------------
    def figures(self) -> Dict[str, FigureMPL]:
        """ Figures."""
        # FIXME: generic handling of figures (i.e., support of multiple backends)
        logger.debug(f"No figures defined for '{self.sid}'.")
        return {}

    def _check(self):
        """Check that everything is okay with the experiment."""
        for key in self.datasets.keys():
            if not isinstance(key, str):
                raise ValueError(f"Dataset keys must be str: '{key} -> {type(key)}'")
        for key in self.simulations.keys():
            if not isinstance(key, str):
                raise ValueError(f"Simulation keys must be str: '{key} -> {type(key)}'")
        for key in self.scans.keys():
            if not isinstance(key, str):
                raise ValueError(f"Scan keys must be str: '{key} -> {type(key)}'")
        for key in self.figures.keys():
            if not isinstance(key, str):
                raise ValueError(f"Figure keys must be str: '{key} -> {type(key)}'")

    # --- EXECUTE -------------------------------------------------------------
    def execute(self,
                output_path: Path,
                model_path: Path,
                data_path: Path,
                show_figures: bool = True) -> ExperimentResult:
        """
        Executes given experiment.
        Returns info dictionary.
        """
        # path operations
        if not Path.exists(model_path):
            raise IOError(f"'model_path' does not exist: '{model_path}'")
        if not Path.exists(data_path):
            raise IOError(f"'data_path' does not exist: '{data_path}'")
        if not Path.is_dir(data_path):
            raise IOError(f"'data_path' is not a directory: '{data_path}'")

        if not Path.exists(output_path):
            Path.mkdir(output_path, parents=True)
            logging.info(f"'output_path' created: '{output_path}'")

        # create experiment
        exp = cls_experiment(model_path=model_path,
                             data_path=data_path)  # type: SimulationExperiment

        # validation
        exp._check()

        # run simulations
        exp._run_tasks()

        # create and save figures
        exp.save_figures(output_path)

        # save results
        results_path = output_path
        if not results_path.exists():
            os.mkdir(results_path)
        exp.save_results(results_path)

        # create and save data sets
        exp.save_datasets(results_path)

        # save json representation
        from pprint import pprint
        pprint(exp.to_dict())
        exp.to_json(output_path / f"{exp.sid}.json")

        # display figures
        if show_figures:
            plt.show()

        # create markdown report
        # exp.to_markdown(output_path)

        return ExperimentResult(
            experiment=exp,
            output_path=output_path,
            model_path=model_path,
            data_path=data_path,
            results_path=results_path,
        )

    def _run_tasks(self, Simulator=SimulatorSerial,
                   absolute_tolerance=1E-14,
                   relative_tolerance=1E-14):
        """Run simulations & scans.

        This should not be called directly, but the results of the simulations
        should be requested by the results property.
        This allows to hash exectuted simulations without the need for
        reececuting them
        """
        if not self.model_path:
            raise ValueError("'model_path' must be set to run 'simulate'")

        simulator = Simulator(self.model_path,
                              absolute_tolerance=absolute_tolerance,
                              relative_tolerance=relative_tolerance)  # reinitialize due to object store

        # FIXME: this can be parallized
        # run timecourse simulations
        if self._results is None:
            self._results = {}
        for key, sim_def in self.simulations.items():
            logger.info(f"Simulate timecourse: '{key}'")
            # normalize the units
            sim_def.normalize(udict=self.udict, ureg=self.ureg)
            # run simulations
            self._results[key] = simulator.timecourses(sim_def)

        # run scans
        if self._scan_results is None:
            self._scan_results = {}

        for key, scan_def in self.scans.items():
            logger.info(f"Simulate scan: '{key}'")
            # normalize the units
            scan_def.normalize(udict=self.udict, ureg=self.ureg)
            # run simulations
            self._scan_results[key] = simulator.scan(scan_def)

        return None

    # --- SERIALIZATION -------------------------------------------------------


    def default(self, o):
        """json encoder"""
        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)

    def to_dict(self):
        """Convert to dictionary.
        This is the basis for the JSON serialization.
        """
        simulations = self.simulations
        for key, tcsim in simulations.items():
            tcsim.normalize(udict=self.udict, ureg=self.ureg)

        # FIXME: resolve paths relative to base_paths
        d = {
            "experiment_id": self.sid,
            "model_path": Path(self.model_path).resolve(),
            "data_path": Path(self.data_path).resolve(),
            "simulations": simulations,
            "figures": self.figures,
        }

        return d

    def to_markdown(self, output_path=None):
        """ Create markdown report of the simulation experiment.

        :param path: path for file, if None JSON str is returned
        :return:
        """
        pass

    def to_json(self, path=None):
        """ Convert experiment to JSON for exchange.

        :param path: path for file, if None JSON str is returned
        :return:
        """
        if path is None:
            return json.dumps(self.to_dict(), cls=ObjectJSONEncoder, indent=2)
        else:
            with open(path, "w") as f_json:
                json.dump(self.to_dict(), fp=f_json, cls=ObjectJSONEncoder, indent=2)

    @classmethod
    def from_json(cls, json_info) -> 'SimulationExperiment':
        """Load experiment from json path or str"""
        if isinstance(json_info, Path):
            with open(json_info, "r") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)

        return JSONExperiment.from_dict(d)

    def save_simulations(self, results_path, normalize=False):
        """ Save simulations

        :param results_path:
        :param normalize: Normalize all values to model units.
        :return:
        """
        for skey, tcsim in self.simulations.items():
            if normalize:
                tcsim.normalize(udict=self.udict, ureg=self.ureg)
            tcsim.to_json(results_path / f"{self.sid}_{skey}.json")

    def save_figures(self, results_path):
        """ Save figures.
        :param results_path:
        :return:
        """
        paths = []
        for fkey, fig in self.figures.items():
            path_svg = results_path / f"{self.sid}_{fkey}.svg"

            if isinstance(fig, FigureSEDML):
                fig_mpl = to_figure(fig)
            else:
                fig_mpl = fig

            fig_mpl.savefig(path_svg, dpi=150, bbox_inches="tight")

            paths.append(path_svg)
        return paths

    def save_results(self, results_path):
        """ Save results (mean timecourse)

        :param results_path:
        :return:
        """
        for rkey, result in self.results.items():
            result.to_hdf5(results_path / f"{self.sid}_sim_{rkey}.h5")

        for rkey, result in self._scan_results.items():
            result.to_hdf5(results_path / f"{self.sid}_scan_{rkey}.h5")

    def save_datasets(self, results_path):
        """ Save datasets

        :param results_path:
        :return:
        """
        for dkey, dset in self.datasets.items():

            # TODO: Save datafiles with the correct units
            dset.to_csv(results_path / f"{self.sid}_data_{dkey}.tsv",
                        sep="\t", index=False)




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


# TODO: implement loading of DataSets with units

def load_data(sid, data_path, sep="\t", comment="#", **kwargs):
    """ Loads data from given figure/table id."""
    study = sid.split('_')[0]
    path = data_path / study / f'{sid}.tsv'

    if not path.exists():
        path = data_path / study / f'.{sid}.tsv'

    return pd.read_csv(path, sep=sep, comment=comment, **kwargs)


def function_name():
    """Returns current function name"""
    frame = inspect.currentframe()
    return inspect.getframeinfo(frame).function


