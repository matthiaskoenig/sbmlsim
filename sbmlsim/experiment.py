import logging

from pathlib import Path
import os
import json
import pandas as pd
import inspect


from sbmlsim.model import load_model
from sbmlsim.simulation_serial import SimulatorSerial
from sbmlsim.timecourse import TimecourseSim
from sbmlsim.plotting_matplotlib import plt
from sbmlsim.units import Units
from json import JSONEncoder
from sbmlsim.serialization import ObjectJSONEncoder
from typing import Dict
from sbmlsim.logging_utils import bcolors
from sbmlsim.result import Result
from sbmlsim.data import DataSet

from matplotlib.pyplot import Figure

logger = logging.getLogger(__name__)


class SimulationExperiment(object):
    """Generic simulation experiment.

    Consists of model, list of timecourse simulations, and corresponding results.
    """
    def __init__(self, model_path=None, data_path=None):
        self.sid = self.__class__.__name__
        self.model_path = model_path
        self.data_path = data_path
        self._results = None
        self._datasets = None
        self._figures = None

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, model_path):
        self._model_path = model_path
        if model_path:
            logging.warning(f"Reading model: {model_path.resolve()}")
            self.r = load_model(self._model_path)
            self.udict, self.ureg = Units.get_units_from_sbml(self._model_path)
        else:
            self.r = None
            self.udict = None
            self.ureg = None

    @property
    def simulations(self) -> Dict[str, TimecourseSim]:
        """ Simulation definitions. """
        raise NotImplementedError

    @property
    def datasets(self) -> Dict[str, pd.DataFrame]:
        """ Datasets. """
        raise NotImplementedError

    @property
    def figures(self) -> Dict[str, Figure]:
        """ Figures."""
        raise NotImplementedError

    def load_data(self, sid, **kwargs):
        """ Loads data from given figure/table id."""
        df = load_data(sid=sid, data_path=self.data_path, **kwargs)
        # TODO: implement loading of DataSets with units
        return df

    def load_data_pkdb(self, sid, **kwargs):
        """Load timecourse data with units."""
        df = load_data(sid=sid, data_path=self.data_path, **kwargs)
        dframes = {}
        for substance in df.substance.unique():
            dframes[substance] = df[df.substance == substance]
        return dframes

    def load_units(self, sids, df=None, units_dict=None):
        """ Loads units from given dataframe."""
        if df is not None:
             udict = {key: df[f"{key}_unit"].unique()[0] for key in sids}
        elif units_dict is not None:
            udict = {}
            for sid in sids:
                udict[sid] = units_dict[sid]
        return udict

    @property
    def results(self) -> Dict[str, Result]:
        if self._results is None:
            self._results = self.simulate()
        return self._results

    def simulate(self, Simulator=SimulatorSerial,
                  absolute_tolerance=1E-12,
                  relative_tolerance=1E-12):
        """Run simulations.

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
        if self._results is None:
            self._results = {}
        for key, sim_def in self.simulations.items():
            logger.warning(f"Simulate {key}")
            # normalize the units
            sim_def.normalize(udict=self.udict, ureg=self.ureg)
            # run simulations
            self._results[key] = simulator.timecourses(sim_def)
        return None

    def _figure(self, xlabel, ylabel, title=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return fig, ax

    def default(self, o):
        """json encoder"""
        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)

    def to_dict(self):
        simulations = self.simulations
        for key, tcsim in simulations.items():
            tcsim.normalize(udict=self.udict, ureg=self.ureg)

        # FIXME: resolve paths relative to base_paths
        d = {
            "experiment_id": self.sid,
            "model_path": Path(self.model_path).resolve(),
            "data_path": Path(self.data_path).resolve(),
            "simulations": simulations,
        }

        return d

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
            path = results_path / f"{self.sid}_{fkey}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            paths.append(path)
        return paths


    def save_results(self, results_path):
        """ Save results (mean timecourse)

        :param results_path:
        :return:
        """
        for rkey, result in self.results.items():
            result.mean.to_csv(results_path / f"{self.sid}_{rkey}.tsv",
                               sep="\t", index=False)

    def save_datasets(self, results_path):
        """ Save datasets

        :param results_path:
        :return:
        """
        for dkey, dset in self.datasets.items():
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


def run_experiment(cls_experiment, output_path, model_path, data_path):
    # create experiment
    exp = cls_experiment(model_path=model_path,
                         data_path=data_path)  # type: SimulationExperiment
    # run simulations
    exp.simulate()

    # create and save figures
    exp.save_figures(output_path)

    # save results
    path_results = output_path / "sbmlsim"
    if not path_results.exists():
        os.mkdir(path_results)
    exp.save_results(path_results)

    # create and save data sets
    exp.save_datasets(path_results)

    exp.to_json(output_path / f"{exp.sid}.json")
    plt.show()
