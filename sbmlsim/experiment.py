import logging

from pathlib import Path
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
    def __init__(self, model_path=None, data_path=None, Simulator=SimulatorSerial):
        self.sid = self.__class__.__name__
        # model
        self.model_path = model_path
        self.data_path = data_path
        self.Simulator = Simulator
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

    @property
    def results(self) -> Dict[str, Result]:
        if self._results is None:
            self._results = self.simulate()
        return self._results

    def simulate(self):
        """Run simulations."""
        if not self.model_path:
            raise ValueError("'model_path' must be set to run 'simulate'")

        simulator = self.Simulator(self.model_path)  # reinitialize due to object store

        results = dict()
        # FIXME: this can be parallized
        for key, sim_def in self.simulations.items():
            logger.warning(f"Simulate {key}")
            # normalize the units
            sim_def.normalize(udict=self.udict, ureg=self.ureg)
            # run simulations
            results[key] = simulator.timecourses(sim_def)
        return results

    def _figure(self, xlabel, ylabel, title=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        return fig, ax

    @staticmethod
    def save_fig(fig, fid, results_path):
        fig.savefig(results_path / f"{fid}.png", dpi=150, bbox_inches="tight")

    def default(self, o):
        """json encoder"""
        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)

    def to_dict(self):
        tcsim = self.timecourse_sim()
        tcsim.normalize(udict=self.udict, ureg=self.ureg)

        d = {
            "experiment_id": self.sid,
            "model_path": Path(self.model_path).resolve(),
            "simulations": tcsim,
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


    '''
    def save_result(self, results_path):
        # FIXME: make work for multiple simulations
        self.result.mean.to_csv(results_path / f"{self.sid}.tsv", sep="\t", index=False)

    def save_data(self, results_path):
        # FIXME
        df.to_csv(results_path / f"{self.sid}_data.tsv", sep="\t", index=False)
    '''

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