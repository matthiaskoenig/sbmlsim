import logging
from pathlib import Path
import pandas as pd
from sbmlsim.model import load_model
from sbmlsim.simulation_serial import SimulatorSerial
from sbmlsim.timecourse import TimecourseSim
from sbmlsim.plotting_matplotlib import plt
from sbmlsim.units import Units

logger = logging.getLogger(__name__)


class SimulationExperiment(object):
    """Generic simulation experiment.

    Consists of model, list of timecourse simulations, and corresponding results.

    """

    def __init__(self, model_path=None, data_path=None):
        self.sid = self.__class__.__name__
        # model
        self.model_path = model_path
        self.data_path = data_path

        self.timecourses = None
        self.figures = None
        self.results = None

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

    def timecourse_sim(self) -> TimecourseSim:
        """ Definition of timecourse experiment"""
        raise NotImplementedError

    def simulate(self, Simulator=SimulatorSerial):
        if not self.model_path:
            raise ValueError("'model_path' must be set to run 'simulate'")

        tcsim = self.timecourse_sim()
        tcsim.normalize(udict=self.udict, ureg=self.ureg)
        simulator = Simulator(self.model_path)  # reinitialize due to object store
        self.result = simulator.timecourses(tcsim)

    def plot_data(self):
        raise NotImplementedError

    def plot_sim(self):
        raise NotImplementedError

    def figure(self, xlabel, ylabel):
        fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        self.plot_data(ax)

        if self.result:
            self.plot_sim(ax)
        else:
            logger.warning("No simulation results, run simulation first.")

        ax.set_title('{}'.format(self.sid))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend()
        return fig

    def save_result(self, results_path):
        self.result.mean.to_csv(results_path / f"{self.sid}.tsv", sep="\t", index=False)

    def save_data(self, results_path):
        # FIXME
        df.to_csv(results_path / f"{self.sid}_data.tsv", sep="\t", index=False)

    @staticmethod
    def save_fig(fig, fid, results_path):
        fig.savefig(results_path / f"{fid}.png", dpi=150, bbox_inches="tight")

    def load_data(self, sep="\t"):
        """ Loads data from given figure/table id."""
        return load_data(sid=self.sid, data_path=self.data_path)


def load_data(sid, data_path, sep="\t"):
    """ Loads data from given figure/table id."""
    study = sid.split('_')[0]
    path = data_path / study / f'{sid}.tsv'

    if not path.exists():
        path = data_path / study / f'.{sid}.tsv'

    return pd.read_csv(path, sep=sep, comment="#")
