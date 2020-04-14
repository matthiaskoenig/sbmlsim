from sbmlsim.data import Data
from sbmlsim.experiment import SimulationExperiment


class FitData(object):

    def __init__(self, x: str, y: str,
                 x_sd: str=None, x_se: str=None,
                 y_sd: str=None, y_se: str=None,
                 dataset: str=None, task: str=None, function: str=None):
        self.x = x
        self.x_sd = x_sd
        self.x_se = x_se
        self.y = y
        self.y_sd = y_sd
        self.y_se = y_se


class FitMapping(object):

    def __init__(self, experiment: SimulationExperiment,
                 reference: FitData, observable: FitData):
        """

        :param reference: experimental reference data
        :param simulation: simulation outcome
        """
        self.experiment = experiment
        self.reference = reference
        self.observable = observable

        # TODO: create data on experiment
