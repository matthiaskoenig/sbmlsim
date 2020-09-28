"""
Definition of FitProblem.
"""
import logging
from typing import Dict, Iterable, List, Set, Union

import numpy as np

from sbmlsim.data import Data


logger = logging.getLogger(__name__)


class FitExperiment(object):
    """
    A Simulation Experiment used in a fitting.
    """

    def __init__(
        self,
        experiment,
        mappings: List[str] = None,
        weights: List[float] = 1.0,
        fit_parameters: Dict[str, List["FitParameter"]] = None,
    ):
        """A Simulation experiment used in a fitting.

        weights must be updated according to the mappings

        :param experiment:
        :param weights: weight of mappings
        :param mappings: mappings to use from experiments (None uses all mappings)
        :param fit_parameters: LOCAL parameters only changed in this simulation
                                experiment
        """
        self.experiment_class = experiment

        if (weights is None) or (mappings is None):
            weights = []
        if mappings is None:
            mappings = []

        if len(mappings) > len(set(mappings)):
            raise ValueError(
                f"Duplicate fit mapping keys are not allowed. Use weighting for "
                f"changing weights of single mappings: {self.experiment_class.__name__}: '{sorted(mappings)}'"
            )

        self.mappings = mappings
        if isinstance(weights, (float, int)):
            self.weights = [weights] * len(mappings)
        else:
            # list of weights
            if len(weights) != len(mappings):
                raise ValueError(
                    f"Mapping weights '{weights}' must have same length as "
                    f"mappings '{mappings}'."
                )
            self.weights = weights
        if fit_parameters is None:
            self.fit_parameters = {}
        else:
            ValueError("Local parameters in FitExperiment not yet supported.")
            # FIXME: implement

    @staticmethod
    def reduce(fit_experiments: Iterable["FitExperiment"]) -> List["FitExperiment"]:
        """Collects fit mappings of multiple FitExperiments"""
        red_experiments = {}
        for fit_exp in fit_experiments:
            sid = fit_exp.experiment_class.__name__
            if sid not in red_experiments:
                red_experiments[sid] = fit_exp
            else:
                # combine the experiments
                red_exp = red_experiments[sid]
                red_exp.mappings = red_exp.mappings + fit_exp.mappings
                red_exp.weights = red_exp.weights + fit_exp.weights

        return list(red_experiments.values())

    def __str__(self):
        return f"{self.__class__.__name__}({self.experiment_class} {self.mappings})"


class FitMapping(object):
    """Mapping of reference data to obeservables in the model."""

    def __init__(
        self,
        experiment: "sbmlsim.experiment.SimulationExperiment",
        reference: "FitData",
        observable: "FitData",
        count: int = None,
    ):
        """FitMapping.

        :param reference: reference data (mostly experimental data)
        :param observable: observable in model
        """
        self.experiment = experiment
        self.reference = reference
        self.observable = observable
        self.count = count


class FitParameter(object):
    """Parameter adjusted in a parameter optimization."""

    def __init__(
        self,
        parameter_id: str,
        start_value: float = None,
        lower_bound: float = -np.Inf,
        upper_bound: float = np.Inf,
        unit: str = None,
    ):
        """FitParameter.

        :param parameter_id: id of parameter in the model
        :param start_value: initial value for fitting
        :param lower_bound: bounds for fitting
        :param upper_bound: bounds for fitting

        """
        self.pid = parameter_id
        self.start_value = start_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.unit = unit
        if unit is None:
            logger.error(
                f"No unit provided for FitParameter '{self.pid}', assuming "
                f"model units."
            )

    def __str__(self):
        return (
            f"{self.__class__.__name__}<{self.pid} = {self.start_value} "
            f"[{self.lower_bound} - {self.upper_bound}]>"
        )


class FitData(object):
    """Data used in a fit.

    This is either data from a dataset, a simulation results from
    a task or functional data, i.e. calculated from other data.
    """

    def __init__(
        self,
        experiment: "SimulationExperiment",
        xid: str,
        yid: str,
        xid_sd: str = None,
        xid_se: str = None,
        yid_sd: str = None,
        yid_se: str = None,
        count: Union[int, str] = None,
        dataset: str = None,
        task: str = None,
        function: str = None,
    ):

        self.dset_id = dataset
        self.task_id = task
        self.function = function

        # FIXME: support setting count on FitData
        self.count = count

        # actual Data
        # FIXME: simplify
        self.x = Data(
            experiment=experiment,
            index=xid,
            task=self.task_id,
            dataset=self.dset_id,
            function=self.function,
        )
        self.y = Data(
            experiment=experiment,
            index=yid,
            task=self.task_id,
            dataset=self.dset_id,
            function=self.function,
        )
        self.x_sd = None
        self.x_se = None
        self.y_sd = None
        self.y_se = None
        if xid_sd:
            if xid_sd.endswith("se"):
                logger.warning("SD error column ends with 'se', check names.")
            self.x_sd = Data(
                experiment=experiment,
                index=xid_sd,
                task=self.task_id,
                dataset=self.dset_id,
                function=self.function,
            )
        if xid_se:
            if xid_se.endswith("sd"):
                logger.warning("SE error column ends with 'sd', check names.")
            self.x_se = Data(
                experiment=experiment,
                index=xid_se,
                task=self.task_id,
                dataset=self.dset_id,
                function=self.function,
            )
        if yid_sd:
            if yid_sd.endswith("se"):
                logger.warning("SD error column ends with 'se', check names.")
            self.y_sd = Data(
                experiment=experiment,
                index=yid_sd,
                task=self.task_id,
                dataset=self.dset_id,
                function=self.function,
            )
        if yid_se:
            if yid_se.endswith("sd"):
                logger.warning("SE error column ends with 'sd', check names.")
            self.y_se = Data(
                experiment=experiment,
                index=yid_se,
                task=self.task_id,
                dataset=self.dset_id,
                function=self.function,
            )

    def is_task(self):
        return self.task_id is not None

    def is_dataset(self):
        return self.dset_id is not None

    def is_function(self):
        return self.function is not None

    @property
    def dtype(self):
        if self.task_id:
            dtype = Data.Types.TASK
        elif self.dset_id:
            dtype = Data.Types.DATASET
        elif self.function:
            dtype = Data.Types.FUNCTION
        else:
            raise ValueError("DataType could not be determined!")
        return dtype

    def get_data(self) -> Dict:
        """Returns actual data."""
        result = FitDataResult()
        for key in ["x", "y", "x_sd", "x_se", "y_sd", "y_se"]:
            d = getattr(self, key)
            if d is not None:
                setattr(result, key, d.data)

        return result


class FitDataResult(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.x_sd = None
        self.x_se = None
        self.y_sd = None
        self.y_se = None

    def __str__(self):
        return str(self.__dict__)
