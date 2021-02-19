"""
Definition of FitProblem.
"""
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from sbmlsim.data import Data
from sbmlsim.serialization import ObjectJSONEncoder


logger = logging.getLogger(__name__)


class FitExperiment(object):
    """A parameter fitting experiment.

    A parameter fitting experiment consists of multiple mapping (reference data to
    observable). The individual mappings can be weighted differently in the fitting.
    """

    def __init__(
        self,
        experiment,
        mappings: List[str] = None,
        weights: Union[float, List[float]] = None,
        use_mapping_weights: bool = False,
        fit_parameters: Dict[str, List["FitParameter"]] = None,
    ):
        """A Simulation experiment used in a fitting.

        weights must be updated according to the mappings

        :param experiment:
        :param mappings: mappings to use from experiments (None uses all mappings)
        :param weights: weight of mappings, the larger the value the larger the weight
        :param use_mapping_weights: uses weights of mapping
        :param fit_parameters: LOCAL parameters only changed in this simulation
                                experiment
        """
        self._weights = None
        self.experiment_class = experiment

        if len(mappings) > len(set(mappings)):
            raise ValueError(
                f"Duplicate fit mapping keys are not allowed. Use weighting for "
                f"changing weights of single mappings: {self.experiment_class.__name__}: '{sorted(mappings)}'"
            )
        self.mappings = mappings
        self.use_mapping_weights = use_mapping_weights
        self.weights = weights

        if fit_parameters is None:
            self.fit_parameters = {}
        else:
            self.fit_parameters = fit_parameters
            # TODO: implement
            raise ValueError(
                "Local parameters in FitExperiment not yet supported, see "
                "https://github.com/matthiaskoenig/sbmlsim/issues/85"
            )

    @property
    def weights(self) -> List[float]:
        """Weights of fit mappings."""
        return self._weights

    @weights.setter
    def weights(self, weights: Union[float, List[float]] = None) -> None:
        """Set weights for mappings in fit experiment."""
        if self.use_mapping_weights:
            if weights is not None:
                raise ValueError(
                    f"Either 'weights' can be set on a FitExperiment or the weight of "
                    f"the FitMapping can be used via the 'use_mapping_weights=True' "
                    f"flag. Weights were provided: '{weights}' in {str(self)}"
                )

            # have to be calculated dynamically
            self._weights = None

        # weights processing
        if weights is None:
            weights = 1.0

        if isinstance(weights, (float, int)):
            return [weights] * len(self.mappings)
        elif isinstance(weights, (list, tuple)):
            # list of weights
            if len(weights) != len(self.mappings):
                raise ValueError(
                    f"Mapping weights '{weights}' must have same length as "
                    f"mappings '{self.mappings}'."
                )
            self._weights = weights

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
        # FIXME: print weights
        return f"{self.__class__.__name__}({self.experiment_class} {self.mappings})"


class FitMapping(object):
    """Mapping of reference data to observable data.

    In the optimization the difference between the reference data
    (ground truth) and the observable (predicted data) is minimized.
    The weight allows to weight the FitMapping.
    """

    def __init__(
        self,
        experiment: "sbmlsim.experiment.SimulationExperiment",
        reference: "FitData",
        observable: "FitData",
        weight: float = None,
    ):
        """FitMapping.

        To use the weight in the fit mapping the `use_mapping_weights` flag
        must be set on the FitExperiment.

        :param reference: reference data (mostly experimental data)
        :param observable: observable in model
        :param weight: weight of fit mapping (default=1.0)
        """
        self.experiment = experiment
        self.reference = reference
        self.observable = observable
        self._weight = weight

    @property
    def weight(self):
        """Returns defined weight or count of the reference."""
        if self._weight is not None:
            return self._weight
        else:
            return self.reference.count


class FitParameter(object):
    """Parameter adjusted in a parameter optimization.

    The bounds define the box in which the parameter can be varied.
    The start value is the initial value in the parameter fitting for
    algorithms which use it.
    """

    def __init__(
        self,
        pid: str,
        start_value: float = None,
        lower_bound: float = -np.Inf,
        upper_bound: float = np.Inf,
        unit: str = None,
    ):
        """Initialize FitParameter.

        :param pid: id of parameter in the model
        :param start_value: initial value for fitting
        :param lower_bound: bounds for fitting
        :param upper_bound: bounds for fitting
        """
        self.pid = pid
        self.start_value = start_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.unit = unit
        if unit is None:
            logger.warning(
                f"No unit provided for FitParameter '{self.pid}', assuming "
                f"model units."
            )

    def __eq__(self, other: "FitParameter") -> bool:
        """Check for equality.

        Uses `math.isclose` for all comparisons of numerical values.
        """
        if not isinstance(other, FitParameter):
            return NotImplemented

        return (
            self.pid == other.pid
            and math.isclose(self.start_value, other.start_value)
            and math.isclose(self.lower_bound, other.lower_bound)
            and math.isclose(self.upper_bound, other.upper_bound)
            and self.unit == other.unit
        )

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"{self.__class__.__name__}<{self.pid} = {self.start_value} "
            f"[{self.lower_bound} - {self.upper_bound}]>"
        )

    def to_json(self, path: Path = None) -> Optional[str]:
        """Serialize to JSON.

        Serializes to file if path is provided, otherwise returns JSON string.
        """
        if path is None:
            return json.dumps(self, cls=ObjectJSONEncoder, indent=2)
        else:
            with open(path, "w") as f_json:
                json.dump(self, fp=f_json, cls=ObjectJSONEncoder, indent=2)

    @staticmethod
    def from_json(json_info: Union[str, Path]) -> "FitParameter":
        """Load from JSON."""
        if isinstance(json_info, Path):
            with open(json_info, "r") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)
        return FitParameter(**d)


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

        if count is not None:
            if dataset is None:
                raise ValueError("'count' can only be set on FitData with dataset")
            else:
                # FIXME: remove duplication with add_data in plotting
                if isinstance(count, int):
                    pass
                elif isinstance(count, str):
                    # resolve count data from dataset
                    count_data = Data(
                        experiment, index=count, dataset=dataset, task=task
                    )
                    counts = count_data.data
                    counts_unique = np.unique(counts.magnitude)
                    if counts_unique.size > 1:
                        logger.warning(f"count is not unique for dataset: '{counts}'")
                    count = int(counts[0].magnitude)
                else:
                    raise ValueError(
                        f"'count' must be integer or a column in a "
                        f"dataset, but type '{type(count)}'."
                    )
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
        result = FitDataInitialized()
        for key in ["x", "y", "x_sd", "x_se", "y_sd", "y_se"]:
            d = getattr(self, key)
            if d is not None:
                setattr(result, key, d.data)

        return result


class FitDataInitialized(object):
    """Initialized FitData with actual data content.

    Data is create from simulation experiment.
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.x_sd = None
        self.x_se = None
        self.y_sd = None
        self.y_se = None

    def __str__(self):
        return str(self.__dict__)
