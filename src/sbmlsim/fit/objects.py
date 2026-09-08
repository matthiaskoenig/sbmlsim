"""Definition of Objects used in FitProblems and optimization."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sbmlsim.data import Data
from sbmlsim.serialization import to_json
from sbmlsim.units import Quantity

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment

logger = logging.getLogger(__name__)


def _isclose(a: float | None, b: float | None) -> bool:
    """Compare two optional floats, None only equals None."""
    if a is None or b is None:
        return a is b
    return math.isclose(a, b)


class FitExperiment:
    """A parameter fitting experiment.

    A parameter fitting experiment consists of multiple mapping (reference data to
    observable). The individual mappings can be weighted differently in the fitting.
    """

    def __init__(
        self,
        experiment: type[SimulationExperiment],
        mappings: list[str] | None = None,
        weights: float | list[float] | None = None,
        use_mapping_weights: bool = False,
        fit_parameters: dict[str, list[FitParameter]] | None = None,
        exclude: bool = False,
    ):
        """Initialize simulation experiment used in a fitting.

        The weights must be updated according to the mappings.

        Args:
            experiment: simulation experiment class of the fit experiment.
            mappings: mappings to use from the experiment. `None` or an empty list
                uses all mappings of the experiment, they are resolved in
                `OptimizationProblem.initialize`, see `resolve_mappings`.
            weights: weight of the mappings, the larger the value the larger the
                weight. A single value is used for all mappings.
            use_mapping_weights: use the weights of the mappings instead of `weights`.
            fit_parameters: LOCAL parameters only changed in this simulation
                experiment.
            exclude: flag to exclude the experiment from the fitting.

        Raises:
            ValueError: for duplicate mappings or unsupported local fit parameters.
        """
        self.experiment_class: type[SimulationExperiment] = experiment
        if mappings is None:
            mappings = []

        if len(mappings) > len(set(mappings)):
            raise ValueError(
                f"Duplicate fit mapping keys are not allowed. Use weighting for "
                f"changing weights of single mappings: "
                f"{self.experiment_class.__name__}: '{sorted(mappings)}'"
            )
        self.mappings: list[str] = mappings
        self.use_mapping_weights = use_mapping_weights
        self.weights = weights
        self.exclude: bool = exclude

        if fit_parameters:
            # TODO: implement
            raise ValueError(
                "Local parameters in FitExperiment not yet supported, see "
                "https://github.com/matthiaskoenig/sbmlsim/issues/85"
            )
        self.fit_parameters: dict[str, list[FitParameter]] = {}

    @property
    def weights(self) -> list[float | None]:
        """Weights of fit mappings, None if the weight of the mapping is used."""
        return self._weights

    @weights.setter
    def weights(self, weights: float | list[float] | None = None) -> None:
        """Set weights for mappings in fit experiment."""
        weights_processed: list[float | None]
        if self.use_mapping_weights is True:
            mapping_weights: list[float | None] = [None] * len(self.mappings)
            # no weights provided use default empty weights
            weights_processed = (
                mapping_weights if weights is None else list(weights)  # ty: ignore[invalid-argument-type]
            )

            # all weights have to be None, i.e [None, ..., None].
            # the weights are calculated dynamically by evaluating the fit mappings.
            if weights_processed != mapping_weights:
                raise ValueError(
                    f"{self.experiment_class.__name__}: either 'weights' are set on "
                    f"a FitExperiment or the weights of the FitMappings are used via "
                    f"'use_mapping_weights=True', but both were given: '{weights}'."
                )
        else:
            # weights processing
            if weights is None:
                weights = 1.0

            if isinstance(weights, (float, int)):
                weights_processed = [weights] * len(self.mappings)
            elif isinstance(weights, (list, tuple)):
                # list of weights
                if len(weights) != len(self.mappings):
                    raise ValueError(
                        f"Mapping weights '{weights}' must have same length as "
                        f"mappings '{self.mappings}'."
                    )
                weights_processed = list(weights)
            else:
                raise ValueError(f"Unsupported weights: '{weights}'")

        self._weights: list[float | None] = weights_processed

    def resolve_mappings(self, mapping_keys: Iterable[str]) -> None:
        """Use all mappings of the experiment if no mappings were selected.

        A `FitExperiment` without mappings uses all fit mappings of its simulation
        experiment. The keys are only known once the experiment is instantiated,
        so the mappings and their weights are resolved in the initialization of the
        `OptimizationProblem`.

        Args:
            mapping_keys: keys of all fit mappings of the simulation experiment.
        """
        if self.mappings:
            return

        self.mappings = list(mapping_keys)
        # the setter expands the weights to the resolved mappings
        self.weights = None

    @staticmethod
    def reduce(fit_experiments: Iterable[FitExperiment]) -> list[FitExperiment]:
        """Combine the fit mappings of the FitExperiments of the same experiment.

        The mappings and their weights are concatenated, the inputs are not modified.

        Raises:
            ValueError: if experiments of the same class cannot be combined, i.e.,
                they use different weighting or repeat a mapping.
        """
        reduced: dict[str, FitExperiment] = {}
        for fit_exp in fit_experiments:
            sid = fit_exp.experiment_class.__name__
            if sid not in reduced:
                reduced[sid] = FitExperiment(
                    experiment=fit_exp.experiment_class,
                    mappings=list(fit_exp.mappings),
                    weights=list(fit_exp.weights),  # ty: ignore[invalid-argument-type]
                    use_mapping_weights=fit_exp.use_mapping_weights,
                    exclude=fit_exp.exclude,
                )
                continue

            red_exp = reduced[sid]
            if red_exp.use_mapping_weights != fit_exp.use_mapping_weights:
                raise ValueError(
                    f"FitExperiments of '{sid}' cannot be combined, they differ in "
                    f"'use_mapping_weights'."
                )
            duplicates = set(red_exp.mappings) & set(fit_exp.mappings)
            if duplicates:
                raise ValueError(
                    f"FitExperiments of '{sid}' cannot be combined, the mappings "
                    f"'{sorted(duplicates)}' occur in both."
                )
            red_exp.mappings = red_exp.mappings + list(fit_exp.mappings)
            red_exp._weights = red_exp._weights + list(fit_exp.weights)
            red_exp.exclude = red_exp.exclude and fit_exp.exclude

        return list(reduced.values())

    def __repr__(self) -> str:
        """Get representation."""
        return (
            f"{self.__class__.__name__}({self.experiment_class.__name__} "
            f"{[f'{m} x {w}' for (m, w) in list(zip(self.mappings, self.weights, strict=False))]})"
        )

    def __str__(self) -> str:
        """Get string."""
        info = [
            "*** FitExperiment ***",
            f"experiment: {self.experiment_class.__name__}",
            f"mappings: {self.mappings}",
            f"weights: {self.weights}",
            f"use_mapping_weights: {self.use_mapping_weights}",
            f"fit_parameters: {self.fit_parameters}",
        ]
        return "\n".join(info)


@dataclass(kw_only=True)
class MappingMetaData:
    """Metadata for mapping.

    Applications derive their metadata from this class, e.g., the tissue, the
    dosing or the group of a study; `outlier` marks a mapping which is excluded
    from the fit.

    The fields are keyword only, so that a subclass can add fields without a
    default after `outlier`, which has one. A subclass must not redeclare
    `outlier`, this would make it a positional field again.
    """

    outlier: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return dict(self.__dict__)


class FitMapping:
    """Mapping of reference data to observable data.

    In the optimization the difference between the reference data
    (ground truth) and the observable (predicted data) is minimized.
    The weight allows to weight the FitMapping.
    """

    def __init__(
        self,
        experiment: Any,  # SimulationExperiment (avoid circular import)
        reference: FitData,
        observable: FitData,
        weight: float | None = None,
        metadata: MappingMetaData | None = None,
    ):
        """Initialize FitMapping.

        To use the weight in the fit mapping the `use_mapping_weights` flag
        must be set on the FitExperiment.

        Args:
            experiment: simulation experiment of the mapping.
            reference: reference data (mostly experimental data).
            observable: observable in the model.
            weight: weight of the fit mapping, the count of the reference data
                is used if no weight is given.
            metadata: metadata of the mapping.
        """
        self.experiment = experiment
        self.reference = reference
        self.observable = observable
        self._weight = weight
        self.metadata = metadata

    @property
    def weight(self) -> float:
        """Return the defined weight or the count of the reference data.

        Raises:
            ValueError: if neither a weight nor a count is available.
        """
        if self._weight is not None:
            return self._weight
        if self.reference.count is None:
            raise ValueError(
                f"FitMapping requires either a 'weight' or a 'count' on its "
                f"reference data: '{self}'"
            )
        return float(self.reference.count)

    def __str__(self) -> str:
        """Get string."""
        return (
            f"FitMapping({self.experiment.sid}, "
            f"reference={self.reference}, observable={self.observable})"
        )


class FitParameter:
    """Parameter adjusted in a parameter optimization.

    The bounds define the box in which the parameter can be varied.
    The start value is the initial value in the parameter fitting for
    algorithms which use it.
    """

    def __init__(
        self,
        pid: str,
        start_value: float | None = None,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        unit: str | None = None,
    ):
        """Initialize FitParameter.

        Args:
            pid: id of the parameter in the model.
            start_value: initial value for the fitting.
            lower_bound: lower bound for the fitting.
            upper_bound: upper bound for the fitting.
            unit: unit of the parameter, the model unit is assumed if not given.

        Raises:
            ValueError: if the bounds or the start value are inconsistent.
        """
        if lower_bound > upper_bound:
            raise ValueError(
                f"FitParameter '{pid}': lower bound '{lower_bound}' is larger than "
                f"upper bound '{upper_bound}'."
            )
        if start_value is not None and not lower_bound <= start_value <= upper_bound:
            raise ValueError(
                f"FitParameter '{pid}': start value '{start_value}' is outside of "
                f"the bounds [{lower_bound} - {upper_bound}]."
            )

        self.pid = pid
        self.start_value = start_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.unit = unit
        if unit is None:
            logger.warning(
                "No unit provided for FitParameter '%s', assuming model units.",
                self.pid,
            )

    def __eq__(self, other: object) -> bool:
        """Check for equality.

        Uses `math.isclose` for all comparisons of numerical values.
        """
        if not isinstance(other, FitParameter):
            return NotImplemented

        return (
            self.pid == other.pid
            and _isclose(self.start_value, other.start_value)
            and _isclose(self.lower_bound, other.lower_bound)
            and _isclose(self.upper_bound, other.upper_bound)
            and self.unit == other.unit
        )

    def __hash__(self) -> int:
        """Get hash of the parameter id."""
        return hash(self.pid)

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"{self.__class__.__name__}<{self.pid} = {self.start_value} "
            f"[{self.lower_bound} - {self.upper_bound}]>"
        )

    def to_json(self, path: Path | None = None) -> str | Path:
        """Serialize to JSON.

        Serializes to file if path is provided, otherwise returns JSON string.
        """
        return to_json(object=self, path=path)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pid": self.pid,
            "start_value": self.start_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "unit": self.unit,
        }

    @staticmethod
    def from_json(json_info: str | Path) -> FitParameter:
        """Load from JSON."""
        if isinstance(json_info, Path):
            with open(json_info, encoding="utf-8") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)
        return FitParameter(**d)

    @staticmethod
    def parameters_to_df(parameters: Iterable[FitParameter]) -> pd.DataFrame:
        """DataFrame of parameters."""
        return pd.DataFrame([p.to_dict() for p in parameters])


class FitData:
    """Data used in a fit.

    This is either data from a dataset, a simulation results from
    a task or functional data, i.e. calculated from other data.
    """

    def __init__(
        self,
        experiment: Any,  # SimulationExperiment (avoid circular import)
        xid: str,
        yid: str,
        xid_sd: str | None = None,
        xid_se: str | None = None,
        yid_sd: str | None = None,
        yid_se: str | None = None,
        count: int | str | None = None,
        dataset: str | None = None,
        task: str | None = None,
        function: str | None = None,
    ):
        """Initialize FitData.

        Args:
            experiment: simulation experiment the data belongs to.
            xid: index of the x data.
            yid: index of the y data.
            xid_sd: index of the standard deviation of the x data.
            xid_se: index of the standard error of the x data.
            yid_sd: index of the standard deviation of the y data.
            yid_se: index of the standard error of the y data.
            count: number of subjects, either an integer or a column of the dataset.
            dataset: id of the dataset the data comes from.
            task: id of the task the data comes from.
            function: id of the function the data is calculated with.

        Raises:
            ValueError: if `count` is set without a dataset or has a wrong type.
        """
        self.experiment = experiment
        self.dset_id = dataset
        self.task_id = task
        self.function = function
        self.count: int | None = self._resolve_count(count)

        # actual data
        self.x = self._data(xid)
        self.y = self._data(yid)
        self.x_sd = self._error_data(xid_sd, error_type="sd")
        self.x_se = self._error_data(xid_se, error_type="se")
        self.y_sd = self._error_data(yid_sd, error_type="sd")
        self.y_se = self._error_data(yid_se, error_type="se")

    def _data(self, index: str) -> Data:
        """Create the data promise for an index."""
        return Data(
            index=index,
            task=self.task_id,
            dataset=self.dset_id,
            function=self.function,
        )

    def _error_data(self, index: str | None, error_type: str) -> Data | None:
        """Create the data promise for an error column, warning on a wrong suffix."""
        if not index:
            return None
        other = "se" if error_type == "sd" else "sd"
        if index.endswith(other):
            logger.warning(
                "%s error column '%s' ends with '%s', check names.",
                error_type.upper(),
                index,
                other,
            )
        return self._data(index)

    def _resolve_count(self, count: int | str | None) -> int | None:
        """Resolve the count from an integer or a column of the dataset."""
        if count is None:
            return None
        if self.dset_id is None:
            raise ValueError("'count' can only be set on FitData with dataset")
        if isinstance(count, int):
            return count
        if not isinstance(count, str):
            raise ValueError(
                f"'count' must be integer or a column in a "
                f"dataset, but type '{type(count)}'."
            )

        # resolve count data from dataset
        # FIXME: remove duplication with add_data in plotting
        count_data = Data(index=count, dataset=self.dset_id, task=self.task_id)
        counts = count_data.get_data(self.experiment)
        counts_unique = np.unique(counts.magnitude)
        if counts_unique.size > 1:
            logger.warning("count is not unique for dataset: '%s'", counts)
        return int(counts_unique[0])

    def __str__(self) -> str:
        """Get string."""
        return (
            f"FitData(experiment={self.experiment.__class__.__name__} dset_id={self.dset_id} "
            f"task_id={self.task_id} function={self.function})"
        )

    def is_task(self) -> bool:
        """Check if FitData comes from a task (simulation)."""
        return self.task_id is not None

    def is_dataset(self) -> bool:
        """Check if FitData comes from a dataset."""
        return self.dset_id is not None

    def is_function(self) -> bool:
        """Check if FitData comes from a function."""
        return self.function is not None

    @property
    def dtype(self) -> Data.Types:
        """Get data type.

        Raises:
            ValueError: if the data is neither from a task, dataset nor function.
        """
        if self.task_id:
            return Data.Types.TASK
        if self.dset_id:
            return Data.Types.DATASET
        if self.function:
            return Data.Types.FUNCTION
        raise ValueError("DataType could not be determined!")

    def get_data(self) -> FitDataInitialized:
        """Return actual data.

        Numerical values are resolved using the executed simulation experiment.
        """
        result = FitDataInitialized()
        for key in FitDataInitialized.KEYS:
            logger.debug("FitData.get_data: %s.%s", self, key)
            d: Data | None = getattr(self, key)
            if d is not None:
                setattr(result, key, d.get_data(self.experiment))

        return result


@dataclass
class FitDataInitialized:
    """Initialized FitData with actual data content.

    The data is created from the simulation experiment, the values are quantities
    with the units of the data.
    """

    KEYS = ("x", "y", "x_sd", "x_se", "y_sd", "y_se")

    x: Quantity | None = None
    y: Quantity | None = None
    x_sd: Quantity | None = None
    x_se: Quantity | None = None
    y_sd: Quantity | None = None
    y_se: Quantity | None = None

    def __str__(self) -> str:
        """Get string representation."""
        return str(self.__dict__)
