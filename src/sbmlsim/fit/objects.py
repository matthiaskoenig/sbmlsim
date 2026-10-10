"""Definition of Objects used in FitProblems and optimization."""

from __future__ import annotations

import functools
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sbmlsim.data import Data, to_quantity
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.serialization import to_json
from sbmlsim.simulator.formula import compile_formula
from sbmlsim.units import Quantity

if TYPE_CHECKING:
    from sbmlsim.experiment import SimulationExperiment

logger = logging.getLogger(__name__)

#: prefix of the target of a parameter which is not an entity of the model.
#: No change of the simulation is written for it, the derived changes of the
#: problem read its value, see `sbmlsim.fit.derived`
EXTERNAL_PREFIX = "sciml:"


def _isclose(a: float | None, b: float | None) -> bool:
    """Compare two optional floats, None only equals None."""
    if a is None or b is None:
        return a is b
    return math.isclose(a, b)


class MappingKind(StrEnum):
    """How the data of the fit mappings of a `FitMappingCollection` is used.

    `training` (default) : the mappings are fitted, i.e., their residuals enter
    the cost of the optimization.

    `validation` : the mappings are not fitted. They are simulated and
    evaluated with the training data when a fit is reported, which shows how
    the fitted parameters describe data they were not fitted on.

    `outlier` : the mappings are not fitted, because the data is not usable,
    e.g. a curve which contradicts the rest of the data. They are simulated and
    evaluated like the validation data, so a report says how far the data a fit
    dropped is from the model and the decision to drop it can be checked.

    `excluded` : the mappings are not used at all, and for another reason than
    an outlier: the model does not describe them, e.g. a study arm with a
    coadministration the model has no interaction for. The data is fine, the
    model is not the one for it, so it is not an outlier: an outlier is a
    decision about the data and an exclusion is a decision about the model.
    Excluded mappings are not resolved, so they have no metrics.

    The kind is set on the `FitMappingCollection`, i.e., when the data of a fit is
    selected, not on the fit mappings of a simulation experiment: a mapping
    describes a curve, the kind describes what a fit does with it, and the same
    curve is training data of one fit and validation data of another.
    """

    TRAINING = "training"
    VALIDATION = "validation"
    OUTLIER = "outlier"
    EXCLUDED = "excluded"


#: kinds which are resolved, simulated and evaluated. Only the training data
#: enters the cost, the validation data and the outliers are evaluated so that
#: a report says how the fit describes the data it was not fitted on
EVALUATED_KINDS: tuple[MappingKind, ...] = (
    MappingKind.TRAINING,
    MappingKind.VALIDATION,
    MappingKind.OUTLIER,
)

#: kinds which a fit does not use at all, i.e. which are not even resolved
UNUSED_KINDS: tuple[MappingKind, ...] = (MappingKind.EXCLUDED,)


class NoiseDistribution(StrEnum):
    """Distribution of the noise of a measurement.

    These are the distributions of PEtab v2. The simulation is the median of
    the distribution and the noise formula gives its scale: the standard
    deviation of `normal`, the standard deviation of the logarithm of
    `log-normal` and the scale `b` of `laplace` and, on the logarithm, of
    `log-laplace`.
    """

    NORMAL = "normal"
    LOG_NORMAL = "log-normal"
    LAPLACE = "laplace"
    LOG_LAPLACE = "log-laplace"

    @property
    def is_log(self) -> bool:
        """Check whether the noise acts on the logarithm of the measurement."""
        return self in {NoiseDistribution.LOG_NORMAL, NoiseDistribution.LOG_LAPLACE}


class PriorDistribution(StrEnum):
    """Distribution of the prior of a parameter.

    These are the priors of PEtab v2, with the parameters of PEtab, e.g. the
    mean and the standard deviation of `normal` and the bounds of `uniform`.
    """

    CAUCHY = "cauchy"
    CHISQUARE = "chisquare"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    LAPLACE = "laplace"
    LOG_LAPLACE = "log-laplace"
    LOG_NORMAL = "log-normal"
    LOG_UNIFORM = "log-uniform"
    NORMAL = "normal"
    RAYLEIGH = "rayleigh"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class Prior:
    """The prior of a parameter.

    The prior is a distribution of PEtab v2 over the value of the parameter in
    its unit, truncated at the bounds of the parameter, i.e. normalized over
    them. A parameter without a prior has the uniform prior over its bounds.

    Attributes:
        distribution: the distribution.
        parameters: the parameters of the distribution, see
            `PriorDistribution`.
    """

    distribution: PriorDistribution
    parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        """Coerce the fields and check them.

        Raises:
            ValueError: if the distribution is not one of PEtab or a parameter
                is not a number.
        """
        try:
            distribution = PriorDistribution(self.distribution)
        except ValueError as err:
            raise ValueError(
                f"The prior distribution '{self.distribution}' is not one of "
                f"PEtab, which are "
                f"'{', '.join(d.value for d in PriorDistribution)}'."
            ) from err
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(
            self, "parameters", tuple(float(value) for value in self.parameters)
        )

    def log_density(self, value: float, lower: float, upper: float) -> float:
        """Get the log density of the prior at a value.

        The distributions are the ones of `petab`, which the PEtab test suite
        is calculated with.

        Args:
            value: the value of the parameter.
            lower: the lower bound of the parameter.
            upper: the upper bound of the parameter.

        Returns:
            The log density of the prior truncated at the bounds, `-inf`
            outside of them.

        Raises:
            ValueError: if the parameters do not fit the distribution.
        """
        from petab.v2 import Parameter as PetabParameter

        distribution = PetabParameter(
            id="p",
            lb=lower,
            ub=upper,
            estimate=True,
            prior_distribution=self.distribution.value,
            prior_parameters=list(self.parameters),
        ).prior_dist
        if distribution is None:
            raise ValueError(f"The prior '{self}' has no distribution.")
        density = float(distribution.pdf(value))
        return math.log(density) if density > 0.0 else -math.inf

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "distribution": self.distribution.value,
            "parameters": list(self.parameters),
        }


@dataclass(frozen=True)
class NoiseParameter:
    """A parameter of a noise formula which is not an entity of a model.

    Attributes:
        pid: id of the parameter.
        value: nominal value, which the log-likelihood uses unless the
            parameter set it is evaluated at has a value for `pid`.
        estimate: whether the problem the noise model was read from estimates
            the parameter. `sbmlsim` does not estimate it, the flag and the
            bounds are kept so that the problem is written as it was read.
        lower_bound: lower bound of the estimation, `None` if there is none.
        upper_bound: upper bound of the estimation, `None` if there is none.
    """

    pid: str
    value: float
    estimate: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class NoiseModel:
    """The noise of the measurements of a fit mapping.

    The noise model does not enter the cost of a fit, which is a weighted
    least squares fit. It is what the log-likelihood of a problem is
    calculated with, see `sbmlsim.fit.petab_v2.likelihood`.

    Attributes:
        formula: the noise formula in the math of PEtab over the selections of
            roadrunner, e.g. `0.05`, `sd` or `sigma_a + 0.1 * sd`, see
            `ObservableModel`. Its symbols are the `placeholders`, the
            `observable` and the `symbols`, i.e. selections of the simulation
            or `parameters`.
        distribution: distribution of the noise.
        placeholders: symbols of the formula which have a value per
            measurement.
        placeholder_values: for every measurement the values of the
            placeholders, in the order of the measurements of the mapping. A
            value is a number or a formula of the selections.
        parameters: the parameters of the formula and of the placeholder
            values with their nominal value, for a symbol which the
            simulation does not select.
        observable: symbol of the formula which stands for the simulation,
            `None` if the formula has none.
    """

    formula: str
    distribution: NoiseDistribution = NoiseDistribution.NORMAL
    placeholders: tuple[str, ...] = ()
    placeholder_values: tuple[tuple[float | str, ...], ...] = ()
    parameters: tuple[NoiseParameter, ...] = ()
    observable: str | None = None

    def __post_init__(self) -> None:
        """Coerce the fields and check them.

        The distribution is coerced to the enum and the sequences to tuples,
        so a noise model which is given a string and lists compares, hashes
        and is written like any other.

        Raises:
            ValueError: if the formula is empty, if the distribution is not
                one of PEtab, or if a measurement has more or fewer values
                than the noise model has placeholders.
        """
        if not str(self.formula).strip():
            raise ValueError("The noise formula of a noise model must not be empty.")
        try:
            distribution = NoiseDistribution(self.distribution)
        except ValueError as err:
            raise ValueError(
                f"The noise distribution '{self.distribution}' is not one of "
                f"PEtab, which are "
                f"'{', '.join(d.value for d in NoiseDistribution)}'."
            ) from err
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "placeholders", tuple(self.placeholders))
        object.__setattr__(
            self,
            "placeholder_values",
            tuple(tuple(values) for values in self.placeholder_values),
        )
        object.__setattr__(self, "parameters", tuple(self.parameters))

        for k, values in enumerate(self.placeholder_values):
            if len(values) != len(self.placeholders):
                raise ValueError(
                    f"The noise formula '{self.formula}' has the placeholders "
                    f"'{list(self.placeholders)}', but the measurement '{k}' "
                    f"has the values '{list(values)}'."
                )

    @functools.cached_property
    def formula_model(self) -> ObservableModel:
        """Get the noise formula with its placeholders as a formula model.

        Raises:
            ValueError: if the formula or a placeholder value is not valid
                math.
        """
        return ObservableModel(
            formula=self.formula,
            placeholders=self.placeholders,
            placeholder_values=self.placeholder_values,
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        """Get the symbols the noise reads besides placeholders and observable.

        These are selections of the simulation or `parameters`, sorted.
        """
        return tuple(s for s in self.formula_model.symbols if s != self.observable)

    def select(self, mask: np.ndarray) -> NoiseModel:
        """Get the noise model of a selection of the measurements.

        Args:
            mask: whether a measurement is selected, one entry per
                measurement.

        Returns:
            The noise model with the placeholder values of the selected
            measurements.
        """
        if not self.placeholders:
            return self
        return replace(
            self,
            placeholder_values=tuple(
                values
                for values, keep in zip(self.placeholder_values, mask, strict=True)
                if keep
            ),
        )


@dataclass(frozen=True)
class ObservableModel:
    """An observable which is a formula of the selections of a simulation.

    The formula is the math of PEtab over the selections of roadrunner, i.e.
    `[S]` is the concentration of a species and `S` its amount, e.g.
    `scale * ([pSTAT] + [ppSTAT])`, see `sbmlsim.simulator.formula`. A
    placeholder is a symbol which has a value per measurement, e.g. an offset
    which differs between the measurements of one observable; its value is a
    number or a formula of the selections, typically a parameter of the model.
    The fit evaluates the observable at the data, so the values of the
    placeholders belong to the data points of the reference of the mapping.

    Attributes:
        formula: the formula of the observable.
        placeholders: symbols of the formula which have a value per
            measurement.
        placeholder_values: for every measurement the values of the
            placeholders, in the order of the reference data of the mapping.
            A value is a number or a formula of the selections.
        unit: unit of the value of the formula, the reference data is
            converted into it.
    """

    formula: str
    placeholders: tuple[str, ...] = ()
    placeholder_values: tuple[tuple[float | str, ...], ...] = ()
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        """Coerce the fields and check them.

        Raises:
            ValueError: if the formula or the formula of a placeholder value
                is not valid math, or if a measurement has more or fewer
                values than the observable has placeholders.
        """
        object.__setattr__(self, "placeholders", tuple(self.placeholders))
        object.__setattr__(
            self,
            "placeholder_values",
            tuple(
                tuple(
                    value if isinstance(value, str) else float(value)
                    for value in values
                )
                for values in self.placeholder_values
            ),
        )
        for k, values in enumerate(self.placeholder_values):
            if len(values) != len(self.placeholders):
                raise ValueError(
                    f"The observable '{self.formula}' has the placeholders "
                    f"'{list(self.placeholders)}', but the measurement '{k}' "
                    f"has the values '{list(values)}'."
                )
        compile_formula(self.formula)
        for value in self._formula_values():
            compile_formula(value)

    def _formula_values(self) -> set[str]:
        """Get the placeholder values which are formulas."""
        return {
            value
            for values in self.placeholder_values
            for value in values
            if isinstance(value, str)
        }

    @property
    def symbols(self) -> tuple[str, ...]:
        """Get the selections the observable reads, sorted.

        These are the symbols of the formula which are not placeholders and
        the symbols of the placeholder values which are formulas.
        """
        names = set(compile_formula(self.formula).symbols) - set(self.placeholders)
        for value in self._formula_values():
            names.update(compile_formula(value).symbols)
        return tuple(sorted(names))

    def select(self, mask: np.ndarray) -> ObservableModel:
        """Get the observable of a selection of the measurements.

        Args:
            mask: whether a measurement is selected, one entry per
                measurement.

        Returns:
            The observable with the placeholder values of the selected
            measurements.
        """
        if not self.placeholders:
            return self
        return replace(
            self,
            placeholder_values=tuple(
                values
                for values, keep in zip(self.placeholder_values, mask, strict=True)
                if keep
            ),
        )

    @functools.cached_property
    def _columns(self) -> tuple[tuple[np.ndarray, dict[str, np.ndarray]], ...]:
        """Get the values of every placeholder over the measurements.

        Returns:
            Per placeholder the numbers, `NaN` where the value is a formula,
            and the indices of the measurements of every formula.
        """
        columns: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
        for j in range(len(self.placeholders)):
            values = [row[j] for row in self.placeholder_values]
            numbers = np.array(
                [np.nan if isinstance(v, str) else v for v in values], dtype=float
            )
            formulas: dict[str, list[int]] = {}
            for i, value in enumerate(values):
                if isinstance(value, str):
                    formulas.setdefault(value, []).append(i)
            columns.append(
                (numbers, {f: np.asarray(ix, dtype=int) for f, ix in formulas.items()})
            )
        return tuple(columns)

    def evaluate(self, values: Mapping[str, np.ndarray], size: int) -> np.ndarray:
        """Evaluate the observable at the measurements.

        Args:
            values: the values of the `symbols` at the measurements, one array
                of `size` values per symbol.
            size: the number of measurements.

        Returns:
            The value of the observable at every measurement.

        Raises:
            ValueError: if the observable has placeholders and not the values
                of `size` measurements.
        """
        if self.placeholders and len(self.placeholder_values) != size:
            raise ValueError(
                f"The observable '{self.formula}' has the placeholder values of "
                f"'{len(self.placeholder_values)}' measurements, but '{size}' "
                f"measurements are evaluated."
            )
        arguments = dict(values)
        for placeholder, (numbers, formulas) in zip(
            self.placeholders, self._columns, strict=True
        ):
            column = numbers.copy()
            for value, indices in formulas.items():
                compiled = compile_formula(value)
                column[indices] = compiled.evaluate_array(
                    [np.asarray(values[s])[indices] for s in compiled.symbols],
                    size=indices.size,
                )
            arguments[placeholder] = column
        compiled = compile_formula(self.formula)
        return compiled.evaluate_array(
            [arguments[s] for s in compiled.symbols], size=size
        )


class FitMappingCollection:
    """The fit mappings of a simulation experiment which a fit uses together.

    A collection selects mappings of one `SimulationExperiment`, says what a fit
    does with them (`MappingKind`) and how they are weighted. It is the unit a
    fit is defined in and the unit a PEtab problem is built from: the mappings
    of a collection which share a simulation are one experiment of PEtab, see
    `sbmlsim.fit.petab_v2`.
    """

    def __init__(
        self,
        experiment: type[SimulationExperiment],
        mappings: list[str] | None = None,
        sid: str | None = None,
        weights: float | list[float] | None = None,
        use_mapping_weights: bool = False,
        fit_parameters: dict[str, list[FitParameter]] | None = None,
        exclude: bool = False,
        kind: MappingKind = MappingKind.TRAINING,
    ):
        """Initialize simulation experiment used in a fitting.

        The weights must be updated according to the mappings.

        Args:
            experiment: simulation experiment class the mappings belong to.
            mappings: mappings to use from the experiment. `None` or an empty list
                uses all mappings of the experiment, they are resolved in
                `OptimizationProblem.initialize`, see `resolve_mappings`.
            sid: id of the collection, which names the experiments of a PEtab
                problem. The name of the simulation experiment class and the
                kind of its mappings by default, i.e. `Beermann1976_training`.
            weights: weight of the mappings, the larger the value the larger the
                weight. A single value is used for all mappings.
            use_mapping_weights: use the weights of the mappings instead of `weights`.
            fit_parameters: LOCAL parameters only changed in this simulation
                experiment.
            exclude: flag to exclude the experiment from the fitting.
            kind: what a fit does with these mappings, i.e., whether they are
                fitted, only evaluated or not used at all.

        Raises:
            ValueError: for duplicate mappings or unsupported local fit parameters.
        """
        self.experiment_class: type[SimulationExperiment] = experiment
        self.sid: str = sid or f"{experiment.__name__}_{kind.value}"
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
        self.kind: MappingKind = kind

        if fit_parameters:
            # TODO: implement
            raise ValueError(
                "Local parameters in FitMappingCollection not yet supported, see "
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
                    f"a FitMappingCollection or the weights of the FitMappings are used via "
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

        A `FitMappingCollection` without mappings uses all fit mappings of its simulation
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
    def reduce(
        mapping_collections: Iterable[FitMappingCollection],
    ) -> list[FitMappingCollection]:
        """Combine the fit mappings of the FitMappingCollections of the same experiment.

        Experiments are combined per simulation experiment and `MappingKind`,
        so that the training and the validation data stay apart. The mappings
        and their weights are concatenated, the inputs are not modified.

        Raises:
            ValueError: if experiments of the same class cannot be combined, i.e.,
                they use different weighting or repeat a mapping.
        """
        reduced: dict[tuple[str, MappingKind], FitMappingCollection] = {}
        for collection in mapping_collections:
            sid = collection.experiment_class.__name__
            # the training and the validation data of an experiment stay apart
            key = (sid, collection.kind)
            if key not in reduced:
                reduced[key] = FitMappingCollection(
                    experiment=collection.experiment_class,
                    mappings=list(collection.mappings),
                    weights=list(collection.weights),  # ty: ignore[invalid-argument-type]
                    use_mapping_weights=collection.use_mapping_weights,
                    exclude=collection.exclude,
                    kind=collection.kind,
                )
                continue

            red_exp = reduced[key]
            if red_exp.use_mapping_weights != collection.use_mapping_weights:
                raise ValueError(
                    f"FitMappingCollections of '{sid}' cannot be combined, they differ in "
                    f"'use_mapping_weights'."
                )
            duplicates = set(red_exp.mappings) & set(collection.mappings)
            if duplicates:
                raise ValueError(
                    f"FitMappingCollections of '{sid}' cannot be combined, the mappings "
                    f"'{sorted(duplicates)}' occur in both."
                )
            red_exp.mappings = red_exp.mappings + list(collection.mappings)
            red_exp._weights = red_exp._weights + list(collection.weights)
            red_exp.exclude = red_exp.exclude and collection.exclude

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
            "*** FitMappingCollection ***",
            f"experiment: {self.experiment_class.__name__}",
            f"mappings: {self.mappings}",
            f"weights: {self.weights}",
            f"use_mapping_weights: {self.use_mapping_weights}",
            f"kind: {self.kind.value}",
            f"fit_parameters: {self.fit_parameters}",
        ]
        return "\n".join(info)


@dataclass(kw_only=True)
class MappingMetaData:
    """Metadata for mapping.

    Applications derive their metadata from this class to describe the curve,
    e.g., the tissue, the route, the dosing or the health of the group of a
    study. The metadata describes the data, not what a fit does with it: how a
    curve is used is the `MappingKind` of the `FitMappingCollection` which selects it.

    The fields are keyword only, so that a subclass can add fields without a
    default.
    """

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
        noise: NoiseModel | None = None,
        observable_model: ObservableModel | None = None,
    ):
        """Initialize FitMapping.

        To use the weight in the fit mapping the `use_mapping_weights` flag
        must be set on the FitMappingCollection.

        Args:
            experiment: simulation experiment of the mapping.
            reference: reference data (mostly experimental data).
            observable: observable in the model.
            weight: weight of the fit mapping, the count of the reference data
                is used if no weight is given.
            metadata: metadata of the mapping.
            noise: noise model of the measurements, which the log-likelihood
                of the problem uses. Without one the noise is normal with the
                standard deviation of the reference data, see
                `sbmlsim.fit.petab_v2.likelihood.default_noise_model`.
            observable_model: the observable as a formula of the selections
                of the task of `observable`, evaluated at the reference data.
                The `y` of `observable` is then the name of the observable and
                not a selection, and the unit of the observable is the unit of
                the observable model.
        """
        self.experiment = experiment
        self.reference = reference
        self.observable = observable
        self._weight = weight
        self.metadata = metadata
        self.noise = noise
        self.observable_model = observable_model

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
        target: str | None = None,
        mappings: Any = None,
        scale: ParameterScaleType | str | None = None,
        prior: Prior | Mapping[str, Any] | None = None,
    ):
        """Initialize FitParameter.

        Args:
            pid: id of the estimated parameter. It is the name in the parameter
                vector, in the parameter sets and in the profiles; it is the id
                of the entity of the model unless `target` says otherwise.
            start_value: initial value for the fitting.
            lower_bound: lower bound for the fitting.
            upper_bound: upper bound for the fitting.
            unit: unit of the parameter, the model unit is assumed if not given.
            target: entity of the model the value is written to. `None` means
                the parameter is the entity, i.e. `target_id` is `pid`. Several
                parameters write one target when each of them selects a part of
                the data, see `mappings`.
            mappings: `MappingFilter` or an iterable of them which select the
                fit mappings the parameter applies to; a mapping passes when it
                passes every filter. `None` applies the parameter everywhere.
                A selector is a callable and is not serialized: it must be a
                module level function, because the workers of a parallel fit
                unpickle the parameters.
            scale: space the optimizer searches the parameter in, or its name.
                `None` is the `parameter_scale` of the `FitSettings`. A
                parameter which is negative or zero, e.g. a weight of a
                network, is searched on the linear scale.
            prior: the prior of the parameter or its dictionary, see `Prior`.
                `None` is the uniform prior over the bounds. The objective of
                a fit does not use it, `log_prior` evaluates it.

        Raises:
            ValueError: if the bounds or the start value are inconsistent, if
                a value is not a number, if the scale is not a scale, or if
                the target is the prefix of an external target alone.
        """
        for key, value in (("lower_bound", lower_bound), ("upper_bound", upper_bound)):
            if value is None or np.isnan(value):
                raise ValueError(
                    f"FitParameter '{pid}': the '{key}' is '{value}', which is "
                    f"not a number. A parameter without a bound has an "
                    f"infinite one."
                )
        if start_value is not None and not np.isfinite(start_value):
            raise ValueError(
                f"FitParameter '{pid}': the start value '{start_value}' is not "
                f"a finite number."
            )
        if isinstance(scale, str):
            if scale not in ParameterScaleType.__members__:
                raise ValueError(
                    f"FitParameter '{pid}': the scale '{scale}' is not one of "
                    f"{list(ParameterScaleType.__members__)}."
                )
            scale = ParameterScaleType[scale]
        if scale is not None and not isinstance(scale, ParameterScaleType):
            raise ValueError(
                f"FitParameter '{pid}': the scale '{scale}' is not a "
                f"`ParameterScaleType`."
            )
        if target == EXTERNAL_PREFIX:
            raise ValueError(
                f"FitParameter '{pid}': the target '{target}' names nothing, an "
                f"external target is '{EXTERNAL_PREFIX}<id>'."
            )
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
        self.target = target
        self.mappings = mappings
        self.scale: ParameterScaleType | None = scale
        if isinstance(prior, Mapping):
            prior = Prior(
                distribution=prior["distribution"], parameters=prior["parameters"]
            )
        self.prior: Prior | None = prior
        if unit is None:
            logger.warning(
                "No unit provided for FitParameter '%s', assuming model units.",
                self.pid,
            )

    @property
    def target_id(self) -> str:
        """Get the entity of the model the value is written to."""
        return self.target if self.target is not None else self.pid

    @property
    def is_external(self) -> bool:
        """Check whether the parameter is not an entity of the model.

        The target of such a parameter has the prefix `EXTERNAL_PREFIX`. The
        fit writes no change for it, the derived changes of the problem read
        its value.
        """
        return self.target_id.startswith(EXTERNAL_PREFIX)

    @property
    def entity_id(self) -> str:
        """Get the target without the prefix of an external target."""
        return self.target_id.removeprefix(EXTERNAL_PREFIX)

    @property
    def is_versioned(self) -> bool:
        """Check whether the parameter applies to a part of the data only."""
        return self.mappings is not None

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
            and self.target_id == other.target_id
            and self.scale == other.scale
            and self.prior == other.prior
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
            "target": self.target,
            "scale": None if self.scale is None else self.scale.name,
            "prior": None if self.prior is None else self.prior.to_dict(),
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


def describe_array(
    label: str,
    elements: int,
    members: Sequence[FitParameter],
    values: Sequence[float | None],
) -> str:
    """Describe an array of elements, e.g. of a network, in one line of text.

    The console and the text reports show an array in place of its elements,
    because a network has hundreds of them.

    Args:
        label: what the array is, e.g. `net1.layer1.weight`.
        elements: number of elements of the array, estimated or not.
        members: the parameters of the fit which are elements of the array. A
            versioned element has one parameter per version.
        values: the value of every member, e.g. its start value, `None` for a
            member without one.

    Returns:
        The number of estimated elements, the minimum, the maximum and the
        norm of the values of the members and the bounds when the members
        agree on them.
    """
    estimated = len({p.entity_id for p in members})
    noun = "element" if elements == 1 else "elements"
    text = f"{label}: {estimated} of {elements} {noun} estimated"
    if not members:
        return text
    array = np.asarray(values, dtype=float)
    lower = {p.lower_bound for p in members}
    upper = {p.upper_bound for p in members}
    text += (
        f", min {array.min():.4g}, max {array.max():.4g}, "
        f"norm {np.linalg.norm(array):.4g}"
    )
    if len(lower) == 1 and len(upper) == 1:
        text += f", bounds [{lower.pop():.4g}, {upper.pop():.4g}]"
    return text


class FitData:
    """Data used in a fit.

    This is either data from a dataset, a simulation results from
    a task or functional data, i.e. calculated from other data.
    """

    def __init__(
        self,
        experiment: Any,  # SimulationExperiment (avoid circular import)
        xid: str | None,
        yid: str,
        xid_sd: str | None = None,
        xid_se: str | None = None,
        yid_sd: str | None = None,
        yid_se: str | None = None,
        count: int | str | None = None,
        dataset: str | None = None,
        task: str | None = None,
        function: str | None = None,
        sel: Mapping[str, Any] | None = None,
    ):
        """Initialize FitData.

        Args:
            experiment: simulation experiment the data belongs to.
            xid: index of the x data, `None` for data without x, i.e. a value per
                simulation.
            yid: index of the y data.
            xid_sd: index of the standard deviation of the x data.
            xid_se: index of the standard error of the x data.
            yid_sd: index of the standard deviation of the y data.
            yid_se: index of the standard error of the y data.
            count: number of subjects, either an integer or a column of the dataset.
            dataset: id of the dataset the data comes from.
            task: id of the task the data comes from.
            function: id of the function the data is calculated with.
            sel: labels of the dimensions of a task, or column values of the rows
                of a dataset, for every data of the fit data, see `Data`.

        Raises:
            ValueError: if `count` is set without a dataset or has a wrong type.
        """
        self.experiment = experiment
        self.dset_id = dataset
        self.task_id = task
        self.function = function
        self.sel: dict[str, Any] = dict(sel) if sel else {}
        self.count: int | None = self._resolve_count(count)

        # actual data
        self.x: Data | None = self._data(xid) if xid is not None else None
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
            sel=self.sel or None,
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
        count_data = Data(
            index=count,
            dataset=self.dset_id,
            task=self.task_id,
            sel=self.sel or None,
        )
        counts = count_data.get_data(self.experiment)
        counts_unique = np.unique(np.asarray(counts.values))
        if counts_unique.size > 1:
            logger.warning("count is not unique for dataset: '%s'", counts)
        return int(counts_unique[0])

    def __str__(self) -> str:
        """Get the source, the x and y and the selection of the data."""
        xid = self.x.index if self.x is not None else None
        return (
            f"FitData(experiment={self.experiment.__class__.__name__} "
            f"dset_id={self.dset_id} task_id={self.task_id} "
            f"function={self.function} xid={xid} yid={self.y.index} "
            f"sel={self.sel})"
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
                setattr(
                    result,
                    key,
                    to_quantity(d.get_data(self.experiment), self.experiment.ureg),
                )

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
