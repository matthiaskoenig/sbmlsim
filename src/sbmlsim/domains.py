"""The domain of a quantity of a model, from distrib.

A domain is the interval of values a model accepts for one of its quantities: a
fraction lies in `[0, 1]`, a volume or a rate in `[0, inf)`. A model states it
as an uncertainty of the SBML package distrib named `DOMAIN` whose single
`UncertSpan` is of the type `range`, on the parameter, compartment or species it
constrains; the factory of a model writes it (the PBPK library of pkpdsim writes
one for every constant, compartment and species of its models), `domain_of` and
`domains` read it back.

A domain also states which values of its interval the model accepts (`Values`):
every real number, the integers of the interval (an index, a count or a class)
or the two truth values of a flag. An integer or boolean domain states it with a
further child of the uncertainty, an `externalParameter` whose `definitionURL` is
the XSD datatype of the value set; a continuous domain states none, so a document
written before the value sets keeps its meaning.

The domain of a quantity is not the uncertainty of a value: the confidence
interval of a literature value and the distribution of a fitted parameter
(`sbmlsim.distrib`, `sbmlsim.simulation.sampling.parameter_distributions`) are
statements of their own, in their own uncertainty elements, which sit next to the
domain on the same element. The reader finds the domain by its name and the type
of its span and ignores every other uncertainty.

A model which states no domain says nothing about its quantities: `domain_of`
gives `None`, `domains` leaves the quantity out, and there is no default. What
reads a domain is the sampler, which keeps the values of a design inside it
(`sbmlsim.simulation.sampling.local` clamps the steps of a local design).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import overload

import libsbml

__all__ = [
    "DOMAIN",
    "VALUE_SET_URL",
    "Domain",
    "Values",
    "domain_of",
    "domains",
]

logger = logging.getLogger(__name__)

DOMAIN = "domain"
"""The name of the uncertainty which carries a domain."""


class Values(StrEnum):
    """The values inside the interval of a domain which a model accepts.

    A flag is kept apart from an integer domain of the same interval on
    purpose: a tool which offers a flag as a checkbox and a class as a spinner
    has to know which of the two it has.
    """

    CONTINUOUS = "continuous"
    """Every real number of the interval, the default."""
    INTEGER = "integer"
    """The integers of the interval: an index, a count, a stage or a class."""
    BOOLEAN = "boolean"
    """The integers 0 and 1 as the truth values: a flag which is off or on."""


VALUE_SET_URL: dict[Values, str] = {
    Values.INTEGER: "http://www.w3.org/2001/XMLSchema#integer",
    Values.BOOLEAN: "http://www.w3.org/2001/XMLSchema#boolean",
}
"""The `definitionURL` which states a value set, the XSD datatype of the values;
`CONTINUOUS` has none, a continuous domain states no such child."""


def _bound(x: float) -> str:
    """Render a bound: an integral value without the trailing `.0`, else its text."""
    return str(int(x)) if math.isfinite(x) and x == int(x) else str(x)


@dataclass(frozen=True)
class Domain:
    """The closed interval of the accepted values of a quantity, a bound may be infinite.

    Attributes:
        lower: the lower bound.
        upper: the upper bound.
        values: which values of the interval are accepted.
    """

    lower: float
    upper: float
    values: Values = Values.CONTINUOUS

    @overload
    def clip(self, lower: float, upper: float) -> tuple[float, float]: ...

    @overload
    def clip(self, lower: float, upper: None = None) -> float: ...

    def clip(
        self, lower: float, upper: float | None = None
    ) -> tuple[float, float] | float:
        """Move a value or an interval which lies outside the domain onto it.

        An interval has each of its bounds limited to the interval of the
        domain, each on its own side, so an interval whose bounds are reversed
        is returned as it is; a caller may rely on that
        (`sampling.set_relative_bounds` builds an inverted pair for a negative
        value on purpose and `sampling.distributions` refuses it by name
        later). An interval of an integer or boolean domain which is not
        reversed has each finite bound widened onto a value the domain
        contains, the lower one down and the upper one up, which is logged: a
        fractional span of a switch or of a count is a modelling error, not a
        rounding one. A single value is limited to the interval and, for an
        integer or boolean domain, rounded to the nearest integer the domain
        contains.

        Args:
            lower: the value, or the lower bound of an interval.
            upper: the upper bound of the interval, `None` for a single value.

        Returns:
            The interval with each bound limited to the domain, or the value
            limited to it. An infinite value is returned at the infinite bound
            of a domain which has one, so the result lies inside the interval
            but is not necessarily a value `contains` accepts.

        Raises:
            ValueError: if a single value is not a number.
        """
        if upper is not None:
            low, high = max(lower, self.lower), min(upper, self.upper)
            if self.values is Values.CONTINUOUS or low > high:
                return low, high
            return self._onto_values(low, high)
        if math.isnan(lower):
            raise ValueError("a value which is not a number cannot be clipped")
        x = min(max(lower, self.lower), self.upper)
        if self.values is Values.CONTINUOUS or not math.isfinite(x):
            return x
        x = float(round(x))
        if x < self.lower:
            x = float(math.ceil(self.lower))
        if x > self.upper:
            x = float(math.floor(self.upper))
        return x

    def _onto_values(self, low: float, high: float) -> tuple[float, float]:
        """Widen the bounds of an interval onto the values this domain holds.

        The lower bound moves down and the upper bound moves up, so the
        interval which was asked for stays inside the one which is returned
        and the result is never empty, and neither bound leaves the interval
        of the domain. A bound which is not finite stays as it is, since no
        integral value lies nearer to it than it does itself.

        Args:
            low: the lower bound, inside the domain.
            high: the upper bound, inside the domain and not below `low`.

        Returns:
            The bounds, each on a value of the domain.
        """
        wide_low, wide_high = low, high
        if math.isfinite(low):
            wide_low = float(math.floor(low))
            if wide_low < self.lower:
                wide_low = float(math.ceil(self.lower))
        if math.isfinite(high):
            wide_high = float(math.ceil(high))
            if wide_high > self.upper:
                wide_high = float(math.floor(self.upper))
        if (wide_low, wide_high) != (low, high):
            logger.warning(
                "The domain %s holds integral values only, so the interval "
                "(%s, %s) is widened to (%s, %s). A fractional span of a "
                "switch or of a count is a modelling error, not a rounding "
                "one.",
                self,
                low,
                high,
                wide_low,
                wide_high,
            )
        return wide_low, wide_high

    def contains(self, value: float) -> bool:
        """Get whether a value is one the domain accepts, the bounds included.

        An integer or boolean domain accepts only an integral value. No domain
        accepts a value which is not finite: an infinite bound is open
        (`[0, inf)`), so `inf` is no value of it, and `nan` is no number at all.

        Args:
            value: the value.

        Returns:
            Whether the domain accepts it.
        """
        if not math.isfinite(value) or not self.lower <= value <= self.upper:
            return False
        return self.values is Values.CONTINUOUS or value == int(value)

    def __str__(self) -> str:
        """Get the interval notation; a finite bound is inclusive, an infinite one open.

        The value set follows when it is not continuous: `[0, 3] integer`.
        """
        left = "[" if math.isfinite(self.lower) else "("
        right = "]" if math.isfinite(self.upper) else ")"
        text = f"{left}{_bound(self.lower)}, {_bound(self.upper)}{right}"
        if self.values is Values.CONTINUOUS:
            return text
        return f"{text} {self.values.value}"


def _values_of(uncertainty: libsbml.Uncertainty) -> Values:
    """Get the value set a domain states, by the URL of its child.

    Args:
        uncertainty: the uncertainty which carries the domain.

    Returns:
        The value set; `CONTINUOUS` for a domain with no such child.
    """
    by_url = {url: values for values, url in VALUE_SET_URL.items()}
    for j in range(uncertainty.getNumUncertParameters()):
        child: libsbml.UncertParameter = uncertainty.getUncertParameter(j)
        if child.getType() != libsbml.DISTRIB_UNCERTTYPE_EXTERNALPARAMETER:
            continue
        values = by_url.get(child.getDefinitionURL())
        if values is not None:
            return values
    return Values.CONTINUOUS


def domain_of(sbase: libsbml.SBase) -> Domain | None:
    """Read the domain of an element of a model.

    Args:
        sbase: a parameter, compartment or species.

    Returns:
        The domain, or `None` if the element states none. A bound which is not
        set reads as the infinity on its side.
    """
    plugin = sbase.getPlugin("distrib")
    if plugin is None:
        return None
    for i in range(plugin.getNumUncertainties()):
        uncertainty: libsbml.Uncertainty = plugin.getUncertainty(i)
        if uncertainty.getName() != DOMAIN:
            continue
        values = _values_of(uncertainty)
        for j in range(uncertainty.getNumUncertParameters()):
            span = uncertainty.getUncertParameter(j)
            if not isinstance(span, libsbml.UncertSpan):
                continue
            if span.getType() != libsbml.DISTRIB_UNCERTTYPE_RANGE:
                continue
            lower = span.getValueLower() if span.isSetValueLower() else -math.inf
            upper = span.getValueUpper() if span.isSetValueUpper() else math.inf
            return Domain(float(lower), float(upper), values)
    return None


def domains(sbml: str | Path) -> dict[str, Domain]:
    """Read the domains of the parameters, compartments and species of a model.

    Only the top-level model is read, the elements of a submodel of a
    hierarchical document are not in the result; the flat SBML of a model, which
    a simulation runs (`sbmlsim.model.model_sbml.SBMLModel.sbml`), has them.

    Args:
        sbml: the SBML of the model itself or the path of its file.

    Returns:
        The domain of every element which states one, by its id; empty for a
        model which states none, and the silence about an element is no
        statement that it has no domain.

    Raises:
        FileNotFoundError: for a path which is no file.
        ValueError: for an SBML which is no model.
    """
    content = sbml if isinstance(sbml, str) and sbml.lstrip().startswith("<") else None
    if content is None and not Path(sbml).is_file():
        raise FileNotFoundError(f"The SBML file '{sbml}' does not exist.")
    document: libsbml.SBMLDocument = (
        libsbml.readSBMLFromString(content)
        if content is not None
        else libsbml.readSBMLFromFile(str(sbml))
    )
    model: libsbml.Model | None = document.getModel()
    if model is None:
        where = "The SBML" if content is not None else f"'{sbml}'"
        raise ValueError(f"{where} is no SBML model, it states no domains.")
    result: dict[str, Domain] = {}
    for elements in (
        model.getListOfParameters(),
        model.getListOfCompartments(),
        model.getListOfSpecies(),
    ):
        for element in elements:
            domain = domain_of(element)
            if domain is not None:
                result[element.getId()] = domain
    return result
