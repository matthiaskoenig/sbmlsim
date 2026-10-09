"""The distributions of the sampler.

A distribution gives the values of a target at probabilities `u` of the unit
cube, `ppf(u, reference)`, the inverse of its cumulative distribution
function, so every design (random draws, Latin hypercubes, the designs of
SALib) works with every marginal. Its numbers are floats in the unit of the
target in the model or quantities; a plain number next to a quantity is in the
unit of the quantity. A distribution without a location (`Uniform(relative=)`,
`LogUniform(factor=)`, `Normal(cv=)`, `LogNormal(cv=)`) is relative to the
reference of its target, the value the model gives it after the
pre-initialization, see `sbmlsim.simulation.sampling.references`.

The values are a quantity when the distribution has a unit (of a quantity or
of its reference) and plain floats otherwise. The probabilities 0 and 1 of an
unbounded distribution are clipped to `EPS` from the bounds of the unit
interval, so its values are finite.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, override

import numpy as np
from scipy import stats

from sbmlsim.simulation.definition import _encode
from sbmlsim.units import Quantity, ureg

#: the distance of the clipped probabilities 0 and 1 from the bounds
EPS: float = 1e-12

#: a number in the unit of a target in the model, or a quantity
Number = float | Quantity


def _unit_of(*values: Any) -> str | None:
    """Get the unit of the first quantity, `None` without one."""
    return next((str(v.units) for v in values if isinstance(v, Quantity)), None)


def _magnitude(value: Number, unit: str | None) -> float:
    """Get a number in a unit; a plain number is in it already."""
    if isinstance(value, Quantity):
        return float(value.to(unit).magnitude) if unit else float(value.magnitude)
    return float(value)


def _values(magnitudes: np.ndarray, unit: str | None) -> np.ndarray | Quantity:
    """Give the values their unit, plain floats without one."""
    return ureg.Quantity(magnitudes, unit) if unit else magnitudes


def _split(values: np.ndarray | Quantity) -> tuple[np.ndarray, str | None]:
    """Split values into their magnitudes and their unit."""
    if isinstance(values, Quantity):
        return np.asarray(values.magnitude, dtype=float), str(values.units)
    return np.asarray(values, dtype=float), None


def _reference(
    reference: Number | None, distribution: Distribution
) -> tuple[float, str | None]:
    """Split a reference into its number and its unit.

    Raises:
        ValueError: without a reference.
    """
    if reference is None:
        raise ValueError(
            f"{distribution!r} is relative to the reference of its target; give the "
            f"design the model (model=) to read it."
        )
    if isinstance(reference, Quantity):
        return float(reference.magnitude), str(reference.units)
    return float(reference), None


def _positive(value: float, distribution: Distribution) -> float:
    """Check that a location of a logarithmic distribution is positive.

    Raises:
        ValueError: if it is not.
    """
    if not value > 0.0:
        raise ValueError(f"{distribution!r} needs a positive location, not {value}.")
    return value


def _clip(u: Any) -> np.ndarray:
    """Clip probabilities into the open unit interval."""
    return np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)


class Distribution(ABC):
    """A distribution of the values of a target, see the module."""

    @property
    def is_relative(self) -> bool:
        """Check whether the location is the reference of the target."""
        return False

    @abstractmethod
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""

    @abstractmethod
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values at probabilities of the unit interval.

        Args:
            u: the probabilities, an array.
            reference: the reference of the target, which a relative
                distribution needs.

        Returns:
            The values, a quantity where the distribution has a unit.

        Raises:
            ValueError: if a relative distribution has no reference or its
                reference does not fit.
        """

    @abstractmethod
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values, the inverse of `ppf`.

        Args:
            x: the values, numbers in the unit of the distribution.
            reference: the reference of the target.

        Returns:
            The probabilities.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform in `[lower, upper]`, or in the reference times `[1 - relative, 1 + relative]`.

    Attributes:
        lower: the lower bound.
        upper: the upper bound, not below `lower`.
        relative: the relative half width around the reference.
    """

    lower: Number | None = None
    upper: Number | None = None
    relative: float | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: unless either both bounds or a positive `relative`
                are given, or if the bounds are no interval.
        """
        if self.relative is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError(f"{self!r} has bounds or a relative width, not both.")
            if not self.relative > 0.0:
                raise ValueError(f"{self!r} needs a positive relative width.")
        elif self.lower is None or self.upper is None:
            raise ValueError(
                f"{self!r} needs a lower and an upper bound, or relative=."
            )
        else:
            lower, upper = self._bounds(None)
            if lower > upper:
                raise ValueError(f"{self!r}: the lower bound is above the upper one.")

    @property
    @override
    def is_relative(self) -> bool:
        """Check whether the bounds are relative to the reference."""
        return self.relative is not None

    def _bounds(self, reference: Number | None) -> tuple[float, float]:
        """Get the bounds in the unit of the distribution."""
        if self.relative is not None:
            value, _ = _reference(reference, self)
            a, b = value * (1.0 - self.relative), value * (1.0 + self.relative)
            return min(a, b), max(a, b)
        if self.lower is None or self.upper is None:
            raise ValueError(f"{self!r} needs a lower and an upper bound.")
        unit = _unit_of(self.lower, self.upper)
        return _magnitude(self.lower, unit), _magnitude(self.upper, unit)

    @override
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        if self.relative is not None:
            return _reference(reference, self)[1]
        return _unit_of(self.lower, self.upper)

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values, `lower + u (upper - lower)`."""
        lower, upper = self._bounds(reference)
        return _values(
            lower + np.asarray(u, dtype=float) * (upper - lower), self._unit(reference)
        )

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values."""
        lower, upper = self._bounds(reference)
        x = np.asarray(x, dtype=float)
        if upper == lower:
            return (x >= lower).astype(float)
        return np.clip((x - lower) / (upper - lower), 0.0, 1.0)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        return {
            "type": "Uniform",
            "lower": _encode(self.lower),
            "upper": _encode(self.upper),
            "relative": self.relative,
        }


@dataclass(frozen=True)
class LogUniform(Distribution):
    """Uniform in the decadic logarithm, in `[lower, upper]` or in the reference divided and multiplied by `factor`.

    Attributes:
        lower: the positive lower bound.
        upper: the upper bound, not below `lower`.
        factor: the factor above 1 around the (positive) reference.
    """

    lower: Number | None = None
    upper: Number | None = None
    factor: float | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: unless either positive bounds or a `factor` above 1
                are given.
        """
        if self.factor is not None:
            if self.lower is not None or self.upper is not None:
                raise ValueError(f"{self!r} has bounds or a factor, not both.")
            if not self.factor > 1.0:
                raise ValueError(f"{self!r} needs a factor above 1.")
        elif self.lower is None or self.upper is None:
            raise ValueError(f"{self!r} needs a lower and an upper bound, or factor=.")
        else:
            lower, upper = self._bounds(None)
            if not 0.0 < lower <= upper:
                raise ValueError(f"{self!r} needs positive bounds with lower <= upper.")

    @property
    @override
    def is_relative(self) -> bool:
        """Check whether the bounds are relative to the reference."""
        return self.factor is not None

    def _bounds(self, reference: Number | None) -> tuple[float, float]:
        """Get the bounds in the unit of the distribution."""
        if self.factor is not None:
            value = _positive(_reference(reference, self)[0], self)
            return value / self.factor, value * self.factor
        if self.lower is None or self.upper is None:
            raise ValueError(f"{self!r} needs a lower and an upper bound.")
        unit = _unit_of(self.lower, self.upper)
        return _magnitude(self.lower, unit), _magnitude(self.upper, unit)

    @override
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        if self.factor is not None:
            return _reference(reference, self)[1]
        return _unit_of(self.lower, self.upper)

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values, `10 ** (log10(lower) + u (log10(upper) - log10(lower)))`."""
        lower, upper = self._bounds(reference)
        log_lower, log_upper = np.log10(lower), np.log10(upper)
        values = np.power(
            10.0, log_lower + np.asarray(u, dtype=float) * (log_upper - log_lower)
        )
        return _values(values, self._unit(reference))

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values."""
        lower, upper = self._bounds(reference)
        x = np.asarray(x, dtype=float)
        if upper == lower:
            return (x >= lower).astype(float)
        positive = x > 0.0
        log_x = np.log10(np.where(positive, x, 1.0))
        log_lower, log_upper = np.log10(lower), np.log10(upper)
        p = np.clip((log_x - log_lower) / (log_upper - log_lower), 0.0, 1.0)
        return np.where(positive, p, 0.0)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        return {
            "type": "LogUniform",
            "lower": _encode(self.lower),
            "upper": _encode(self.upper),
            "factor": self.factor,
        }


@dataclass(frozen=True)
class Normal(Distribution):
    """Normal with a standard deviation or a coefficient of variation.

    Without a `mean` the mean is the reference of the target.

    Attributes:
        mean: the mean.
        sd: the positive standard deviation.
        cv: the positive coefficient of variation, `sd = cv |mean|`.
    """

    mean: Number | None = None
    sd: Number | None = None
    cv: float | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: unless exactly one positive `sd` or `cv` is given.
        """
        if (self.sd is None) == (self.cv is None):
            raise ValueError(f"{self!r} needs exactly one of sd and cv.")
        if self.sd is not None and not _magnitude(self.sd, None) > 0.0:
            raise ValueError(f"{self!r} needs a positive standard deviation.")
        if self.cv is not None and not self.cv > 0.0:
            raise ValueError(f"{self!r} needs a positive coefficient of variation.")

    @property
    @override
    def is_relative(self) -> bool:
        """Check whether the mean is the reference."""
        return self.mean is None

    @override
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        if self.mean is None:
            return _unit_of(self.sd) or _reference(reference, self)[1]
        return _unit_of(self.mean, self.sd)

    def _parameters(self, reference: Number | None) -> tuple[float, float]:
        """Get the mean and the standard deviation in the unit of the distribution."""
        unit = self._unit(reference)
        if self.mean is None:
            mean = _reference(reference, self)[0]
        else:
            mean = _magnitude(self.mean, unit)
        sd = (
            _magnitude(self.sd, unit)
            if self.sd is not None
            else (self.cv or 0.0) * abs(mean)
        )
        if not sd > 0.0:
            raise ValueError(
                f"{self!r} has a standard deviation of {sd} at the mean {mean}."
            )
        return mean, sd

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values, `mean + sd z(u)`."""
        mean, sd = self._parameters(reference)
        return _values(mean + sd * stats.norm.ppf(_clip(u)), self._unit(reference))

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values."""
        mean, sd = self._parameters(reference)
        return np.asarray(stats.norm.cdf((np.asarray(x, dtype=float) - mean) / sd))

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        return {
            "type": "Normal",
            "mean": _encode(self.mean),
            "sd": _encode(self.sd),
            "cv": self.cv,
        }


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal with a median and a coefficient of variation.

    Without a `median` the median is the reference of the target.

    Attributes:
        median: the positive median.
        cv: the positive coefficient of variation of the distribution, the
            sigma of the logarithm is `sqrt(ln(1 + cv^2))`.
    """

    median: Number | None = None
    cv: float | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: without a positive `cv` or with a median which is
                not positive.
        """
        if self.cv is None or not self.cv > 0.0:
            raise ValueError(f"{self!r} needs a positive coefficient of variation.")
        if self.median is not None and not _magnitude(self.median, None) > 0.0:
            raise ValueError(f"{self!r} needs a positive median.")

    @property
    @override
    def is_relative(self) -> bool:
        """Check whether the median is the reference."""
        return self.median is None

    @override
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        if self.median is None:
            return _reference(reference, self)[1]
        return _unit_of(self.median)

    def _parameters(self, reference: Number | None) -> tuple[float, float]:
        """Get the median and the sigma of the logarithm."""
        if self.median is None:
            median = _positive(_reference(reference, self)[0], self)
        else:
            median = _magnitude(self.median, self._unit(reference))
        return median, math.sqrt(math.log1p((self.cv or 0.0) ** 2))

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values, `median exp(sigma z(u))`."""
        median, sigma = self._parameters(reference)
        return _values(
            median * np.exp(sigma * stats.norm.ppf(_clip(u))), self._unit(reference)
        )

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values."""
        median, sigma = self._parameters(reference)
        x = np.asarray(x, dtype=float)
        positive = x > 0.0
        p = stats.norm.cdf(np.log(np.where(positive, x, 1.0) / median) / sigma)
        return np.where(positive, p, 0.0)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        return {"type": "LogNormal", "median": _encode(self.median), "cv": self.cv}


@dataclass(frozen=True)
class Truncated(Distribution):
    """Another distribution restricted to `[lower, upper]`.

    The probabilities are mapped onto the part of the inner distribution
    between the bounds, so the values keep its shape. The bounds are numbers
    in the unit of the inner distribution or quantities.

    Attributes:
        distribution: the inner distribution.
        lower: the lower bound, unbounded below without.
        upper: the upper bound, unbounded above without.
    """

    distribution: Distribution
    lower: Number | None = None
    upper: Number | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: without a bound, with bounds which are no interval or
                an interval without probability.
        """
        if self.lower is None and self.upper is None:
            raise ValueError(f"{self!r} needs a lower or an upper bound.")
        if self.lower is not None and self.upper is not None:
            unit = _unit_of(self.lower, self.upper)
            if not _magnitude(self.lower, unit) < _magnitude(self.upper, unit):
                raise ValueError(
                    f"{self!r}: the lower bound must be below the upper one."
                )
        if not self.distribution.is_relative:
            self._probabilities(None)

    @property
    @override
    def is_relative(self) -> bool:
        """Check whether the inner distribution is relative."""
        return self.distribution.is_relative

    @override
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        return self.distribution._unit(reference) or _unit_of(self.lower, self.upper)

    def _bounds(self, reference: Number | None) -> tuple[float | None, float | None]:
        """Get the bounds in the unit of the distribution."""
        unit = self._unit(reference)
        lower = None if self.lower is None else _magnitude(self.lower, unit)
        upper = None if self.upper is None else _magnitude(self.upper, unit)
        return lower, upper

    def _probabilities(self, reference: Number | None) -> tuple[float, float]:
        """Get the probabilities of the bounds under the inner distribution.

        Raises:
            ValueError: if there is no probability between the bounds.
        """
        lower, upper = self._bounds(reference)
        low = (
            0.0
            if lower is None
            else float(self.distribution.cdf(np.array(lower), reference))
        )
        high = (
            1.0
            if upper is None
            else float(self.distribution.cdf(np.array(upper), reference))
        )
        if not high > low:
            raise ValueError(
                f"{self!r}: the interval between the bounds has no probability."
            )
        return low, high

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values, the inner `ppf` at `F(lower) + u (F(upper) - F(lower))`."""
        low, high = self._probabilities(reference)
        values, unit = _split(
            self.distribution.ppf(
                low + np.asarray(u, dtype=float) * (high - low), reference
            )
        )
        lower, upper = self._bounds(reference)
        return _values(np.clip(values, lower, upper), unit or self._unit(reference))

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values."""
        low, high = self._probabilities(reference)
        p = (self.distribution.cdf(x, reference) - low) / (high - low)
        return np.clip(p, 0.0, 1.0)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        return {
            "type": "Truncated",
            "distribution": self.distribution.to_dict(),
            "lower": _encode(self.lower),
            "upper": _encode(self.upper),
        }


@dataclass(frozen=True)
class Empirical(Distribution):
    """The sorted values of a sample, each with the same probability.

    Attributes:
        values: the values, a sequence of numbers or a quantity array; sorted
            and kept as a tuple of floats.
        unit: the unit of the values, `None` for plain numbers.
    """

    values: Sequence[float] | Quantity
    unit: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Sort the values and keep their unit.

        Raises:
            ValueError: without values.
        """
        unit = _unit_of(self.values)
        magnitudes = np.asarray(
            self.values.magnitude if isinstance(self.values, Quantity) else self.values,
            dtype=float,
        )
        if magnitudes.size == 0:
            raise ValueError("Empirical needs at least one value.")
        object.__setattr__(
            self, "values", tuple(float(v) for v in np.sort(magnitudes.ravel()))
        )
        object.__setattr__(self, "unit", unit)

    @override
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        return self.unit

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the values, the one at the position `floor(u n)`."""
        values = np.asarray(self.values, dtype=float)
        n = len(values)
        index = np.clip(np.floor(np.asarray(u, dtype=float) * n).astype(int), 0, n - 1)
        return _values(values[index], self.unit)

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values, the share of the values up to them."""
        values = np.asarray(self.values, dtype=float)
        return np.searchsorted(values, np.asarray(x, dtype=float), side="right") / len(
            values
        )

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        values = list(self.values)  # ty: ignore[invalid-argument-type]
        return {
            "type": "Empirical",
            "values": _encode(ureg.Quantity(values, self.unit))
            if self.unit
            else values,
        }


@dataclass(frozen=True)
class Fixed(Distribution):
    """A single value, for a target which a design keeps at a value.

    Attributes:
        value: the value.
    """

    value: Number

    @override
    def _unit(self, reference: Number | None) -> str | None:
        """Get the unit of the values."""
        return _unit_of(self.value)

    @override
    def ppf(self, u: Any, reference: Number | None = None) -> np.ndarray | Quantity:
        """Get the value for every probability."""
        unit = self._unit(reference)
        return _values(np.full(np.shape(u), _magnitude(self.value, unit)), unit)

    @override
    def cdf(self, x: Any, reference: Number | None = None) -> np.ndarray:
        """Get the probabilities of values, 1 from the value on."""
        return (
            np.asarray(x, dtype=float) >= _magnitude(self.value, self._unit(reference))
        ).astype(float)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the distribution as a dictionary of JSON types."""
        return {"type": "Fixed", "value": _encode(self.value)}
