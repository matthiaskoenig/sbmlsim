"""Uncertain values: a nominal value with its distribution.

An `Uncertain` is given wherever a value is given, e.g. the value of a
`Change`, of a pre-initialization change or the prior of a `FitParameter`. A
simulation simulates its `nominal` value, so declaring a value uncertain
changes nothing until it is propagated, see `sbmlsim.uncertainty.propagate`.

The spread is given as reported:

| given | distribution | nominal |
|---|---|---|
| `sd=` or `cv=` of a positive value (`shape="auto"`) | log-normal with the mean and the SD matched | the value, the mean |
| `sd=`/`cv=` with `shape="normal"`, or a value which may be negative | normal | the mean |
| `low=`/`high=` with them | the distribution above, truncated | the mean |
| `Uncertain.range(low, high)` | uniform (`log=True`: log-uniform) | the (geometric) midpoint |
| `Uncertain.interval(low, high, level)` | log-normal (positive) or normal with these quantiles | the median |
| `Uncertain.se(mean, se, n)` | the SD `se * sqrt(n)`, as the fit converts it | the mean |
| `distribution=` | the given one | the value |

A log-normal with the arithmetic mean `m` and the coefficient of variation
`cv` has the median `m / sqrt(1 + cv^2)` and the standard deviation of the
logarithm `sqrt(ln(1 + cv^2))` (Limpert et al. 2001). It is the default of a
positive quantity, e.g. a dose, a body weight, a volume or a rate, which a
normal distribution of a large CV would give negative values.

`kind` is `uncertainty` (epistemic: what is not known about the value) or
`variability` (aleatory: the value differs between subjects); it matters for
a nested propagation and for virtual trials. `name` makes two uses one random
variable: one object used twice, or two objects of one name, are one variable
of an analysis; different objects are independent unless a `Covariance`
correlates them.

Limpert E, Stahel WA, Abbt M (2001). Log-normal distributions across the
sciences: keys and clues. BioScience 51:341-352.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pint.errors import DimensionalityError
from scipy import integrate, stats

from sbmlsim.simulation.definition import _decode, _encode
from sbmlsim.simulation.sampling.distributions import (
    Distribution,
    LogNormal,
    LogUniform,
    Normal,
    Truncated,
    Uniform,
    distribution_from_dict,
)
from sbmlsim.units import Quantity, ureg

#: the kinds of an uncertain value
KINDS: tuple[str, ...] = ("uncertainty", "variability")

#: a number, in the unit of its target, or a quantity
Number = float | Quantity


def _unit(value: Number) -> str | None:
    """Get the unit of a value, `None` for a number."""
    return str(value.units) if isinstance(value, Quantity) else None


def _in_unit(value: Number, unit: str | None, what: str) -> float:
    """Get the magnitude of a value in a unit; a number is in it already.

    Raises:
        ValueError: if a quantity cannot be converted into the unit.
    """
    if isinstance(value, Quantity):
        if unit is None:
            if value.dimensionless:
                return float(value.to("dimensionless").magnitude)
            raise ValueError(f"The {what} {value} has a unit, the value has none.")
        try:
            return float(value.to(unit).magnitude)
        except DimensionalityError as err:
            raise ValueError(
                f"The {what} {value} does not fit the unit {unit} of the value."
            ) from err
    return float(value)


def _with_unit(magnitude: float, unit: str | None) -> Number:
    """Get a magnitude as a quantity of a unit, a number without."""
    return ureg.Quantity(magnitude, unit) if unit else float(magnitude)


def _check_kind(kind: str) -> None:
    """Check the kind of an uncertain value.

    Raises:
        ValueError: if it is none of `KINDS`.
    """
    if kind not in KINDS:
        raise ValueError(f"The kind {kind!r} of an uncertain value is none of {KINDS}.")


@dataclass(frozen=True, eq=False)
class Uncertain:
    """A value with its distribution, see the module.

    Attributes:
        nominal: the value a simulation simulates, a quantity or a number.
        distribution: the distribution of the sampler.
        shape: `log-normal`, `normal`, `uniform`, `log-uniform` or `given`.
        kind: `uncertainty` or `variability`.
        name: the name of the random variable, `None` for the object itself.
    """

    nominal: Number
    distribution: Distribution
    shape: str
    kind: Literal["uncertainty", "variability"] = "uncertainty"
    name: str | None = None

    def __init__(
        self,
        value: Number,
        *,
        sd: Number | None = None,
        cv: float | None = None,
        low: Number | None = None,
        high: Number | None = None,
        shape: Literal["auto", "normal", "log-normal"] = "auto",
        distribution: Distribution | None = None,
        kind: Literal["uncertainty", "variability"] = "uncertainty",
        name: str | None = None,
    ) -> None:
        """Create an uncertain value from its mean and spread, see the module.

        Args:
            value: the nominal value, the mean of `sd` and `cv`.
            sd: the standard deviation.
            cv: the coefficient of variation, `sd / |mean|`.
            low: the lower bound of a truncation.
            high: the upper bound of a truncation.
            shape: `auto` (log-normal for a positive value, else normal),
                `normal` or `log-normal`.
            distribution: the distribution, instead of `sd` and `cv`.
            kind: `uncertainty` or `variability`.
            name: the name of the random variable.

        Raises:
            ValueError: unless exactly one of `sd`, `cv` and `distribution`
                is given, for a log-normal of a value which is not positive,
                or for units which do not fit.
        """
        if sum(x is not None for x in (sd, cv, distribution)) != 1:
            raise ValueError(
                f"The uncertain value {value} needs exactly one of sd, cv and "
                f"distribution."
            )
        _check_kind(kind)
        unit = _unit(value)
        if distribution is not None:
            resolved, chosen = distribution, "given"
        else:
            mean = _in_unit(value, unit, "value")
            spread = (
                _in_unit(sd, unit, "standard deviation")
                if sd is not None
                else float(cv) * abs(mean)  # ty: ignore[invalid-argument-type]
            )
            if not spread > 0.0:
                raise ValueError(
                    f"The uncertain value {value} needs a positive spread, not "
                    f"{spread}."
                )
            chosen = (
                ("log-normal" if mean > 0.0 else "normal") if shape == "auto" else shape
            )
            if chosen == "log-normal":
                if not mean > 0.0:
                    raise ValueError(f"The log-normal value {value} must be positive.")
                ratio = spread / mean
                resolved = LogNormal(
                    median=_with_unit(mean / math.sqrt(1.0 + ratio**2), unit), cv=ratio
                )
            elif chosen == "normal":
                resolved = Normal(
                    mean=_with_unit(mean, unit), sd=_with_unit(spread, unit)
                )
            else:
                raise ValueError(
                    f"The shape {shape!r} is none of 'auto', 'normal' and 'log-normal'."
                )
        if low is not None or high is not None:
            resolved = Truncated(
                resolved,
                lower=None
                if low is None
                else _with_unit(_in_unit(low, unit, "bound"), unit),
                upper=None
                if high is None
                else _with_unit(_in_unit(high, unit, "bound"), unit),
            )
        object.__setattr__(self, "nominal", value)
        object.__setattr__(self, "distribution", resolved)
        object.__setattr__(self, "shape", chosen)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)

    @classmethod
    def _of(
        cls,
        nominal: Number,
        distribution: Distribution,
        shape: str,
        kind: Literal["uncertainty", "variability"],
        name: str | None,
    ) -> Uncertain:
        """Create an uncertain value of its fields."""
        _check_kind(kind)
        value = cls.__new__(cls)
        object.__setattr__(value, "nominal", nominal)
        object.__setattr__(value, "distribution", distribution)
        object.__setattr__(value, "shape", shape)
        object.__setattr__(value, "kind", kind)
        object.__setattr__(value, "name", name)
        return value

    @classmethod
    def range(
        cls,
        low: Number,
        high: Number,
        *,
        log: bool = False,
        kind: Literal["uncertainty", "variability"] = "uncertainty",
        name: str | None = None,
    ) -> Uncertain:
        """Create an uncertain value which lies between a minimum and a maximum.

        Args:
            low: the minimum.
            high: the maximum, larger than the minimum.
            log: log-uniform instead of uniform, for a positive range of
                orders of magnitude.
            kind: `uncertainty` or `variability`.
            name: the name of the random variable.

        Returns:
            The uncertain value, its nominal the midpoint (the geometric one
            for `log`).

        Raises:
            ValueError: for a range which is empty or, for `log`, not positive.
        """
        unit = _unit(low)
        lo, hi = _in_unit(low, unit, "minimum"), _in_unit(high, unit, "maximum")
        if not hi > lo:
            raise ValueError(f"The range [{low}, {high}] is empty.")
        if log:
            if not lo > 0.0:
                raise ValueError(f"The log range [{low}, {high}] must be positive.")
            nominal = math.sqrt(lo * hi)
            distribution: Distribution = LogUniform(
                lower=_with_unit(lo, unit), upper=_with_unit(hi, unit)
            )
        else:
            nominal = 0.5 * (lo + hi)
            distribution = Uniform(
                lower=_with_unit(lo, unit), upper=_with_unit(hi, unit)
            )
        return cls._of(
            _with_unit(nominal, unit),
            distribution,
            "log-uniform" if log else "uniform",
            kind,
            name,
        )

    @classmethod
    def interval(
        cls,
        low: Number,
        high: Number,
        *,
        level: float = 0.95,
        shape: Literal["auto", "normal", "log-normal"] = "auto",
        kind: Literal["uncertainty", "variability"] = "uncertainty",
        name: str | None = None,
    ) -> Uncertain:
        """Create an uncertain value of a reported interval.

        The quantiles `(1 - level) / 2` and `(1 + level) / 2` of the
        distribution are the bounds of the interval: of a log-normal for a
        positive interval (`shape="auto"`), of a normal otherwise.

        Args:
            low: the lower bound of the interval.
            high: the upper bound.
            level: the level of the interval, e.g. 0.95.
            shape: `auto`, `normal` or `log-normal`.
            kind: `uncertainty` or `variability`.
            name: the name of the random variable.

        Returns:
            The uncertain value, its nominal the median.

        Raises:
            ValueError: for an empty interval, a level outside (0, 1) or a
                log-normal interval which is not positive.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"The level {level} of an interval must be in (0, 1).")
        unit = _unit(low)
        lo, hi = _in_unit(low, unit, "lower bound"), _in_unit(high, unit, "upper bound")
        if not hi > lo:
            raise ValueError(f"The interval [{low}, {high}] is empty.")
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        chosen = ("log-normal" if lo > 0.0 else "normal") if shape == "auto" else shape
        if chosen == "log-normal":
            if not lo > 0.0:
                raise ValueError(
                    f"The log-normal interval [{low}, {high}] must be positive."
                )
            mu = 0.5 * (math.log(lo) + math.log(hi))
            sigma = (math.log(hi) - math.log(lo)) / (2.0 * z)
            median = math.exp(mu)
            distribution: Distribution = LogNormal(
                median=_with_unit(median, unit), cv=math.sqrt(math.expm1(sigma**2))
            )
            nominal = median
        else:
            nominal = 0.5 * (lo + hi)
            distribution = Normal(
                mean=_with_unit(nominal, unit),
                sd=_with_unit((hi - lo) / (2.0 * z), unit),
            )
        return cls._of(_with_unit(nominal, unit), distribution, chosen, kind, name)

    @classmethod
    def se(cls, mean: Number, se: Number, n: float, **kwargs: Any) -> Uncertain:
        """Create an uncertain value of a mean, its standard error and the count.

        The standard deviation is `se * sqrt(n)`, as a fit converts the
        standard errors of its data.

        Args:
            mean: the mean.
            se: the standard error of the mean.
            n: the count, the number of subjects.
            kwargs: the other arguments of `Uncertain`.

        Returns:
            The uncertain value.

        Raises:
            ValueError: for a count which is not positive.
        """
        if not n > 0:
            raise ValueError(f"The count {n} must be positive.")
        return cls(mean, sd=se * math.sqrt(n), **kwargs)

    @property
    def unit(self) -> str | None:
        """Get the unit of the value, `None` for a number."""
        return _unit(self.nominal)

    def _moments(self) -> tuple[float, float]:
        """Get the mean and the standard deviation in the unit of the value."""
        d, unit = self.distribution, self.unit

        def magnitude(x: Any) -> float:
            return _in_unit(x, unit, "value") if isinstance(x, Quantity) else float(x)

        if isinstance(d, Normal) and d.mean is not None and d.sd is not None:
            return magnitude(d.mean), magnitude(d.sd)
        if isinstance(d, LogNormal) and d.median is not None and d.cv is not None:
            median, sigma2 = magnitude(d.median), math.log1p(d.cv**2)
            mean = median * math.exp(sigma2 / 2.0)
            return mean, mean * d.cv

        # the moments of the inverse of the distribution function over (0, 1)
        def value(u: float) -> float:
            x = d.ppf(np.array([u]))
            if isinstance(x, Quantity):
                return float(
                    np.asarray(x.to(unit).magnitude if unit else x.magnitude)[0]
                )
            return float(np.asarray(x)[0])

        mean = integrate.quad(value, 0.0, 1.0, limit=200)[0]
        second = integrate.quad(lambda u: value(u) ** 2, 0.0, 1.0, limit=200)[0]
        return mean, math.sqrt(max(second - mean**2, 0.0))

    def mean(self) -> Number:
        """Get the mean of the distribution, in the unit of the value."""
        return _with_unit(self._moments()[0], self.unit)

    def sd(self) -> Number:
        """Get the standard deviation of the distribution, in the unit of the value."""
        return _with_unit(self._moments()[1], self.unit)

    def to_dict(self) -> dict[str, Any]:
        """Get the uncertain value as a dictionary of JSON types."""
        return {
            "nominal": _encode(self.nominal),
            "distribution": self.distribution.to_dict(),
            "shape": self.shape,
            "kind": self.kind,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Uncertain:
        """Create an uncertain value of `to_dict`."""
        return cls._of(
            _decode(d["nominal"]),
            distribution_from_dict(d["distribution"]),
            d["shape"],
            d.get("kind", "uncertainty"),
            d.get("name"),
        )

    def __repr__(self) -> str:
        """Get the nominal value, the shape and the name."""
        name = f", name={self.name!r}" if self.name else ""
        return f"Uncertain({self.nominal}, {self.shape}{name})"
