"""Observables: what a scan computes from every simulation.

An observable reads the selections of roadrunner (`S` amount, `[S]`
concentration, `time`, parameters, compartments, reactions) and other
observables by id:

- `Formula(id, formula, unit=None)`: the math of PEtab with the reductions
  over the time of a simulation, `max`, `min`, `mean` and `at`, see
  `sbmlsim.simulator.formula`; a timecourse, or a value per simulation when
  it reads only values per simulation, e.g. `max([glc])`;
- `PK(id, selection, *, dose=None, route=None, options=None, parameters=None)`:
  the non-compartmental analysis of pkpdutils of a timecourse, a value per
  simulation `<id>.<parameter>` for every parameter, e.g. `hctz.cmax`;
- `Custom(id, function, unit, *, symbols, kind=SCALAR)`: a function of a
  module, called with the time points and the values of its symbols of every
  simulation.

The definitions are frozen and pickle. `Simulator.run(model, scan,
observables)` compiles them against the models of the run, see
`sbmlsim.simulator.observables`, and evaluates them in the workers on the
native solution of every simulation.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, override

import numpy as np

from sbmlsim.simulation.scan import RESERVED
from sbmlsim.units import Quantity, ureg

if TYPE_CHECKING:
    from pkpdutils import NCAOptions

#: the id of an observable
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ObservableKind(StrEnum):
    """What an observable is per simulation."""

    #: a value at every time point
    TIMECOURSE = "timecourse"
    #: one value
    SCALAR = "scalar"


def _check_id(name: str, kind: str) -> None:
    """Check the id of an observable.

    Raises:
        ValueError: if the id is no identifier or a name of the result.
    """
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise ValueError(
            f"The id of a {kind} observable is an identifier of letters, digits "
            f"and underscores, not {name!r}."
        )
    if name in RESERVED:
        raise ValueError(
            f"The id '{name}' is a name of the result ({sorted(RESERVED)}), choose "
            f"another one."
        )


def _check_unit(unit: str, name: str) -> None:
    """Check that pint reads a unit.

    Raises:
        ValueError: if pint does not read it.
    """
    try:
        ureg.parse_units(unit)
    except Exception as err:
        raise ValueError(
            f"The unit '{unit}' of the observable '{name}' is no unit: {err}"
        ) from err


def _check_names(names: Any, what: str, name: str) -> tuple[str, ...]:
    """Check a sequence of unique names.

    Raises:
        TypeError: if the names are a string.
        ValueError: if a name is empty or appears twice.
    """
    if isinstance(names, str):
        raise TypeError(
            f"The {what} of the observable '{name}' are a sequence of names, not "
            f"the string {names!r}."
        )
    names = tuple(names)
    if len(set(names)) != len(names) or not all(
        isinstance(n, str) and n for n in names
    ):
        raise ValueError(
            f"The {what} of the observable '{name}' are unique names, not {names}."
        )
    return names


def _check_function(function: Any, name: str) -> None:
    """Check that a function is defined at the top level of a module.

    Raises:
        TypeError: if it is not callable.
        ValueError: if it is a lambda, a closure or not found in its module.
    """
    if not callable(function):
        raise TypeError(
            f"The function of the observable '{name}' is not callable: {function!r}."
        )
    qualname = getattr(function, "__qualname__", "")
    module = sys.modules.get(getattr(function, "__module__", None) or "")
    found: Any = module
    for part in qualname.split("."):
        found = getattr(found, part, None)
    if module is None or "<" in qualname or found is not function:
        raise ValueError(
            f"The function '{qualname}' of the observable '{name}' is no function "
            f"of a module but a lambda or a closure; define it at the top level "
            f"of a module, so that it pickles for the workers of a scan."
        )


@dataclass(frozen=True)
class Observable:
    """An observable, the base of `Formula`, `PK` and `Custom`.

    Attributes:
        id: the id, which the result and the other observables use.
    """

    id: str

    @property
    def reads(self) -> tuple[str, ...]:
        """Get the selections and the observables it reads."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types, the provenance of a result."""
        raise NotImplementedError


@dataclass(frozen=True)
class Formula(Observable):
    """A formula of the math of PEtab with reductions over time, see the module.

    Attributes:
        formula: the formula over selections and the ids of other observables.
        unit: the unit of the values: the unit the formula has is converted
            into it; where pint cannot derive one, e.g. for `piecewise` or a
            comparison, it is the unit of the formula, which then needs it.
    """

    formula: str
    unit: str | None = None

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            ValueError: if the id, the formula or the unit is not valid.
        """
        _check_id(self.id, "Formula")
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ValueError(
                f"The formula of the observable '{self.id}' is a non-empty string, "
                f"not {self.formula!r}."
            )
        if self.unit is not None:
            _check_unit(self.unit, self.id)
        # the simulator package imports the definitions
        from sbmlsim.simulator.formula import reduce_formula

        reduce_formula(self.formula)

    @property
    @override
    def reads(self) -> tuple[str, ...]:
        """Get the selections and the observables the formula reads."""
        from sbmlsim.simulator.formula import reduce_formula

        return reduce_formula(self.formula).symbols

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types."""
        return {
            "type": "Formula",
            "id": self.id,
            "formula": self.formula,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class PK(Observable):
    """The non-compartmental analysis of a timecourse with pkpdutils.

    Every parameter pkpdutils derives is a value per simulation
    `<id>.<parameter>`, e.g. `hctz.cmax`, `hctz.auc_inf_obs`, `hctz.thalf`,
    with the unit pkpdutils gives it, and `<id>.flags` the flags of the
    analysis (`pkpdutils.NCAFlag`).

    Attributes:
        selection: the timecourse, a selection of the model or the id of a
            timecourse observable, e.g. a concentration.
        dose: the target of the model which the simulations dose: its values
            in the plan of every point are the doses and the times of these
            values their times, so the dose of a dimension and the changes of
            a multiple dosing are found; a quantity is a fixed dose at the
            start; `None` analyses without a dose and leaves out the
            parameters which need one.
        route: the route of the dose, a `pkpdutils.Route` or its name, e.g.
            `"oral"` or `"iv_bolus"`; a dose needs it.
        options: the options of the analysis, a `pkpdutils.NCAOptions`.
        parameters: the parameters to keep, every parameter by default.
    """

    selection: str
    dose: str | Quantity | None = field(default=None, kw_only=True)
    route: str | None = field(default=None, kw_only=True)
    options: NCAOptions | None = field(default=None, kw_only=True)
    parameters: tuple[str, ...] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            TypeError: if the dose, the route or the parameters have the wrong type.
            ValueError: if the id or the selection is not valid, a fixed dose is
                no single amount, a dose has no route or a parameter repeats.
        """
        _check_id(self.id, "PK")
        if not isinstance(self.selection, str) or not self.selection:
            raise ValueError(
                f"The timecourse of the observable '{self.id}' is a selection or "
                f"an observable, not {self.selection!r}."
            )
        if self.dose is not None:
            if isinstance(self.dose, Quantity):
                if np.ndim(self.dose.magnitude) != 0:
                    raise ValueError(
                        f"The fixed dose of the observable '{self.id}' is one "
                        f"amount, not {self.dose}."
                    )
            elif not isinstance(self.dose, str) or not self.dose:
                raise TypeError(
                    f"The dose of the observable '{self.id}' is a target of the "
                    f"model or a quantity, not {self.dose!r}."
                )
            if self.route is None:
                raise ValueError(
                    f"The dose of the observable '{self.id}' needs its route, e.g. "
                    f"route='oral'."
                )
        if self.route is not None and not isinstance(self.route, str):
            raise TypeError(
                f"The route of the observable '{self.id}' is a name of a route of "
                f"pkpdutils, not {self.route!r}."
            )
        if self.parameters is not None:
            parameters = _check_names(self.parameters, "parameters", self.id)
            if not parameters:
                raise ValueError(f"The observable '{self.id}' keeps no parameters.")
            object.__setattr__(self, "parameters", parameters)

    @property
    @override
    def reads(self) -> tuple[str, ...]:
        """Get the timecourse the analysis reads."""
        return (self.selection,)

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types."""
        dose: Any = self.dose
        if isinstance(self.dose, Quantity):
            dose = {"value": float(self.dose.magnitude), "unit": str(self.dose.units)}
        return {
            "type": "PK",
            "id": self.id,
            "selection": self.selection,
            "dose": dose,
            "route": None if self.route is None else str(self.route),
            "options": None
            if self.options is None
            else self.options.model_dump(mode="json"),
            "parameters": None if self.parameters is None else list(self.parameters),
        }


@dataclass(frozen=True)
class Custom(Observable):
    """A function of a module, called once per simulation.

    `function(time, values)` gets the time points of a simulation, without
    padding, and `values`, its symbols to their values: an array of the length
    of `time` for a timecourse and a float for a value per simulation. It
    returns a float for a value per simulation and an array of the length of
    `time` for a timecourse.

    Attributes:
        function: the function, defined at the top level of a module so that
            it pickles for the workers of a scan.
        unit: the unit of its values.
        symbols: the selections and observables it reads.
        kind: a value per simulation (`SCALAR`) or a timecourse.
    """

    function: Callable[[np.ndarray, dict[str, Any]], Any]
    unit: str
    symbols: tuple[str, ...] = field(kw_only=True)
    kind: ObservableKind = field(default=ObservableKind.SCALAR, kw_only=True)

    def __post_init__(self) -> None:
        """Check the definition.

        Raises:
            TypeError: if the function is not callable or the symbols are a string.
            ValueError: if the id, the function, the unit or the symbols are not
                valid.
        """
        _check_id(self.id, "Custom")
        _check_function(self.function, self.id)
        if not isinstance(self.unit, str):
            raise TypeError(
                f"The unit of the observable '{self.id}' is a string, not {self.unit!r}."
            )
        _check_unit(self.unit, self.id)
        object.__setattr__(
            self, "symbols", _check_names(self.symbols, "symbols", self.id)
        )
        object.__setattr__(self, "kind", ObservableKind(self.kind))

    @property
    @override
    def reads(self) -> tuple[str, ...]:
        """Get the symbols the function reads."""
        return self.symbols

    @override
    def to_dict(self) -> dict[str, Any]:
        """Get the definition as JSON types; the function as `module:name`."""
        function = (
            f"{getattr(self.function, '__module__', '')}:"
            f"{getattr(self.function, '__qualname__', '')}"
        )
        return {
            "type": "Custom",
            "id": self.id,
            "function": function,
            "unit": self.unit,
            "symbols": list(self.symbols),
            "kind": str(self.kind),
        }
