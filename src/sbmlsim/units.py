"""Manage units and units conversions.

Used for model and data unit conversions.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator, MutableMapping
from pathlib import Path

import libsbml
import numpy as np
from sbmlutils.io import read_sbml

# Disable Pint's old fallback behavior (must come before importing Pint)
os.environ["PINT_ARRAY_PROTOCOL_FALLBACK"] = "0"


from typing import ClassVar

import pint
from pint import UnitRegistry
from pint.errors import DimensionalityError, UndefinedUnitError
from pint.facets.plain import PlainQuantity

#: type of the quantities of a unit registry; `pint.Quantity` is a subclass, the
#: registry itself creates `PlainQuantity` instances
Quantity = PlainQuantity

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

logger = logging.getLogger(__name__)

UdictType = dict[str, str]


def _create_registry() -> UnitRegistry:
    """Create the unit registry of the package with the units sbmlsim defines.

    The units of a model are not defined in it: a model stores the expression
    of every unit definition (`Units.udef_to_str`), so two models which use
    one id for different units never share a definition.
    """
    registry = pint.UnitRegistry(on_redefinition="ignore")
    registry.define("none = count")
    registry.define("item = count")
    registry.define("percent = 0.01*count")
    # FIXME: the international unit is specific to a substance, these are the
    # ones of insulin
    registry.define("IU = 0.0347 * mg")
    registry.define("IU_per_ml = 0.0347 * mg/ml")
    return registry


#: the unit registry of the package, every model, experiment and fit uses it
ureg: UnitRegistry = _create_registry()

#: the quantities of the registry of the package, `Q(10, "mg")`
Q = ureg.Quantity


def _package_registry() -> UnitRegistry:
    """Get the unit registry of the package, for a function with a `ureg` argument."""
    return ureg


class UnitsInformation(MutableMapping):
    """Storage of units information.

    Used for models or datasets.
    """

    def __init__(self, udict: UdictType, ureg: UnitRegistry, *args, **kwargs):
        """Initialize UnitsInformation.

        Behaves like a dict which allows to lookup units by id.
        """
        self.udict: UdictType = udict
        self.ureg: UnitRegistry = ureg
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key: str) -> str:
        """Get item."""
        return self.udict[self._keytransform(key)]

    def __setitem__(self, key: str, value: str) -> None:
        """Set item."""
        self.udict[self._keytransform(key)] = value

    def __delitem__(self, key) -> None:
        """Delete item."""
        del self.udict[self._keytransform(key)]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.udict)

    def __len__(self) -> int:
        """Get length."""
        return len(self.udict)

    def _keytransform(self, key: str) -> str:
        """Transform key."""
        return key

    def __str__(self) -> str:
        """Get string."""
        items = [f"{k}: {self[k]}" for k in self.keys()]
        return "\n".join(items)

    @property
    def Q_(self):
        """Get quantity for generating quantities."""
        return self.ureg.Quantity

    @staticmethod
    def from_sbml(
        sbml: str | Path, ureg: UnitRegistry | None = None
    ) -> UnitsInformation:
        """Get pint UnitsInformation for model."""
        doc: libsbml.SBMLDocument = read_sbml(sbml)
        return UnitsInformation.from_sbml_doc(doc, ureg=ureg)

    sbml_uids: ClassVar[list[str]] = [
        "ampere",
        "farad",
        "joule",
        "lux",
        "radian",
        "volt",
        "avogadro",
        "gram",
        "katal",
        "metre",
        "second",
        "watt",
        "becquerel",
        "gray",
        "kelvin",
        "mole",
        "siemens",
        "weber",
        "candela",
        "henry",
        "kilogram",
        "newton",
        "sievert",
        "coulomb",
        "hertz",
        "litre",
        "ohm",
        "steradian",
        "dimensionless",
        "item",
        "lumen",
        "pascal",
        "tesla",
    ]

    @staticmethod
    def model_uid_dict(model: libsbml.Model, ureg: UnitRegistry) -> dict[str, str]:
        """Get the expression of every unit id a model can use.

        The expression is a unit pint parses: the id itself for a unit kind of
        SBML pint knows and for a unit definition whose id pint parses as the
        same unit, the expression of the unit definition (`Units.udef_to_str`)
        otherwise. Nothing is defined in the registry.

        Args:
            model: the model.
            ureg: the registry the expressions are checked with.

        Returns:
            unit id -> expression.
        """
        uid_dict: dict[str, str] = {}

        # the unit kinds of SBML
        for key in UnitsInformation.sbml_uids:
            try:
                _ = ureg(key)
                uid_dict[key] = key
            except UndefinedUnitError:
                logger.debug("SBML unit kind can not be used in pint: '%s'", key)

        # map no units on dimensionless
        uid_dict[""] = "dimensionless"

        # the predefined units of SBML Level 2
        uid_dict.update(
            {
                "substance": "mole",
                "volume": "litre",
                "area": "meter^2",
                "length": "meter",
                "time": "second",
            }
        )

        udef: libsbml.UnitDefinition
        for udef in model.getListOfUnitDefinitions():
            uid = udef.getId()
            unit_str = Units.udef_to_str(udef)
            if not UnitsInformation._is_unit(unit_str, ureg):
                # a factor which is no prefix, e.g. the 133.322 of a mmHg: pint
                # has no unit for it, so it is defined, under a name which no
                # other definition uses
                unit_str = UnitsInformation._define(uid, unit_str, ureg)
            logger.debug("%s = %s", uid, unit_str)
            uid_dict[uid] = unit_str

        return uid_dict

    @staticmethod
    def _is_unit(expression: str, ureg: UnitRegistry) -> bool:
        """Check whether an expression is a unit of pint without a factor."""
        try:
            ureg.parse_units(expression)
        except Exception:  # pint raises many types
            return False
        return True

    @staticmethod
    def _define(uid: str, expression: str, ureg: UnitRegistry) -> str:
        """Define a unit with a factor in the registry under a free name.

        Args:
            uid: the id of the unit definition, the name if it is free.
            expression: the expression of the unit, with its factor.
            ureg: the registry.

        Returns:
            The name of the unit: `uid` if the registry does not know it or
            knows it as the same unit, `uid_<n>` with the first `n` which is
            free or the same unit otherwise.
        """
        quantity = ureg(expression)
        name = uid
        n = 1
        while True:
            try:
                defined = ureg(name)
            except Exception:  # pint raises many types
                ureg.define(f"{name} = {expression}")
                return name
            if (
                np.isclose(
                    defined.to_base_units().magnitude,
                    quantity.to_base_units().magnitude,
                )
                and defined.dimensionality == quantity.dimensionality
            ):
                return name
            n += 1
            name = f"{uid}_{n}"

    @staticmethod
    def from_sbml_doc(
        doc: libsbml.SBMLDocument, ureg: UnitRegistry | None = None
    ) -> UnitsInformation:
        """Get pint UnitsInformation for model in document.

        An entity without units, i.e. the time without time units, a species
        without the units of its substance or its compartment, or an entity
        whose derived unit is no unit definition of the model, is logged at
        the level `DEBUG`, and the model once at `INFO` with their number.

        Raises:
            ValueError: if the document has no model.
        """
        if ureg is None:
            ureg = _package_registry()

        # create sid to unit mapping
        model: libsbml.Model = doc.getModel()
        if not model:
            raise ValueError(f"No model found in SBMLDocument: {doc}")

        uid_dict: dict[str, str] = UnitsInformation.model_uid_dict(model, ureg=ureg)

        # add additional units
        udict: dict[str, str] = {}
        # the entities without units
        missing: list[str] = []

        # add time unit
        def expression(uid: str) -> str:
            """Get the expression of a unit id, the id if the model lacks it."""
            return uid_dict.get(uid, uid)

        time_uid: str = model.getTimeUnits()
        if time_uid:
            udict["time"] = expression(time_uid)
        if not time_uid:
            logger.debug("No time units defined for 'time', falling back to 'second'")
            missing.append("time")
            udict["time"] = "second"

        # get all objects in model
        if not model.isPopulatedAllElementIdList():
            model.populateAllElementIdList()
        sid_list: libsbml.IdList = model.getAllElementIdList()

        for k in range(sid_list.size()):
            sid = sid_list.at(k)
            element: libsbml.SBase = model.getElementBySId(sid)
            if element:
                # in case of reactions we have to derive units from the kinetic law
                if isinstance(element, libsbml.Reaction):
                    if element.isSetKineticLaw():
                        element = element.getKineticLaw()
                    else:
                        continue

                # for species the amount and concentration units have to be added
                if isinstance(element, libsbml.Species):
                    # amount units
                    substance_uid = element.getSubstanceUnits()
                    udict[sid] = expression(substance_uid) if substance_uid else ""

                    compartment: libsbml.Compartment = model.getCompartment(
                        element.getCompartment()
                    )
                    volume_uid = compartment.getUnits()

                    # store concentration
                    if substance_uid and volume_uid:
                        udict[f"[{sid}]"] = Units.quotient(
                            expression(substance_uid), expression(volume_uid)
                        )
                    elif not substance_uid:
                        logger.debug(
                            "Substance unit missing, undefined concentration unit "
                            "for '[%s]'",
                            sid,
                        )
                        missing.append(f"[{sid}]")
                        udict[f"[{sid}]"] = ""
                    elif not volume_uid:
                        logger.debug(
                            "Volume unit missing, undefined concentration unit "
                            "for '[%s]'",
                            sid,
                        )
                        missing.append(f"[{sid}]")
                        udict[f"[{sid}]"] = ""

                elif isinstance(element, (libsbml.Compartment, libsbml.Parameter)):
                    uid_element = element.getUnits()
                    udict[sid] = expression(uid_element) if uid_element else ""
                else:
                    udef: libsbml.UnitDefinition = element.getDerivedUnitDefinition()
                    if udef is None:
                        continue

                    # find the correct unit definition
                    uid: str | None = None
                    udef_test: libsbml.UnitDefinition
                    for udef_test in model.getListOfUnitDefinitions():
                        if libsbml.UnitDefinition.areIdentical(udef_test, udef):
                            uid = udef_test.getId()
                            break

                    if uid:
                        udict[sid] = expression(uid)
                    else:
                        logger.debug(
                            "DerivedUnit of '%s' not in UnitDefinitions: '%s'",
                            sid,
                            Units.udef_to_str(udef),
                        )
                        missing.append(sid)
                        udict[sid] = Units.udef_to_str(udef)

            else:
                # check if sid is a unit
                udef = model.getUnitDefinition(sid)
                if udef is None:
                    # elements in packages
                    logger.debug("No element found for id '%s'", sid)

        if missing:
            logger.info(
                "The model '%s' has %s entities without units or with a derived "
                "unit which is no unit definition, they are logged at DEBUG",
                model.getId(),
                len(missing),
            )

        return UnitsInformation(udict=udict, ureg=ureg)

    @staticmethod
    def normalize_changes(
        changes: dict[str, Quantity | float], uinfo: UnitsInformation
    ) -> dict[str, Quantity | float]:
        """Normalize all changes to units in given units dictionary.

        This is a major helper function allowing to convert changes
        to the requested units.
        """
        Q_ = uinfo.ureg.Quantity
        changes_normed = {}
        for key, item in changes.items():
            if isinstance(item, Quantity):
                try:
                    # convert to model units
                    item = item.to(uinfo[key])
                except DimensionalityError as err:
                    logger.error(
                        "DimensionalityError '%s = %s'. Check that model units fit with changes units.\n%s",
                        key,
                        item,
                        err,
                    )
                    raise err
                except KeyError as err:
                    logger.error(
                        "KeyError: '%s' does not exist in unit dictionary of model.",
                        key,
                    )
                    raise err
            else:
                logger.warning(
                    "No units provided, assuming dictionary units: %s = %s", key, item
                )
                try:
                    # convert to model units
                    item = Q_(item, uinfo[key])
                except DimensionalityError as err:
                    logger.error("DimensionalityError '%s = %s'.\n%s", key, item, err)

            changes_normed[key] = item

        return changes_normed


class Units:
    """Units class.

    Container for unit related functionality.
    Allows to read the unit information from SBML models and provides
    helpers for the unit conversion.
    """

    #: the symbols of the unit kinds of SBML in pint, a kind which is not
    #: listed is written with its name
    _symbols: ClassVar[dict[str, str]] = {
        "kilogram": "kg",
        "gram": "g",
        "metre": "m",
        "meter": "m",
        "second": "s",
        "mole": "mol",
        "litre": "l",
        "liter": "l",
        "katal": "kat",
        "dimensionless": "",
    }

    #: the SI prefixes of the scales of a unit
    _prefixes: ClassVar[dict[int, str]] = {
        9: "G",
        6: "M",
        3: "k",
        2: "h",
        1: "da",
        -1: "d",
        -2: "c",
        -3: "m",
        -6: "u",
        -9: "n",
        -12: "p",
        -15: "f",
    }

    #: the names of the multiples of a second
    _seconds: ClassVar[dict[float, str]] = {
        60.0: "min",
        3600.0: "hr",
        86400.0: "day",
        604800.0: "week",
    }

    @classmethod
    def _unit_to_str(cls, u: libsbml.Unit) -> str:
        """Format a unit of a unit definition without the sign of its exponent.

        Args:
            u: the unit, `(multiplier * 10^scale * kind)^exponent`.

        Returns:
            The unit with the prefix and the name of pint where there is one,
            e.g. `mmol`, `min` or `mm^2`, a product with the multiplier and
            the power of ten otherwise. Empty for a dimensionless unit.
        """
        kind = libsbml.UnitKind_toString(u.getKind())
        symbol = cls._symbols.get(kind, kind)
        multiplier = u.getMultiplier()
        scale = u.getScale()
        exponent = abs(u.getExponentAsDouble())

        if not symbol and np.isclose(multiplier, 1.0) and scale == 0:
            return ""

        # a multiplier which is a power of ten is a scale, e.g. 0.001 * mole
        value = multiplier * 10.0**scale
        power = np.log10(value) if value > 0 else np.nan
        if np.isfinite(power) and np.isclose(power, round(power), atol=1e-9):
            scale, multiplier = round(power), 1.0

        name: str
        factor = ""
        seconds = (
            cls._seconds.get(round(multiplier * 10.0**scale, 9))
            if kind == "second"
            else None
        )
        if seconds is not None:
            name = seconds
        else:
            if scale == 0:
                name = symbol
            elif scale in cls._prefixes and kind != "kilogram" and symbol:
                name = f"{cls._prefixes[scale]}{symbol}"
            elif kind == "kilogram" and scale + 3 in cls._prefixes:
                name = f"{cls._prefixes[scale + 3]}g" if scale + 3 else "g"
            else:
                name = f"10^{scale}*{symbol}" if symbol else f"10^{scale}"
            if not np.isclose(multiplier, 1.0):
                factor = f"{multiplier}*"

        term = f"{factor}{name}"
        compound = "*" in term
        if np.isclose(exponent, 1.0):
            return f"({term})" if compound else term
        exponent_str = f"{exponent:g}"
        return f"({term})^{exponent_str}" if compound else f"{term}^{exponent_str}"

    @classmethod
    def udef_to_str(cls, udef: libsbml.UnitDefinition) -> str:
        """Format SBML unitDefinition as string.

        Units have the general format
            (multiplier * 10^scale *ukind)^exponent
            (m * 10^s *k)^e

        The string is a unit pint parses, with the prefixes and names of pint
        where there are some, e.g. `mmol/min` for a millimole per minute.

        Returns the string "None" in case no UnitDefinition was provided.
        """
        if udef is None:
            return "None"

        # order the unit definition
        libsbml.UnitDefinition.reorder(udef)

        # collect formated nominators and denominators
        nom: list[str] = []
        denom: list[str] = []
        for u in udef.getListOfUnits():
            term = cls._unit_to_str(u)
            if not term:
                continue
            if u.getExponentAsDouble() >= 0.0:
                nom.append(term)
            else:
                denom.append(term)

        nom_str = " * ".join(nom)
        denom_str = " * ".join(denom)
        if len(denom) > 1:
            denom_str = f"({denom_str})"
        if nom_str and denom_str:
            return f"{nom_str}/{denom_str}"
        if nom_str:
            return nom_str
        if denom_str:
            return f"1/{denom_str}"
        return ""

    @staticmethod
    def quotient(numerator: str, denominator: str) -> str:
        """Get the quotient of two unit expressions, with the brackets it needs.

        Args:
            numerator: expression of the numerator, e.g. `mmol`.
            denominator: expression of the denominator, e.g. `l` or `m^3`.

        Returns:
            The expression of the quotient, e.g. `mmol/l`.
        """

        def group(expression: str) -> str:
            simple = all(c.isalnum() or c in "_^." for c in expression)
            return expression if simple or _enclosed(expression) else f"({expression})"

        return f"{group(numerator)}/{group(denominator)}"


def _enclosed(expression: str) -> bool:
    """Check whether an expression is one pair of brackets around the rest."""
    if not (expression.startswith("(") and expression.endswith(")")):
        return False
    depth = 0
    for k, c in enumerate(expression):
        depth += {"(": 1, ")": -1}.get(c, 0)
        if depth == 0 and k < len(expression) - 1:
            return False
    return True
