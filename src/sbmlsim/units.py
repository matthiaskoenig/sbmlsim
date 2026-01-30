"""Manage units and units conversions.

Used for model and data unit conversions.
"""

from __future__ import annotations
import os
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Iterator, Optional, Union

import libsbml
import numpy as np
from pymetadata import log
from pymetadata.console import console
from sbmlutils.io import read_sbml


# Disable Pint's old fallback behavior (must come before importing Pint)
os.environ["PINT_ARRAY_PROTOCOL_FALLBACK"] = "0"


import pint  # noqa: E402
from pint import Quantity, UnitRegistry  # noqa: E402
from pint.errors import DimensionalityError, UndefinedUnitError  # noqa: E402


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

logger = log.get_logger(__name__)
UdictType = dict[str, str]


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
        sbml: Union[str, Path], ureg: Optional[UnitRegistry] = None
    ) -> UnitsInformation:
        """Get pint UnitsInformation for model."""

        doc: libsbml.SBMLDocument = read_sbml(sbml)
        return UnitsInformation.from_sbml_doc(doc, ureg=ureg)

    sbml_uids = [
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
        """Populate the model uid dict for lookup."""

        uid_dict: dict[str, str] = {}

        # add SBML definitions
        for key in UnitsInformation.sbml_uids:
            try:
                _ = ureg(key)
                uid_dict[key] = key
            except UndefinedUnitError:
                logger.debug(f"SBML unit kind can not be used in pint: '{key}'")

        # map no units on dimensionless
        uid_dict[""] = "dimensionless"

        # add predefined units (SBML Level 2)
        for uid, unit_str in {
            "substance": "mole",
            "volume": "litre",
            "area": "meter^2",
            "length": "meter",
            "time": "second",
        }.items():
            ureg.define(f"{uid} = {unit_str}")

        udef: libsbml.UnitDefinition
        for udef in model.getListOfUnitDefinitions():
            uid = udef.getId()
            unit_str = Units.udef_to_str(udef)
            q = ureg(unit_str)
            try:
                # check if uid is existing unit registry definition (short name)
                q_uid = ureg(uid)

                # check if identical
                if q_uid != q:
                    logger.debug(
                        f"SBML uid interpretation of '{uid}' does not match unit "
                        f"registry: '{uid} = {q} != {q_uid}'."
                    )
                else:
                    unit_str = uid

            except UndefinedUnitError:
                definition = f"{uid} = {unit_str}"
                ureg.define(definition)

            logger.debug(f"{uid} = {unit_str} ({q})")
            uid_dict[uid] = unit_str

        return uid_dict

    @staticmethod
    def from_sbml_doc(
        doc: libsbml.SBMLDocument, ureg: Optional[UnitRegistry] = None
    ) -> UnitsInformation:
        """Get pint UnitsInformation for model in document."""

        if ureg is None:
            ureg = UnitsInformation._default_ureg()

        # create sid to unit mapping
        model: libsbml.Model = doc.getModel()
        if not model:
            ValueError(f"No model found in SBMLDocument: {doc}")

        uid_dict: dict[str, str] = UnitsInformation.model_uid_dict(model, ureg=ureg)

        # add additional units
        udict: dict[str, str] = {}

        # add time unit
        time_uid: str = model.getTimeUnits()
        if time_uid:
            udict["time"] = uid_dict[time_uid]
        if not time_uid:
            logger.warning("No time units defined in model, falling back to 'second'.")
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
                    # udict[sid] = uid_dict[substance_uid]
                    udict[sid] = substance_uid

                    compartment: libsbml.Compartment = model.getCompartment(
                        element.getCompartment()
                    )
                    volume_uid = compartment.getUnits()

                    # store concentration
                    if substance_uid and volume_uid:
                        udict[f"[{sid}]"] = f"{substance_uid}/{volume_uid}"
                    elif not substance_uid:
                        logger.warning(
                            f"Substance unit missing, "
                            f"undefined concentration unit for '[{sid}]')"
                        )
                        udict[f"[{sid}]"] = ""
                    elif not volume_uid:
                        logger.warning(
                            f"Volume unit missing, "
                            f"undefined concentration unit for '[{sid}]')"
                        )
                        udict[f"[{sid}]"] = ""

                elif isinstance(element, (libsbml.Compartment, libsbml.Parameter)):
                    # udict[sid] = uid_dict[element.getUnits()]
                    udict[sid] = element.getUnits()
                else:
                    udef: libsbml.UnitDefinition = element.getDerivedUnitDefinition()
                    if udef is None:
                        continue

                    # find the correct unit definition
                    uid: Optional[str] = None
                    udef_test: libsbml.UnitDefinition
                    for udef_test in model.getListOfUnitDefinitions():
                        if libsbml.UnitDefinition.areIdentical(udef_test, udef):
                            uid = udef_test.getId()
                            break

                    if uid:
                        # udict[sid] = uid_dict[uid]
                        udict[sid] = uid
                    else:
                        logger.warning(
                            f"DerivedUnit not in UnitDefinitions: "
                            f"'{Units.udef_to_str(udef)}'"
                        )
                        udict[sid] = Units.udef_to_str(udef)

            else:
                # check if sid is a unit
                udef = model.getUnitDefinition(sid)
                if udef is None:
                    # elements in packages
                    logger.debug(f"No element found for id '{sid}'")

        return UnitsInformation(udict=udict, ureg=ureg)

    @staticmethod
    def _default_ureg() -> pint.UnitRegistry:
        """Get default unit registry."""
        ureg = pint.UnitRegistry(on_redefinition="ignore")
        ureg.define("none = count")
        ureg.define("item = count")
        ureg.define("percent = 0.01*count")

        # FIXME: manual conversion
        ureg.define(
            "IU = 0.0347 * mg"
        )  # IU for insulin ! FIXME better handling of general IU
        ureg.define(
            "IU_per_ml = 0.0347 * mg/ml"
        )  # IU for insulin ! FIXME better handling of general IU
        return ureg

    @staticmethod
    def normalize_changes(
        changes: dict[str, Quantity], uinfo: "UnitsInformation"
    ) -> dict[str, Quantity]:
        """Normalize all changes to units in given units dictionary.

        This is a major helper function allowing to convert changes
        to the requested units.
        """
        Q_ = uinfo.ureg.Quantity
        changes_normed = {}
        for key, item in changes.items():
            if hasattr(item, "units"):
                try:
                    # convert to model units
                    item = item.to(uinfo[key])
                except DimensionalityError as err:
                    logger.error(
                        f"DimensionalityError "
                        f"'{key} = {item}'. Check that model "
                        f"units fit with changes units."
                        f"\n{err}"
                    )
                    raise err
                except KeyError as err:
                    logger.error(
                        f"KeyError: '{key}' does not exist in unit "
                        f"dictionary of model."
                    )
                    raise err
            else:
                logger.warning(
                    f"No units provided, assuming dictionary units: {key} = {item}"
                )
                try:
                    # convert to model units
                    item = Q_(item, uinfo[key])
                except DimensionalityError as err:
                    logger.error(
                        f"DimensionalityError " f"'{key} = {item}'." f"\n{err}"
                    )

            changes_normed[key] = item

        return changes_normed


class Units:
    """Units class.

    Container for unit related functionality.
    Allows to read the unit information from SBML models and provides
    helpers for the unit conversion.
    """

    # abbreviation dictionary for string representation
    _units_abbreviation = {
        "kilogram": "kg",
        "meter": "m",
        "metre": "m",
        "second": "s",
        "hour": "hr",
        "dimensionless": "",
        "katal": "kat",
        "gram": "g",
    }

    @classmethod
    def udef_to_str(cls, udef: libsbml.UnitDefinition) -> str:
        """Format SBML unitDefinition as string.

        Units have the general format
            (multiplier * 10^scale *ukind)^exponent
            (m * 10^s *k)^e

        Returns the string "None" in case no UnitDefinition was provided.

        """
        if udef is None:
            return "None"

        # order the unit definition
        libsbml.UnitDefinition.reorder(udef)

        # collect formated nominators and denominators
        nom = []
        denom = []
        for u in udef.getListOfUnits():
            m = u.getMultiplier()
            s = u.getScale()
            e = u.getExponent()
            k = libsbml.UnitKind_toString(u.getKind())

            # get better name for unit
            k_str = cls._units_abbreviation.get(k, k)

            # (m * 10^s *k)^e

            # handle m
            if np.isclose(m, 1.0):
                m_str = ""
            else:
                m_str = str(m) + "*"

            if np.isclose(abs(e), 1.0):
                e_str = ""
            else:
                e_str = "^" + str(abs(e))

            # FIXME: handle unit prefixes;

            if np.isclose(s, 0.0):
                if not m_str and not e_str:
                    string = k_str
                else:
                    string = "({}{}{})".format(m_str, k_str, e_str)
            else:
                if e_str == "":
                    string = "({}10^{}*{})".format(m_str, s, k_str)
                else:
                    string = "(({}10^{}*{})^{})".format(m_str, s, k_str, e_str)

            # collect the terms
            if e >= 0.0:
                nom.append(string)
            else:
                denom.append(string)

        nom_str = " * ".join(nom)
        denom_str = " * ".join(denom)
        if len(denom) > 1:
            denom_str = f"({denom_str})"
        if (len(nom_str) > 0) and (len(denom_str) > 0):
            return f"{nom_str}/{denom_str}"
        if (len(nom_str) > 0) and (len(denom_str) == 0):
            return nom_str
        if (len(nom_str) == 0) and (len(denom_str) > 0):
            return f"1/{denom_str}"
        return ""


if __name__ == "__main__":
    from sbmlsim.resources import DEMO_SBML

    ureg = UnitRegistry(on_redefinition="ignore")
    uinfo = UnitsInformation.from_sbml(DEMO_SBML, ureg=ureg)
    console.log(uinfo.udict)
