"""Manage units and units conversions.

Used for model and data unit conversions.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import libsbml


# Disable Pint's old fallback behavior (must come before importing Pint)
os.environ["PINT_ARRAY_PROTOCOL_FALLBACK"] = "0"

import warnings  # noqa: E402

import pint  # noqa: E402
from pint import Quantity, UnitRegistry  # noqa: E402
from pint.errors import DimensionalityError, UndefinedUnitError  # noqa: E402


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

logger = logging.getLogger(__name__)


class Units:
    """Units class.

    Container for unit related functionality.
    """

    UNIT_ABBREVIATIONS = {
        "kilogram": "kg",
        "meter": "m",
        "metre": "m",
        "second": "s",
        "dimensionless": "",
        "katal": "kat",
        "gram": "g",
    }

    @classmethod
    def default_ureg(cls) -> pint.UnitRegistry:
        """Get default unit registry."""
        ureg = pint.UnitRegistry()
        ureg.define("none = count")
        ureg.define("item = count")
        ureg.define("percent = 0.01*count")
        # FIXME: manual conversion
        ureg.define(
            "IU = 0.0347 * mg"
        )  # IU for insulin ! FIXME better handling of general IU
        ureg.define(
            "IU/ml = 0.0347 * mg/ml"
        )  # IU for insulin ! FIXME better handling of general IU
        return ureg

    @classmethod
    def ureg_from_sbml(cls, doc: libsbml.SBMLDocument, ureg: UnitRegistry = None):
        """Create a pint unit registry for the given SBML.

        :param model_path:
        :return:
        """
        # get all units defined in the model (unit definitions)
        model: libsbml.Model = doc.getModel()

        # add all UnitDefinitions to unit registry
        if not ureg:
            ureg = Units.default_ureg()

        udef: libsbml.UnitDefinition
        for udef in model.getListOfUnitDefinitions():
            uid = udef.getId()
            udef_str = cls.unitDefinitionToString(udef)
            try:
                # check if unit registry definition
                q1 = ureg(uid)
                # SBML definition
                q2 = ureg(udef_str)
                # check if identical
                if q1 != q2:
                    logger.debug(
                        f"SBML uid '{uid}' cannot be looked up in UnitsRegistry: "
                        f"'{uid} = {q1} != {q2}"
                    )
            except UndefinedUnitError:
                definition = f"{uid} = {udef_str}"
                ureg.define(definition)

        return ureg

    @classmethod
    def get_units_from_sbml(
        cls, model_path: Path, ureg: UnitRegistry = None
    ) -> Tuple[Dict[str, str], UnitRegistry]:
        """Get pint unit dictionary for given model.

        :param model_path: path to SBML model
        :return: udict, ureg
        """
        if isinstance(model_path, Path):
            doc = libsbml.readSBMLFromFile(
                str(model_path)
            )  # type: libsbml.SBMLDocument
        elif isinstance(model_path, str):
            doc = libsbml.readSBMLFromFile(model_path)

        # parse unit registry
        ureg = cls.ureg_from_sbml(doc, ureg)

        # get all units defined in the model (unit definitions)
        model = doc.getModel()  # type: libsbml.Model

        # create sid to unit mapping
        udict = {}
        if not model.isPopulatedAllElementIdList():
            model.populateAllElementIdList()

        # add time unit
        time_uid = model.getTimeUnits()
        if not time_uid:
            time_uid = "second"
        udict["time"] = time_uid

        sid_list = model.getAllElementIdList()  # type: libsbml.IdList
        for k in range(sid_list.size()):
            sid = sid_list.at(k)
            element = model.getElementBySId(sid)  # type: libsbml.SBase
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
                    udict[sid] = substance_uid

                    compartment = model.getCompartment(
                        element.getCompartment()
                    )  # type: libsbml.Compartment
                    volume_uid = compartment.getUnits()

                    # store concentration
                    if substance_uid and volume_uid:
                        udict[f"[{sid}]"] = f"{substance_uid}/{volume_uid}"
                    else:
                        logger.warning(
                            f"Substance or volume unit missing, "
                            f"cannot determine concentration "
                            f"unit for '[{sid}]')"
                        )
                        udict[f"[{sid}]"] = ""

                elif isinstance(element, (libsbml.Compartment, libsbml.Parameter)):
                    udict[sid] = element.getUnits()
                else:
                    udef = element.getDerivedUnitDefinition()
                    if udef is None:
                        continue
                    uid = None
                    # find the correct unit definition
                    for (
                        udef_test
                    ) in (
                        model.getListOfUnitDefinitions()
                    ):  # type: libsbml.UnitDefinition
                        if libsbml.UnitDefinition_areEquivalent(udef_test, udef):
                            uid = udef_test.getId()
                            break
                    if uid is None:
                        logger.warning(
                            f"DerivedUnit not in UnitDefinitions: "
                            f"'{Units.unitDefinitionToString(udef)}'"
                        )
                        udict[sid] = Units.unitDefinitionToString(udef)
                    else:
                        udict[sid] = uid

            else:
                # check if sid is a unit
                udef = model.getUnitDefinition(sid)
                if udef is None:
                    # elements in packages
                    logger.debug(f"No element found for id '{sid}'")

        return udict, ureg

    @classmethod
    def unitIdNormalization(cls, uid: str) -> str:
        """Normalize unit ids."""
        # FIXME: this is very specific to the uids in the model
        uid_in = uid[:]
        if "__" in uid:
            uid = "__".join(uid.split("__")[1:])
        uid = uid.replace("_per_", "/")
        uid = uid.replace("_", "*")
        if uid is not uid_in:
            logger.debug(f"uid normalization: {uid_in} -> {uid}")
        return uid

    @classmethod
    def unitDefinitionToString(cls, udef: libsbml.UnitDefinition) -> str:
        """Format SBML unitDefinition as string.

        Units have the general format
            (multiplier * 10^scale *ukind)^exponent
            (m * 10^s *k)^e

        """
        if udef is None:
            return "None"

        libsbml.UnitDefinition_reorder(udef)
        # collect formated nominators and denominators
        nom = []
        denom = []
        for u in udef.getListOfUnits():
            m = u.getMultiplier()
            s = u.getScale()
            e = u.getExponent()
            k = libsbml.UnitKind_toString(u.getKind())

            # get better name for unit
            k_str = cls.UNIT_ABBREVIATIONS.get(k, k)

            # (m * 10^s *k)^e

            # handle m
            if cls._isclose(m, 1.0):
                m_str = ""
            else:
                m_str = str(m) + "*"

            if cls._isclose(abs(e), 1.0):
                e_str = ""
            else:
                e_str = "^" + str(abs(e))

            if cls._isclose(s, 0.0):
                string = "{}{}{}".format(m_str, k_str, e_str)
            else:
                if e_str == "":
                    string = "({}10^{})*{}".format(m_str, s, k_str)
                else:
                    string = "(({}10^{})*{}){}".format(m_str, s, k_str, e_str)

            # collect the terms
            if e >= 0.0:
                nom.append(string)
            else:
                denom.append(string)

        nom_str = " * ".join(nom)
        denom_str = " * ".join(denom)
        if (len(nom_str) > 0) and (len(denom_str) > 0):
            return "({})/({})".format(nom_str, denom_str)
        if (len(nom_str) > 0) and (len(denom_str) == 0):
            return nom_str
        if (len(nom_str) == 0) and (len(denom_str) > 0):
            return "1/({})".format(denom_str)
        return ""

    @staticmethod
    def _isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        """Calculate the two floats are identical."""
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    @staticmethod
    def normalize_changes(
        changes: Dict, udict: Dict, ureg: UnitRegistry
    ) -> Dict[str, Quantity]:
        """Normalize all changes to units in units dictionary.

        :param changes:
        :param udict:
        :param ureg:
        :return:
        """
        Q_ = ureg.Quantity
        changes_normed = {}
        for key, item in changes.items():
            if hasattr(item, "units"):
                try:
                    # convert to model units
                    item = item.to(udict[key])
                except DimensionalityError as err:
                    logger.error(f"DimensionalityError " f"'{key} = {item}'. {err}")
                    raise err
                except KeyError as err:
                    logger.error(
                        f"KeyError: '{key}' does not exist in unit "
                        f"dictionary of model."
                    )
                    raise err
            else:
                item = Q_(item, udict[key])
                logger.warning(
                    f"No units provided, assuming dictionary units: " f"{key} = {item}"
                )
            changes_normed[key] = item

        return changes_normed
