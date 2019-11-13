"""
Managing units and units conversions in models.

FIXME: have a look at uncertainties
https://pythonhosted.org/uncertainties/
"""
import libsbml
from sbmlsim.tests.constants import MODEL_REPRESSILATOR, MODEL_GLCWB
from sbmlsim.model import load_model

from pathlib import Path

import pint
ureg = pint.UnitRegistry()
ureg.define('none = count')
ureg.define('item = count')
ureg.define('yr = year')
ureg.define('percent = 0.01*count')

import logging
logger = logging.getLogger(__name__)


class Units(object):

    UNIT_ABBREVIATIONS = {
        'kilogram': 'kg',
        'meter': 'm',
        'metre': 'm',
        'second': 's',
        'dimensionless': '',
        'katal': 'kat',
        'gram': 'g',
    }

    @classmethod
    def get_units_from_sbml(cls, model_path):
        """ Get pint unit dictionary for given model.

        :param model_path: path to SBML model
        :return:
        """

        # FIXME: must create a unit registry for the respective model
        # ! DO FORBID OVERWRITING OF BASE UNITS !


        if isinstance(model_path, Path):
            doc = libsbml.readSBMLFromFile(str(model_path))  # type: libsbml.SBMLDocument
        elif isinstance(model_path, str):
            doc = libsbml.readSBMLFromString(model_path)

        # get all units defined in the model (unit definitions)
        model = doc.getModel()  # type: libsbml.Model

        # check that all units can be converted to pint
        udef_to_ureg = {}


        for udef in model.getListOfUnitDefinitions():  # type: libsbml.UnitDefinition
            udef_str = cls.unitDefinitionToString(udef)
            quantity = ureg(udef_str)
            udef_to_ureg[udef.getId()] = quantity.to_compact()

        # create id ureg mapping
        if not model.isPopulatedAllElementIdList():
            model.populateAllElementIdList()

        def unit_str(uid):
            """ Convert an SBML unit identifier into parsable string by pint."""
            udef = model.getUnitDefinition(uid)
            if udef:
                return cls.unitDefinitionToString(udef)
            else:
                return uid

        sid_to_ureg = {}
        # add time unit
        time_uid = model.getTimeUnits()
        if not time_uid:
            time_uid = "second"
        time_ustr = unit_str(time_uid)
        sid_to_ureg["time"] = ureg(time_ustr)

        sid_list = model.getAllElementIdList()  # type: libsbml.IdList
        for k in range(sid_list.size()):
            sid = sid_list.at(k)
            element = model.getElementBySId(sid)  # type: libsbml.SBase
            if element:
                if isinstance(element, libsbml.Reaction):
                    if element.isSetKineticLaw():
                        # in case of reactions we have to derive units from the kinetic law
                        element = element.getKineticLaw()
                    else:
                        continue

                # for species the amount and concentration units have to be added
                if isinstance(element, libsbml.Species):
                    # amount units
                    substance_uid = element.getSubstanceUnits()
                    substance_ustr = unit_str(substance_uid)
                    # store amount
                    sid_to_ureg[sid] = ureg(substance_uid)

                    compartment = model.getCompartment(element.getCompartment())  # type: libsbml.Compartment
                    volume_uid = compartment.getUnits()
                    volume_ustr = unit_str(volume_uid)

                    # store concentration
                    sid_to_ureg[f"[{sid}]"] = ureg(f"({substance_ustr})/({volume_ustr})")

                elif isinstance(element, libsbml.Compartment):
                    compartment_uid = element.getUnits()
                    compartment_ustr = unit_str(compartment_uid)
                    sid_to_ureg[sid] = ureg(compartment_ustr)

                else:
                    udef = element.getDerivedUnitDefinition()
                    sid_to_ureg[sid] = ureg(cls.unitDefinitionToString(udef))

            else:
                # check if sid is a unit
                udef = model.getUnitDefinition(sid)
                if udef is None:
                    logger.error(f"No element found for id '{sid}'")

        return sid_to_ureg



    @classmethod
    def unitDefinitionToString(cls, udef: libsbml.UnitDefinition) -> str:
        """ Formating of units.
        Units have the general format
            (multiplier * 10^scale *ukind)^exponent
            (m * 10^s *k)^e

        """
        if udef is None:
            return 'None'

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
                m_str = ''
            else:
                m_str = str(m) + '*'

            if cls._isclose(abs(e), 1.0):
                e_str = ''
            else:
                e_str = '^' + str(abs(e))

            if cls._isclose(s, 0.0):
                string = '{}{}{}'.format(m_str, k_str, e_str)
            else:
                if e_str == '':
                    string = '({}10^{})*{}'.format(m_str, s, k_str)
                else:
                    string = '(({}10^{})*{}){}'.format(m_str, s, k_str, e_str)

            # collect the terms
            if e >= 0.0:
                nom.append(string)
            else:
                denom.append(string)

        nom_str = ' * '.join(nom)
        denom_str = ' * '.join(denom)
        if (len(nom_str) > 0) and (len(denom_str) > 0):
            return '({})/({})'.format(nom_str, denom_str)
        if (len(nom_str) > 0) and (len(denom_str) == 0):
            return nom_str
        if (len(nom_str) == 0) and (len(denom_str) > 0):
            return '1/({})'.format(denom_str)
        return ''

    @staticmethod
    def _isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        """ Calculate the two floats are identical. """
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


if __name__ == "__main__":
    # r = load_model(MODEL_REPRESSILATOR)

    data = 3 * ureg.meter + 4 * ureg.cm
    print(data)

    sid_to_ureg = Units.get_units_from_sbml(MODEL_GLCWB)
    from pprint import pprint
    pprint(sid_to_ureg)

    '''
    distance = 24.0 * ureg.meter
    print(type(distance))
    print(distance.magnitude)
    print(distance.units)
    print(distance.dimensionality)
    time = 8.0 * ureg.second
    speed = distance / time
    print(speed.to(ureg.inch / ureg.minute))
    '''








