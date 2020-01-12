"""
Managing units and units conversions in models.

FIXME: have a look at uncertainties
https://pythonhosted.org/uncertainties/
"""
from pathlib import Path
import pint
import logging
import libsbml
from sbmlsim.tests.constants import MODEL_REPRESSILATOR, MODEL_GLCWB
from pprint import pprint
from pint.errors import UndefinedUnitError

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
    def default_ureg(cls):
        ureg = pint.UnitRegistry()
        ureg.define('none = count')
        ureg.define('item = count')
        ureg.define('yr = year')
        ureg.define('percent = 0.01*count')
        ureg.define('IU = 0.0347 * mg')  # IU for insulin ! (FIXME better handling of general IU)
        ureg.define('IU/ml = 0.0347 * mg/ml')  # IU for insulin ! (FIXME better handling of general IU)
        return ureg

    @classmethod
    def ureg_from_sbml(cls, doc: libsbml.SBMLDocument):
        """ Creates a pint unit registry for the given SBML.

        :param model_path:
        :return:
        """
        # get all units defined in the model (unit definitions)
        model = doc.getModel()  # type: libsbml.Model

        # add all UnitDefinitions to unit registry
        ureg = Units.default_ureg()
        for udef in model.getListOfUnitDefinitions():  # type: libsbml.UnitDefinition
            uid = udef.getId()
            udef_str = cls.unitDefinitionToString(udef)
            try:
                # check if unit registry definition
                q1 = ureg(uid)
                # SBML definition
                q2 = ureg(udef_str)
                # check if identical
                if q1 != q2:
                    logger.error(f"SBML uid '{uid}' defined differently in UnitsRegistry: '{uid} = {q1} != {q2}")
            except UndefinedUnitError as err:

                definition = f"{uid} = {udef_str}"
                ureg.define(definition)

        return ureg

    @classmethod
    def get_units_from_sbml(cls, model_path: Path):
        """ Get pint unit dictionary for given model.

        :param model_path: path to SBML model
        :return:
        """
        if isinstance(model_path, Path):
            doc = libsbml.readSBMLFromFile(str(model_path))  # type: libsbml.SBMLDocument
        elif isinstance(model_path, str):
            doc = libsbml.readSBMLFromFile(model_path)

        # parse unit registry
        ureg = cls.ureg_from_sbml(doc)

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

                    compartment = model.getCompartment(element.getCompartment())  # type: libsbml.Compartment
                    volume_uid = compartment.getUnits()

                    # store concentration
                    if substance_uid and volume_uid:
                        udict[f"[{sid}]"] = f"{substance_uid}/{volume_uid}"
                    else:
                        logger.warning(f"substance or volume unit missing, "
                                       f"impossible to determine concentration "
                                       f"unit for [{sid}])")
                        udict[f"[{sid}]"] = ''

                elif isinstance(element, (libsbml.Compartment, libsbml.Parameter)):
                    udict[sid] = element.getUnits()
                else:
                    udef = element.getDerivedUnitDefinition()
                    if udef is None:
                        continue
                    uid = None
                    # find the correct unit definition
                    for udef_test in model.getListOfUnitDefinitions():  # type: libsbml.UnitDefinition
                        if libsbml.UnitDefinition_areEquivalent(udef_test, udef):
                            uid = udef_test.getId()
                            break
                    if uid is None:
                        logger.warning(f"DerivedUnit not found in UnitDefinitions: {Units.unitDefinitionToString(udef)}")
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
    ureg = pint.UnitRegistry()

    udict, ureg = Units.get_units_from_sbml(MODEL_GLCWB)
    from pprint import pprint
    # pprint(udict)
    # print(ureg["mmole_per_min"])

    q1 = 1 * ureg("mole/s")
    print(q1)
    print(q1.to("mmole_per_min"))

    print('-' * 80)

    q2 = ureg("(mole)/(60000.0*s)")
    print(q2)
    print(q2.to("mmole_per_min"))











