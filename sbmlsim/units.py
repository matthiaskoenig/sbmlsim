"""
Managing units and units conversions in models.

FIXME: have a look at uncertainties
https://pythonhosted.org/uncertainties/
"""
import libsbml
from sbmlsim.tests.constants import MODEL_REPRESSILATOR, MODEL_GLCWB
from sbmlsim.model import load_model

import pint
ureg = pint.UnitRegistry()
ureg.define('none = count')
ureg.define('yr = year')
ureg.define('percent = 0.01*count')

import logging
logger = logging.getLogger(__name__)



UNIT_ABBREVIATIONS = {
    'kilogram': 'kg',
    'meter': 'm',
    'metre': 'm',
    'second': 's',
    'dimensionless': '',
    'katal': 'kat',
    'gram': 'g',
}


def _isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """ Calculate the two floats are identical. """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def unitDefinitionToString(udef):
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
        k_str = UNIT_ABBREVIATIONS.get(k, k)

        # (m * 10^s *k)^e

        # handle m
        if _isclose(m, 1.0):
            m_str = ''
        else:
            m_str = str(m) + '*'

        if _isclose(abs(e), 1.0):
            e_str = ''
        else:
            e_str = '^' + str(abs(e))

        if _isclose(s, 0.0):
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

def get_units(model_path):
    """ Calculate all units from given model.

    :param model_path: path to SBML model
    :return:
    """
    doc = libsbml.readSBMLFromFile(str(model_path))  # type: libsbml.SBMLDocument

    # get all units defined in the model (unit definitions)
    model = doc.getModel()  # type: libsbml.Model

    units = ()
    for udef in model.getListOfUnitDefinitions():  # type: libsbml.UnitDefinition
        print(str(udef))

        udef.getListOfUnits()


    return units

if __name__ == "__main__":
    # r = load_model(MODEL_REPRESSILATOR)

    data = 3 * ureg.meter + 4 * ureg.cm
    print(data)

    units = get_units(MODEL_GLCWB)







