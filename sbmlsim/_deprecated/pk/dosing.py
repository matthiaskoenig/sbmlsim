

"""
Dosing in pkdb models.

Helpers for running simulations with whole-body model.

**Simulation experiments**

* setting concentrations of substances in all compartments/tissues (for initialization)
* oral dose of substance (OGTT)
* bolus iv dose (with injection time, ivGTT)
* constant iv doses for given intervals (lactate, glucose, insulin, glucagon infusions)
* clamping values via infusions (hyperglycemic, euglycemic clamps with set points)

**Sampled simulation experiments**

* sampling from initial concentrations, parameter values, ...

"""
import os
import re
import logging
import roadrunner
import pandas as pd
from roadrunner import SelectionRecord


# -------------------------------------------------------------------------------------------------
# Dosing
# -------------------------------------------------------------------------------------------------
class Dosing(object):
    """ Classes and methods related to dosing experiments.

    Bolus injections, ...
    """
    def __init__(self, substance, route, dose, unit):
        self.substance = substance
        self.route = route
        self.dose = dose
        self.unit = unit

    def __repr__(self):
        return "{} [{}] {}".format(self.dose, self.unit, self.route)

    @staticmethod
    def print_doses(r, name=None):
        """ Prints the complete dose information of the model. """
        if name:
            print('***', name, '***')
        for key in Dosing.get_doses_keys(r):
            print('{}\t{}'.format(key, r.getValue(key)))


    @staticmethod
    def get_doses_keys(r):
        """ Returns parameter ids corresponding to dosing parameters.

        :param r: roadrunner model instance
        :return:
        """
        doses_keys = []
        for pid in r.model.getGlobalParameterIds():
            if pid.startswith("PODOSE_") or pid.startswith("IVDOSE_"):
                doses_keys.append(pid)

        return doses_keys

    @staticmethod
    def set_dosing(r, dosing, bodyweight=None, show=False):
        """ Sets dosing in roadrunner model.

        Doses per bodyweight are scaled with given body weight, or body weight of the respective model.
        Doses are in the units of the dosing keys.

        :param r:
        :param dosing:
        :param bodyweight:
        :param show:
        :return:
        """
        if dosing.route == "oral":
            pid = "PODOSE_{}".format(dosing.substance)
        elif dosing.route == "iv":
            pid = "IVDOSE_{}".format(dosing.substance)
        else:
            raise ValueError("Invalid dosing route: {}".format(dosing.route))

        # get dose in [mg]
        dose = dosing.dose
        if dosing.unit.endswith("kg"):
            if bodyweight is None:
                # use bodyweight from model
                bodyweight = r.BW
            dose = dose * bodyweight

        # reset the model
        r.reset()
        r.setValue('init({})'.format(pid), dose)  # set dose in [mg]
        r.reset(SelectionRecord.GLOBAL_PARAMETER)
        r.reset()

        if show:
            Dosing.print_doses(r)


def oral_dose(IV):
    pass




# dosing
if dosing is not None:
    # get bodyweight
    if "BW" in changes:
        bodyweight = changes["BW"]
    else:
        bodyweight = r.BW

    set_dosing(r, dosing, bodyweight=bodyweight)
