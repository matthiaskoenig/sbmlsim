"""
Functions for loading models and setting selections.
"""
import logging
import roadrunner
import re
import numpy as np


def load_model(path, selections=True):
    """ Loads the latest model version.

    :param path: path to SBML model
    :param selections: boolean flag to set selections
    :return: roadrunner instance
    """
    logging.info("Loading: '{}'".format(path))
    r = roadrunner.RoadRunner(path)
    if selections:
        set_selections(r)
    return r


def set_selections(r):
    """ Sets the full model selections. """
    r.timeCourseSelections = ["time"] \
                             + r.model.getFloatingSpeciesIds() \
                             + r.model.getBoundarySpeciesIds() \
                             + r.model.getGlobalParameterIds() \
                             + r.model.getReactionIds() \
                             + r.model.getCompartmentIds()
    r.timeCourseSelections += [f'[{key}]' for key in (r.model.getFloatingSpeciesIds() + r.model.getBoundarySpeciesIds())]


'''
# TODO handle the sensitivity changesets
  for pid in parameters.keys():
            for change in [1.0 + sensitivity, 1.0 - sensitivity]:
                resetAll(r)
                reset_doses(r)
                # dosing
                if dosing:
                    set_dosing(r, dosing, bodyweight=bodyweight)
                # general changes
                for key, value in changes.items():
                    r[key] = value
                # parameter changes
                value = r[pid]
                new_value = value * change
                r[pid] = new_value

                s = r.simulate(start=0, end=tend, steps=steps)
                if yfun:
                    # conversion function
                    s = pd.DataFrame(s, columns=s.colnames)
                    yfun(s)
                    s_data[:, :, idx] = s
                else:
                    s_data[:, :, idx] = s
                idx += 1
'''


def exlude_pkdb_parameter_filter(pid):
    """ Returns True if excluded, False otherwise

    :param pid:
    :return:
    """
    # TODO: implement
    # dose parameters
    if (pid.startswith("IVDOSE_")) or (pid.startswith("PODOSE_")):
        return True

    # physical parameters
    if (pid.startswith("Mr_")) or pid in ["R_PDB"]:
        return True
    return False


def _parameters_for_sensitivity(r, exclude_filter=None, exclude_zero=True, zero_eps=1E-8):
    """ Get parameter ids for sensitivity analysis.

    Values around current model state are used.

    :param r:
    :param exclude_filter: filter function to exclude parameters
    :param exclude_zero: exclude parameters which are zero
    :return:
    """
    import libsbml

    print(r.getSBML())
    doc = libsbml.readSBMLFromString(r.getSBML())  # type: libsbml.SBMLDocument
    print(doc)
    model = doc.getModel()  # type: libsbml.Model
    print(model)

    # constant parameters
    pids_const = []
    for p in model.getListOfParameters():
        if p.getConstant() is True:
            pids_const.append(p.getId())

    # filter parameters
    parameters = {}
    for pid in pids_const:
        if exclude_filter and exclude_filter(pid):
            continue

        value = r[pid]
        if exclude_zero:
            if np.abs(value) < zero_eps:
                continue

        parameters[pid] = value

    return parameters


def set_initial_concentrations(r, skey, value):
    """ Set initial concentrations for skey.

    :param r: roadrunner model
    :param skey: substance key
    :param value: new value in model units
    :return:
    """
    return _set_initial_values(r, skey, value, method="concentration")

def set_initial_amounts(r, skey, value):
    """ Set initial amounts for skey.

    :param r: roadrunner model
    :param skey:
    :param value:
    :return:
    """
    return _set_initial_values(r, skey, value, method="amount")


def _set_initial_values(r, sid, value, method="concentration"):
    """ Setting the initial concentration of a distributing substance.

    Takes care of all the compartment values so starting close/in steady state.
    Units are in model units

    return: species keys which have been set
    """
    if method not in ["amount", "concentration"]:
        raise ValueError

    species_ids = r.model.getFloatingSpeciesIds() + r.model.getBoundarySpeciesIds()
    species_keys = get_species_keys(sid, species_ids)
    print(species_keys)
    for key in species_keys:
        if 'urine' in key:
            logging.warning("urinary values are not set")
            continue
        if method == "concentration":
            rkey = f'init([{key}])'
        elif method == "amount":
            rkey = f'init({value})'
        # print(f'{rkey} <- {value}')

        r.setValue(rkey, value)

    return species_keys