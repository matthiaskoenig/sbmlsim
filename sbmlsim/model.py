"""
Functions for loading models and setting selections.
"""
import logging
import roadrunner
import re


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


def


def parameters_for_sensitivity(r, model_path):
    """ Get the parameter ids for the sensitivity analysis.

    This includes all constant parameters (not changed via assignments),
    excluding
    - parameters with value=0 (no effect on model, dummy parameter)
    - parameters which are physical constants, e.g., molecular weights
    """
    try:
        import tesbml as libsbml
    except ImportError:
        import libsbml

    doc = libsbml.readSBMLFromFile(model_path)
    model = doc.getModel()

    # constant parameters in model
    pids_const = []
    for p in model.getListOfParameters():
        if p.getConstant() == True:
            pids_const.append(p.getId())

    # print('constant parameters:', len(pids_const))

    # filter parameters
    parameters = {}
    for pid in pids_const:
        # dose parameters
        if (pid.startswith("IVDOSE_")) or (pid.startswith("PODOSE_")):
            continue

        # physical parameters
        if (pid.startswith("Mr_")) or pid in ["R_PDB"]:
            continue

        # zero parameters
        value = r[pid]
        if np.abs(value) < 1E-8:
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