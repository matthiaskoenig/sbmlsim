"""
Methods specific to pkdb models
"""
import logging
import re

# -------------------------------------------------------------------------------------------------
# Initial values
# -------------------------------------------------------------------------------------------------
# Helper functions for setting initial values in the model.
# Substances which are transported in the body can hereby be initialized in all
# tissues to identical values (which removes the distribution kinetics).
# -------------------------------------------------------------------------------------------------

def set_initial_concentrations(r, skey, value: float):
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

    return: changeset for model
    """
    if method not in ["amount", "concentration"]:
        raise ValueError

    species_ids = r.model.getFloatingSpeciesIds() + r.model.getBoundarySpeciesIds()
    species_keys = get_species_keys(sid, species_ids)

    changeset = {}
    for key in species_keys:
        if 'urine' in key:
            logging.warning("urinary values are not set")
            continue
        if method == "concentration":
            rkey = f'init([{key}])'
        elif method == "amount":
            rkey = f'init({value})'
        # print(f'{rkey} <- {value}')

        changeset[rkey] = value

    return changeset


def get_species_keys(skey, species_ids):
    """ Get keys of substance in given list of ids.

    Relies on naming patterns of ids. This does not get the species ids of the submodels,
    but only of the top model.

    :param skey: substance key
    :param species_ids: list of species ids to filter from
    :return:
    """
    keys = []
    for species_id in species_ids:
        # use regular expression to find ids
        pattern = r'^A[a-z]+(_blood)*\_{}$'.format(skey)
        match = re.search(pattern, species_id)
        if match:
            # print("match:", species_id)
            keys.append(species_id)

    return keys