"""
Methods specific to pkdb models
"""
import logging
import re
import roadrunner

from sbmlsim.units import Units, ureg
from pint.errors import DimensionalityError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Initial values
# -------------------------------------------------------------------------------------------------
# Helper functions for setting initial values in the model.
# Substances which are transported in the body can hereby be initialized in all
# tissues to identical values (which removes the distribution kinetics).
# -------------------------------------------------------------------------------------------------

def init_concentrations_changes(r: roadrunner.RoadRunner, skey, value: float):
    """ Changes to set initial concentrations for skey.

    :param r: roadrunner model
    :param skey: substance key
    :param value: new value in model units
    :return:
    """
    return _set_initial_values(r, skey, value, method="concentration")


def init_amounts_changes(r: roadrunner.RoadRunner, skey, value):
    """ Set initial amounts for skey.

    :param r: roadrunner model
    :param skey:
    :param value:
    :return:
    """
    return _set_initial_values(r, skey, value, method="amount")


def _set_initial_values(r: roadrunner.RoadRunner, sid, value, method="concentration") -> dict:
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

    # get units from model
    units = Units.get_units_from_sbml(r.getCurrentSBML())

    for key in species_keys:

        if method == "concentration":
            rkey = f'[{key}]'

        # inject units
        if hasattr(value, "units"):
            try:
                # check that unit can be converted
                value.to(units[rkey])
            except DimensionalityError as err:
                logger.error(f"DimensionalityError "
                             f"'{rkey} = {value}'. {err}")
                raise err
        else:
            value = value * units[rkey]
            logger.warning(f"Not possible to check units, model units assumed: '{rkey} = {value}'")

        if 'urine' in rkey:
            # FIXME
            logging.warning("urinary values are not set")
            continue

        # FIXME: bugfix for json export
        if hasattr(value, "units"):
            logger.warning("units ignored ! FIXME")
            value = value.magnitude

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