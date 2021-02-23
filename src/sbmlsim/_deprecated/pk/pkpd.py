"""Methods specific to pkdb models."""
import logging
import re
from typing import Dict

import roadrunner


logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Initial values
# -------------------------------------------------------------------------------------------------
# Helper functions for setting initial values in the model.
# Substances which are transported in the body can hereby be initialized in all
# tissues to identical values (which removes the distribution kinetics).
# -------------------------------------------------------------------------------------------------


def init_concentrations_changes(r: roadrunner.RoadRunner, skey, value: float):
    """Get changes to set initial concentrations for skey."""
    return _set_initial_values(r, skey, value, method="concentration")


def init_amounts_changes(r: roadrunner.RoadRunner, skey, value):
    """Set initial amounts for skey."""
    return _set_initial_values(r, skey, value, method="amount")


def _set_initial_values(
    r: roadrunner.RoadRunner, sid, value, method="concentration"
) -> Dict:
    """Set the initial concentration of a distributing substance.

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

        if method == "concentration":
            rkey = f"[{key}]"

        if "urine" in rkey:
            logging.debug("urinary values are not set")
            continue

        changeset[rkey] = value

    return changeset


def get_species_keys(skey, species_ids):
    """Get keys of substance in given list of ids.

    Relies on naming patterns of ids. This does not get the species ids of the submodels,
    but only of the top model.

    :param skey: substance key
    :param species_ids: list of species ids to filter from
    :return:
    """
    keys = []
    for species_id in species_ids:
        # use regular expression to find ids
        # This pattern is not very robust !!! FIXME (e.g. blood vs. plasma)
        pattern = r"^[AC][a-z]+(_plasma)*\_{}$".format(skey)
        match = re.search(pattern, species_id)
        if match:
            # print("match:", species_id)
            keys.append(species_id)

    return keys
