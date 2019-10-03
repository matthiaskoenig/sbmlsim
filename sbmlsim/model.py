"""
Functions for model loading and model manipulation.
"""
import logging
import roadrunner


def load_model(path, selections: bool = True) -> roadrunner.RoadRunner:
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


def set_selections(r: roadrunner.RoadRunner) -> None:
    """ Sets the full model selections. """
    r.timeCourseSelections = ["time"] \
                             + r.model.getFloatingSpeciesIds() \
                             + r.model.getBoundarySpeciesIds() \
                             + r.model.getGlobalParameterIds() \
                             + r.model.getReactionIds() \
                             + r.model.getCompartmentIds()
    r.timeCourseSelections += [f'[{key}]' for key in (
                r.model.getFloatingSpeciesIds() + r.model.getBoundarySpeciesIds())]
