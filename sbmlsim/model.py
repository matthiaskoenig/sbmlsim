"""
Functions for model loading, model manipulation and settings on the integrator.
"""
from pathlib import Path
import logging
import roadrunner
import libsbml
import numpy as np
import pandas as pd
from typing import List, Tuple

MODEL_CHANGE_BOUNDARY_CONDITION = "boundary_condition"


def load_model(path: Path, selections: List[str] = None) -> roadrunner.RoadRunner:
    """ Loads the latest model version.

    :param path: path to SBML model or SBML string
    :param selections: boolean flag to set selections
    :return: roadrunner instance
    """
    logging.info("Loading: '{}'".format(path))
    if isinstance(path, Path):
        path = str(path)

    r = roadrunner.RoadRunner(path)
    set_timecourse_selections(r, selections)
    return r


def copy_model(r: roadrunner.RoadRunner) -> roadrunner.RoadRunner:
    """Copy current model.

    :param r:
    :return:
    """
    # independent copy by parsing SBML
    sbml_str = r.getCurrentSBML()  # type: str
    r_copy = roadrunner.RoadRunner(sbml_str)

    # copy state of instance
    copy_model_state(r_from=r, r_to=r_copy)
    return r_copy


def copy_model_state(r_from: roadrunner.RoadRunner, r_to: roadrunner.RoadRunner,
                     copy_selections=True,
                     copy_integrator=True,
                     copy_states=True):
    """ Copy roadrunner state between model instances

    :param r_from:
    :param r_to:
    :return:
    """
    if copy_selections:
        # copy of selections (by value)
        r_to.timeCourseSelections = r_from.timeCourseSelections
        r_to.steadyStateSelections = r_from.steadyStateSelections

    if copy_integrator:
        # copy integrator state
        integrator = r_from.getIntegrator()  # type: roadrunner.Integrator
        integrator_name = integrator.getName()
        r_to.setIntegrator(integrator_name)

        settings_keys = integrator.getSettings()  # type: Tuple[str]
        print(settings_keys)
        for key in settings_keys:
            r_to.integrator.setValue(key, integrator.getValue(key))

    if copy_states:
        # FIXME: implement: copying of current state to initial state
        # for state variables
        pass


def clamp_species(r: roadrunner.RoadRunner, sids,
                  boundary_condition=True) -> roadrunner.RoadRunner:
    """ Clamp/Free specie(s) via setting boundaryCondition=True/False.

    This requires changing the SBML and ODE system.

    :param r: roadrunner.RoadRunner
    :param sids: sid or iterable of sids
    :param boundary_condition: boolean flag to clamp (True) or free (False) species
    :return: modified roadrunner.RoadRunner
    """
    # get model for current SBML state
    sbml_str = r.getCurrentSBML()
    # FIXME: bug in concentrations!

    # print(sbml_str)

    doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
    model = doc.getModel()  # type: libsbml.Model

    if isinstance(sids, str):
        sids = [sids]

    for sid in sids:
        # set boundary conditions
        sbase = model.getElementBySId(sid)  # type: libsbml.SBase
        if not sbase:
            logging.error("No element for SId in model: {}".format(sid))
            return None
        else:
            if sbase.getTypeCode() == libsbml.SBML_SPECIES:
                species = sbase  # type: libsbml.Species
                species.setBoundaryCondition(boundary_condition)
            else:
                logging.error(
                    "SId in clamp does not match species: {}".format(sbase))
                return None

    # create modified roadrunner instance
    sbmlmod_str = libsbml.writeSBMLToString(doc)
    rmod = load_model(sbmlmod_str)  # type: roadrunner.RoadRunner
    set_timecourse_selections(rmod, r.timeCourseSelections)

    return rmod


# --------------------------------------
# Selections
# --------------------------------------
def set_timecourse_selections(r: roadrunner.RoadRunner,
                              selections: List[str] = None) -> None:
    """ Sets the full model selections. """
    if not selections:
        r_model = r.model  # type: roadrunner.ExecutableModel

        r.timeCourseSelections = ["time"] \
                                 + r_model.getFloatingSpeciesIds() \
                                 + r_model.getBoundarySpeciesIds() \
                                 + r_model.getGlobalParameterIds() \
                                 + r_model.getReactionIds() \
                                 + r_model.getCompartmentIds()
        r.timeCourseSelections += [f'[{key}]' for key in (
                r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds())]
    else:
        r.timeCourseSelections = selections


# --------------------------------------
# Resets
# --------------------------------------
def reset_all(r):
    """ Reset all model variables to CURRENT init(X) values.

    This resets all variables, S1, S2 etc to the CURRENT init(X) values. It also resets all
    parameters back to the values they had when the model was first loaded.
    """
    # FIXME: check if this is still needed
    logging.warning(
        "reset_all is deprecated",
        DeprecationWarning
    )
    r.reset(roadrunner.SelectionRecord.TIME |
            roadrunner.SelectionRecord.RATE |
            roadrunner.SelectionRecord.FLOATING |
            roadrunner.SelectionRecord.GLOBAL_PARAMETER)


# --------------------------------
# Model information
# --------------------------------
def parameter_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
    """
    Create GlobalParameter DataFrame.
    :return: pandas DataFrame
    """
    r_model = r.model  # type: roadrunner.ExecutableModel
    doc = libsbml.readSBMLFromString(
        r.getCurrentSBML())  # type: libsbml.SBMLDocument
    model = doc.getModel()  # type: libsbml.Model
    sids = r_model.getGlobalParameterIds()
    parameters = [model.getParameter(sid) for sid in
                  sids]  # type: List[libsbml.Parameter]
    data = {
        'sid': sids,
        'value': r_model.getGlobalParameterValues(),
        'unit': [p.units for p in parameters],
        'constant': [p.constant for p in parameters],
        'name': [p.name for p in parameters],
    }
    df = pd.DataFrame(data,
                      columns=['sid', 'value', 'unit', 'constant', 'name'])
    return df


def species_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
    """
    Create FloatingSpecies DataFrame.
    :return: pandas DataFrame
    """
    r_model = r.model  # type: roadrunner.ExecutableModel
    sbml_str = r.getCurrentSBML()

    doc = libsbml.readSBMLFromString(sbml_str)  # type: libsbml.SBMLDocument
    model = doc.getModel()  # type: libsbml.Model

    sids = r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
    species = [model.getSpecies(sid) for sid in
               sids]  # type: List[libsbml.Species]

    data = {
        'sid': sids,
        'concentration': np.concatenate([
            r_model.getFloatingSpeciesConcentrations(),
            r_model.getBoundarySpeciesConcentrations()
        ], axis=0),
        'amount': np.concatenate([
            r.model.getFloatingSpeciesAmounts(),
            r.model.getBoundarySpeciesAmounts()
        ], axis=0),
        'unit': [s.getUnits() for s in species],
        'constant': [s.getConstant() for s in species],
        'boundaryCondition': [s.getBoundaryCondition() for s in species],
        'name': [s.getName() for s in species],
    }

    return pd.DataFrame(data, columns=['sid', 'concentration', 'amount', 'unit',
                                       'constant',
                                       'boundaryCondition', 'species', 'name'])


if __name__ == "__main__":
    from sbmlsim.tests.constants import MODEL_REPRESSILATOR

    from sbmlsim.simulation_serial import SimulatorSerial
    from sbmlsim.result import Result
    from sbmlsim.timecourse import Timecourse, TimecourseSim
    from matplotlib import pyplot as plt

    # running first simulation
    simulator = SimulatorSerial(MODEL_REPRESSILATOR)
    result = simulator.timecourse(Timecourse(0, 100, 201, ))

    # make a copy of current model with state
    r_copy = copy_model(simulator.r)

    # continue simulation
    result2 = simulator.timecourse(
        TimecourseSim(Timecourse(100, 200, 201), reset=False))

    plt.plot(result.time, result.X)
    plt.plot(result2.time, result2.X)
    plt.show()

    if True:
        # r = simulator.r
        simulator2 = SimulatorSerial(path=None)
        simulator2.r = r_copy
        result3 = simulator2.timecourse(TimecourseSim(Timecourse(100, 200, 201),
                                                      reset=False))
        plt.plot(result.time, result.X)
        plt.plot(result3.time, result3.X)
        plt.show()
