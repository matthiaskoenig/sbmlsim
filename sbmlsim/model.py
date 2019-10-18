"""
Functions for model loading, model manipulation and settings on the integrator.
"""
import logging
import roadrunner
import libsbml
import numpy as np
import pandas as pd

MODEL_CHANGE_BOUNDARY_CONDITION = "boundary_condition"

def load_model(path, selections: bool = True) -> roadrunner.RoadRunner:
    """ Loads the latest model version.

    :param path: path to SBML model or SBML string
    :param selections: boolean flag to set selections
    :return: roadrunner instance
    """
    logging.info("Loading: '{}'".format(path))
    r = roadrunner.RoadRunner(path)
    if selections:
        set_timecourse_selections(r)
    return r

# --------------------------------------
# Selections
# --------------------------------------
def set_timecourse_selections(r: roadrunner.RoadRunner, selections=None) -> None:
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
# Integrator settings
# --------------------------------
# FIXME: implement setting of ode solver properties: variable_step_size, stiff, absolute_tolerance, relative_tolerance

def set_integrator_settings(r: roadrunner.RoadRunner, **kwargs) -> None:
    """ Set integrator settings. """
    integrator = r.getIntegrator()
    for key, value in kwargs.items():
        # adapt the absolute_tolerance relative to the amounts
        if key == "absolute_tolerance":
            value = value * min(r.model.getCompartmentVolumes())
        integrator.setValue(key, value)
    return integrator


def set_default_settings(self):
    """ Set default settings of integrator. """
    self.set_integrator_settings(
            variable_step_size=True,
            stiff=True,
            absolute_tolerance=1E-8,
            relative_tolerance=1E-8
    )




# --------------------------------
# Model information
# --------------------------------
def parameter_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
    """
    Create GlobalParameter DataFrame.
    :return: pandas DataFrame
    """
    r_model = r.model  # type: roadrunner.ExecutableModel
    doc = libsbml.readSBMLFromString(r.getCurrentSBML())  # type: libsbml.SBMLDocument
    model = doc.getModel()  # type: libsbml.Model
    sids = r_model.getGlobalParameterIds()
    parameters = [model.getParameter(sid) for sid in sids]  # type: List[libsbml.Parameter]
    data = {
        'sid': sids,
        'value': r_model.getGlobalParameterValues(),
        'unit': [p.units for p in parameters],
        'constant': [p.constant for p in parameters],
        'name': [p.name for p in parameters],
        }
    df = pd.DataFrame(data, columns=['sid', 'value', 'unit', 'constant', 'name'])
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
    species = [model.getSpecies(sid) for sid in sids]  # type: List[libsbml.Species]

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

    return pd.DataFrame(data, columns=['sid', 'concentration', 'amount', 'unit', 'constant',
                                'boundaryCondition', 'species', 'name'])


# --------------------------------
# Model manipulation
# --------------------------------
def clamp_species(r: roadrunner.RoadRunner, sids, boundary_condition=True) -> roadrunner.RoadRunner:
    """ Clamp/Free specie(s) via setting boundaryCondition=True/False.

    This requires changing the SBML and ODE system.

    :param r: roadrunner.RoadRunner
    :param sids: sid or iterable of sids
    :param boundary_condition: boolean flag to clamp (True) or free (False) species
    :return: modified roadrunner.RoadRunner
    """
    # get model for current SBML state
    sbml_str = r.getCurrentSBML()
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
                logging.error("SId in clamp does not match species: {}".format(sbase))
                return None

    # create modified roadrunner instance
    sbmlmod_str = libsbml.writeSBMLToString(doc)
    rmod = load_model(sbmlmod_str)  # type: roadrunner.RoadRunner
    set_timecourse_selections(rmod, r.timeCourseSelections)

    return rmod


if __name__ == "__main__":
    from sbmlsim.tests.constants import MODEL_REPRESSILATOR

    r = load_model(MODEL_REPRESSILATOR)


    print("-" * 80)
    df = parameter_df(r)
    print(df)
    print("-" * 80)
    df = species_df(r)
    print(df)
    print("-" * 80)