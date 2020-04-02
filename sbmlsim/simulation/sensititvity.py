"""
Helpers for calculating model sensitivities.

Allows to get sets of changes from given model instance.
"""
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.simulation import ScanSim, Dimension, Timecourse, TimecourseSim

from copy import deepcopy
import libsbml
import numpy as np
from enum import Enum


class SensitivityType(Enum):
    PARAMETER_SENSITIVITY = 1
    SPECIES_SENSITIVITY = 2


class ModelSensitivity(object):

    @staticmethod
    def apply_change(ref_dict, change: float = 0.1):
        """ Applies relative change to reference dictionary.

        :param ref_dict: {key: value} dictionary to change
        :param change: relative change to apply.
        :return:
        """
        d = ref_dict.copy()
        d = {k: v * (1.0 + change) for k, v in d.items()}
        return d

    @staticmethod
    def reference_values(model: RoadrunnerSBMLModel,
                          stype: SensitivityType = SensitivityType.PARAMETER_SENSITIVITY,
                          exclude_filter=None,
                          exclude_zero: bool = True,
                          zero_eps: float = 1E-8):
        """Returns keys and values dict for sensitivity analysis of the reference
        state. Values in current model state are used.

        :param model:
        :param exclude_filter: filter function to exclude parameters,
            excludes parameter id if the filter function is True
        :param exclude_zero: exclude parameters which are zero
        :return:
        """

        doc = libsbml.readSBMLFromString(model.model.getSBML())  # type: libsbml.SBMLDocument
        sbml_model = doc.getModel()  # type: libsbml.Model

        ids = []

        if stype == SensitivityType.PARAMETER_SENSITIVITY:
            # constant parameters
            for p in sbml_model.getListOfParameters():  # type: libsbml.Parameter
                if p.getConstant() is True:
                    ids.append(p.getId())
        elif stype == SensitivityType.SPECIES_SENSITIVITY:
            # initial species amount
            for s in sbml_model.getListOfSpecies():  # type: libsbml.Species
                ids.append(s.getId())

        def value_dict(ids):
            """Key: value dict from current model state.
            Non-zero and exclude filtering is applied.
            """
            ureg = model.ureg
            udict = model.udict
            Q_ = model.Q_

            d = {}
            for id in sorted(ids):
                if exclude_filter and exclude_filter(id):
                    continue

                value = model.model[id]
                if exclude_zero:
                    if np.abs(value) < zero_eps:
                        continue
                d[id] = Q_(value, udict[id])
            return d

        return value_dict(ids)

    def create_simulations(self, simulation, value_dict):
        """ Creates list of simulations from given simulation.
        Every key:value pair creates a new simulation.

        :param simulation:
        :param change_dict:
        :return:
        """
        simulations = []
        for key, value in value_dict.items:

            sim_new = deepcopy(simulation)
            # changes are mixed in the first timecourse
            tc = sim_new.run_timecourse[0]
            tc.add_change(key, value)
            simulations.append(sim_new)

        return simulations


if __name__ == "__main__":
    from pprint import pprint
    from sbmlsim.tests.constants import MODEL_REPRESSILATOR
    model = RoadrunnerSBMLModel(MODEL_REPRESSILATOR)

    p_ref = ModelSensitivity.reference_values(
        model=model,
        stype=SensitivityType.PARAMETER_SENSITIVITY
    )
    s_ref = ModelSensitivity.reference_values(
        model=model,
        stype=SensitivityType.SPECIES_SENSITIVITY
    )

    pprint(p_ref)
    pprint(ModelSensitivity.apply_change(p_ref, change=0.1))
    pprint(ModelSensitivity.apply_change(p_ref, change=-0.1))
    exit()

    pprint(s_ref)
    pprint(ModelSensitivity.apply_change(s_ref, change=0.1))
    pprint(ModelSensitivity.apply_change(s_ref, change=-0.1))

    simulation = TimecourseSim([
            Timecourse(0, 100, steps=100, changes={})
        ]
    )
    simulations = ModelSensitivity.create_simulations(
        simulation=simulation,
        change_dict=ModelSensitivity.apply_change(p_ref, change=0.1)
    )
    print(simulations)

