"""Helpers for calculating model sensitivities and uncertainties.

Allows to get sets of changes from given model instance.
"""

from enum import Enum
from typing import Iterable

import libsbml
import numpy as np
from pymetadata import log
from pymetadata.console import console
import roadrunner

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Dimension, ScanSim, TimecourseSim


logger = log.get_logger(__name__)


class SensitivityType(Enum):
    """Type of sensitivity."""

    PARAMETER_SENSITIVITY = 1
    SPECIES_SENSITIVITY = 2
    All_SENSITIVITY = 3


class DistributionType(Enum):
    """Type of supported distributions.

    # FIXME: support lognormal
    """

    NORMAL_DISTRIBUTION = 1


class ModelSensitivity(object):
    """Helpers for calculating model sensitivity."""

    @staticmethod
    def difference_sensitivity_scan(
        model: roadrunner.RoadRunner,
        simulation: TimecourseSim,
        difference: float = 0.1,
        stype: SensitivityType = SensitivityType.PARAMETER_SENSITIVITY,
        exclude_filter=None,
        exclude_zero: bool = True,
        zero_eps: float = 1e-8,
    ) -> ScanSim:
        """Create a parameter sensitivity scan for given TimecourseSimulation.

        :param model: model for execution (needed to select parameters)
        :param simulation: timecourse simulation to scan
        :param difference: change in parameters, i.e. every parameter (which is not excluded) is changed to
                           '(1.0 - difference) * value' and '(1.0 + difference) * value'
        :param stype: which sensitivity (parameters or species)
        :param exclude_filter: filter function which defines which parameters should be excluded from scan
        :param exclude_zero: parameters with a value of abs(value)<zero_eps are excluded from scan
        :param zero_eps: epsilon for zero values
        :return:
        """

        dim = ModelSensitivity.create_difference_dimension(
            model=model,
            changes=simulation.timecourses[0].changes,
            difference=difference,
            stype=stype,
            exclude_filter=exclude_filter,
            exclude_zero=exclude_zero,
            zero_eps=zero_eps,
        )
        scan = ScanSim(
            simulation=simulation,
            dimensions=[
                dim,
            ],
            mapping={"dim_sens": 0},
        )
        return scan

    @staticmethod
    def distribution_sensitivity_scan(
        model: roadrunner.RoadRunner,
        simulation: TimecourseSim,
        cv: float = 0.1,
        size: int = 10,
        distribution: DistributionType = DistributionType.NORMAL_DISTRIBUTION,
        stype: SensitivityType = SensitivityType.PARAMETER_SENSITIVITY,
        exclude_filter=None,
        exclude_zero: bool = True,
        zero_eps: float = 1e-8,
    ) -> ScanSim:
        """Get sensitivity scan based on distributions for values."""
        dim = ModelSensitivity.create_sampling_dimension(
            model=model,
            changes=simulation.timecourses[0].changes,
            cv=cv,
            size=size,
            distribution=distribution,
            stype=stype,
            exclude_filter=exclude_filter,
            exclude_zero=exclude_zero,
            zero_eps=zero_eps,
        )
        scan = ScanSim(
            simulation=simulation,
            dimensions=[
                dim,
            ],
            mapping={"dim_sens": 0},
        )
        return scan

    @staticmethod
    def create_sampling_dimension(
        model: roadrunner.RoadRunner,
        changes: dict = None,
        cv: float = 0.1,
        size: int = 10,
        distribution: DistributionType = DistributionType.NORMAL_DISTRIBUTION,
        stype: SensitivityType = SensitivityType.PARAMETER_SENSITIVITY,
        exclude_filter=None,
        exclude_zero: bool = True,
        zero_eps: float = 1e-8,
    ) -> Dimension:
        """Create list of dimensions for sampling parameter values.

        Only parameters relevant for "GU_", "LI_" and "KI_" models are
        sampled.

        cv: coeffient of variation (sigma/mean) -> sigma = cv*mean
        """
        p_ref = ModelSensitivity.reference_dict(
            model=model,
            changes=changes,
            stype=stype,
            exclude_filter=exclude_filter,
            exclude_zero=exclude_zero,
            zero_eps=zero_eps,
        )
        Q_ = model.Q_

        changes = {}
        for key, value in p_ref.items():
            magnitude = value.magnitude
            # FIXME: use lognormal to avoid negative values, or remove negative samples
            if distribution == DistributionType.NORMAL_DISTRIBUTION:
                values = np.random.normal(magnitude, scale=magnitude * cv, size=size)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            changes[key] = Q_(values, value.units)

        return Dimension("dim_sens", changes=changes)

    @staticmethod
    def create_difference_dimension(
        model: roadrunner.RoadRunner,
        changes: dict = None,
        difference: float = 0.1,
        stype: SensitivityType = SensitivityType.PARAMETER_SENSITIVITY,
        exclude_filter=None,
        exclude_zero: bool = True,
        zero_eps: float = 1e-8,
    ) -> Dimension:
        """Create list of dimensions for sampling parameter values.

        Only parameters relevant for "GU_", "LI_" and "KI_" models are
        sampled.

        cv: coeffient of variation (sigma/mean) -> sigma = cv*mean
        """
        p_ref = ModelSensitivity.reference_dict(
            model=model,
            changes=changes,
            stype=stype,
            exclude_filter=exclude_filter,
            exclude_zero=exclude_zero,
            zero_eps=zero_eps,
        )
        Q_ = model.Q_

        changes = {}
        index = 0
        num_pars = len(p_ref)
        for key, value in p_ref.items():
            values = np.ones(shape=(2 * num_pars,)) * value.magnitude
            # change parameters in correct position
            values[index] = value.magnitude * (1.0 + difference)
            values[index + num_pars] = value.magnitude * (1.0 - difference)
            changes[key] = Q_(values, value.units)
            index += 1
        return Dimension("dim_sens", changes=changes)

    @staticmethod
    def reference_dict(
        model: RoadrunnerSBMLModel,
        changes: dict = None,
        stype: SensitivityType = SensitivityType.PARAMETER_SENSITIVITY,
        exclude_filter=None,
        exclude_zero: bool = True,
        zero_eps: float = 1e-8,
    ) -> dict:
        """Get key:value dict for sensitivity analysis.

        Values are based on the reference state of the model with the applied
        changes. Values in current model state are used.

        :param model:
        :param exclude_filter: filter function to exclude parameters,
            excludes parameter id if the filter function is True
        :param exclude_zero: exclude parameters which are zero
        :return:
        """
        # reset model
        model.r.resetAll()

        # apply normalized model changes
        if changes is None:
            changes = {}
        for key, item in changes.items():
            try:
                model.r[key] = item.magnitude
            except AttributeError as err:
                logger.error(
                    f"Change is not a Quantity with unit: '{key} = {item}'. "
                    f"Add units to all changes."
                )
                raise err

        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(model.r.getSBML())
        sbml_model: libsbml.Model = doc.getModel()

        ids = []

        if stype in {
            SensitivityType.PARAMETER_SENSITIVITY,
            SensitivityType.All_SENSITIVITY,
        }:
            # constant parameters
            p: libsbml.Parameter
            for p in sbml_model.getListOfParameters():
                if p.getConstant() is True:
                    ids.append(p.getId())
        if stype in {
            SensitivityType.SPECIES_SENSITIVITY,
            SensitivityType.All_SENSITIVITY,
        }:
            # initial species amount
            s: libsbml.Species
            for s in sbml_model.getListOfSpecies():
                ids.append(s.getId())

        def value_dict(ids: Iterable[str]) -> dict[str, float]:
            """Key: value dict from current model state.

            Non-zero and exclude filtering is applied.
            """
            d: dict[str, float] = {}
            for key in sorted(ids):
                if exclude_filter and exclude_filter(key):
                    continue

                value = model.r[key]
                if exclude_zero:
                    if np.abs(value) < zero_eps:
                        continue
                d[key] = value
            return d

        return value_dict(ids)

    @staticmethod
    def apply_change_to_dict(ref_dict, change: float = 0.1):
        """Apply relative change to reference dictionary.

        :param ref_dict: {key: value} dictionary to change
        :param change: relative change to apply.
        :return:
        """
        d = ref_dict.copy()
        d = {k: v * (1.0 + change) for k, v in d.items()}
        return d


if __name__ == "__main__":
    from sbmlsim.resources import REPRESSILATOR_SBML

    console.print("Loading model")
    model = roadrunner.RoadRunner(str(REPRESSILATOR_SBML))

    console.print("Reference dict")
    p_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.PARAMETER_SENSITIVITY
    )
    s_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.SPECIES_SENSITIVITY
    )

    console.print("Apply changes")
    console.print(p_ref)
    console.print(ModelSensitivity.apply_change_to_dict(p_ref, change=0.1))
    console.print(ModelSensitivity.apply_change_to_dict(p_ref, change=-0.1))

    console.print(s_ref)
    console.print(ModelSensitivity.apply_change_to_dict(s_ref, change=0.1))
    console.print(ModelSensitivity.apply_change_to_dict(s_ref, change=-0.1))
