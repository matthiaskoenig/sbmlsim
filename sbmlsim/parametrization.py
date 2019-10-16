"""
Parametrization handles collections of changes to the model
before simulation. Helper functions exist for the creation
and combination of such changes for simulations.
"""
# TODO: support of distributions, refactor in factories which create the
# respective changesets

from collections import OrderedDict
import logging
import roadrunner
import numpy as np
import libsbml


class ChangeSet(object):
    def init(self):
        pass

    @staticmethod
    def scan_changeset(selector: str, values):
        """Create changeset to scan parameter.

        :param r: RoadRunner model
        :param selector: selector in model
        :param values:
        :return: changeset
        """
        return [{selector: value} for value in values]

    @classmethod
    def parameter_sensitivity_changeset(cls, r: roadrunner.RoadRunner,
                                        sensitivity: float=0.1):
        """ Create changeset to calculate parameter sensitivity.

        :param r: model
        :param sensitivity: change for calculation of sensitivity
        :return: changeset
        """
        p_dict = cls._parameters_for_sensitivity(r)
        changeset = []
        for pid, value in p_dict.items():
            for change in [1.0 + sensitivity, 1.0 - sensitivity]:
                changeset.append(
                    {pid: change*value}
                )
        return changeset

    @staticmethod
    def _parameters_for_sensitivity(r, exclude_filter=None,
                                    exclude_zero: bool = True,
                                    zero_eps: float = 1E-8):
        """ Get parameter ids for sensitivity analysis.

        Values around current model state are used.

        :param r:
        :param exclude_filter: filter function to exclude parameters
        :param exclude_zero: exclude parameters which are zero
        :return:
        """
        doc = libsbml.readSBMLFromString(
            r.getSBML())  # type: libsbml.SBMLDocument
        model = doc.getModel()  # type: libsbml.Model

        # constant parameters
        pids_const = []
        for p in model.getListOfParameters():
            if p.getConstant() is True:
                pids_const.append(p.getId())

        # filter parameters
        parameters = OrderedDict()
        for pid in sorted(pids_const):
            if exclude_filter and exclude_filter(pid):
                continue

            value = r[pid]
            if exclude_zero:
                if np.abs(value) < zero_eps:
                    continue

            parameters[pid] = value

        return parameters

    @staticmethod
    def species_sensitivity_changeset(r, sensitivity=0.1):
        # TODO: initial condition sensitivity
        raise NotImplemented



