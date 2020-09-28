"""
Module for Creation and managment of Samples and the SampleParameters
used in simulations.

 Sample parameter defines the value of a single parameter.
    In most cases these parameters corresponds to parameters in an SBML.
    Depending on the ptype different behavior can be implemented, for instance
    in the integration.
    key, value, unit correspond to id, value, unit in the SBML.
"""

# TODO: rename Sample -> ParameterCollection (??) , better naming. This is not really describing what it is doing.
from __future__ import division, print_function

from copy import deepcopy

import django


django.setup()
from simapp.models import Parameter, ParameterType


class SampleParameterException(Exception):
    """ Exception for any problem with parameter samples. """

    pass


class SampleParameter(object):
    """Class for storing key = value [unit] settings for simulation.
    The key corresponds to the identifier of the object to set and is
    in most cases an SBML SBase identifier.
    Allowed types are the allowed parameter types defined in the
    django model.
    """

    def __init__(self, key, value, unit, parameter_type):
        self.key = key
        self.value = value
        self.unit = unit
        self.parameter_type = parameter_type

    @classmethod
    def from_parameter(cls, p):
        """ Works with SampleParameter or django parameter. """
        if hasattr(p, "key"):
            # Sample parameter
            return cls(p.key, p.value, p.unit, p.parameter_type)
        else:
            # django parameter
            return cls(p.key, p.value, p.unit, p.parameter_type)

    def __repr__(self):
        return "<{} = {:.3E} [{}] ({})>".format(
            self.key, self.value, self.unit, ParameterType.labels[self.parameter_type]
        )


class Sample(dict):
    """ A sample is a collection of SampleParameters. """

    def add_parameters(self, parameters):
        for p in parameters:
            self.add_parameter(p)

    def add_parameter(self, p):
        if isinstance(p, Parameter):
            p = SampleParameter.from_parameter(p)
        if not isinstance(p, SampleParameter):
            raise SampleParameterException
        self[p.key] = p

    def get_parameter(self, key):
        return self[key]

    @property
    def parameters(self):
        return self.values()

    @staticmethod
    def from_parameters(parameters):
        s = Sample()
        s.add_parameters(parameters)
        return s

    @classmethod
    def set_parameter_in_samples(cls, sample_par, samples):
        """Set SampleParameter in all samples.
        Is SampleParameter with given key already exists it is overwritten."""
        for s in samples:
            s.add_parameter(sample_par)
        return samples

    @staticmethod
    def deepcopy_samples(samples):
        """Returns a deepcopy of the list of samples.
        Required for the creation of derived samples
        """
        # TODO: check if this works
        return deepcopy(samples)


def set_parameters_in_samples_XXX(parameters, samples):
    """
    This functionality has to be much clearer and must be documented much better.
    What is this doing exactly ??
    ? how is the parameters structured ?"""
    # TODO: refactor
    for pset in parameters:
        ParameterType.check_type(pset["parameter_type"])

    Np = len(parameters)  # numbers of parameters to set
    Nval = len(parameters[0]["values"])  # number of values from first p_dict

    new_samples = []
    for s in samples:
        for k in range(Nval):
            # make a copy of the dictionary
            snew = s.copy()
            # set all the information
            for i in range(Np):
                p_dict = parameters[i]
                snew[p_dict["pid"]] = (
                    p_dict["pid"],
                    p_dict["values"][k],
                    p_dict["unit"],
                    p_dict["parameter_type"],
                )
            new_samples.append(snew)
    return new_samples
