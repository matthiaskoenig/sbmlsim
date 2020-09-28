"""
Definition of parameter distributions.

Parameter distributions are used for the sampling of parameters in
models, but could also be applied in other fields.
A single sample is a combination of parameters to be set in the model.

Only fixed parameters in the model should be sampled, i.e.
model parameters which are constant and not calculated by initial assignment
or based on other parameters.


Parameter distributions are defined in the best case scenario
in an external file, for instance csv and loaded from this source.
An important part is that the units of the parameter distribution have
to be identical to the units of the actual parameter being set, i.e.
the parameter distributions have to fit to the actual parameters in
the model.
Parameters should be given in SI units (but have to be at least the units defined
in the SBML so that no additional conversions of units are necessary.
"""

# FIXME: implememt distributions via changesets.
# - support different sampling regimes
# - support dist package


# TODO: In future versions it should also be possible to set non-terminal parameters
# in SBML models. It will be necessary to change the model structure to allow for that.
# For instance via replacement of initial assignments via fixed parameters.
# The replacements have to be performed in the model when these parameters are set.
from __future__ import division, print_function

import math
from enum import Enum

import numpy as np
from multiscale.util.util_classes import EnumType
from samples import SampleParameter


class DistributionType(EnumType, Enum):
    CONSTANT = 0  # (MEAN) required
    NORMAL = 1  # (MEAN, LOG) required
    LOGNORMAL = 2  # (MEANLOG, STDLOG) required


class DistributionParameterType(EnumType, Enum):
    MEAN = 0
    STD = 1
    MEANLOG = 2
    STDLOG = 3


class Distribution(object):
    """Class handles the distribution parameters.
    For every distributed parameter a single Distribution object is generated.
    """

    # TODO: generate subclasses for the different distribution types, i.e. NormalDistribution, LogNormalDistribution
    class DistException(Exception):
        pass

    def __init__(self, distribution_type, parameters):
        self.distribution_type = distribution_type
        self.parameters = parameters

        if self.distribution_type == DistributionType.LOGNORMAL:
            self.convert_lognormal_mean_std()

        self.check()
        self.check_parameters()

    @property
    def key(self):
        """Return the common key of the parameters.
        I.e. the key of the parameter which is set via the distribution."""
        return self.parameters.values()[0].key

    @property
    def unit(self):
        return self.parameters.values()[0].unit

    @property
    def parameter_type(self):
        return self.parameters.values()[0].parameter_type

    def samples(self, n_samples=1):
        """ Create samples from the distribution. """
        if self.distribution_type == DistributionType.CONSTANT:
            data = self.parameters[DistributionParameterType.MEAN].value * np.ones(
                n_samples
            )

        elif self.distribution_type == DistributionType.NORMAL:
            data = np.random.normal(
                self.parameters[DistributionParameterType.MEAN].value,
                self.parameters[DistributionParameterType.STD].value,
                n_samples,
            )
        elif self.distribution_type == DistributionType.LOGNORMAL:
            data = np.random.lognormal(
                self.parameters[DistributionParameterType.MEANLOG].value,
                self.parameters[DistributionParameterType.STDLOG].value,
                n_samples,
            )
        else:
            raise Distribution.DistException(
                "DistType not supported: {}".format(self.distribution_type)
            )

        if n_samples == 1:
            return data[0]
        return data

    def mean(self):
        """ Mean value of distribution for mean sampling. """
        if self.distribution_type in (
            DistributionType.CONSTANT,
            DistributionType.NORMAL,
            DistributionType.LOGNORMAL,
        ):
            return self.parameters[DistributionParameterType.MEAN].value
        else:
            raise Distribution.DistException(
                "DistType not supported: {}".format(self.distribution_type)
            )

    def convert_lognormal_mean_std(self):
        """ Convert lognormal mean, std => meanlog and stdlog. """
        if (
            DistributionParameterType.MEAN in self.parameters
            and DistributionParameterType.STD in self.parameters
        ):
            # get the old sample parameter
            sp_mean = self.parameters[DistributionParameterType.MEAN]
            sp_std = self.parameters[DistributionParameterType.STD]
            # calculate meanlog and stdlog
            meanlog = Distribution.calc_meanlog(sp_mean.value, sp_std.value)
            stdlog = Distribution.calc_stdlog(sp_mean.value, sp_std.value)
            # store new parameters
            self.parameters[DistributionParameterType.MEANLOG] = SampleParameter(
                sp_mean.key, meanlog, sp_mean.unit, sp_mean.parameter_type
            )
            self.parameters[DistributionParameterType.STDLOG] = SampleParameter(
                sp_std.key, stdlog, sp_std.unit, sp_std.parameter_type
            )
            # remove old paramters
            # del self.parameters[DistParsType.MEAN]
            # del self.parameters[DistParsType.STD]

    def check(self):
        """ Check consistency of the defined distributions. """
        if self.distribution_type == DistributionType.CONSTANT:
            if len(self.parameters) != 1:
                raise Distribution.DistException(
                    "Constant distribution has 1 parameter."
                )
            self.parameters[DistributionParameterType.MEAN]

        elif self.distribution_type == DistributionType.NORMAL:
            if len(self.parameters) != 2:
                raise Distribution.DistException("Normal distribution has 2 parameter.")
            self.parameters[DistributionParameterType.MEAN]
            self.parameters[DistributionParameterType.STD]

        elif self.distribution_type == DistributionType.LOGNORMAL:
            if len(self.parameters) < 2:
                raise Distribution.DistException(
                    "LogNormal distribution has 2 parameter."
                )
            self.parameters[DistributionParameterType.MEANLOG]
            self.parameters[DistributionParameterType.STDLOG]
        else:
            raise Distribution.DistException(
                "DistType not supported: {}".format(self.distribution_type)
            )

    def check_parameters(self):
        """ Check consistency of parameters within distribution. """
        # check that the keys are identical for all parameters in distribution
        key = None
        unit = None
        parameter_type = None
        for p in self.parameters.values():
            if not key:
                key = p.key
                unit = p.unit
                parameter_type = p.parameter_type
                continue
            if p.key != key:
                raise Distribution.DistException(
                    "All parameters of distribution need same key"
                )
            if p.unit != unit:
                raise Distribution.DistException(
                    "All parameters of distribution need same unit"
                )
            if p.parameter_type != parameter_type:
                raise Distribution.DistException(
                    "All parameters of distribution need same parameter_type"
                )

        if self.distribution_type == DistributionType.CONSTANT:
            pass

        elif self.distribution_type == DistributionType.NORMAL:
            pass

        elif self.distribution_type == DistributionType.LOGNORMAL:
            pass

    def __repr__(self):
        return "{} : {}".format(self.distribution_type, self.parameters)

    @staticmethod
    def calc_meanlog(mean, std):
        """Calculatate meanlog from mean and std.
        :param mean:
        :param std:
        :return:
        """
        return math.log(mean ** 2 / math.sqrt(std ** 2 + mean ** 2))

    @staticmethod
    def calc_stdlog(mean, std):
        """Calculate stdlog from mean and std.
        :param mean:
        :param std:
        :return:
        """
        return math.sqrt(math.log(std ** 2 / mean ** 2 + 1))
