"""
Implementation of various sampling methods for sample generation
from given distributions.
Supported methods are DISTRIBUTION, MEAN, LHS or MIXED.

[DISTRIBUTION]
Samples from the given distribution.

[LHS]
Latin Hypercube sampling
This 'multi-start' approach facilitates a broad coverage of the
parameter search space in order to find the global optimum.
Latin hypercube sampling [17] of the initial parameter guesses
can be used to guarantee that each parameter estimation run
starts in a different region in the high-dimensional parameter
space. This method prohibits that randomly selected starting
points are accidentally close to each other. Therefore, Latin
hypercube sampling provides a better coverage of the space.
"""

from __future__ import division, print_function

from multiscale.util.util_classes import EnumType
from samples import Sample, SampleParameter


class SamplingType(EnumType):
    """ Supported types of sampling. """

    DISTRIBUTION = 0
    MEAN = 1
    # LHS = 2
    # MIXED = 3


class SamplingException(Exception):
    pass


class Sampling(object):
    """Sample from given distributions via the defined sampling type.
    If keys are provided only the subset of distributions corresponding
    to the keys are sampled.
    """

    def __init__(
        self, distributions, sampling_type=SamplingType.DISTRIBUTION, keys=None
    ):
        self.distributions = distributions
        self.sampling_type = sampling_type
        self.keys = keys

    def sample(self, n_samples):
        """Creates samples.
        Master function which switches between methods for sample creation.
        This function should be called from other modules.
        """
        if self.sampling_type == SamplingType.DISTRIBUTION:
            samples = self.sample_from_distribution(n_samples)
        elif self.sampling_type == SamplingType.MEAN:
            samples = self.sample_from_mean(n_samples)
        #     elif (sampling_type == SamplingType.LHS):
        #         samples = _createSamplesByLHS(distributions, n_samples, keys);
        #     elif (sampling_type == SamplingType.MIXED):
        #         samples1 = _createSamplesByDistribution(distributions, n_samples/2, keys);
        #         samples2 = _createSamplesByLHS(distributions, n_samples/2, keys);
        #         samples = samples1 + samples2
        else:
            raise SamplingException(
                "SamplingType not supported: {}".format(self.sampling_type)
            )
        return samples

    def sample_from_distribution(self, n_samples):
        """
        Returns parameter samples from given distributions.
        If keys are provided, only the subset existing in keys is sampled.
        The generation of database samples is done in the SimulationFactory
        and via the simulation definitions.
        """
        samples = []
        for _ in xrange(n_samples):
            s = Sample()
            for dist in self.distributions:
                if self.keys and (dist.key not in keys):
                    continue
                s.add_parameter(
                    SampleParameter(
                        dist.key,
                        value=dist.samples(n_samples=1),
                        unit=dist.unit,
                        parameter_type=dist.parameter_type,
                    )
                )
            samples.append(s)
        return samples

    def sample_from_mean(self, n_samples=1):
        """ Returns mean parameters for the given distribution distribution_data. """
        samples = []
        for _ in xrange(n_samples):
            s = Sample()
            for dist in self.distributions:
                if self.keys and (dist.key not in keys):
                    continue
                s.add_parameter(
                    SampleParameter(
                        dist.key,
                        value=dist.mean(),
                        unit=dist.unit,
                        parameter_type=dist.parameter_type,
                    )
                )
            samples.append(s)
        return samples

    # def _createSamplesByLHS(dist_data, N, keys=None):
    #     '''
    #     Returns the parameter samples via LHS sampling.
    #     The boundaries of the samples are defined via the given distributions for the normal state.
    #     The lower and upper bounds have to account for the ranges with nonzero probability.
    #     Necessary to have the lower and upper bounds for sampling.
    #     '''
    #     # Get the LHS boundaries for all parameter dimensions and get
    #     # the values (always sample down to zero, the upper sample boundary
    #     # depends on the mean and sd of the values
    #     pointsLHS = dict()
    #     for pid in dist_data.keys():
    #         if keys and (pid not in keys):
    #             continue
    #         dtmp = dist_data[pid]
    #         minLHS = dtmp['llb'];           # 0.01
    #         maxLHS = dtmp['uub'];           # 0.99
    #         pointValues = calculatePointsByLHS(N, minLHS, maxLHS)
    #         random.shuffle(pointValues)
    #         pointsLHS[pid] = pointValues
    #
    #     # put the LHS dimensions together
    #     samples = []
    #     for ks in xrange(N):
    #         s = Sample()
    #         for pid in dist_data.keys():
    #             if keys and (pid not in keys):
    #                 continue
    #             pointValues = pointsLHS[pid]
    #             value = pointValues[ks]
    #             s.add_parameter(SampleParameter(pid, value,
    #                                          dtmp['unit'], GLOBAL_PARAMETER))
    #         samples.append(s)
    #
    #     return samples
    #
    #
    # def calculatePointsByLHS(N, variableMax, variableMin):
    #     '''
    #         This is the 1D solution.
    #         Necessary to have the
    #         ! PointValues are in the order of the segments, which has to be taken into account
    #         when generating the multi-dimensional LHS.
    #
    #         In Monte Carlo odesim, we want the entire distribution to be used evenly.
    #         We usually use a large number of samples to reduce actual randomness,
    #         but the latin hypercube sampling permits us to get the ideal randomness
    #         without so much of a calculation.
    #         https://mathieu.fenniak.net/latin-hypercube-sampling/
    #
    #         Latin hypercube sampling is capable of reducing the number of runs necessary
    #         to stablize a Monte Carlo odesim by a huge factor. Some simulations may take
    #         up to 30% fewer calculations to create a smooth distribution of outputs.
    #         The process is quick, simple, and easy to implement. It helps ensure that the Monte Carlo odesim
    #         is run over the entire length of the variable distributions,
    #         taking even unlikely extremities into account as one would desire.
    #     '''
    #     segmentSize = 1/float(N)
    #     pointValues = []
    #     for i in range(N):
    #         # Get the random point
    #         segmentMin = float(i) * segmentSize
    #         # segmentMax = float(i+1) * segmentSize
    #         point = segmentMin + (random.random() * segmentSize)
    #
    #         # Transform to the variable range
    #         pointValue = variableMin + point *(variableMax - variableMin)
    #         pointValues.append(pointValue)
    #     return pointValues
