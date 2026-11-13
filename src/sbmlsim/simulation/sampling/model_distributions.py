"""The distributions of the parameters of an SBML model, from distrib.

A fitted parameter of a model carries the distribution of its draws as the
uncertainty of the SBML package distrib, an uncert parameter of the type
`distribution` whose math is a formula of `sbmlsim.distrib`, e.g.
`lognormal(mu, sigma, lower, upper)`; the factory of a model writes it from the
record of an adopted fit (`sbmlsim.fit.records.AdoptedRecords.parameter`).
`parameter_distributions` reads these distributions back as distributions of
`sbmlsim.simulation.sampling` and `uncertainty_dimension` draws them into the
dimension `UNCERTAINTY_DIMENSION` of a scan. distrib states one distribution
per parameter, so the draws are independent unless a correlation of their
normal scores is given, e.g. the one of the record of an adopted fit
(`sbmlsim.fit.records.AdoptedRecord.correlation`), which the Latin hypercube
takes as the rank correlation of a Gaussian copula.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import libsbml
import numpy as np

from sbmlsim import distrib
from sbmlsim.distrib import UNCERTAINTY_DIMENSION
from sbmlsim.simulation.sampling.designs import lhs
from sbmlsim.simulation.sampling.distributions import (
    Distribution,
    LogNormal,
    Normal,
    Truncated,
    Uniform,
)
from sbmlsim.simulation.scan import Dimension

if TYPE_CHECKING:
    from sbmlsim.uncertainty.covariance import Covariance

__all__ = [
    "MAX_SIGMA",
    "UNCERTAINTY_DIMENSION",
    "distribution_from_formula",
    "parameter_distributions",
    "uncertainty_dimension",
]


#: the largest sigma of a lognormal whose variance is a number:
#: `expm1(sigma**2)` overflows above it, and a lognormal that wide over any
#: bounds is the uniform of a parameter the data do not constrain
MAX_SIGMA = 26.0


def distribution_from_formula(formula: str) -> Distribution:
    """Get the distribution of a formula of distrib.

    Args:
        formula: `lognormal(mu, sigma[, lower, upper])` (mu and sigma of the
            natural logarithm), `normal(mean, sd[, lower, upper])` or
            `uniform(lower, upper)`, with numbers as arguments (the grammar of
            `sbmlsim.distrib`).

    Returns:
        The distribution, truncated if the formula has bounds.

    Raises:
        ValueError: for another formula, or for a lognormal whose sigma is so
            large that its variance is no number (`MAX_SIGMA`).
    """
    parsed = distrib.parse(formula)
    args = parsed.args
    if parsed.name == "uniform":
        return Uniform(args[0], args[1])
    inner: Distribution
    if parsed.name == "lognormal":
        mu, sigma = args[0], args[1]
        if sigma >= MAX_SIGMA:
            raise ValueError(
                f"the sigma of a lognormal must be below {MAX_SIGMA}, got "
                f"'{sigma}' in '{formula}'; a parameter which the data do not "
                f"constrain is a uniform over its bounds, see "
                f"`sbmlsim.fit.records`."
            )
        inner = LogNormal(math.exp(mu), math.sqrt(math.expm1(sigma**2)))
    else:
        inner = Normal(args[0], args[1])
    bounds = parsed.bounds
    if bounds is not None:
        return Truncated(inner, *bounds)
    return inner


def parameter_distributions(sbml_path: Path) -> dict[str, Distribution]:
    """Get the distributions of the parameters of a model.

    Every parameter with a distrib uncertainty which states a distribution
    (an uncert parameter of the type `distribution`) has one.

    Args:
        sbml_path: the SBML model, the flat model a simulation runs.

    Returns:
        The distribution of every such parameter by its id.

    Raises:
        ValueError: if the file has no model, a distribution is not
            supported, see `distribution_from_formula`, or a parameter states
            two.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model | None = doc.getModel()
    if model is None:
        raise ValueError(f"'{sbml_path}' has no SBML model.")
    distributions: dict[str, Distribution] = {}
    for parameter in model.getListOfParameters():
        plugin = parameter.getPlugin("distrib")
        if plugin is None:
            continue
        for k in range(plugin.getNumUncertainties()):
            uncertainty = plugin.getUncertainty(k)
            for j in range(uncertainty.getNumUncertParameters()):
                child = uncertainty.getUncertParameter(j)
                if child.getType() != libsbml.DISTRIB_UNCERTTYPE_DISTRIBUTION:
                    continue
                if not child.isSetMath():
                    continue
                pid = parameter.getId()
                if pid in distributions:
                    raise ValueError(f"The parameter '{pid}' states two distributions.")
                formula = libsbml.formulaToL3String(child.getMath())
                distributions[pid] = distribution_from_formula(formula)
    return distributions


def uncertainty_dimension(
    sbml_path: Path,
    draws: int,
    seed: int,
    id: str = UNCERTAINTY_DIMENSION,
    correlation: Sequence[Covariance] = (),
) -> Dimension:
    """Draw the parameters of a model which have a distribution.

    The draws are a Latin hypercube of the distributions, see
    `parameter_distributions`, with the correlation of the normal scores of
    `correlation` as the Spearman rank correlation `6 / π asin(r / 2)` of the
    values (Iman and Conover, see `lhs`), the correlation of a Gaussian copula
    of the distributions. A parameter which no correlation names, or which has
    no distribution, is independent.

    Args:
        sbml_path: the SBML model.
        draws: the number of draws, with a correlation more than the
            number of parameters with a distribution.
        seed: the seed of the Latin hypercube.
        id: the id of the dimension.
        correlation: correlations of the normal scores of parameters
            (`sbmlsim.uncertainty.Covariance` of the kind `correlation`), of
            disjoint parameters.

    Returns:
        The dimension of the draws.

    Raises:
        ValueError: if no parameter of the model has a distribution, two
            correlations share a parameter or a correlation is not one of the
            kind `correlation`.
    """
    distributions = parameter_distributions(sbml_path)
    if not distributions:
        raise ValueError(
            f"No parameter of '{sbml_path}' has a distribution: the uncertainty "
            f"of the fitted parameters comes from the adopted fits of the model."
        )
    index = {pid: k for k, pid in enumerate(distributions)}
    matrix = np.eye(len(index))
    seen: set[str] = set()
    for c in correlation:
        if c.kind != "correlation":
            raise ValueError(
                f"The correlation of {list(c.targets)} is a covariance, the draws "
                f"take a correlation of the normal scores."
            )
        shared = seen & set(c.targets)
        if shared:
            raise ValueError(
                f"The parameters {sorted(shared)} are in two correlations."
            )
        seen |= set(c.targets)
        positions = [(k, index[t]) for k, t in enumerate(c.targets) if t in index]
        for i, a in positions:
            for j, b in positions:
                if a != b:
                    r = float(c.matrix[i, j])
                    matrix[a, b] = (
                        6.0 / math.pi * math.asin(max(-1.0, min(1.0, r)) / 2.0)
                    )
    if not np.any(matrix != np.eye(len(index))):
        return lhs(distributions, draws, seed=seed, id=id)
    return lhs(distributions, draws, seed=seed, id=id, correlation=matrix)
