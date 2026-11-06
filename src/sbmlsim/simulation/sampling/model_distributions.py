"""The distributions of the parameters of an SBML model, from distrib.

A fitted parameter of a model carries the distribution of its draws as the
uncertainty of the SBML package distrib, an uncert parameter of the type
`distribution` whose math is a formula of `sbmlsim.distrib`, e.g.
`lognormal(mu, sigma, lower, upper)`; the factory of a model writes it from the
record of an adopted fit (`sbmlsim.fit.records.AdoptedRecords.parameter`).
`parameter_distributions` reads these distributions back as distributions of
`sbmlsim.simulation.sampling` and `uncertainty_dimension` draws them into the
dimension `UNCERTAINTY_DIMENSION` of a scan. The draws are independent per
parameter: distrib states one distribution per parameter.
"""

import math
from pathlib import Path

import libsbml

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

__all__ = [
    "UNCERTAINTY_DIMENSION",
    "distribution_from_formula",
    "parameter_distributions",
    "uncertainty_dimension",
]


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
        ValueError: for another formula.
    """
    parsed = distrib.parse(formula)
    args = parsed.args
    if parsed.name == "uniform":
        return Uniform(args[0], args[1])
    inner: Distribution
    if parsed.name == "lognormal":
        mu, sigma = args[0], args[1]
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
    sbml_path: Path, draws: int, seed: int, id: str = UNCERTAINTY_DIMENSION
) -> Dimension:
    """Draw the parameters of a model which have a distribution.

    The draws are a Latin hypercube of the distributions, see
    `parameter_distributions`.

    Args:
        sbml_path: the SBML model.
        draws: the number of draws.
        seed: the seed of the Latin hypercube.
        id: the id of the dimension.

    Returns:
        The dimension of the draws.

    Raises:
        ValueError: if no parameter of the model has a distribution.
    """
    distributions = parameter_distributions(sbml_path)
    if not distributions:
        raise ValueError(
            f"No parameter of '{sbml_path}' has a distribution: the uncertainty "
            f"of the fitted parameters comes from the adopted fits of the model."
        )
    return lhs(distributions, draws, seed=seed, id=id)
