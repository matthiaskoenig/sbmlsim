"""Methods of a parameter fit as blocks of a report document.

`methods_section` writes the methods of a fit from what the fit used: its
problem, its settings, the optimizer of its runs, the analyses of its
identifiability and its provenance. The text is specific to the settings and
states what the code does, with the formulas of the objective, the residuals,
the weights and the metrics, so it can go into the supplement of a manuscript
as it is. Every value of a setting has its text in a table of this module; a
value without one raises `UnsupportedSetting` when the methods are written,
so a new setting cannot be reported silently wrong.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

import libsbml

from sbmlsim.fit.derived import hook_summaries
from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import Identifiability, IdentifiabilityResult
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    DataAtChangeType,
    FitSettings,
    LossFunctionType,
    OptimizationAlgorithmType,
    ParameterScaleType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.provenance import FitProvenance
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.sampling import SamplingType
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.report.document import (
    Block,
    Cite,
    Code,
    Equation,
    Inline,
    Math,
    Paragraph,
    Ref,
    Section,
    Table,
    p,
)

__all__ = [
    "ReportContext",
    "UnsupportedSetting",
    "describe_parameter",
    "methods_section",
]

Part = str | Inline

SBMLSIM = Cite(("Koenig2026_sbmlsim",))
SCIPY = Cite(("Virtanen2020",))
LHS = Cite(("McKay1979",))


@dataclass(frozen=True)
class ReportContext:
    """What a report says about a fit beyond the fit itself.

    The context is plain data which the caller provides, e.g. pkpdsim from the
    definition of a model; sbmlsim reports a fit with an empty context.

    Attributes:
        title: title of the fit, e.g. "Pharmacokinetics of HCTZ".
        description: what the fit is for, the first sentence of the methods.
        parameter_descriptions: description of a parameter by parameter id,
            preferred over `FitParameter.description`.
        notes: further sentences of the methods of the parameters and data.
    """

    title: str = ""
    description: str = ""
    parameter_descriptions: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


class UnsupportedSetting(ValueError):
    """A value of a setting which the methods have no text for."""


def _text[K, V](table: Mapping[K, V], value: K, setting: str) -> V:
    """Get the text of the value of a setting.

    Raises:
        UnsupportedSetting: if the table has no text for the value.
    """
    try:
        return table[value]
    except KeyError:
        name = getattr(value, "name", value)
        raise UnsupportedSetting(
            f"The methods have no text for the {setting} '{name}'."
        ) from None


#: the loss function, rho of the squared weighted residual
LOSS_TEXT: dict[LossFunctionType, tuple[Part, ...]] = {
    LossFunctionType.LINEAR: (
        "The loss function is linear, ",
        Math(r"\rho(z) = z", "rho(z) = z"),
        ", i.e. a standard least squares problem.",
    ),
    LossFunctionType.SOFT_L1: (
        "The loss function is the soft L1 loss ",
        Math(r"\rho(z) = 2\left(\sqrt{1+z} - 1\right)", "rho(z) = 2(sqrt(1+z) - 1)"),
        ", a smooth approximation of the absolute value which reduces the "
        "influence of outliers.",
    ),
    LossFunctionType.CAUCHY: (
        "The loss function is the Cauchy loss ",
        Math(r"\rho(z) = \ln(1+z)", "rho(z) = ln(1+z)"),
        ", which severely weakens the influence of outliers.",
    ),
    LossFunctionType.ARCTAN: (
        "The loss function is the arctan loss ",
        Math(r"\rho(z) = \arctan(z)", "rho(z) = arctan(z)"),
        ", which limits the loss of a single residual.",
    ),
}

_F = (r"f_k(x_{i,k}, \vec{p})", "f_k(x_(i,k), arrow(p))")
_F1 = (r"f_k(x_{1,k}, \vec{p})", "f_k(x_(1,k), arrow(p))")
_DIFF = (
    rf"\left(y_{{i,k}} - y_{{1,k}}\right) - \left({_F[0]} - {_F1[0]}\right)",
    f"(y_(i,k) - y_(1,k)) - ({_F[1]} - {_F1[1]})",
)

#: the residual of a data point: the equation and its sentence
RESIDUAL_TEXT: dict[ResidualType, tuple[Equation, tuple[Part, ...]]] = {
    ResidualType.ABSOLUTE: (
        Equation(rf"r_{{i,k}} = y_{{i,k}} - {_F[0]}", f"r_(i,k) = y_(i,k) - {_F[1]}"),
        (
            "The residuals are absolute, the difference of the data point ",
            Math("y_{i,k}", "y_(i,k)"),
            " at ",
            Math("x_{i,k}", "x_(i,k)"),
            " and the prediction ",
            Math("f_k", "f_k"),
            " of the model:",
        ),
    ),
    ResidualType.NORMALIZED: (
        Equation(
            rf"r_{{i,k}} = \frac{{y_{{i,k}} - {_F[0]}}}{{\overline{{y}}_k}}",
            f"r_(i,k) = (y_(i,k) - {_F[1]}) / overline(y)_k",
        ),
        (
            "The residuals are normalized by the mean ",
            Math(r"\overline{y}_k", "overline(y)_k"),
            " of the data of their curve, which makes curves of different "
            "quantities and magnitudes comparable:",
        ),
    ),
    ResidualType.ABSOLUTE_TO_BASELINE: (
        Equation(rf"r_{{i,k}} = {_DIFF[0]}", f"r_(i,k) = {_DIFF[1]}"),
        (
            "The residuals are the absolute changes to the baseline, the first "
            "data point ",
            Math("y_{1,k}", "y_(1,k)"),
            " of a curve and the prediction at it:",
        ),
    ),
    ResidualType.NORMALIZED_TO_BASELINE: (
        Equation(
            rf"r_{{i,k}} = \frac{{{_DIFF[0]}}}{{\overline{{\Delta y}}_k}}",
            f"r_(i,k) = ({_DIFF[1]}) / overline(Delta y)_k",
        ),
        (
            "The residuals are the changes to the baseline, the first data point ",
            Math("y_{1,k}", "y_(1,k)"),
            " of a curve and the prediction at it, normalized by the mean ",
            Math(r"\overline{\Delta y}_k", "overline(Delta y)_k"),
            " of the changes of the data of the curve, which makes curves of "
            "different quantities and magnitudes comparable:",
        ),
    ),
}

#: residuals whose data and prediction are changes to the first data point
BASELINE_RESIDUALS = frozenset(
    {ResidualType.ABSOLUTE_TO_BASELINE, ResidualType.NORMALIZED_TO_BASELINE}
)

_MAPPING_WEIGHT = (
    " of its fit mapping (the weight its fit mapping collection assigns, 1 by "
    "default, or, for a collection which uses the weights of its mappings, the "
    "weight of the mapping, by default the number of subjects of its data)"
)

#: the weight of a curve, by the set of curve weightings
CURVE_WEIGHT_TEXT: dict[frozenset[WeightingCurvesType], tuple[Part, ...]] = {
    frozenset(): (Math("w_k = 1", "w_k = 1"), ", i.e. all curves have the same weight"),
    frozenset({WeightingCurvesType.MAPPING}): (
        Math("w_k = n_k", "w_k = n_k"),
        ", the weight",
        _MAPPING_WEIGHT,
    ),
    frozenset({WeightingCurvesType.POINTS}): (
        Math(r"w_k = 1 / N_k", "w_k = 1 / N_k"),
        " with the number ",
        Math("N_k", "N_k"),
        " of its data points, so that a curve contributes the mean of its "
        "squared residuals whatever its number of points",
    ),
    frozenset({WeightingCurvesType.MAPPING, WeightingCurvesType.POINTS}): (
        Math(r"w_k = n_k / N_k", "w_k = n_k / N_k"),
        ", the weight ",
        Math("n_k", "n_k"),
        _MAPPING_WEIGHT,
        " divided by the number ",
        Math("N_k", "N_k"),
        " of its data points",
    ),
}

#: the weight of a data point within its curve
POINT_WEIGHT_TEXT: dict[WeightingPointsType, tuple[Part, ...]] = {
    WeightingPointsType.NO_WEIGHTING: (
        "The data points of a curve have the same weight, ",
        Math("w^{p}_{i,k} = 1", "w^p_(i,k) = 1"),
        ".",
    ),
    WeightingPointsType.ERROR_WEIGHTING: (
        "A data point is weighted with the inverse of its coefficient of variation, ",
        Math(
            r"w^{p}_{i,k} = |y_{i,k}| / \sigma_{i,k}",
            "w^p_(i,k) = |y_(i,k)| / sigma_(i,k)",
        ),
        ", with ",
        Math(r"\sigma_{i,k}", "sigma_(i,k)"),
        " the standard deviation of the data point, or its standard error for "
        "data without a standard deviation. An error of zero is missing, a "
        "missing error of a curve with errors is replaced by the largest error "
        "of the curve, and the data points of a curve without errors have the "
        "weight ",
        Math("w^{p}_{i,k} = 2", "w^p_(i,k) = 2"),
        ", the weight of a coefficient of variation of 0.5.",
    ),
}

#: the space the optimizer searches the parameters in
SCALE_TEXT: dict[ParameterScaleType, str] = {
    ParameterScaleType.LOG10: "in log10 space",
    ParameterScaleType.LOG: "in natural log space",
    ParameterScaleType.LINEAR: "on the linear scale",
}

#: the unit of a step of a profile in the space of the optimizer
STEP_UNIT_TEXT: dict[ParameterScaleType, str] = {
    ParameterScaleType.LOG10: "decades of the parameter",
    ParameterScaleType.LOG: "units of the natural logarithm of the parameter",
    ParameterScaleType.LINEAR: "units of the parameter",
}

#: the sampling of the start values of the runs of the local optimizer
SAMPLING_TEXT: dict[SamplingType, tuple[Part, ...]] = {
    SamplingType.LOGUNIFORM_LHS: (
        "The start values of the runs were drawn by latin hypercube sampling ",
        LHS,
        " log-uniformly, i.e. uniformly in the logarithm of the parameters, "
        "within their bounds",
    ),
    SamplingType.UNIFORM_LHS: (
        "The start values of the runs were drawn by latin hypercube sampling ",
        LHS,
        " uniformly within the bounds of the parameters",
    ),
    SamplingType.LOGUNIFORM: (
        "The start values of the runs were drawn at random log-uniformly, i.e. "
        "uniformly in the logarithm of the parameters, within their bounds",
    ),
    SamplingType.UNIFORM: (
        "The start values of the runs were drawn at random uniformly within the "
        "bounds of the parameters",
    ),
    SamplingType.START: ("Every run started from the start values of the parameters",),
}

#: the methods of `scipy.optimize.least_squares`
LEAST_SQUARES_TEXT: dict[str, tuple[Part, ...]] = {
    "trf": ("the trust region reflective method ", Cite(("Branch1999",))),
    "dogbox": ("the dogleg method with rectangular trust regions",),
    "lm": ("the Levenberg-Marquardt method of MINPACK, which does not use the bounds",),
}

#: the finite differences of the jacobian of `scipy.optimize.least_squares`
JACOBIAN_TEXT: dict[str, str] = {
    "2-point": "forward differences",
    "3-point": "central differences",
    "cs": "complex steps",
}

#: the initial population of `scipy.optimize.differential_evolution`
POPULATION_TEXT: dict[str, tuple[Part, ...]] = {
    "latinhypercube": ("by latin hypercube sampling ", LHS),
    "sobol": ("from a Sobol sequence",),
    "halton": ("from a Halton sequence",),
    "random": ("at random",),
}

#: the state a data point at the time of a change is compared with
DATA_AT_CHANGE_TEXT: dict[DataAtChangeType, str] = {
    DataAtChangeType.BEFORE: (
        "A data point at the time of a change after the start of a simulation, "
        "e.g. a dose or the reset of a collection, was compared with the state "
        "before the change, except the first data point of a curve with data at "
        "later times, which starts a collection and was compared with the state "
        "after the change."
    ),
    DataAtChangeType.AFTER: (
        "A data point at the time of a change was compared with the state after "
        "the change, the semantics of PEtab v2."
    ),
}

#: the meaning of the kinds of fit mappings, how a fit uses their data
KIND_TEXT: dict[MappingKind, str] = {
    MappingKind.TRAINING: "whose residuals enter the objective",
    MappingKind.VALIDATION: (
        "which were simulated and evaluated with the fitted parameters but not fitted"
    ),
    MappingKind.OUTLIER: (
        "data the fit dropped as not usable, which were evaluated like the "
        "validation data so that the decision to drop them can be checked"
    ),
}

#: the name of a kind of fit mappings
KIND_NAME: dict[MappingKind, str] = {
    MappingKind.TRAINING: "training data",
    MappingKind.VALIDATION: "validation data",
    MappingKind.OUTLIER: "outliers",
}

#: the classification of a profile; `{max_points}` is the largest number of
#: points of a scan in one direction
IDENTIFIABILITY_TEXT: dict[Identifiability, str] = {
    Identifiability.IDENTIFIABLE: (
        "identifiable if its profile reaches the threshold within the scan on "
        "both sides of the optimum, i.e. its confidence interval is finite"
    ),
    Identifiability.NON_IDENTIFIABLE_LOWER: (
        "practically non-identifiable towards the lower bound if its profile "
        "reaches the threshold above the optimum but not below it within the "
        "scan, which ends at the lower bound or after {max_points} points"
    ),
    Identifiability.NON_IDENTIFIABLE_UPPER: (
        "practically non-identifiable towards the upper bound if its profile "
        "reaches the threshold below the optimum but not above it within the "
        "scan, which ends at the upper bound or after {max_points} points"
    ),
    Identifiability.NON_IDENTIFIABLE: (
        "practically non-identifiable if its profile reaches the threshold on "
        "neither side within the scans, which end at the bounds or after "
        "{max_points} points, and is not flat"
    ),
    Identifiability.STRUCTURAL: (
        "structurally non-identifiable if its profile is flat on both sides"
    ),
}


def _math(x: float) -> Math:
    """Format a number as a formula, a very small or large one as a power of ten."""
    if math.isinf(x):
        return Math(r"\infty" if x > 0 else r"-\infty", "oo" if x > 0 else "-oo")
    if x == 0.0 or 1e-3 <= abs(x) < 1e5 or math.isnan(x):
        return Math(f"{x:.6g}", f"{x:.6g}")
    # the exponent form of every decade, three significant digits
    digits, exponent = f"{x:.2e}".split("e")
    mantissa = f"{float(digits):g}"
    power = int(exponent)
    if mantissa in ("1", "-1"):
        sign = "-" if mantissa == "-1" else ""
        return Math(f"{sign}10^{{{power}}}", f"{sign}10^({power})")
    return Math(rf"{mantissa} \cdot 10^{{{power}}}", f"{mantissa} dot 10^({power})")


def _number(value: Any) -> str | Math:
    """Format a number as text, a very small or large one as a power of ten."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    x = float(value)
    if x == 0.0 or 1e-3 <= abs(x) < 1e5 or math.isnan(x):
        return f"{x:.6g}"
    return _math(x)


def _sentence(text: str) -> str:
    """Get a text as a sentence which ends with a full stop."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _count(n: int, singular: str, plural: str = "") -> str:
    """Get a number with its noun."""
    return f"{n} {singular if n == 1 else plural or singular + 's'}"


def _join(
    items: Sequence[Sequence[Part]], sep: str = ", ", last: str = " and "
) -> list[Part]:
    """Join enumerated parts with a separator and a last conjunction."""
    parts: list[Part] = []
    for k, item in enumerate(items):
        if k:
            parts.append(last if k == len(items) - 1 else sep)
        parts.extend(item)
    return parts


def _section(title: str, key: str, blocks: Sequence[Block]) -> Section:
    """Get a sub-section of the methods."""
    return Section(title, tuple(blocks), level=2, key=key)


@lru_cache(maxsize=32)
def _sbml_names(sbml: str) -> dict[str, str]:
    """Get the name of every element of an SBML model which has one, by id."""
    document = libsbml.readSBMLFromString(sbml)
    model = document.getModel()
    names: dict[str, str] = {}
    if model is None:
        return names
    elements = model.getListOfAllElements()
    for k in range(elements.getSize()):
        element = elements.get(k)
        if element.isSetId() and element.isSetName() and element.getName():
            names.setdefault(element.getId(), element.getName())
    return names


def describe_parameter(
    problem: OptimizationProblem, pid: str, context: ReportContext
) -> str:
    """Describe a fit parameter.

    Args:
        problem: the optimization problem of the parameter; an uninitialized
            problem is initialized with its settings or the default settings,
            the models are read in the initialization.
        pid: id of the fit parameter.
        context: the context of the report.

    Returns:
        The description of the context, else the description of the fit
        parameter, else the SBML name of its target in the models of the
        problem, else an empty string.
    """
    if pid in context.parameter_descriptions:
        return context.parameter_descriptions[pid]
    parameter = next((fp for fp in problem.parameters if fp.pid == pid), None)
    if parameter is None:
        return ""
    if parameter.description:
        return parameter.description
    if not problem.is_initialized:
        problem.initialize(problem.settings if problem.settings else FitSettings())
    for model in {id(m): m for m in problem.models}.values():
        name = _sbml_names(model.sbml).get(parameter.entity_id)
        if name:
            return name
    return ""


def methods_section(
    *,
    problem: OptimizationProblem,
    settings: FitSettings,
    result: OptimizationResult | None,
    identifiability: IdentifiabilityResult | None,
    fisher: FisherInformation | None,
    context: ReportContext,
) -> Section:
    """Write the methods of a fit.

    Args:
        problem: the optimization problem; an uninitialized problem is
            initialized with the settings, the data of the fit mappings is
            resolved in the initialization.
        settings: the settings of the fit.
        result: the result of the fit, its runs say which optimizer ran with
            which options; `None` leaves the optimizer out.
        identifiability: the profile likelihood analysis of the fit, if any.
        fisher: the Fisher information of the fit, if any.
        context: what the report says about the fit beyond the fit.

    Returns:
        The section of the methods with the sub-sections `methods:parameters`,
        `methods:objective`, `methods:optimization`, `methods:metrics`,
        `methods:identifiability` (with an identifiability analysis or a
        Fisher information) and `methods:provenance`.

    Raises:
        UnsupportedSetting: if a value of a setting has no text.
    """
    if not problem.is_initialized:
        problem.initialize(settings)
    blocks = [
        _parameters(problem, context),
        _objective(settings),
        _optimization(problem, settings, result),
        _metrics(problem, settings),
    ]
    if identifiability is not None or fisher is not None:
        blocks.append(_identifiability(identifiability, fisher))
    blocks.append(_provenance(result))
    title = f"Methods: {context.title}" if context.title else "Methods"
    return Section(title, tuple(blocks), level=1, key="methods")


def _parameters(problem: OptimizationProblem, context: ReportContext) -> Section:
    """Write the parameters and the data of the fit."""
    parts: list[Part] = []
    if context.description:
        parts.append(_sentence(context.description) + " ")
    n = len(problem.parameters)
    parts.append(f"{_count(n, 'parameter')} {'was' if n == 1 else 'were'} optimized: ")
    items: list[list[Part]] = []
    for parameter in problem.parameters:
        description = describe_parameter(problem, parameter.pid, context)
        if description:
            items.append([f"{description} (", Code(parameter.pid), ")"])
        else:
            items.append([Code(parameter.pid)])
    parts.extend(_join(items))
    parts.append(".")
    for note in context.notes:
        parts.append(" " + _sentence(note))

    kinds: list[list[Part]] = []
    for kind, count in problem.mapping_counts().items():
        if not count:
            continue
        points = sum(len(problem.y_references[k]) for k in problem.indices(kind))
        kinds.append(
            [
                f"{_count(count, 'fit mapping')} of {_text(KIND_NAME, kind, 'kind')} "
                f"with {_count(points, 'data point')}, "
                f"{_text(KIND_TEXT, kind, 'kind')}"
            ]
        )
    data: list[Part] = [
        "A fit mapping is a curve of a study, the data and the observable of "
        "the model it is compared with. The fit has ",
        *_join(kinds, sep="; ", last="; and "),
        ". The fit was set up and run with sbmlsim ",
        SBMLSIM,
        ".",
    ]
    return _section("Parameters and data", "methods:parameters", [p(*parts), p(*data)])


def _objective(settings: FitSettings) -> Section:
    """Write the objective function with its residuals, weights and loss."""
    residual_equation, residual_text = _text(
        RESIDUAL_TEXT, settings.residual, "residual"
    )
    curves = frozenset(settings.weighting_curves)
    curve_text = _text(CURVE_WEIGHT_TEXT, curves, "weighting of the curves")
    point_text = _text(
        POINT_WEIGHT_TEXT, settings.weighting_points, "weighting of the points"
    )
    loss_text = _text(LOSS_TEXT, settings.loss_function, "loss function")

    blocks: list[Block] = [
        p("The parameters were estimated by minimizing the objective function"),
        Equation(
            r"F(\vec{p}) = \frac{1}{2} \sum_{k} \sum_{i} "
            r"\rho\left(w_{i,k}\, r_{i,k}(\vec{p})^2\right)",
            "F(arrow(p)) = 1/2 sum_k sum_i rho(w_(i,k) r_(i,k)(arrow(p))^2)",
        ),
        p(
            "where ",
            Math("k", "k"),
            " runs over the fit mappings of the training data and ",
            Math("i", "i"),
            " over their data points, ",
            Math(r"r_{i,k}", "r_(i,k)"),
            " is the residual, ",
            Math(r"w_{i,k} = w_k\, w^{p}_{i,k}", "w_(i,k) = w_k w^p_(i,k)"),
            " the weight of the data point and ",
            Math(r"\rho", "rho"),
            " the loss function, applied to the squared weighted residual ",
            Math(r"z = w_{i,k}\, r_{i,k}^2", "z = w_(i,k) r_(i,k)^2"),
            ". ",
            *loss_text,
        ),
        p(*residual_text),
        residual_equation,
        p("The weight of curve ", Math("k", "k"), " is ", *curve_text, "."),
    ]
    points: list[Part] = list(point_text)
    if (
        settings.residual in BASELINE_RESIDUALS
        and settings.weighting_points is WeightingPointsType.ERROR_WEIGHTING
    ):
        points.append(
            " The data point in the weight is the change to the baseline, so the "
            "baseline itself has the weight 0."
        )
    blocks.append(p(*points))
    return _section("Objective function", "methods:objective", blocks)


def _optimization(
    problem: OptimizationProblem,
    settings: FitSettings,
    result: OptimizationResult | None,
) -> Section:
    """Write the optimizer, its runs and the simulator of the fit."""
    scale = settings.parameter_scale
    scale_parts: list[Part] = [
        f"The optimizer searched the parameters {_text(SCALE_TEXT, scale, 'scale')} "
        "within their bounds"
    ]
    own = [
        fp for fp in problem.parameters if fp.scale is not None and fp.scale != scale
    ]
    if own:
        scale_parts.append(", except ")
        scale_parts.extend(
            _join(
                [
                    [Code(fp.pid), f" {_text(SCALE_TEXT, fp.scale, 'scale')}"]
                    for fp in own
                    if fp.scale is not None
                ]
            )
        )
    scale_parts.append("; bounds and values are given in the units of the model.")
    blocks: list[Block] = [p(*scale_parts)]

    if result is not None and result.fits:
        blocks.extend(_runs(problem, result))

    blocks.append(_integrator(settings))
    if problem.hybridizations:
        summaries = hook_summaries(problem.hybridizations)
        blocks.append(
            p(
                "The simulations have the hybridizations ",
                *_join(
                    [[Code(s.name), f" ({s.kind}: {s.description})"] for s in summaries]
                ),
                ", changes of the simulations which are calculated from the "
                "fitted parameters.",
            )
        )
    return _section("Optimization", "methods:optimization", blocks)


def _algorithm(result: OptimizationResult) -> OptimizationAlgorithmType | None:
    """Get the algorithm of the runs of a result, `None` if not recorded."""
    name = result.fits[0].get("algorithm")
    if name is None and result.provenance is not None:
        name = result.provenance.arguments.get("algorithm")
    if name is None:
        return None
    return _text(OptimizationAlgorithmType.__members__, str(name), "algorithm")


def _runs(problem: OptimizationProblem, result: OptimizationResult) -> list[Paragraph]:
    """Write the optimizer of the runs, their start values and the convergence."""
    algorithm = _algorithm(result)
    kwargs: dict[str, Any] = dict(result.fits[0].get("optimizer_kwargs") or {})
    provenance = result.provenance
    n = len(result.fits)
    blocks: list[Paragraph] = []
    if algorithm is None:
        blocks.append(p("The optimizer of the runs of this fit was not recorded."))
    elif algorithm is OptimizationAlgorithmType.LEAST_SQUARE:
        blocks.append(_least_squares(kwargs, n))
        blocks.append(_start_values(problem, provenance))
    else:
        blocks.append(_differential_evolution(kwargs, n, provenance))
    if n > 1:
        blocks.append(
            p(
                "The convergence of the runs was assessed from the traces of their "
                "cost and the waterfall plot of their final costs, and the "
                "parameter set with the lowest cost was used."
            )
        )
    else:
        blocks.append(
            p(
                "The convergence of the run was assessed from the trace of its "
                "cost, and its parameter set was used."
            )
        )
    return blocks


def _least_squares(kwargs: Mapping[str, Any], n: int) -> Paragraph:
    """Write the options of `scipy.optimize.least_squares`."""
    method = _text(LEAST_SQUARES_TEXT, str(kwargs.get("method", "trf")), "method")
    jac = kwargs.get("jac", "2-point")
    parts: list[Part] = [
        f"The objective was minimized in {_count(n, 'run')} with ",
        *method,
        " of the least squares optimizer of SciPy ",
        SCIPY,
        ", with a Jacobian of ",
    ]
    if isinstance(jac, str):
        parts.append(f"finite {_text(JACOBIAN_TEXT, jac, 'Jacobian')}")
        diff_step = kwargs.get("diff_step")
        if diff_step is None:
            parts.append(" with the default relative step of SciPy")
        else:
            parts.extend([" with the relative step ", _number(diff_step)])
    else:
        parts.append("the function of the fit")
    tolerances = [
        [f"{name} = ", _number(kwargs[name])]
        for name in ("ftol", "xtol", "gtol")
        if kwargs.get(name) is not None
    ]
    if tolerances:
        parts.append(
            ". A run stopped at the tolerances of the change of the cost, of the "
            "parameters and of the gradient, "
        )
        parts.extend(_join(tolerances))
    if kwargs.get("max_nfev") is not None:
        parts.extend([", or after ", _count(int(kwargs["max_nfev"]), "evaluation")])
    parts.append(".")
    return p(*parts)


def _start_values(
    problem: OptimizationProblem, provenance: FitProvenance | None
) -> Paragraph:
    """Write the sampling of the start values of the local optimizer."""
    sampling: SamplingType | None = None
    if provenance is not None:
        recorded = provenance.arguments.get("optimizer", {}).get("sampling")
        if recorded is not None:
            name = str(recorded).rsplit(".", maxsplit=1)[-1]
            sampling = _text(SamplingType.__members__, name, "sampling")
    if sampling is None:
        parts: list[Part] = ["The start values were sampled within the bounds"]
    else:
        parts = list(_text(SAMPLING_TEXT, sampling, "sampling"))
        if sampling is not SamplingType.START:
            linear = [
                [Code(fp.pid)]
                for fp in problem.parameters
                if sampling.is_log and fp.scale is ParameterScaleType.LINEAR
            ]
            if linear:
                parts.extend([", uniformly for ", *_join(linear)])
            infinite = [
                [Code(fp.pid)]
                for fp in problem.parameters
                if fp.lower_bound == float("-inf") or fp.upper_bound == float("inf")
            ]
            if infinite:
                parts.extend([", and from their start values for ", *_join(infinite)])
    if provenance is not None and sampling is not SamplingType.START:
        if provenance.seed is None:
            parts.append(", without a seed")
        else:
            parts.append(f", with the seed {provenance.seed}")
    parts.append(".")
    return p(*parts)


def _differential_evolution(
    kwargs: Mapping[str, Any], n: int, provenance: FitProvenance | None
) -> Paragraph:
    """Write the options of `scipy.optimize.differential_evolution`."""
    parts: list[Part] = [
        f"The objective was minimized in {_count(n, 'run')} with differential "
        "evolution ",
        Cite(("Storn1997",)),
        " of SciPy ",
        SCIPY,
        " within the bounds of the parameters",
    ]
    options: list[list[Part]] = []
    if "strategy" in kwargs:
        options.append(["the strategy ", Code(str(kwargs["strategy"]))])
    if "popsize" in kwargs:
        options.append(
            [
                "a population of ",
                _number(kwargs["popsize"]),
                " times the number of parameters",
            ]
        )
    if "maxiter" in kwargs:
        options.append(["at most ", _number(kwargs["maxiter"]), " generations"])
    if "tol" in kwargs:
        options.append(
            ["the relative tolerance ", _number(kwargs["tol"]), " of convergence"]
        )
    if "recombination" in kwargs:
        options.append(
            ["the recombination constant ", _number(kwargs["recombination"])]
        )
    mutation = kwargs.get("mutation")
    if isinstance(mutation, list) and len(mutation) == 2:
        options.append(
            [
                "a mutation dithered between ",
                _number(mutation[0]),
                " and ",
                _number(mutation[1]),
            ]
        )
    elif mutation is not None:
        options.append(["the mutation constant ", _number(mutation)])
    if options:
        parts.append(", with ")
        parts.extend(_join(options))
    parts.append(".")
    init = kwargs.get("init")
    if isinstance(init, str):
        parts.append(" The initial population of a run was drawn ")
        parts.extend(_text(POPULATION_TEXT, init, "initial population"))
        if provenance is not None and provenance.seed is not None:
            parts.append(
                f", with a seed of every run derived from the seed {provenance.seed} "
                "of the fit"
            )
        parts.append(".")
    polish = kwargs.get("polish")
    if polish is True:
        parts.append(
            " The best member of the final population was polished with the "
            "L-BFGS-B method of SciPy."
        )
    elif isinstance(polish, dict):
        parts.append(
            " The best member of the final population was polished with a "
            "function of the fit."
        )
    return p(*parts)


def _tolerance(tolerance: AbsoluteTolerance) -> list[Part]:
    """Write the absolute tolerance of the states."""
    kinds = {tolerance.amount, tolerance.concentration, tolerance.other}
    parts: list[Part]
    if len(kinds) == 1:
        parts = ["the absolute tolerance ", _number(tolerance.amount)]
    else:
        parts = [
            "the absolute tolerances ",
            _number(tolerance.amount),
            " of an amount, ",
            _number(tolerance.concentration),
            " of a concentration and ",
            _number(tolerance.other),
            " of the other states",
        ]
    parts.append(
        ", which for a species in a concentration is multiplied by the initial "
        "volume of its compartment"
    )
    if tolerance.ids:
        parts.append(f", and {_count(len(tolerance.ids), 'state')} of its own")
    return parts


def _integrator(settings: FitSettings) -> Paragraph:
    """Write the simulator and its integrator."""
    step = (
        " It ran with the variable step size of libRoadRunner, i.e. with an output "
        "at every step of the integrator."
        if settings.variable_step_size
        else " It ran without the variable step size of libRoadRunner, i.e. with "
        "the output at the time points of the simulation."
    )
    parts: list[Part] = [
        "The models were simulated with libRoadRunner ",
        Cite(("Welsh2023",)),
        " and its integrator CVODE ",
        Cite(("Hindmarsh2005",)),
        " with the relative tolerance ",
        _number(settings.relative_tolerance),
        " and ",
        *_tolerance(AbsoluteTolerance.of(settings.absolute_tolerance)),
        ".",
        step,
    ]
    if settings.initial_time_step > 0:
        parts.extend(
            [
                " The first step after the start and after every change was ",
                _number(settings.initial_time_step),
                " in the time unit of the model.",
            ]
        )
    parts.append(
        " " + _text(DATA_AT_CHANGE_TEXT, settings.data_at_change, "data at a change")
    )
    return p(*parts)


def _metrics(problem: OptimizationProblem, settings: FitSettings) -> Section:
    """Write the metrics of the goodness of fit."""
    kinds = _join(
        [
            [_text(KIND_NAME, kind, "kind")]
            for kind, n in problem.mapping_counts().items()
            if n
        ]
    )
    blocks: list[Block] = [
        p(
            "The goodness of fit is given per kind of data (",
            *kinds,
            ") by metrics over its ",
            Math("n", "n"),
            " data points with the residuals ",
            Math(
                r"e_i = y_i - f(x_i, \hat{\vec{p}})",
                "e_i = y_i - f(x_i, hat(arrow(p)))",
            ),
            " at the fitted parameters ",
            Math(r"\hat{\vec{p}}", "hat(arrow(p))"),
            ", the mean squared error, its root and the root mean squared error "
            "normalized by the mean ",
            Math(r"\overline{y}_{k(i)}", "overline(y)_(k(i))"),
            " of the data of the curve of a point:",
        ),
        Equation(
            r"\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} e_i^2, \quad "
            r"\mathrm{RMSE} = \sqrt{\mathrm{MSE}}, \quad "
            r"\mathrm{NRMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} "
            r"\left(\frac{e_i}{\overline{y}_{k(i)}}\right)^2}",
            '"MSE" = 1/n sum_(i=1)^n e_i^2, quad "RMSE" = sqrt("MSE"), quad '
            '"NRMSE" = sqrt(1/n sum_(i=1)^n (e_i / overline(y)_(k(i)))^2)',
        ),
        p(
            "The root mean square of the residuals of the objective, i.e. what "
            "the fit minimizes, and the coefficient of determination:"
        ),
        Equation(
            r"\mathrm{RMSE}_w = \sqrt{\frac{1}{n} \sum_{i=1}^{n} "
            r"\rho\left(w_i\, r_i^2\right)}, \quad "
            r"R^2 = 1 - \frac{\sum_{i} e_i^2}{\sum_{i} \left(y_i - \overline{y}\right)^2}",
            '"RMSE"_w = sqrt(1/n sum_(i=1)^n rho(w_i r_i^2)), quad '
            "R^2 = 1 - (sum_i e_i^2) / (sum_i (y_i - overline(y))^2)",
        ),
        p(
            "The predictions of a non-linear model are no linear regression of "
            "the data, so ",
            Math("R^2", "R^2"),
            " can be negative: the prediction is worse than the mean of the data. "
            "The information criteria of Akaike ",
            Cite(("Akaike1974",)),
            " and Schwarz ",
            Cite(("Schwarz1978",)),
            " with the number ",
            Math("n_p", "n_p"),
            " of fitted parameters are those of a least squares fit with normally "
            "distributed residuals, up to an additive constant, so only "
            "differences between models fitted to the same data are meaningful:",
        ),
        Equation(
            r"\mathrm{AIC} = n \ln(\mathrm{MSE}) + 2 n_p, \quad "
            r"\mathrm{BIC} = n \ln(\mathrm{MSE}) + n_p \ln(n)",
            '"AIC" = n ln("MSE") + 2 n_p, quad "BIC" = n ln("MSE") + n_p ln(n)',
        ),
        p(
            "The percentage of the data points whose prediction ",
            Math(
                r"f_i = f(x_i, \hat{\vec{p}})",
                "f_i = f(x_i, hat(arrow(p)))",
            ),
            " lies within twofold of the data is taken over the ",
            Math("n_+", "n_(+)"),
            " data points with positive data ",
            Math("y_i > 0", "y_i > 0"),
            " and a positive prediction ",
            Math("f_i > 0", "f_i > 0"),
            "; the others have no ratio and are not counted, and a kind of data "
            "without such a point has no percentage:",
        ),
        Equation(
            r"P_2 = \frac{100}{n_+} \left|\left\{ i : \frac{1}{2} \le "
            r"\frac{f_i}{y_i} \le 2 \right\}\right|",
            "P_2 = 100 / n_(+) abs(\\{ i : 1/2 <= f_i / y_i <= 2 \\})",
        ),
    ]
    if settings.residual in BASELINE_RESIDUALS:
        blocks.append(
            p(
                "With the residuals to the baseline the data and the predictions "
                "of the metrics are the changes to the baseline."
            )
        )
    blocks.append(
        p(
            "The goodness of fit plot shows the predictions against the data on "
            "logarithmic axes, with the lines of twofold ",
            Math("f = 2y", "f = 2y"),
            " and ",
            Math("f = y/2", "f = y/2"),
            ". The Bland-Altman plot ",
            Cite(("Bland1986",)),
            " shows the ratio of prediction and data as ",
            Math(r"d_i = \log_{10}(f_i / y_i)", "d_i = log_10(f_i / y_i)"),
            " against the geometric mean ",
            Math(r"\sqrt{f_i\, y_i}", "sqrt(f_i y_i)"),
            " for the data points with positive values, with the bias ",
            Math(r"\overline{d}", "overline(d)"),
            " and the limits of agreement ",
            Math(r"\overline{d} \pm 1.96\, s_d", "overline(d) plus.minus 1.96 s_d"),
            ", with the sample standard deviation ",
            Math("s_d", "s_d"),
            " of ",
            Math("d_i", "d_i"),
            ", of the training data; they are read as the fold factors ",
            Math(r"10^{\overline{d}}", "10^(overline(d))"),
            " and ",
            Math(
                r"10^{\overline{d} \pm 1.96\, s_d}",
                "10^(overline(d) plus.minus 1.96 s_d)",
            ),
            ".",
        )
    )
    return _section("Goodness of fit", "methods:metrics", blocks)


def _identifiability(
    identifiability: IdentifiabilityResult | None,
    fisher: FisherInformation | None,
) -> Section:
    """Write the profile likelihood and the Fisher information."""
    blocks: list[Block] = []
    if identifiability is not None:
        blocks.extend(_profiles(identifiability))
    if fisher is not None:
        blocks.extend(_fisher(fisher, preferred=identifiability is not None))
    return _section("Identifiability", "methods:identifiability", blocks)


def _profiles(result: IdentifiabilityResult) -> list[Block]:
    """Write the profile likelihood analysis."""
    ps = result.settings
    blocks: list[Block] = [
        p(
            "The identifiability of the parameters was analysed with the profile "
            "likelihood ",
            Cite(("Raue2009", "Kreutz2013")),
            ". The profile of a parameter is the lowest objective with the "
            "parameter fixed:",
        ),
        Equation(
            r"\mathrm{PL}(p_j) = \min_{p_{l \neq j}} F(\vec{p})",
            '"PL"(p_j) = min_(p_(l != j)) F(arrow(p))',
        ),
    ]
    scan: list[Part]
    if ps.reoptimize:
        scan = [
            "At every point of a scan the other parameters were re-optimized from "
            "the previous point with the least squares optimizer of SciPy"
        ]
        if ps.optimizer_kwargs.get("diff_step") is not None:
            scan.extend(
                [" with the relative step ", _number(ps.optimizer_kwargs["diff_step"])]
            )
        scan.append(".")
    else:
        scan = [
            "The other parameters were kept at the optimum, i.e. the profile is a "
            "scan of the cost and a lower bound of the profile likelihood."
        ]
    scales = {result.scale(pid) for pid in result.pids}
    units = _join(
        [[_text(STEP_UNIT_TEXT, s, "scale")] for s in sorted(scales, key=str)]
    )
    scan.extend(
        [
            " A scan ran from the optimum in both directions in the space of the "
            "optimizer, with steps in ",
            *units,
            ", starting at ",
            _number(ps.initial_step),
            f". A step which raised the objective by more than "
            f"{ps.max_cost_fraction:g} of the distance between the objective at the "
            "optimum and the threshold was reduced by the factor ",
            _number(ps.step_factor),
            " and repeated, down to ",
            _number(ps.min_step),
            ", and the step after one which raised it by less than that divided "
            "by the factor was enlarged by the factor, up to ",
            _number(ps.max_step),
            ". A scan stopped at the threshold, at the bound of the parameter, at "
            f"a point which failed or after {ps.max_points} points.",
        ]
    )
    blocks.append(p(*scan))
    delta = ps.delta / 2.0
    blocks.extend(
        [
            p(
                "The confidence intervals are the values at which the profile "
                "crosses the threshold of the likelihood ratio test"
            ),
            Equation(
                r"F_{\mathrm{thr}} = F_{\min} + \frac{\chi^2_{\alpha}(\mathrm{df})}{2}",
                'F_"thr" = F_"min" + chi^2_alpha ("df") / 2',
            ),
            p(
                "with ",
                Math(r"F_{\min}", 'F_"min"'),
                " the lowest objective of the fit and of all profiles and ",
                Math(r"\chi^2_{\alpha}(\mathrm{df})", 'chi^2_alpha ("df")'),
                f" the quantile of the χ² distribution at the confidence level "
                f"α = {ps.alpha:g} with df = {ps.degrees_of_freedom} degrees of "
                f"freedom, i.e. {delta:.3g} above the lowest objective. A "
                "crossing is interpolated between two points of the profile, on "
                "the logarithm of a parameter on a logarithmic scale. A parameter "
                "is ",
                *_join(
                    [
                        [
                            _text(IDENTIFIABILITY_TEXT, kind, "identifiability").format(
                                max_points=ps.max_points
                            )
                        ]
                        for kind in Identifiability
                    ],
                    sep="; ",
                    last="; and ",
                ),
                f". A side of a profile is flat if it rises by less than "
                f"{ps.flatness:g} of the distance between the lowest objective and "
                "the threshold, and a profile with a point which failed or did not "
                "converge is not classified. The profile likelihood is preferred "
                "over the intervals of the Fisher information, its intervals are "
                "invariant under a transformation of the parameters and can be "
                "asymmetric ",
                Cite(("Wieland2021",)),
                ".",
            ),
        ]
    )
    return blocks


def _fisher(fisher: FisherInformation, preferred: bool) -> list[Block]:
    """Write the Fisher information."""
    step = _math(fisher.step)
    blocks: list[Block] = [
        p(
            "The Fisher information "
            + ("is the local alternative to the profiles: it " if preferred else ""),
            "is the Gauss-Newton approximation of the curvature of the objective "
            "at the optimum, from the Jacobian ",
            Math("J", "J"),
            " of the weighted residuals of the training data with respect to the "
            "parameters ",
            Math(r"\vec{p}", "arrow(p)"),
            " in the space of the optimizer, computed by central finite differences "
            "with the step ",
            Math(
                rf"h_j = \epsilon \max(|p_j|, 1),\ \epsilon = {step.latex}",
                f"h_j = epsilon max(|p_j|, 1), space epsilon = {step.typst}",
            ),
            " of the parameter ",
            Math("p_j", "p_j"),
            ":",
        ),
        Equation(
            r"I = J^\top J, \quad \mathrm{cov} = \sigma^2 I^{-1}, \quad "
            r"\sigma^2 = \frac{2 F}{n - n_p}",
            'I = J^top J, quad "cov" = sigma^2 I^(-1), quad '
            "sigma^2 = (2 F) / (n - n_p)",
        ),
        p(
            "with the number ",
            Math("n", "n"),
            " of data points of the training data and the number ",
            Math("n_p", "n_p"),
            " of parameters. "
            f"The information has rank {fisher.rank} of {fisher.k}, the number of "
            "its eigenvalues above ",
            _number(fisher.rank_tolerance),
            " of the largest one, and the condition number ",
            _number(float(f"{fisher.condition_number:.3g}")),
            "; the covariance of an information of a lower rank is computed with "
            "the pseudo-inverse. The Wald "
            "intervals are ",
            Math(
                r"\hat{p}_j \pm t_{(1+\alpha)/2,\, n-n_p}\, \mathrm{SE}_j",
                'hat(p)_j plus.minus t_((1+alpha)/2, n-n_p) "SE"_j',
            ),
            " with the standard errors ",
            Math(
                r"\mathrm{SE}_j = \sqrt{\mathrm{cov}_{jj}}",
                '"SE"_j = sqrt("cov"_(j j))',
            ),
            f" at the confidence level α = {fisher.alpha:g}, transformed back into "
            "the units of the model.",
        ),
    ]
    return blocks


#: the number of characters of a checksum in the table of the checksums, which
#: identify a file; a full SHA-256 of 64 characters does not fit on a page
CHECKSUM_CHARS = 12


def _label(text: str) -> str:
    """Get a label of the characters every format accepts."""
    return re.sub(r"[^A-Za-z0-9_.:-]", "-", text)


def _created(created: str) -> str:
    """Format the date and time of the provenance."""
    try:
        moment = datetime.fromisoformat(created)
    except ValueError:
        return created
    zone = f" {moment.tzname()}" if moment.tzinfo else ""
    return f"{moment:%Y-%m-%d %H:%M:%S}{zone}"


def _provenance(result: OptimizationResult | None) -> Section:
    """Write the provenance of the fit and the table of its checksums."""
    provenance = result.provenance if result is not None else None
    if result is None or provenance is None:
        return _section(
            "Provenance",
            "methods:provenance",
            [p("The provenance of this fit was not recorded.")],
        )
    versions = provenance.versions

    def version(name: str) -> str:
        return versions.get(name, "not recorded")

    seed = (
        f"seed {provenance.seed}" if provenance.seed is not None else "without a seed"
    )
    parts: list[Part] = [
        f"The fit {result.sid} was run on {_created(provenance.created)} with "
        f"sbmlsim {version('sbmlsim')} ",
        SBMLSIM,
        f", libRoadRunner {version('libroadrunner')} ",
        Cite(("Welsh2023",)),
        f", SciPy {version('scipy')} ",
        SCIPY,
        f", NumPy {version('numpy')} ",
        Cite(("Harris2020",)),
        f" and Python {version('python')} ({seed}). ",
    ]
    git = provenance.git
    if git is None:
        parts.append("The models and data were not under version control")
    else:
        parts.extend(
            [
                "The models and data were those of commit ",
                Code(git.sha),
                " of ",
                Code(git.root),
                " with uncommitted changes" if git.dirty else "",
            ]
        )
    blocks: list[Block] = []
    label = _label(f"tab:checksums:{result.sid}")
    if provenance.files:
        parts.extend(["; their SHA-256 checksums are listed in ", Ref(label), "."])
    else:
        parts.append(".")
    if provenance.fixed_values:
        parts.append(" The fit kept the values ")
        parts.extend(
            _join(
                [
                    [Code(sid), " = ", _number(value)]
                    for sid, value in provenance.fixed_values.items()
                ]
            )
        )
        parts.append(" of the model fixed.")
    blocks.append(p(*parts))
    if provenance.files:
        blocks.append(
            Table(
                caption=(
                    p(
                        f"SHA-256 checksums of the models and data of the fit "
                        f"{result.sid}, their first {CHECKSUM_CHARS} characters; "
                        "the full checksums are in the provenance of the result "
                        "(optimization_result.json)."
                    )
                ).content,
                header=("File", "SHA-256"),
                rows=tuple(
                    (Code(path), Code(f"{digest[:CHECKSUM_CHARS]}..."))
                    for path, digest in provenance.files.items()
                ),
                label=label,
                align="ll",
            )
        )
    return _section("Provenance", "methods:provenance", blocks)
