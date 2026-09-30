"""What PEtab v2 does not express about an sbmlsim optimization problem.

PEtab describes a parameter estimation problem as tables: a model, the
conditions and experiments it is simulated under, the observables, the
measurements and the parameters. `sbmlsim` describes more than that, i.e., the
units of everything, what a fit does with a subset of the data and how the
residuals are weighted, and less of it in tables: a `SimulationExperiment` is
python and its data can be a function of other data.

This module is the catalogue of the differences. Every `Gap` says what
`sbmlsim` has, what PEtab v2 offers for it and what the layer does about it,
and `gaps_of_problem` reports the gaps a given problem actually runs into, so
that what a specific export loses is known before it is written.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rich.table import Table

from sbmlsim.fit.objects import EVALUATED_KINDS, MappingKind
from sbmlsim.fit.options import ResidualType, WeightingCurvesType, WeightingPointsType

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem

logger = logging.getLogger(__name__)

#: the layers of a network which behave differently in training mode
EVALUATION_MODE_LAYERS: frozenset[str] = frozenset(
    {
        "Dropout",
        "Dropout1d",
        "Dropout2d",
        "Dropout3d",
        "AlphaDropout",
        "FeatureAlphaDropout",
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
    }
)


class GapKind(StrEnum):
    """What the layer does about a difference to PEtab v2."""

    EXTENSION = "extension"
    """PEtab has no place for it, the `sbmlsim` extension of the problem
    carries it, so the round trip through `sbmlsim` is exact. The extension is
    required, i.e. a tool which does not know it rejects the problem rather
    than reading it without the information; `required_extension=False` writes
    a problem for other tools."""

    LOSSY = "lossy"
    """The information is transformed and the round trip is not exact, i.e.,
    the problem which is read back is not the problem which was written."""

    UNSUPPORTED = "unsupported"
    """The export raises, there is no representation which keeps the meaning of
    the fit."""


@dataclass(frozen=True)
class Gap:
    """A difference between an `sbmlsim` fit and a PEtab v2 problem.

    Attributes:
        id: identifier of the gap.
        kind: what the layer does about it.
        sbmlsim: what `sbmlsim` has.
        petab: what PEtab v2 offers for it, `-` if it offers nothing.
        detail: what the layer does and what it costs.
    """

    id: str
    kind: GapKind
    sbmlsim: str
    petab: str
    detail: str


#: every difference between an `sbmlsim` fit and a PEtab v2 problem
GAPS: tuple[Gap, ...] = (
    Gap(
        id="units",
        kind=GapKind.EXTENSION,
        sbmlsim="pint units on the parameters, the observables and the data",
        petab="-",
        detail="the data is converted to the units of the model, which is what "
        "the objective uses, and the units are written to the extension. A tool "
        "which reads the problem without the extension gets the numbers in model "
        "units, which is what PEtab assumes anyway",
    ),
    Gap(
        id="mapping-kind",
        kind=GapKind.EXTENSION,
        sbmlsim="a fit mapping is training data, validation data, an outlier "
        "or excluded because the model does not describe it",
        petab="every measurement of a problem enters the objective",
        detail="everything a fit evaluates is written, i.e. the training data, "
        "the validation data and the outliers, and the kind of every mapping "
        "goes to the extension, so a round trip keeps the fit. A tool which "
        "reads the problem without the extension fits the validation data and "
        "the outliers as well, which is why the extension is required. The data "
        "the model does not describe is not part of the problem at all",
    ),
    Gap(
        id="fit-settings",
        kind=GapKind.EXTENSION,
        sbmlsim="`FitSettings`, i.e. the residual type, the parameter scale the "
        "optimizer searches in, the loss function, the weighting of curves and "
        "points and the tolerances of the integrator",
        petab="`noiseDistribution` per observable, the objective is the negative "
        "log likelihood. The `parameterScale` column of v1 is gone in v2, which "
        "says the scale is a property of the optimization and not of the problem",
        detail="the settings go to the extension, which is required because of "
        "them: a tool which reads the problem without the settings optimizes a "
        "different objective on the same data. The bounds and the start values "
        "are written on the linear scale, i.e. in the units of the model, which "
        "is what they are in `sbmlsim` as well",
    ),
    Gap(
        id="weights",
        kind=GapKind.LOSSY,
        sbmlsim="a weight per curve and a weight per point, `ERROR_WEIGHTING` "
        "weights a point with `|y/sd|`, i.e. the inverse coefficient of variation",
        petab="`noiseParameters` per measurement, i.e. the standard deviation",
        detail="the standard deviation of the data is written as the noise "
        "parameter, which is the closest PEtab has. The weights of `sbmlsim` are "
        "not a standard deviation, so the objective of a tool which reads the "
        "problem differs from the objective of the fit; the weights themselves "
        "are in the extension",
    ),
    Gap(
        id="residual-baseline",
        kind=GapKind.EXTENSION,
        sbmlsim="`ABSOLUTE_TO_BASELINE` and `NORMALIZED_TO_BASELINE` subtract the "
        "first point of a curve from the data and from the prediction",
        petab="a measurement is the value of the observable",
        detail="the measurements are written as they are, i.e. not shifted to the "
        "baseline, and the residual type goes to the extension. `sbmlsim` "
        "subtracts the baseline again when it reads the problem",
    ),
    Gap(
        id="presimulation",
        kind=GapKind.LOSSY,
        sbmlsim="`Timecourse(discard=True)`, a simulation of a finite duration "
        "whose result is dropped",
        petab="a period at `time=-inf`, i.e. simulation to a steady state",
        detail="a leading discarded timecourse becomes the pre-equilibration "
        "period of the experiment and its duration and its steps are lost; a "
        "discarded timecourse which is not the first is unsupported",
    ),
    Gap(
        id="output-times",
        kind=GapKind.LOSSY,
        sbmlsim="`Timecourse(start, end, steps)` is the output grid of the simulation",
        petab="the simulation is evaluated at the times of the measurements",
        detail="the grid goes to the extension so that `sbmlsim` simulates as "
        "before; another tool simulates the measurement times, which is the same "
        "fit with fewer output points",
    ),
    Gap(
        id="model-changes",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="`model_changes` and `model_manipulations`, i.e. "
        "`ModelChange.clamp_species`, which change the structure of the model",
        petab="a condition changes the value of an entity of the model",
        detail="a structural change is not a value, the export raises. Apply the "
        "change to the model and export the model it produces",
    ),
    Gap(
        id="condition-target",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="a change of a timecourse names what it sets with a selection, "
        "i.e. `[S1]` is the concentration and `S1` the amount of a species",
        petab="a condition assigns an identifier, and what it means is what the "
        "model means: the amount of a species with `hasOnlySubstanceUnits=true` "
        "and the concentration of one with `false`",
        detail="a change which sets the concentration of an amount based species, "
        "or the amount of a concentration based one, is not an identifier PEtab "
        "can assign and the export raises. An observable is a math expression, so "
        "there the same case is written as `S1 / compartment`",
    ),
    Gap(
        id="data-functions",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="`Data` with a `function`, i.e. reference data or an observable "
        "which is calculated from other data by a python function",
        petab="an observable is a formula over the entities of the model",
        detail="a python function is not a formula, the export raises for an "
        "observable which is one. Reference data which is a function is written "
        "as the numbers it produces",
    ),
    Gap(
        id="noise-parameters",
        kind=GapKind.LOSSY,
        sbmlsim="a fit estimates the parameters of a model and weights the data "
        "of a curve, i.e. the spread of the data is data and not a parameter",
        petab="the parameters of the noise and of the observables are estimated "
        "with the parameters of the model, e.g. a `sd_<observable>` which the "
        "objective fits",
        detail="a parameter of a problem which is not an entity of a model is "
        "not fitted and the reader says which; the data of its observable is "
        "weighted by `FitSettings.weighting_points` instead. A fit of such a "
        "problem is therefore not the fit PEtab describes. A parameter of a "
        "noise formula is kept in the noise model of its fit mapping with its "
        "nominal value, its bounds and whether the problem estimates it, so it "
        "is written as it was read; `log_likelihood` evaluates it at the value "
        "of the parameter set it is given and at the nominal value without one",
    ),
    Gap(
        id="noise-model",
        kind=GapKind.LOSSY,
        sbmlsim="the cost of a fit is a weighted sum of squares. The noise "
        "formula and the noise distribution of a fit mapping are kept as its "
        "`NoiseModel`, which `log_likelihood` evaluates and the optimizer does "
        "not use",
        petab="`noiseFormula` and `noiseDistribution` per observable are the "
        "objective, i.e. the negative log likelihood of the measurements",
        detail="the noise model of a problem which is read is written as it "
        "was read, so the tables of a round trip agree and the log-likelihood "
        "of a problem is compared with the one of other tools. The fit stays a "
        "least squares fit, i.e. its optimum is not the maximum of the "
        "likelihood. A fit mapping without a noise model is written with a "
        "normal noise of the standard deviation of its data, or of `1.0` for "
        "data without errors, and has that noise model when it is read back",
    ),
    Gap(
        id="sciml-model-format",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="a network is the NN YAML of PEtab SciML, which "
        "`sbmlsim.sciml` evaluates and compiles",
        petab="PEtab SciML also allows the formats `pytorch`, `equinox` and "
        "`lux.jl`, i.e. a network in the code of a framework",
        detail="the reader raises for a network which is not in the format "
        "`YAML` and names the network: the code of a framework is not read",
    ),
    Gap(
        id="sciml-layer-sbml",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="a network in the right hand side or in an observable is "
        "compiled into the model as assignment rules, which the MathML of "
        "SBML expresses",
        petab="a layer of the NN YAML is any layer of PEtab SciML, and a "
        "network of any layers sits in the right hand side",
        detail="a convolution, a pooling or a normalization layer, and `gelu` "
        "with the error function, have no MathML: the reader raises for such a "
        "network in the right hand side or in an observable and names the "
        "network and the node. Such a network runs before the simulation",
    ),
    Gap(
        id="sciml-training-mode",
        kind=GapKind.LOSSY,
        sbmlsim="a network is evaluated in evaluation mode: dropout is the "
        "identity and the normalization layers use their stored statistics",
        petab="PEtab SciML does not say in which mode a network is evaluated, "
        "the reference values of its test suite are built in training mode "
        "for dropout",
        detail="the values of a problem with such a layer differ from the ones "
        "of a tool which evaluates in training mode",
    ),
    Gap(
        id="sciml-priors",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="the objective of a fit has no priors (issue #190)",
        petab="`priorDistribution` and `priorParameters` of a parameter of a "
        "network, i.e. of all elements a row covers",
        detail="the reader raises for a prior on the parameters of a network "
        "and names the parameter. A problem with priors states a log-posterior, "
        "which the log-likelihood is not",
    ),
    Gap(
        id="sciml-parameter-scale",
        kind=GapKind.EXTENSION,
        sbmlsim="`FitParameter.scale`, the space the optimizer searches one "
        "parameter in: the elements of a network are negative and zero and "
        "are searched on the linear scale",
        petab="PEtab v2 has no scale of a parameter. The problems of PEtab "
        "SciML carry the column `parameterScale` of PEtab v1",
        detail="the reader reads the column of a problem of PEtab SciML, the "
        "elements of a network are on the linear scale, and the scale of a "
        "parameter goes to the extension",
    ),
    Gap(
        id="foreign-extension",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="the reader interprets the `sbmlsim` extension of a problem",
        petab="a problem carries the extensions of any tool, and `required` "
        "says whether it can be interpreted without one of them",
        detail="a problem which requires an extension `sbmlsim` does not know "
        "is not read, the reader raises and names the extension. An extension "
        "which is not required is ignored with a message in the log",
    ),
    Gap(
        id="x-observable",
        kind=GapKind.UNSUPPORTED,
        sbmlsim="the x of a fit mapping is any observable of the task",
        petab="a measurement is a value at a time",
        detail="a mapping whose x is not the time of the simulation, e.g. a dose "
        "response over a concentration, has no PEtab representation and the "
        "export raises",
    ),
    Gap(
        id="selections",
        kind=GapKind.LOSSY,
        sbmlsim="a fit is a set of `SimulationExperiment` classes, and the "
        "selections of a task are the observables of its experiment",
        petab="a problem is one set of tables, the experiment a measurement "
        "belongs to is not part of it",
        detail="the reader builds one simulation experiment for the problem, so "
        "a task selects the observables of the whole problem and not those of "
        "the experiment it came from. roadrunner returns slightly different "
        "values for a different selection list, i.e. the round trip of the HCTZ "
        "example agrees to 7e-6 in the cost and not exactly",
    ),
    Gap(
        id="integrator",
        kind=GapKind.EXTENSION,
        sbmlsim="the integrator settings of a model, i.e. the KISAO terms of "
        "`sbmlsim.simulation.algorithm`",
        petab="-",
        detail="the settings go to the extension",
    ),
    Gap(
        id="metadata",
        kind=GapKind.EXTENSION,
        sbmlsim="`MappingMetaData`, which describes what a curve is",
        petab="-",
        detail="the metadata goes to the extension",
    ),
    Gap(
        id="experiment-split",
        kind=GapKind.LOSSY,
        sbmlsim="one simulation whose fit mappings are training, validation "
        "and outlier data at once, e.g. an outlier curve simulated together "
        "with the curve which is fitted",
        petab="a `FitMappingCollection` is one kind, and "
        "`select_mapping_collections` gives an experiment one collection per "
        "kind it holds",
        detail="a simulation whose mappings span more than one kind is "
        "written as several PEtab experiments, one per kind, which all carry "
        "the same conditions, so the binding of a versioned parameter is "
        "unaffected: every one of them gets the same version. A problem "
        "which is read back therefore has more simulation groups than the "
        "fit which was written, so a coverage count or a group count read "
        "off it is higher than before, even though the simulations "
        "themselves and their result are the same",
    ),
)

#: the gaps by their id
GAPS_BY_ID: dict[str, Gap] = {gap.id: gap for gap in GAPS}


def _has_a_simulation_of_several_kinds(problem: "OptimizationProblem") -> bool:
    """Check whether an exported simulation mixes the kinds of its mappings.

    `PetabExporter` writes one PEtab experiment per collection, and
    `helpers.select_mapping_collections` gives an experiment one collection
    per kind it holds; a simulation whose mappings are of several kinds is
    therefore written as several PEtab experiments which share one
    simulation, see the `experiment-split` gap.

    Args:
        problem: the initialized problem which is exported.

    Returns:
        `True` if a simulation of an exported mapping is shared by mappings
        of more than one of the kinds `sbmlsim.fit.objects.EVALUATED_KINDS`
        writes, `False` otherwise.
    """
    groups: dict[tuple[int, int], set[MappingKind]] = {}
    for k, kind in enumerate(problem.mapping_kinds):
        if kind not in EVALUATED_KINDS:
            # not written by the exporter, see `PetabExporter.indices`
            continue
        key = (id(problem.models[k]), id(problem.simulations[k]))
        groups.setdefault(key, set()).add(kind)
    return any(len(kinds) > 1 for kinds in groups.values())


def gaps_of_problem(problem: "OptimizationProblem") -> list[Gap]:
    """Report the gaps an optimization problem runs into.

    The problem must be initialized, i.e., its mappings are resolved; the gaps
    of the data are only known then.

    Args:
        problem: the problem which is exported.

    Returns:
        The gaps which apply to this problem, in the order of `GAPS`.

    Raises:
        ValueError: if the problem is not initialized.
    """
    if not problem.is_initialized:
        raise ValueError(
            f"'{problem.opid}': the gaps of a problem require the resolved "
            f"mappings, call `initialize(settings)` first."
        )

    hits: set[str] = {"units", "fit-settings", "output-times"}
    if len(set(problem.experiment_keys)) > 1:
        hits.add("selections")

    if len(set(problem.mapping_kinds)) > 1:
        hits.add("mapping-kind")
    if _has_a_simulation_of_several_kinds(problem):
        hits.add("experiment-split")
    if problem.residual in {
        ResidualType.ABSOLUTE_TO_BASELINE,
        ResidualType.NORMALIZED_TO_BASELINE,
    }:
        hits.add("residual-baseline")
    if problem.weighting_points != WeightingPointsType.NO_WEIGHTING or set(
        problem.weighting_curves
    ) - {WeightingCurvesType.POINTS}:
        hits.add("weights")

    for simulation in problem.simulations:
        timecourses = getattr(simulation, "timecourses", [])
        for k, tc in enumerate(timecourses):
            if tc.discard:
                hits.add("presimulation")
            if (tc.model_changes or tc.model_manipulations) and k >= 0:
                hits.add("model-changes")

    for noise in problem.noise_models:
        if noise is None:
            continue
        hits.add("noise-model")
        if any(parameter.estimate for parameter in noise.parameters):
            hits.add("noise-parameters")

    if any(parameter.scale is not None for parameter in problem.parameters):
        hits.add("sciml-parameter-scale")
    for hybridization in problem.hybridizations:
        # the layers of a network of `sbmlsim.sciml`, without importing it
        model = getattr(getattr(hybridization, "network", None), "model", None)
        layer_types = {layer.layer_type for layer in getattr(model, "layers", [])}
        if layer_types & EVALUATION_MODE_LAYERS:
            hits.add("sciml-training-mode")

    for k, xid in enumerate(problem.xid_observable):
        if xid != "time":
            logger.warning(
                "'%s': the x of the mapping '%s' is '%s' and not the time of the "
                "simulation, which PEtab cannot express.",
                problem.opid,
                problem.mapping_keys[k],
                xid,
            )
            hits.add("x-observable")

    return [gap for gap in GAPS if gap.id in hits]


def gaps_table(gaps: Iterable[Gap], title: str | None = None) -> Table:
    """Get the table of the gaps for the console.

    Args:
        gaps: gaps to show.
        title: title of the table.

    Returns:
        The table of the gaps with their kind and what the layer does.
    """
    table = Table(title=title, title_justify="left", header_style="bold")
    table.add_column("gap")
    table.add_column("kind")
    table.add_column("sbmlsim")
    table.add_column("PEtab v2")
    style: dict[GapKind, str] = {
        GapKind.EXTENSION: "green",
        GapKind.LOSSY: "orange3",
        GapKind.UNSUPPORTED: "red",
    }
    for gap in gaps:
        table.add_row(
            gap.id,
            f"[{style[gap.kind]}]{gap.kind.value}[/{style[gap.kind]}]",
            gap.sbmlsim,
            gap.petab,
        )
    return table


def gaps_dict(gaps: Iterable[Gap]) -> list[dict[str, Any]]:
    """Get the gaps as dictionaries, i.e., for a report or the extension."""
    return [
        {
            "id": gap.id,
            "kind": gap.kind.value,
            "sbmlsim": gap.sbmlsim,
            "petab": gap.petab,
            "detail": gap.detail,
        }
        for gap in gaps
    ]
