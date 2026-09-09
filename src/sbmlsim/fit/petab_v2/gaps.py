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

from sbmlsim.fit.options import ResidualType, WeightingCurvesType, WeightingPointsType

if TYPE_CHECKING:
    from sbmlsim.fit.optimization import OptimizationProblem

logger = logging.getLogger(__name__)


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
        detail="the training data is the PEtab problem, the kind of every mapping "
        "goes to the extension. A tool which reads the problem without it fits "
        "the validation data as well, which is why the outliers are not written "
        "at all",
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
        "problem is therefore not the fit PEtab describes",
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
)

#: the gaps by their id
GAPS_BY_ID: dict[str, Gap] = {gap.id: gap for gap in GAPS}


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
