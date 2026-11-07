"""The table of the parameters of a sensitivity analysis, as sbmlsim 0.8 wrote it.

The parameters a sensitivity analysis of a model varies, with their values,
bounds, units, types and references, and the table of them,
`parameters.tsv` and `parameters.tex`, which the runner of the analyses
(`sbmlsim.sensitivity.runner`) writes next to their results. The rows and
columns are the ones of sbmlsim 0.8, so the tables which manuscripts read stay
the same:

```py
parameters = parameters_from_sbml(MODEL_PATH, exclude_ids={"Mr_hctz"})
set_bounds(parameters, [("f_renal_function", 0.85, 1.15, ParameterType.SCALING)])
set_relative_bounds(parameters, 0.15)
write_parameters(parameters, out)
design = sobol(distributions(parameters), n=4096, seed=1234, model=MODEL_PATH)
```

`sbmlsim.simulation.sampling.parameters_of` lists the ids of the constant
parameters of a model for a design; `parameters_from_sbml` is the table of
sbmlsim 0.8 with the names, values and units, which selects the parameters
otherwise (the constant compartments and the constant or boundary species as
well, in the order of the model).

A global design draws the values around the values of the table. A model with
several variants (`sbmlsim.project.Project.variants`, e.g. a variant per population)
samples a group of another variant relative to the values of its variant:
`build_values` reads them, `design_factors` gives the ratios and
`relative_design` scales the design.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import libsbml
import numpy as np
import pandas as pd
from sbmlutils.report.units import udef_to_string

from sbmlsim.atomic import write_text_atomic
from sbmlsim.simulation import Dimension, Simulation
from sbmlsim.simulation.sampling import Uniform

__all__ = [
    "COLUMNS",
    "ParameterType",
    "SensitivityParameter",
    "build_values",
    "design_factors",
    "distributions",
    "free_parameters",
    "parameters_from_sbml",
    "parameters_to_df",
    "relative_design",
    "set_bounds",
    "set_relative_bounds",
    "write_parameters",
]

logger = logging.getLogger(__name__)

#: the columns of the table of the parameters, in the order of sbmlsim 0.8
COLUMNS: tuple[str, ...] = (
    "uid",
    "name",
    "value",
    "lower_bound",
    "upper_bound",
    "unit",
    "type",
    "reference",
)


class ParameterType(StrEnum):
    """Where the value and the bounds of a parameter of an analysis come from."""

    DATA = "data"
    SCALING = "scaling"
    NA = "na"
    FIT = "fitted"


@dataclass(kw_only=True)
class SensitivityParameter:
    """A parameter of a sensitivity analysis.

    Attributes:
        uid: the id in the model.
        name: the name in the model, the id if it has none.
        value: the value in the model after its initialization.
        lower_bound: the lower bound of the analysis.
        upper_bound: the upper bound of the analysis.
        unit: the unit in the model.
        type: where the value and the bounds come from.
        reference: the references of the value, e.g. a LaTeX citation.
    """

    uid: str
    name: str
    value: float = math.nan
    lower_bound: float = math.nan
    upper_bound: float = math.nan
    unit: str | None = None
    type: ParameterType = ParameterType.NA
    reference: str = ""


def parameters_from_sbml(
    sbml_path: Path,
    exclude_ids: Iterable[str] | None = None,
    exclude_na: bool = True,
    exclude_zero: bool = True,
    *,
    exclude_derived: bool = False,
) -> list[SensitivityParameter]:
    """Get the parameters of a model which an analysis varies, as sbmlsim 0.8 did.

    The constant parameters, the constant compartments and the constant or
    boundary species, in the order of the model, with their values after the
    initialization (the concentration of a species with only substance units).
    A parameter whose value is not a number (`exclude_na`) or is close to zero
    (`exclude_zero`), and every id of `exclude_ids`, is left out; so is every
    derived parameter, compartment and species (`exclude_derived`).

    Args:
        sbml_path: the SBML model.
        exclude_ids: the ids to leave out.
        exclude_na: leave out a parameter without a value.
        exclude_zero: leave out a parameter whose value is close to zero.
        exclude_derived: leave out every parameter, compartment and species
            with an initial assignment, e.g. the organ volumes and the
            clearances which a physiologically based model derives from its
            covariates.

    Returns:
        The parameters, without bounds.
    """
    from sbmlsim.engine import JitOptimization
    from sbmlsim.model.model_roadrunner import load_roadrunner

    # the values only: compiled with LLJIT at O0, which compiles fastest and
    # gives the values of roadrunner's default MCJIT bitwise
    r = load_roadrunner(sbml_path, JitOptimization.O0)
    model: libsbml.Model = libsbml.readSBMLFromFile(str(sbml_path)).getModel()
    excluded = set(exclude_ids or ())
    parameters: list[SensitivityParameter] = []

    def parameter(sbase: libsbml.SBase) -> SensitivityParameter:
        uid = sbase.getId()
        selection = uid
        if isinstance(sbase, libsbml.Species) and sbase.getHasOnlySubstanceUnits():
            selection = f"[{uid}]"
        return SensitivityParameter(
            uid=uid,
            name=sbase.getName() if sbase.isSetName() else uid,
            value=float(r.getValue(selection)),
            unit=udef_to_string(
                sbase.getDerivedUnitDefinition(), model=None, format="str"
            ),
        )

    for p in model.getListOfParameters():
        if p.getConstant():
            if exclude_na and np.isnan(p.getValue()):
                excluded.add(p.getId())
            if exclude_zero and np.isclose(r.getValue(p.getId()), 0.0):
                excluded.add(p.getId())
            parameters.append(parameter(p))
    for c in model.getListOfCompartments():
        if c.getConstant():
            if exclude_na and np.isnan(c.getSize()):
                excluded.add(c.getId())
            if exclude_zero and np.isclose(r.getValue(c.getId()), 0.0):
                excluded.add(c.getId())
            parameters.append(parameter(c))
    for s in model.getListOfSpecies():
        amount, concentration = s.isSetInitialAmount(), s.isSetInitialConcentration()
        if exclude_na and (
            (not amount and not concentration)
            or (amount and np.isnan(s.getInitialAmount()))
            or (concentration and np.isnan(s.getInitialConcentration()))
        ):
            excluded.add(s.getId())
        if exclude_zero and (
            (amount and np.isclose(s.getInitialAmount(), 0.0))
            or (concentration and np.isclose(s.getInitialConcentration(), 0.0))
        ):
            excluded.add(s.getId())
        if s.getConstant() or s.getBoundaryCondition():
            parameters.append(parameter(s))
    if exclude_derived:
        excluded.update(
            p.uid for p in parameters if model.getInitialAssignmentBySymbol(p.uid)
        )
    logger.info(
        "Excluded parameters: %s",
        sorted(p.uid for p in parameters if p.uid in excluded),
    )
    return [p for p in parameters if p.uid not in excluded]


def set_bounds(
    parameters: Iterable[SensitivityParameter],
    bounds: Iterable[tuple[str, float, float, ParameterType]],
) -> None:
    """Set the bounds and the type of parameters.

    Args:
        parameters: the parameters, changed in place.
        bounds: `(uid, lower bound, upper bound, type)`; a uid which is no
            parameter is logged and skipped, as sbmlsim 0.8 printed it.
    """
    by_uid = {p.uid: p for p in parameters}
    for uid, lower, upper, kind in bounds:
        if uid not in by_uid:
            logger.warning("unused bounds definition: %s = [%s, %s]", uid, lower, upper)
            continue
        p = by_uid[uid]
        p.lower_bound, p.upper_bound, p.type = lower, upper, kind


def set_relative_bounds(
    parameters: Iterable[SensitivityParameter], fraction: float
) -> None:
    """Let every parameter without bounds vary by `value * (1 +- fraction)`.

    The bounds of a negative value are reversed, the lower bound
    `value * (1 - fraction)` is above the upper bound `value * (1 + fraction)`,
    as in the table of sbmlsim 0.8, which stays the same; `distributions`
    refuses such a parameter, naming it, so a model sets its bounds with
    `set_bounds`. A value of 0 has the bounds 0 and 0.

    Args:
        parameters: the parameters, changed in place.
        fraction: the relative half width.
    """
    for p in parameters:
        if math.isnan(p.lower_bound) and math.isnan(p.upper_bound):
            p.lower_bound = p.value * (1 - fraction)
            p.upper_bound = p.value * (1 + fraction)


def parameters_to_df(
    parameters: Sequence[SensitivityParameter],
    sort: bool = True,
    builds: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Get the table of the parameters.

    Args:
        parameters: the parameters.
        sort: sort the rows by the type and the id, as sbmlsim 0.8 did.
        builds: the values of the parameters in other variants of the model
            than the one of the table, by the name of the variant (the runner
            gives those of the variants of its groups other than the default
            one); each is the column `value_<variant>` after `value`.

    Returns:
        One row per parameter, the columns `COLUMNS` (and `value_<variant>`),
        the type as its value.
    """
    builds = builds or {}
    columns = list(COLUMNS)
    position = columns.index("value") + 1
    columns[position:position] = [f"value_{variant}" for variant in builds]
    rows = [
        {
            **asdict(p),
            "type": str(p.type),
            **{
                f"value_{variant}": values.get(p.uid, math.nan)
                for variant, values in builds.items()
            },
        }
        for p in parameters
    ]
    df = pd.DataFrame(rows, columns=columns)
    if sort:
        df = df.sort_values(by=["type", "uid"], ascending=True, ignore_index=True)
    return df


def write_parameters(
    parameters: Sequence[SensitivityParameter],
    directory: Path,
    builds: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Write the table of the parameters as `parameters.tsv` and `parameters.tex`.

    Each file is written atomically (`sbmlsim.atomic.write_text_atomic`).

    Args:
        parameters: the parameters.
        directory: the directory, created if it does not exist.
        builds: the values in the other variants of the model, see
            `parameters_to_df`.

    Returns:
        The table.
    """
    directory.mkdir(parents=True, exist_ok=True)
    df = parameters_to_df(parameters, builds=builds)
    write_text_atomic(
        directory / "parameters.tsv",
        df.to_csv(None, sep="\t", index=False, lineterminator="\n"),
    )
    tex = df.to_latex(None, index=False, float_format="{:.3g}".format)
    write_text_atomic(directory / "parameters.tex", tex.replace("_", r"\_"))
    return df


def distributions(
    parameters: Sequence[SensitivityParameter],
) -> dict[str, Uniform]:
    """Get the distributions of the parameters of a global analysis, uniform in their bounds.

    Args:
        parameters: the parameters, with bounds.

    Returns:
        The distribution of every parameter, by its id.

    Raises:
        ValueError: naming the parameters without bounds, or with a lower bound
            above the upper one (the relative bounds of a negative value, see
            `set_relative_bounds`).
    """
    missing = sorted(
        p.uid
        for p in parameters
        if math.isnan(p.lower_bound) or math.isnan(p.upper_bound)
    )
    if missing:
        raise ValueError(f"The parameters {missing} have no bounds.")
    reversed_bounds = [
        f"'{p.uid}' ({p.lower_bound:g} > {p.upper_bound:g})"
        for p in sorted(parameters, key=lambda p: p.uid)
        if p.lower_bound > p.upper_bound
    ]
    if reversed_bounds:
        raise ValueError(
            f"The lower bound of the parameters {', '.join(reversed_bounds)} is "
            "above the upper one, e.g. the relative bounds of a negative value: "
            "set their bounds with `set_bounds`."
        )
    return {
        p.uid: Uniform(lower=p.lower_bound, upper=p.upper_bound) for p in parameters
    }


def free_parameters(
    parameters: Sequence[SensitivityParameter], simulations: Iterable[Simulation]
) -> list[SensitivityParameter]:
    """Get the parameters an analysis varies: those which none of its simulations sets.

    A design replaces its targets wherever a simulation sets them, so a
    parameter which the simulation of a group sets (in its `preinit_changes`,
    the `preinit_changes` of its presimulation or in a `Change`, e.g. the
    degree of a renal impairment) would take the values of the design in every
    group, and the groups would be one analysis. sbmlsim 0.8.5 left such a
    parameter out of an analysis, the designs of `sbmlsim.simulation.sampling`
    do not (`Scan` warns about it), so a runner takes the parameters of a
    design from here. The ids are compared as written, plain ids as sbmlsim
    0.8.5 did: a target given as a selection (`[S]`) does not match the
    parameter `S`. Every parameter left out is logged ("Removing fixed
    parameter"), as sbmlsim 0.8.5 printed it.

    Args:
        parameters: the parameters of the analysis.
        simulations: the simulations of the groups of the analysis.

    Returns:
        The parameters which no simulation sets, in their order.
    """
    fixed: set[str] = set()
    for simulation in simulations:
        fixed.update(simulation.preinit_changes)
        if simulation.presimulation is not None:
            fixed.update(simulation.presimulation.preinit_changes)
        for change in simulation.changes:
            fixed.update(change.values)
    for p in parameters:
        if p.uid in fixed:
            logger.info("Removing fixed parameter: %s", p.uid)
    return [p for p in parameters if p.uid not in fixed]


def build_values(sbml_path: Path, uids: Iterable[str]) -> dict[str, float]:
    """Get the values of parameters in a variant of a model, as `parameters_from_sbml` reads them.

    Args:
        sbml_path: the SBML model, e.g. a variant of the model.
        uids: the ids of the parameters, compartments and species.

    Returns:
        The value after the initialization of every id the model has.
    """
    wanted = set(uids)
    return {
        p.uid: p.value
        for p in parameters_from_sbml(sbml_path, exclude_na=False, exclude_zero=False)
        if p.uid in wanted
    }


def design_factors(
    parameters: Sequence[SensitivityParameter], values: Mapping[str, float]
) -> dict[str, float]:
    """Get the factors which carry a design of the table to the values of a variant.

    A global design draws the values of the parameters in their bounds around
    the values of the table, which are those of the default variant of the
    model. A group of another variant samples the same design relative to the
    values of its variant: every value of a parameter is multiplied with the
    ratio of its value in the variant to its value in the table, so e.g. a
    group of a variant varies the body weight around the body weight of its
    variant by the same factors as the default variant around its own. The
    bounds of the table are taken relative to its values.

    Args:
        parameters: the parameters which a design varies.
        values: the values of the parameters in the variant of a group.

    Returns:
        The ratio of every parameter whose value differs, by its id.

    Raises:
        ValueError: for a parameter which has no value in the variant, and for
            a parameter whose value differs and whose bounds cannot be taken
            relative to its value: a value of the table which is zero or not
            finite, or a ratio which is not positive (another sign); naming the
            parameter and both values.
    """
    missing = [p.uid for p in parameters if p.uid not in values]
    if missing:
        raise ValueError(
            f"The varied parameter {', '.join(repr(m) for m in missing)} has no "
            "value in the variant, so the design has no counterpart in it."
        )
    factors: dict[str, float] = {}
    problems: list[str] = []
    for p in parameters:
        value = values[p.uid]
        if value == p.value:
            continue
        ratio = value / p.value if p.value != 0 else math.nan
        if not (math.isfinite(p.value) and math.isfinite(ratio) and ratio > 0):
            problems.append(
                f"'{p.uid}' ({p.value:g} in the table, {value:g} in the variant)"
            )
            continue
        factors[p.uid] = ratio
    if problems:
        raise ValueError(
            f"The values of the parameters {', '.join(problems)} differ between "
            "the table and the variant, and their bounds cannot be taken relative "
            "to the value of the table, so the design has no counterpart in the "
            "variant: give them a value in the table which is neither zero nor of "
            "another sign, or leave them out of the analysis."
        )
    return factors


def relative_design(design: Dimension, factors: Mapping[str, float]) -> Dimension:
    """Get a design with the values of its parameters multiplied by factors.

    The design of a group of another variant, see `design_factors`; the
    bounds of the uniform distributions of its record are multiplied as well.

    Args:
        design: the design over the values of the table.
        factors: the factor of a parameter, by its id; 1 for the others.

    Returns:
        The design of the group, the given one without a factor.
    """
    factors = {uid: f for uid, f in factors.items() if uid in design.values}
    if not factors:
        return design
    record = design.design
    if record is not None:
        scaled = {}
        for uid, d in record.distributions.items():
            if uid in factors:
                d = {
                    **d,
                    **{
                        key: d[key] * factors[uid]
                        for key in ("lower", "upper")
                        if isinstance(d.get(key), int | float)
                    },
                }
            scaled[uid] = d
        record = dataclasses.replace(record, distributions=scaled)
    return Dimension(
        design.id,
        values={
            uid: np.asarray(v) * factors.get(uid, 1.0)
            for uid, v in design.values.items()
        },
        labels=design.labels,
        design=record,
    )
