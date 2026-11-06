"""A fit of a project: its start, the adopted fits it keeps fixed and its remainders.

A fit of one subset of a project (`sbmlsim.project.Project.fits`) runs on the
models of the project, which carry the values of the adopted fits of the other
subsets (`Project.adopted_fits`), so a fit keeps these values fixed and
records them: `fixed_values` gives the adopted fits of the other subsets and
the values of their parameters, which the provenance of the fit records
(`FIXED_FITS`), and the report of the fit names them, see
`sbmlsim.fit.cli.report_context`. Least squares can start from a point instead
of samples of the bounds (`start_point`, `with_start`): the values of a JSON
file, of a finished or of an adopted fit. `remainder_problems` refuses a fit
whose bounds let a remainder `1 - a - b` of the model become negative.

The `fit` kind of `sbmlsim.run` uses them with the fields `start` and
`record_adopted` of its spec, and the kind `adopt` adopts a finished fit.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from sbmlsim.fit.records import convert_value

if TYPE_CHECKING:
    from sbmlsim.fit.cli import FitDefinition, FitRecords
    from sbmlsim.fit.objects import FitParameter
    from sbmlsim.project import Project

__all__ = [
    "FIXED_FITS",
    "current_fixed_fits",
    "fixed_values",
    "recorded_fixed_fits",
    "remainder_problems",
    "start_point",
    "with_start",
]

#: the argument of the provenance of a fit which names the fixed adopted fits
FIXED_FITS = "fixed_fits"

#: the best parameter set of a fit, next to its other records
_PARAMETERS_FILE = "parameters.json"


def _fit_values(
    directory: Path, parameters: Mapping[str, FitParameter], start: str
) -> dict[str, object]:
    """Get the best values of a fit in the units of the fit parameters.

    Raises:
        ValueError: if a value has no unit for a fit parameter with one.
    """
    best = json.loads((directory / _PARAMETERS_FILE).read_text(encoding="utf-8"))
    best = best["sets"][0]
    units = dict(best.get("units") or {})
    values: dict[str, object] = {}
    for pid, value in best["values"].items():
        if pid not in parameters:
            continue
        unit, target = units.get(pid), parameters[pid].unit
        if unit is None and target is not None:
            raise ValueError(
                f"The fit '{start}' has no unit of '{pid}': its value cannot be "
                f"converted into the unit of the fit parameter, {target}."
            )
        values[pid] = convert_value(float(value), unit, target)
    return values


def start_point(
    project: Project,
    definition: FitDefinition,
    start: Mapping[str, float] | str | Path,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Get the parameters a fit starts from.

    The start is a point, the values by parameter id in the units of the fit
    parameters (`FitParameter.unit`), given as a mapping or as a JSON file of
    one; or a fit: the directory of its records, or a name, the id of a fit,
    which is looked up as an adopted fit of the project (`Project.fits_dir`),
    then as a finished fit in `output_dir` when one is given, and only then
    as a path; the best parameters of the fit (`parameters.json`, in the
    units of the model) are converted into the units of the fit parameters. A fit of another problem
    gives the parameters which the definition fits. A parameter which the
    start does not give keeps its own start value.

    Args:
        project: the project.
        definition: the fit problem.
        start: the point, the JSON file, the directory or the id of a fit.
        output_dir: the directory of the finished fits, none without.

    Returns:
        The start value of every parameter the start gives which the
        definition fits.

    Raises:
        ValueError: if the start is neither a point, a file nor a fit, a value
            is not a number, a parameter of the point is not fitted, a value of
            a fit has no unit for a fit parameter with one, or a value is
            outside of the bounds of its parameter.
    """
    parameters = {p.pid: p for p in definition.parameters}
    values: dict[str, object]
    path = Path(start) if not isinstance(start, Mapping) else None
    # a name is a fit id: an adopted fit, a finished fit, then a path of the
    # current directory
    fits = [
        d / path.name
        for d in (project.fits_dir, output_dir)
        if d is not None
        and path is not None
        and not path.is_absolute()
        and len(path.parts) == 1
    ]
    record = next((d for d in fits if (d / _PARAMETERS_FILE).is_file()), None)
    if isinstance(start, Mapping):
        values = dict(start)
        given = "The start point"
    elif record is not None:
        values = _fit_values(record, parameters, str(start))
        given = ""
    elif path is not None and path.is_file():
        values = dict(json.loads(path.read_text(encoding="utf-8")))
        given = f"The start '{start}'"
    elif path is not None and (path / _PARAMETERS_FILE).is_file():
        values = _fit_values(path, parameters, str(start))
        given = ""
    else:
        raise ValueError(
            f"The start '{start}' is neither a json file nor a fit: no records "
            f"of a fit, adopted in {project.fits_dir} or finished in "
            f"{output_dir}."
        )
    if given:
        unknown = sorted(set(values) - set(parameters))
        if unknown:
            raise ValueError(
                f"{given} gives parameters which are not fitted: {unknown}; the "
                f"fitted parameters are {sorted(parameters)}."
            )
    point: dict[str, float] = {}
    for pid, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"The start value of '{pid}' is not a number: {value!r}.")
        parameter = parameters[pid]
        if not parameter.lower_bound <= value <= parameter.upper_bound:
            raise ValueError(
                f"The start value {value} of '{pid}' is outside of its bounds "
                f"[{parameter.lower_bound}, {parameter.upper_bound}]."
            )
        point[pid] = float(value)
    return point


def with_start(definition: FitDefinition, point: Mapping[str, float]) -> FitDefinition:
    """Get a fit problem whose parameters start from a point.

    The parameters are copies, so the definition of the project keeps its
    start values (the definitions are objects of a module).

    Args:
        definition: the fit problem.
        point: the start value by parameter id.

    Returns:
        The fit problem with the start values of the point.
    """
    parameters = []
    for parameter in definition.parameters:
        if parameter.pid in point:
            parameter = copy.copy(parameter)
            parameter.start_value = point[parameter.pid]
        parameters.append(parameter)
    return dataclasses.replace(definition, parameters=parameters)


def current_fixed_fits(project: Project, subset: str) -> list[str]:
    """Get the adopted fits whose parameters a fit of a subset keeps fixed.

    Args:
        project: the project.
        subset: the fit problem, e.g. `PK`.

    Returns:
        The adopted fits of the project of the other subsets, in the order of
        `Project.adopted_fits`.
    """
    return [
        fit_id for other, fit_id in project.adopted_fits().items() if other != subset
    ]


def fixed_values(
    project: Project, subset: str, definition: FitDefinition
) -> tuple[list[str], dict[str, float]]:
    """Get what a fit of a subset keeps fixed: the adopted fits of the other problems.

    Args:
        project: the project.
        subset: the fit problem, e.g. `PK`.
        definition: its definition, whose parameters are fitted and not fixed.

    Returns:
        The adopted fits of the other subsets (`current_fixed_fits`) and the
        values of their parameters in the units of the model, without the
        parameters the definition fits.
    """
    fixed_fits = current_fixed_fits(project, subset)
    if not fixed_fits:
        return [], {}
    records = project.fit_records().records
    fitted = {p.pid for p in definition.parameters}
    values = {
        pid: parameter.value
        for fit_id in fixed_fits
        for pid, parameter in records[fit_id].parameters.items()
        if pid not in fitted
    }
    return fixed_fits, values


def recorded_fixed_fits(records: FitRecords) -> tuple[str, ...]:
    """Get the fixed adopted fits which the provenance of a fit records.

    A fit records them in the arguments of its provenance (`FIXED_FITS`).

    Args:
        records: the records of the fit (`sbmlsim.fit.cli.load_fit_records`).

    Returns:
        The ids of the fixed fits, none for a fit without a result, without a
        provenance or which did not record them.
    """
    opt_result = records.opt_result
    if opt_result is None or opt_result.provenance is None:
        return ()
    return tuple(opt_result.provenance.arguments.get(FIXED_FITS, ()))


def remainder_problems(
    sbml_path: Path,
    parameters: Iterable[FitParameter],
    fixed_values: Mapping[str, float] | None = None,
) -> list[str]:
    """Check that the fitted fractions of a remainder cannot sum above 1.

    A remainder is the initial assignment `1 - a - b - ...` of a constant of
    the model, e.g. the remainder of the product fractions of a metabolism: it
    keeps the sum of the fractions at 1, and is negative, a product formed at a
    negative rate, when the other fractions sum above 1. A fit whose bounds
    allow it is refused before it starts: the upper bounds of the fitted
    fractions of a remainder and the values of the others (the fixed values,
    else those of the model) sum to at most 1.

    Args:
        sbml_path: the model the fit simulates.
        parameters: the parameters of the fit.
        fixed_values: the values of parameters the fit keeps fixed, by id.

    Returns:
        The problems, naming the remainder, the fitted parameters and their
        upper bounds.
    """
    import libsbml

    fixed_values = fixed_values or {}
    upper = {p.pid: p.upper_bound for p in parameters}
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model | None = doc.getModel()
    if model is None:
        return []
    problems: list[str] = []
    for assignment in model.getListOfInitialAssignments():
        terms = _remainder_terms(assignment.getMath())
        if terms is None or not upper.keys() & set(terms):
            continue
        total = 0.0
        for term in terms:
            if term in upper:
                total += upper[term]
            elif term in fixed_values:
                total += fixed_values[term]
            elif (parameter := model.getParameter(term)) is not None:
                total += parameter.getValue()
        if not total <= 1.0 + 1e-9:
            fitted = ", ".join(f"{t} <= {upper[t]:g}" for t in terms if t in upper)
            problems.append(
                f"the remainder '{assignment.getSymbol()}' = 1 - "
                f"{' - '.join(terms)} can be negative: the fractions sum to "
                f"{total:g} > 1 at the upper bounds of the fit ({fitted}); "
                "lower the upper bounds so that they sum to at most 1"
            )
    return problems


def _remainder_terms(ast: object) -> list[str] | None:
    """The ids of a formula `1 - a - b - ...`, `None` for another formula."""
    import libsbml

    if not isinstance(ast, libsbml.ASTNode):
        return None
    terms: list[str] = []
    node = ast
    while node.getType() == libsbml.AST_MINUS and node.getNumChildren() == 2:
        right = node.getChild(1)
        if right.getType() != libsbml.AST_NAME:
            return None
        terms.insert(0, right.getName())
        node = node.getChild(0)
    if not terms or not node.isNumber() or node.getValue() != 1.0:
        return None
    return terms
