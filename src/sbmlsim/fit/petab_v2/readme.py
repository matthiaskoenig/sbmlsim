"""The README of an exported PEtab v2 problem.

`to_petab` writes a `README.md` next to `problem.yaml`, which says in words
what the files of the problem hold and what they do not: the fit problem and
the values it starts from, the data which is written and the data which is
not (the excluded data, the kinds which are not exported and the fit mappings
of an unsupported gap with `unsupported="skip"`), every gap of the problem
with what it means for a tool other than `sbmlsim`, the measurements at the
time of a change and how the problem is read back.

The README is a function of the problem and of the version of `sbmlsim`
only: it has no time and no path of the machine, so an export of the same
problem writes the same file.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any

from petab.v2 import Problem as PetabProblem

from sbmlsim.fit.objects import EVALUATED_KINDS, UNUSED_KINDS, MappingKind
from sbmlsim.fit.options import DataAtChangeType, ObjectiveType
from sbmlsim.fit.petab_v2.gaps import (
    GAPS,
    GAPS_BY_ID,
    Gap,
    GapKind,
    mapping_name,
    measurements_at_changes,
    unsupported_mappings,
)

if TYPE_CHECKING:
    from sbmlsim.fit.petab_v2.export import PetabExporter

#: the name of the README of a problem
README_FILE = "README.md"

#: what a kind of gap means for a tool other than `sbmlsim`
KIND_MEANING: dict[GapKind, str] = {
    GapKind.EXTENSION: "PEtab has no place for it and the extension `sbmlsim` of "
    "`problem.yaml` carries it: `sbmlsim` reads it back exactly, a tool which "
    "does not know the extension does not see it.",
    GapKind.LOSSY: "the export transforms it: what another tool reads differs "
    "from the fit, and for some of them, as the gap says, also what `sbmlsim` "
    "reads back.",
    GapKind.UNSUPPORTED: "PEtab has no representation for it: the fit mappings "
    "are left out of the problem, see [Data which is not written]"
    "(#data-which-is-not-written).",
}

#: what a fit does with the data of a kind
KIND_USE: dict[MappingKind, str] = {
    MappingKind.TRAINING: "fitted",
    MappingKind.VALIDATION: "evaluated, not fitted",
    MappingKind.OUTLIER: "evaluated, not fitted",
}


def _number(value: Any) -> str:
    """Format a number of a table, six significant digits."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.6g}"
    except TypeError, ValueError:
        return str(value)


def _word(value: Any) -> str:
    """Format a setting: the value of a `StrEnum`, the name of an `Enum`."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        words = [_word(v) for v in value]
        return ", ".join(words) if words else "-"
    return str(value)


def _code(text: str) -> str:
    """Format a name as code of markdown."""
    return f"`{text}`"


def _row(*cells: str) -> str:
    """Get a row of a markdown table, `|` of a cell escaped."""
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """Get a markdown table."""
    return [
        _row(*header),
        _row(*(["---"] * len(header))),
        *(_row(*row) for row in rows),
    ]


def _grouped(names: Iterable[str]) -> list[str]:
    """Get the fit mappings `<experiment>.<mapping>` as a list by experiment."""
    by_experiment: dict[str, list[str]] = defaultdict(list)
    for name in names:
        experiment, _, mapping = name.partition(".")
        by_experiment[experiment].append(mapping)
    return [
        f"- {_code(experiment)}: {', '.join(_code(m) for m in mappings)}"
        for experiment, mappings in by_experiment.items()
    ]


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """Get a count with its noun, `1 fit mapping`, `2 fit mappings`."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def problem_readme(
    exporter: PetabExporter,
    petab_problem: PetabProblem,
    files: Sequence[str],
    origin: str | None = None,
    related: Mapping[str, str] | None = None,
) -> str:
    """Get the README of an exported problem.

    Args:
        exporter: the export, after `to_problem`.
        petab_problem: the problem it built.
        files: the files of the problem relative to its directory, the YAML
            file first.
        origin: where the values of the parameters come from, e.g. `the best
            parameters of the fit PK_...`; the start values of the fit
            parameters or a given parameter set without.
        related: the other problems of the same fit by their role, as a path
            relative to the directory of the problem, e.g. `{"validation":
            "../validation"}` for the training problem, see
            `to_petab_problems`.

    Returns:
        The README as markdown.
    """
    import sbmlsim

    problem = exporter.problem
    settings = problem.settings_initialized
    opid = problem.opid
    role = exporter.role
    related = dict(related or {})
    gaps = list(exporter.gaps)
    gap_ids = {gap.id for gap in gaps}
    skipped = exporter.skipped
    written = sorted(exporter.indices)
    if origin is None:
        origin = (
            "the start values of the fit parameters"
            if all(exporter.values[p.pid] == p.start_value for p in problem.parameters)
            else "a parameter set of the fit"
        )

    title = {
        "fit": f"PEtab v2 problem {_code(opid)}",
        "training": f"PEtab v2 training problem of {_code(opid)}",
        "validation": f"PEtab v2 validation problem of {_code(opid)}",
    }[role]
    what = {
        "fit": f"The fit problem {_code(opid)} with the data of the kinds "
        f"{', '.join(_code(k.value) for k in EVALUATED_KINDS if k in exporter.kinds)}",
        "training": f"The training problem of the fit {_code(opid)}: its training "
        "data, the data the fit optimizes, and its parameters to estimate",
        "validation": f"The validation problem of the fit {_code(opid)}: its "
        "validation data, which the fit does not optimize, at the values of the "
        "parameters of the fit, which the problem does not estimate",
    }[role]
    lines: list[str] = [
        f"# {title}",
        "",
        f"{what}, as a problem of [PEtab v2](https://petab.readthedocs.io), "
        "written by `sbmlsim.fit.petab_v2` of [sbmlsim](https://github.com/"
        f"matthiaskoenig/sbmlsim) {sbmlsim.__version__}. This file describes "
        "the problem, the data it holds and does not hold and what PEtab does "
        "not express about the fit; it is no part of the problem.",
        "",
    ]
    if role != "fit":
        lines += [
            "A fit is written as two problems: the training problem, which a tool "
            "fits, and the validation problem, which a tool evaluates at the "
            "values of the fit. PEtab v2 has no collection of problems in one "
            "YAML file (PEtab v1 had `problems`), so they are two problems with "
            "their own files"
            + (
                ": "
                + ", ".join(
                    f"the {other} problem `{path}`" for other, path in related.items()
                )
                + "."
                if related
                else (
                    "; PEtab v2 cannot express any validation data of this fit, "
                    "see [Data which is not written](#data-which-is-not-written), "
                    "so it has no validation problem."
                    if role == "training" and problem.indices(MappingKind.VALIDATION)
                    else "; this fit has no "
                    + ("validation" if role == "training" else "training")
                    + " data, so it has no other problem."
                )
            ),
            "",
        ]
    if role == "validation":
        lines += [
            "The parameters of the fit are not estimated (`estimate=false` of "
            "`parameters.tsv`): the problem checks the fit on data it was not "
            "fitted to, and a problem which estimated them would fit the "
            "validation data, i.e. be another training problem. Their bounds "
            "are kept, and the nominal values are the values of the fit, which "
            "every PEtab tool simulates the problem with. `sbmlsim` reads them "
            "as the parameters of the fit at these values (`role: validation` "
            "of the extension), so the problem it reads has a cost, see "
            "[Reading the problem](#reading-the-problem).",
            "",
        ]

    # --- THE FIT PROBLEM ---
    collections = {
        problem.collection_indices[k]: problem.mapping_collections[
            problem.collection_indices[k]
        ]
        for k in written
    }
    n_measurements = len(petab_problem.measurements)
    fit_ids = {p.pid for p in problem.parameters}
    rows_of_fit = [p for p in petab_problem.parameters if p.id in fit_ids]
    n_estimated = sum(1 for p in petab_problem.parameters if p.estimate)
    objective = {
        ObjectiveType.LEAST_SQUARES: "weighted least squares",
        ObjectiveType.LIKELIHOOD: "negative log-likelihood",
        ObjectiveType.POSTERIOR: "negative log-posterior",
    }.get(settings.objective, _word(settings.objective))
    lines += [
        "## The fit problem",
        "",
        *_table(
            ("", ""),
            [
                ("fit", _code(opid)),
                ("problem", role),
                ("parameter values", origin),
                ("objective", f"{objective} ({_code(_word(settings.objective))})"),
                ("residual", _code(_word(settings.residual))),
                ("loss function", _code(_word(settings.loss_function))),
                ("weighting of the curves", _code(_word(settings.weighting_curves))),
                ("weighting of the points", _code(_word(settings.weighting_points))),
                ("data at a change", _code(_word(settings.data_at_change))),
                ("parameter scale", _code(_word(settings.parameter_scale))),
                (
                    "experiments",
                    f"{len(petab_problem.experiments)} PEtab experiments of "
                    f"{_count(len(collections), 'collection')} of fit mappings",
                ),
                (
                    "observables",
                    f"{len(petab_problem.observables)} of "
                    f"{_count(len(written), 'fit mapping')}",
                ),
                ("measurements", str(n_measurements)),
                (
                    "parameters",
                    f"{len(rows_of_fit)} of the fit, "
                    + (f"{n_estimated} estimated" if n_estimated else "none estimated"),
                ),
                (
                    "extension `sbmlsim`",
                    "required" if exporter.required_extension else "not required",
                ),
            ],
        ),
        "",
        "The settings of the fit are `settings` of the extension `sbmlsim`. The "
        "parameters of the fit with their bounds and the values the problem "
        + ("is evaluated at" if role == "validation" else "starts from")
        + ", the `nominalValue` of `parameters.tsv`, in the units of the model:",
        "",
    ]
    units = {p.pid: p.unit for p in problem.parameters}
    targets = {p.pid: p.target_id for p in problem.parameters}
    lines += _table(
        (
            "parameter",
            "target",
            "unit",
            "lower bound",
            "upper bound",
            "value",
            "estimated",
        ),
        [
            (
                _code(p.id),
                _code(targets.get(p.id, p.id)),
                str(units.get(p.id) or "-"),
                _number(p.lb),
                _number(p.ub),
                _number(p.nominal_value),
                "yes" if p.estimate else "no",
            )
            for p in rows_of_fit
        ],
    )
    lines.append("")
    if problem.fixed_values:
        lines += [
            f"The fit keeps {_count(len(problem.fixed_values), 'entity', 'entities')} "
            "of the model fixed, e.g. at the values of the adopted fits of other "
            "fit problems; the condition of the first period of every experiment "
            "sets them, in the units of the model:",
            "",
            *_table(
                ("entity", "value"),
                [
                    (_code(target), _number(value))
                    for target, value in problem.fixed_values.items()
                ],
            ),
            "",
        ]

    # --- FILES ---
    tables = {
        "conditions.tsv": "the conditions, the changes of the periods of the "
        "experiments",
        "experiments.tsv": "the experiments, a period per start and change of a "
        "simulation",
        "observables.tsv": "the observables, one per fit mapping, named "
        "`<experiment>.<mapping>`, with the noise model",
        "measurements.tsv": "the data of the fit mappings in the units of the model",
        "parameters.tsv": "the parameters with their bounds and nominal values",
        "mapping.tsv": "the mapping of the ids of the problem",
    }
    models = {
        str(getattr(model, "rel_path", "")): model.model_id
        for model in petab_problem.models
    }
    rows: list[tuple[str, str]] = []
    for i, path in enumerate(files):
        if i == 0:
            rows.append(
                (_code(path), "the problem: its files and the extension `sbmlsim`")
            )
        elif path in models:
            rows.append((_code(path), f"the model {_code(models[path])} (SBML)"))
        elif path in tables:
            rows.append((_code(path), tables[path]))
        else:
            rows.append((_code(path), "a file of the networks of PEtab SciML"))
    rows.append((_code(README_FILE), "this description"))
    lines += ["## Files", "", *_table(("file", "content"), rows), ""]

    # --- DATA ---
    lines += _data(exporter, petab_problem, written, collections)
    lines += _not_written(exporter, skipped, related)

    # --- GAPS ---
    lines += [
        "## What PEtab does not express",
        "",
        "The gaps of this problem from the catalogue of the differences between "
        "a fit of `sbmlsim` and PEtab v2, `sbmlsim.fit.petab_v2.gaps`. Their "
        "kind says what they mean for a tool other than `sbmlsim`:",
        "",
        *(f"- {kind.value}: {KIND_MEANING[kind]}" for kind in GapKind),
        "",
    ]
    all_gaps = [gap for gap in GAPS if gap.id in gap_ids or gap.id in skipped]
    if all_gaps:
        lines += _table(
            ("gap", "kind"),
            [(f"[{_code(g.id)}](#{g.id})", g.kind.value) for g in all_gaps],
        )
        lines.append("")
    for gap in all_gaps:
        lines += _gap(gap, len(skipped.get(gap.id, [])))

    # --- MEASUREMENTS AT CHANGES ---
    lines += _at_changes(exporter, written)

    # --- READING ---
    lines += _reading(exporter, gaps)
    return "\n".join(lines).rstrip("\n") + "\n"


def _reading(exporter: PetabExporter, gaps: Sequence[Gap]) -> list[str]:
    """Get the section of how the problem is read and how other tools see it."""
    role = exporter.role
    lossy = [g.id for g in gaps if g.kind is GapKind.LOSSY]
    data = {
        "fit": "the data of the problem",
        "training": "the training data",
        "validation": "the validation data",
    }[role]
    lines = [
        "## Reading the problem",
        "",
        "`sbmlsim` reads the problem with `from_petab`, which returns the "
        "optimization problem and the settings of its fit, and evaluates the "
        f"cost of {data} at the values the problem "
        + ("is evaluated at:" if role == "validation" else "starts from:"),
        "",
        "```python",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "",
        "from sbmlsim.fit.petab_v2 import from_petab, log_likelihood",
        "",
        'problem, settings = from_petab(Path("problem.yaml"))',
        "problem.initialize(settings)",
        "x = np.asarray(problem.x0, dtype=float)",
        "cost = problem.cost_least_square(problem.to_scale(x))",
        "llh = log_likelihood(problem)  # the log-likelihood of PEtab",
        "```",
        "",
        "It reads the tables and the extension `sbmlsim`, so the problem it "
        "reads is the problem which was written, with the settings, the units "
        "and the kind of every fit mapping"
        + (
            "; the validation data is the data of the objective of the "
            "problem which is read, so the cost is the one of the validation "
            "data. "
            if role == "validation"
            else ". "
        )
        + "The cost is the one `sbmlsim` computes for the fit, up to "
        + (
            "what the lossy gaps "
            + ", ".join(f"[{_code(g)}](#{g})" for g in lossy)
            + " say, e.g. the selections of the tasks"
            if "selections" in lossy
            else (
                "what the lossy gaps "
                + ", ".join(f"[{_code(g)}](#{g})" for g in lossy)
                + " say"
                if lossy
                else "the tolerances of the integrator"
            )
        )
        + ". A problem which is read comes back as one collection of fit "
        "mappings per PEtab experiment.",
        "",
    ]
    kinds_note = (
        ", of every measurement of the tables, also of the validation data and "
        "the outliers"
        if any(g.id == "mapping-kind" for g in gaps)
        else ""
    )
    use = (
        "evaluate the log-likelihood of PEtab at the nominal values, the values "
        "of the fit, with the noise model of every observable"
        if role == "validation"
        else "optimize the objective of PEtab, the negative log-likelihood of the "
        f"noise model of every observable{kinds_note}, which is another fit of "
        "the same data"
    )
    if exporter.required_extension:
        lines += [
            "The extension `sbmlsim` is marked as required (`required: true` in "
            "`problem.yaml`, `required_extension=True` of `to_petab`): its "
            "settings are the objective of the fit, so a PEtab tool which does "
            "not know the extension rejects the problem instead of using the "
            "same data with another objective. Written with "
            "`required_extension=False` (the field `required_extension` of the "
            "kind `petab` of `sbmlsim.run`) the problem is one for other tools: "
            f"they read the tables, ignore the extension and {use}.",
            "",
        ]
    else:
        lines += [
            "The extension `sbmlsim` is not required (`required: false` in "
            "`problem.yaml`, `required_extension=False` of `to_petab`): a "
            f"PEtab tool reads the tables, ignores the extension and {use}. "
            "`sbmlsim` reads the extension and uses the objective of the fit.",
            "",
        ]
    return lines


def _data(
    exporter: PetabExporter,
    petab_problem: PetabProblem,
    written: Sequence[int],
    collections: dict[int, Any],
) -> list[str]:
    """Get the section of the data which is written."""
    problem = exporter.problem
    counts: dict[MappingKind, list[int]] = {}
    for k in written:
        kind = problem.mapping_kinds[k]
        counts.setdefault(kind, [0, 0])
        counts[kind][0] += 1
        counts[kind][1] += len(problem.y_references[k])
    lines = [
        "## Data",
        "",
        f"The problem has the data of {_count(len(written), 'fit mapping')} with "
        f"{_count(len(petab_problem.measurements), 'measurement')}, by what the "
        "fit does with them:",
        "",
        *_table(
            ("kind", "fit mappings", "measurements", "the fit"),
            [
                (kind.value, str(n), str(m), KIND_USE.get(kind, "-"))
                for kind in EVALUATED_KINDS
                if kind in counts
                for n, m in [counts[kind]]
            ],
        ),
        "",
    ]
    if len(counts) > 1:
        lines += [
            "PEtab has no kind of a measurement: a tool which reads the tables "
            "without the extension fits every measurement, also the ones `sbmlsim` "
            "only evaluates. The kind of a fit mapping is `kind` of its entry in "
            "`observables` of the extension `sbmlsim`.",
            "",
        ]
    per_collection: dict[int, dict[str, Any]] = {}
    experiments: dict[int, set[str]] = defaultdict(set)
    for k in written:
        c = problem.collection_indices[k]
        entry = per_collection.setdefault(c, {"mappings": 0, "measurements": 0})
        entry["mappings"] += 1
        entry["measurements"] += len(problem.y_references[k])
        experiments[c].add(exporter.experiment_ids[k])
    lines += [
        "The collections of fit mappings of the fit, each one PEtab experiment "
        "per simulation of its mappings:",
        "",
        *_table(
            (
                "collection",
                "simulation experiment",
                "kind",
                "PEtab experiments",
                "fit mappings",
                "measurements",
            ),
            [
                (
                    _code(collections[c].sid),
                    _code(collections[c].experiment_class.__name__),
                    collections[c].kind.value,
                    str(len(experiments[c])),
                    str(per_collection[c]["mappings"]),
                    str(per_collection[c]["measurements"]),
                )
                for c in sorted(per_collection)
            ],
        ),
        "",
    ]
    return lines


def _not_written(
    exporter: PetabExporter,
    skipped: dict[str, list[str]],
    related: Mapping[str, str],
) -> list[str]:
    """Get the section of the data which is not written."""
    problem = exporter.problem
    lines = ["## Data which is not written", ""]
    found = False

    excluded = [
        f"{collection.experiment_class.__name__}.{mapping}"
        for collection in problem.mapping_collections
        if collection.kind in UNUSED_KINDS
        for mapping in collection.mappings
    ]
    if excluded:
        found = True
        lines += [
            f"The fit excludes the data of {_count(len(excluded), 'fit mapping')} "
            "(kind `excluded`), which the model does not describe; it is no part "
            "of the fit and of the problem:",
            "",
            *_grouped(excluded),
            "",
        ]

    for kind in EVALUATED_KINDS:
        if kind in exporter.kinds:
            continue
        names = [
            mapping_name(problem, k)
            for k in range(len(problem.mapping_keys))
            if problem.mapping_kinds[k] is kind
        ]
        if not names:
            continue
        found = True
        where = related.get(kind.value)
        reason = {
            MappingKind.TRAINING: "the training data, which the fit optimizes",
            MappingKind.VALIDATION: "the validation data, which the fit does not "
            "optimize and evaluates",
            MappingKind.OUTLIER: "the outliers, data the fit drops as not usable "
            "and evaluates; no problem of the export holds them",
        }[kind]
        if where:
            location = f", in the {kind.value} problem `{where}`"
        elif exporter.role != "fit" and kind is not MappingKind.OUTLIER:
            indices = [
                k
                for k in range(len(problem.mapping_keys))
                if problem.mapping_kinds[k] is kind
            ]
            hits = unsupported_mappings(problem, indices)
            location = (
                f"; there is no {kind.value} problem, PEtab v2 cannot express "
                f"any of them (gaps {', '.join(_code(gap) for gap in hits)})"
            )
        else:
            location = ""
        verb = "is" if len(names) == 1 else "are"
        lines += [
            f"The {_count(len(names), 'fit mapping')} of the kind {_code(kind.value)} "
            f"{verb} {reason}{location}:",
            "",
            *_grouped(names),
            "",
        ]

    if skipped:
        found = True
        names = {name for gap_names in skipped.values() for name in gap_names}
        lines += [
            f"PEtab v2 cannot express {_count(len(names), 'fit mapping')} of the "
            'fit, which the export leaves out (`unsupported="skip"`): the problem '
            "is the fit without them, and the extension `sbmlsim` names them in "
            "`skipped`. By the gap which keeps them out:",
            "",
        ]
        for gap_id, gap_names in skipped.items():
            gap = GAPS_BY_ID[gap_id]
            lines += [
                f"### {_code(gap_id)}, {_count(len(gap_names), 'fit mapping')}",
                "",
                f"{_sentence(gap.sbmlsim)}; PEtab v2: {gap.petab}.",
                "",
                *_grouped(gap_names),
                "",
            ]

    if not found:
        lines += [
            "Every fit mapping of the fit is written: the fit excludes no data "
            "and PEtab expresses every mapping.",
            "",
        ]
    return lines


def _sentence(text: str) -> str:
    """Get a text as the start of a sentence."""
    return text[:1].upper() + text[1:] if text else text


def _gap(gap: Gap, n_skipped: int) -> list[str]:
    """Get the subsection of a gap."""
    lines = [
        f"### {gap.id}",
        "",
        f"Kind {_code(gap.kind.value)}: {KIND_MEANING[gap.kind]}",
        "",
        f"- sbmlsim: {gap.sbmlsim}",
        f"- PEtab v2: {gap.petab}",
        f"- what the export does: {gap.detail}",
    ]
    if n_skipped:
        lines.append(
            f"- this problem: {_count(n_skipped, 'fit mapping')} left out, see "
            "[Data which is not written](#data-which-is-not-written)"
        )
    lines.append("")
    return lines


def _at_changes(exporter: PetabExporter, written: Sequence[int]) -> list[str]:
    """Get the section of the measurements at the time of a change."""
    problem = exporter.problem
    settings = problem.settings_initialized
    lines = ["## Measurements at the time of a change", ""]
    if settings.data_at_change is not DataAtChangeType.BEFORE:
        return [
            *lines,
            "The fit compares a measurement at the time of a change with the "
            "state after the change (`data_at_change` "
            f"{_code(_word(settings.data_at_change))}), as PEtab does.",
            "",
        ]
    at_changes = measurements_at_changes(problem, written)
    if not at_changes:
        return [
            *lines,
            "No measurement of the problem is at the time of a change after the "
            "start of its simulation, so the fit and PEtab compare every "
            "measurement with the same state.",
            "",
        ]
    units = {
        mapping_name(problem, k): str(
            problem.models[k].uinfo[problem.xid_observable[k]]
        )
        for k in written
    }
    n_points = sum(len(times) for times in at_changes.values())
    lines += [
        "The fit compares a measurement at the time of a change after the start "
        "of its simulation with the state before the change (`data_at_change` "
        "`before`, the default of `sbmlsim`), except the first point of a "
        "curve, e.g. the cumulative amount in the urine at the end of a "
        "collection which the change empties. PEtab compares such a measurement "
        "with the state after the change. `sbmlsim` reads the setting from the "
        "extension and compares as the fit; a tool which does not read the "
        f"extension compares the {_count(n_points, 'measurement')} of "
        f"{_count(len(at_changes), 'fit mapping')} with the state after the "
        "change, which differs for every state the change sets (gap "
        "[`measurement-at-change`](#measurement-at-change)):",
        "",
        *_table(
            ("fit mapping", "times"),
            [
                (
                    _code(name),
                    f"{', '.join(_number(t) for t in times)} {units.get(name, '')}".strip(),
                )
                for name, times in at_changes.items()
            ],
        ),
        "",
    ]
    return lines
