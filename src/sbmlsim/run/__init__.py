"""The run layer: an analysis from a typed spec into an output directory.

An analysis is a `Kind` (`sbmlsim.run.kinds`): a pydantic `Spec` of its
settings and the functions which run it. `execute` runs it whole, `plan`,
`count_units`, `compute_unit` and `merge` run a splittable kind as units which
other processes or machines compute (simrun) and a merge, with the same result
as `execute`. A run writes only into its output directory, see
`sbmlsim.run.context` for its files: `run.json`, `spec.json`, `summary.json`
and the events in `progress.jsonl`.

The command line is `python -m sbmlsim.run`, see `sbmlsim.run.cli`:

    python -m sbmlsim.run <kind> <spec> --out DIR [--workers N] [--resume]
    python -m sbmlsim.run plan <kind> <spec> [-o FILE]
    python -m sbmlsim.run unit <kind> <spec> <index> --out DIR
    python -m sbmlsim.run merge <kind> <spec> --out DIR
    python -m sbmlsim.run list <project> [--json]

`import sbmlsim` does not import this package, and this package imports
nothing of the analyses until it runs one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sbmlsim import events
from sbmlsim.run.context import (
    RunContext,
    load_spec,
    prepare_out,
    read_provenance,
    record_run,
    spec_hash,
)
from sbmlsim.run.kinds import (
    ConfigurationError,
    Kind,
    PlanMismatch,
    Spec,
    SplitSpec,
    UnknownKind,
    get_kind,
    register,
)
from sbmlsim.run.units import MissingUnits, unit_blocks

__all__ = [
    "ConfigurationError",
    "Kind",
    "MissingUnits",
    "PlanMismatch",
    "RunContext",
    "Spec",
    "SplitSpec",
    "UnknownKind",
    "compute_unit",
    "count_units",
    "execute",
    "get_kind",
    "list_analyses",
    "merge",
    "plan",
    "register",
    "spec_hash",
    "unit_blocks",
]

#: a spec as the API takes it: a spec, its settings or a YAML or JSON file
type SpecLike = Spec | Mapping[str, Any] | Path | str


def _planned[S: Spec](kind: Kind[S], spec: S) -> S:
    """Plan a spec with the plan of its kind."""
    planned = kind.plan(spec)
    if not isinstance(planned, kind.spec):
        raise TypeError(
            f"The plan of the kind '{kind.name}' returned a "
            f"'{type(planned).__name__}', not a '{kind.spec.__name__}'."
        )
    return planned


def _n_units(kind: Kind[Any], spec: Spec) -> int:
    """Count the units of a planned spec of a splittable kind."""
    if not kind.splittable:
        raise ConfigurationError(f"The kind '{kind.name}' is not split into units.")
    if isinstance(spec, SplitSpec) and spec.units is None:
        raise ConfigurationError(
            f"The spec of the kind '{kind.name}' is not planned, its units are "
            "not set: plan it first."
        )
    if kind.count_units is not None:
        return kind.count_units(spec)
    if not isinstance(spec, SplitSpec) or spec.units is None:
        raise ConfigurationError(f"The kind '{kind.name}' counts no units.")
    return spec.units


def _check_workers(workers: int | None) -> None:
    """Refuse a number of workers below 1."""
    if workers is not None and workers < 1:
        raise ConfigurationError(f"A run needs at least one worker, not {workers}.")


def execute(
    kind: str,
    spec: SpecLike,
    out: Path,
    *,
    workers: int | None = None,
    resume: bool = False,
    provenance: Path | None = None,
    inputs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Run an analysis whole into an output directory.

    The spec is validated, its inputs set and it is planned, unless the run
    resumes, which runs the planned spec of `out/spec.json` as it is. The run
    writes `spec.json`, `run.json`, `summary.json` and its events into `out`,
    see `sbmlsim.run.context`.

    Args:
        kind: the name of the kind or `module:attribute` of a kind.
        spec: the spec, its settings or a YAML or JSON file of them, whose
            relative paths are relative to the file.
        out: the output directory, created; it must be empty without `resume`.
        workers: the number of processes, `None` for the kind to decide.
        resume: continue the run of the same spec in `out`.
        provenance: the file of the provenance of the sources, the one of
            `SIMRUN_PROVENANCE` without.
        inputs: the paths of the input fields of the spec by their name.

    Returns:
        The summary of the run.

    Raises:
        ConfigurationError: for an unknown kind, a spec which is not read, an
            output which is not empty without `resume`, a resume of another
            spec and a provenance which is not read.
        pydantic.ValidationError: for invalid settings.
    """
    the_kind = get_kind(kind)
    given = load_spec(the_kind, spec, inputs)
    given_hash = spec_hash(given)
    _check_workers(workers)
    sources = read_provenance(provenance)
    out = Path(out)
    stored = prepare_out(the_kind, given_hash, out, resume)
    planned = stored if stored is not None else _planned(the_kind, given)
    ctx = RunContext(out=out, workers=workers, resume=resume, provenance=sources)
    return record_run(
        the_kind, planned, given_hash, ctx, lambda: the_kind.run(planned, ctx)
    )


def plan(kind: str, spec: SpecLike) -> Spec:
    """Plan a spec: the settings every unit of a split run computes alike.

    Args:
        kind: the name of the kind or `module:attribute` of a kind.
        spec: the spec, its settings or a YAML or JSON file of them.

    Returns:
        The planned spec, e.g. with its seed and its `units`.
    """
    the_kind = get_kind(kind)
    return _planned(the_kind, load_spec(the_kind, spec))


def count_units(kind: str, spec: SpecLike) -> int:
    """Count the units of a planned spec.

    Args:
        kind: the name of the kind or `module:attribute` of a kind.
        spec: the planned spec.

    Returns:
        The number of units.

    Raises:
        ConfigurationError: for a kind which is not split and a spec which is
            not planned.
    """
    the_kind = get_kind(kind)
    return _n_units(the_kind, load_spec(the_kind, spec))


def compute_unit(
    kind: str,
    spec: SpecLike,
    index: int,
    out: Path,
    *,
    inputs: Mapping[str, Path] | None = None,
) -> None:
    """Compute one unit of a planned spec into an output directory, serially.

    The units of a run compute into the same output, each of its own files,
    and a unit computed again writes the same files. A unit writes its events
    into `progress/unit-<index>.jsonl` and no `run.json`, which the merge
    writes.

    Args:
        kind: the name of the kind or `module:attribute` of a kind.
        spec: the planned spec.
        index: the index of the unit, in `range(count_units(kind, spec))`.
        out: the output directory, created.
        inputs: the paths of the input fields of the spec by their name.

    Raises:
        ConfigurationError: for a kind which is not split, a spec which is not
            planned and an index outside of the units.
    """
    the_kind = get_kind(kind)
    given = load_spec(the_kind, spec, inputs)
    n_units = _n_units(the_kind, given)
    compute = the_kind.compute_unit
    assert compute is not None
    if not 0 <= index < n_units:
        raise ConfigurationError(
            f"The unit {index} is outside of the units range(0, {n_units})."
        )
    sources = read_provenance(None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(out=out, workers=1, resume=True, provenance=sources, unit=index)
    progress = ctx.progress_path
    progress.parent.mkdir(parents=True, exist_ok=True)
    with events.recording(progress):
        compute(given, index, ctx)


def merge(
    kind: str,
    spec: SpecLike,
    out: Path,
    *,
    inputs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Merge the units of a planned spec in an output directory.

    Writes `spec.json`, `run.json` and `summary.json` as `execute` does.

    Args:
        kind: the name of the kind or `module:attribute` of a kind.
        spec: the planned spec.
        out: the output directory the units were computed into.
        inputs: the paths of the input fields of the spec by their name.

    Returns:
        The summary of the run.

    Raises:
        ConfigurationError: for a kind which is not split and a spec which is
            not planned.
        MissingUnits: from the merge of a kind, naming the missing units.
    """
    the_kind = get_kind(kind)
    given = load_spec(the_kind, spec, inputs)
    _n_units(the_kind, given)
    merge_units = the_kind.merge
    assert merge_units is not None
    sources = read_provenance(None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(out=out, workers=None, resume=True, provenance=sources)
    return record_run(
        the_kind, given, spec_hash(given), ctx, lambda: merge_units(given, ctx)
    )


def list_analyses(project: str) -> list[dict[str, str]]:
    """List the analyses of a project, see `sbmlsim.project.Project.analyses`.

    Args:
        project: the project, `module:attribute`.

    Returns:
        One entry `{"kind", "name", "description"}` per analysis.

    Raises:
        ConfigurationError: for a project which is not loaded.
    """
    from sbmlsim.project import load_project

    try:
        the_project = load_project(project)
    except (ImportError, ValueError) as err:
        raise ConfigurationError(
            f"The project '{project}' is not loaded: {err}"
        ) from err
    return the_project.analyses()
