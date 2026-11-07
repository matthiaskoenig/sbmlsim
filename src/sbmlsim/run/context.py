"""The context of a run: its output directory and the files which describe it.

A run writes only into its output directory `out`:

- `run.json` (`RUN_FILE`), the record of the run: the kind, the hash of the
  given spec (`spec_hash`), the versions of the packages, python, the host,
  the provenance of the sources of a snapshot (`sources`, the file simrun
  writes, see `sbmlsim.fit.provenance.snapshot_provenance`), the git revision
  of the project without sources (`git`), the variables of the task which
  simrun sets in a pod (`ENVIRONMENT`, never a credential), the number of workers, the times and the status (`running`,
  `done` or `failed` with the `error`). It is written when the run starts and
  again when it ends, so an interrupted run leaves the record a resume checks.
- `spec.json` (`SPEC_FILE`), the planned spec, which a resume runs as it is.
- `summary.json` (`SUMMARY_FILE`), the summary the kind returns.
- `progress.jsonl` (`PROGRESS_FILE`), the events of the run (`sbmlsim.events`),
  of a unit `progress/unit-<index>.jsonl`, so the units of a split run append
  to files of their own.

A JSON file is written atomically with the numbers JSON has no value for
(NaN, infinity) as `null`. The functions of this module are the parts of the
API of `sbmlsim.run`, which composes them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import socket
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from sbmlsim import events
from sbmlsim.atomic import write_text_atomic
from sbmlsim.cgroup import available_cpus
from sbmlsim.run.kinds import ConfigurationError, Kind, OutInUse, PlanMismatch, Spec

__all__ = [
    "ENVIRONMENT",
    "PROGRESS_FILE",
    "RUN_FILE",
    "SPEC_FILE",
    "SUMMARY_FILE",
    "RunContext",
    "spec_hash",
]

logger = logging.getLogger(__name__)

#: the record of a run
RUN_FILE = "run.json"
#: the planned spec of a run
SPEC_FILE = "spec.json"
#: the summary of a run
SUMMARY_FILE = "summary.json"
#: the events of a run
PROGRESS_FILE = "progress.jsonl"
#: the directory of the events of the units of a split run
PROGRESS_DIR = "progress"
#: the variables of the environment a run records: the ones simrun sets for the
#: task of a pod (`simrun.podspec.env_values`, `simrun.worker`), never its
#: credentials or the settings of its client, and the optimization level of
#: the JIT of roadrunner (`model_roadrunner.JIT_OPTIMIZATION_VARIABLE`), which
#: changes the results in the last bits and is no part of the spec
ENVIRONMENT = (
    "SBMLSIM_JIT_OPTIMIZATION",
    "SIMRUN_ATTEMPT",
    "SIMRUN_EDITABLE",
    "SIMRUN_INPUTS",
    "SIMRUN_MEMORY_BASE_GB",
    "SIMRUN_MEMORY_PER_WORKER_GB",
    "SIMRUN_NODE",
    "SIMRUN_PROVENANCE",
    "SIMRUN_ROLE",
    "SIMRUN_SCRATCH",
    "SIMRUN_STAGE",
    "SIMRUN_STALE_S",
    "SIMRUN_TASK_DIR",
    "SIMRUN_TASK_TYPE",
    "SIMRUN_WORKER",
)
#: a variable whose name contains one of them is never recorded
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "KEY")


def recorded_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The variables of an environment which `run.json` records.

    Args:
        environ: the environment, e.g. `os.environ`.

    Returns:
        The variables of `ENVIRONMENT` which are set, by name; a name which
        contains one of `SECRET_MARKERS` is never recorded.
    """
    return {
        name: environ[name]
        for name in sorted(ENVIRONMENT)
        if name in environ and not any(m in name.upper() for m in SECRET_MARKERS)
    }


def _jsonable(value: Any) -> Any:
    """Convert a value to JSON values, a number JSON has no value for to `None`."""
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)  # numpy arrays and scalars
    if callable(tolist):
        return _jsonable(tolist())
    return value


def to_json(data: Any) -> str:
    """Write data as indented JSON with sorted keys, see `_jsonable`.

    Args:
        data: JSON values, numpy values, paths and pydantic models.

    Returns:
        The JSON text with a final newline.
    """
    return json.dumps(_jsonable(data), indent=2, sort_keys=True, default=str) + "\n"


@dataclass
class RunContext:
    """Where a run writes and what it may assume.

    Attributes:
        out: the output directory, which exists.
        workers: the number of processes the run may use: the given number,
            else every CPU of the quota (`available_cpus`) for a kind which
            uses several cores and 1 for one which does not, 1 for a unit.
        resume: whether the run continues an earlier one in `out`, so it
            keeps what exists, e.g. the files of the finished repeats.
        provenance: the provenance of the sources of the snapshot the run runs
            on, `None` outside of simrun.
        unit: the index of the unit a split run computes, `None` for a whole
            run or a merge.
    """

    out: Path
    workers: int | None
    resume: bool
    provenance: list[dict[str, Any]] | None
    unit: int | None = None

    def write_json(self, name: str, data: Any) -> Path:
        """Write a JSON file into the output atomically, see `to_json`.

        Args:
            name: the path of the file relative to `out`; its directories are
                created.
            data: the content.

        Returns:
            The path of the file.
        """
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, to_json(data))
        return path

    @property
    def progress_path(self) -> Path:
        """The file of the events: `progress.jsonl`, of a unit `progress/unit-<index>.jsonl`."""
        if self.unit is None:
            return self.out / PROGRESS_FILE
        return self.out / PROGRESS_DIR / f"unit-{self.unit}.jsonl"


def spec_hash(spec: Spec, exclude: Collection[str] = ()) -> str:
    """Get the hash of a spec: the sha256 of its canonical JSON, sorted keys.

    Args:
        spec: the spec.
        exclude: the fields left out, the inputs of its kind (`Kind.inputs`),
            which are identified by their own provenance, so a run whose
            inputs are staged at another path is the same run.

    Returns:
        The hexadecimal digest.
    """
    canonical = json.dumps(
        spec.model_dump(mode="json", exclude=set(exclude)),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated[S: Spec](kind: Kind[S], settings: Mapping[str, Any] | str) -> S:
    """Validate the settings of a spec, a mapping or JSON.

    Raises:
        ConfigurationError: for invalid settings, with the errors of pydantic.
    """
    try:
        if isinstance(settings, str):
            return kind.spec.model_validate_json(settings)
        return kind.spec.model_validate(dict(settings))
    except ValidationError as err:
        raise ConfigurationError(
            f"The spec of the kind '{kind.name}' is invalid: {err}"
        ) from err


def _read_spec_file(path: Path) -> dict[str, Any]:
    """Read the settings of a spec file, JSON for `.json` and YAML otherwise."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise ConfigurationError(f"The spec '{path}' is not read: {err}") from err
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise ConfigurationError(f"The spec '{path}' is no JSON: {err}") from err
    else:
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as err:
            raise ConfigurationError(f"The spec '{path}' is no YAML: {err}") from err
    if not isinstance(data, dict):
        raise ConfigurationError(f"The spec '{path}' is no mapping of settings.")
    return data


def _relative_to(spec: Spec, base: Path) -> Spec:
    """Resolve the relative paths of a spec against a directory.

    Only the fields which are a `Path` or a list or tuple of them are resolved,
    not the paths inside of other values, e.g. of a nested model, and a string
    of a field of `Spec.path_strings`, e.g. the start of a fit, which may be a
    path or a name (`Spec.is_name`), unless it is a name; whether a file
    exists does not matter, so the spec and its hash do not depend on it.
    """

    def resolved(value: Any) -> Any:
        if isinstance(value, Path) and not value.is_absolute():
            return base / value
        return value

    updates: dict[str, Any] = {}
    for name in type(spec).model_fields:
        value = getattr(spec, name)
        if isinstance(value, Path):
            updates[name] = resolved(value)
        elif isinstance(value, list | tuple):
            updates[name] = type(value)(resolved(v) for v in value)
        elif (
            name in type(spec).path_strings
            and isinstance(value, str)
            and not Path(value).is_absolute()
            and not type(spec).is_name(name, value)
        ):
            updates[name] = str(base / value)
    return spec.model_copy(update=updates)


def load_spec[S: Spec](
    kind: Kind[S],
    spec: Spec | Mapping[str, Any] | Path | str,
    inputs: Mapping[str, Path] | None = None,
) -> S:
    """Validate the spec of a kind and set its inputs.

    Args:
        kind: the kind.
        spec: the spec, its settings or a YAML or JSON file of them, whose
            relative paths are relative to the file (the fields which are a
            `Path` or a list or tuple of them).
        inputs: the paths of input fields by their name, see `Kind.inputs`,
            resolved to absolute paths.

    Returns:
        The spec.

    Raises:
        ConfigurationError: for a spec of another kind, a file which is not
            read or no mapping, invalid settings and an input the kind does
            not have.
    """
    if isinstance(spec, Spec):
        if not isinstance(spec, kind.spec):
            raise ConfigurationError(
                f"The spec of the kind '{kind.name}' is a '{kind.spec.__name__}', "
                f"not a '{type(spec).__name__}'."
            )
        loaded: S = spec
    elif isinstance(spec, Mapping):
        loaded = _validated(kind, spec)
    else:
        path = Path(spec)
        validated = _validated(kind, _read_spec_file(path))
        loaded = _validated(
            kind, _relative_to(validated, path.parent.resolve()).model_dump()
        )
    if inputs:
        unknown = sorted(set(inputs) - set(kind.inputs))
        if unknown:
            raise ConfigurationError(
                f"The kind '{kind.name}' has no input {unknown}, its inputs are "
                f"{list(kind.inputs)}."
            )
        settings = loaded.model_dump()
        settings.update({port: Path(path).resolve() for port, path in inputs.items()})
        loaded = _validated(kind, settings)
    return loaded


def with_inputs[S: Spec](kind: Kind[S], spec: S, source: Spec) -> S:
    """Get a spec with the input fields of another spec of its kind.

    Args:
        kind: the kind.
        spec: the spec, e.g. the planned spec of a run to resume.
        source: the spec whose inputs are taken, e.g. the given spec.

    Returns:
        The spec with the inputs of `source`.
    """
    if not kind.inputs:
        return spec
    settings = spec.model_dump()
    settings.update({name: getattr(source, name) for name in kind.inputs})
    return _validated(kind, settings)


def check_plan(kind: Kind[Any], spec: Spec, out: Path) -> None:
    """Fix the planned spec of a split run in its output, or check it.

    The first unit writes `spec.json`; a later unit or the merge of another
    plan, e.g. of a design drawn again, would mix its results with the ones of
    the first plan and is refused. The inputs are not compared, see
    `spec_hash`.

    Args:
        kind: the kind of the run.
        spec: the planned spec.
        out: the output directory, which exists.

    Raises:
        PlanMismatch: if `out` holds the units of another planned spec.
        ConfigurationError: if its `spec.json` is no spec of the kind.
    """
    path = out / SPEC_FILE
    if not path.exists():
        write_text_atomic(path, to_json(spec))
        return
    stored = _validated(kind, path.read_text(encoding="utf-8"))
    if spec_hash(stored, kind.inputs) != spec_hash(spec, kind.inputs):
        raise PlanMismatch(
            f"The output '{out}' holds the units of another planned spec of the "
            f"kind '{kind.name}'; compute the units of a plan into an output of "
            "their own."
        )


def read_provenance(path: Path | None) -> list[dict[str, Any]] | None:
    """Read the provenance of the sources of a snapshot.

    Args:
        path: the file simrun writes, the one of `SIMRUN_PROVENANCE` without.

    Returns:
        The sources, `None` without a file.

    Raises:
        ConfigurationError: for a file which is not read or no provenance.
    """
    from sbmlsim.fit.provenance import snapshot_provenance

    try:
        return snapshot_provenance(path)
    except (OSError, ValueError) as err:
        raise ConfigurationError(
            f"The provenance of the sources is not read: {err}"
        ) from err


def check_out_of(
    kind: Kind[Any], given_hash: str, out: Path, resume: bool
) -> Spec | None:
    """Check that a run may run into an output, without creating or changing it.

    Args:
        kind: the kind of the run.
        given_hash: the hash of the given spec, see `spec_hash`.
        out: the output directory, which need not exist.
        resume: whether the run resumes a run in `out`.

    Returns:
        The planned spec of the run to resume, `None` for a new run.

    Raises:
        OutInUse: a `ConfigurationError` for an output which is no
            directory, which is not empty without `resume`, and with `resume`
            which has no record of a run, an unreadable one or the record of
            a run of another kind or spec.
    """
    if out.exists() and not out.is_dir():
        raise OutInUse(
            f"The output '{out}' is no directory.", out=out, reason="not_a_directory"
        )
    if not out.exists() or not any(out.iterdir()):
        return None
    if not resume:
        raise OutInUse(
            f"The output '{out}' is not empty: resume the run in it (--resume) "
            "or give another output.",
            out=out,
            reason="not_empty",
        )
    run_file = out / RUN_FILE
    try:
        record = json.loads(run_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OutInUse(
            f"The output '{out}' is not empty and has no {RUN_FILE} of a run to resume.",
            out=out,
            reason="no_record",
        ) from None
    except (OSError, json.JSONDecodeError) as err:
        raise OutInUse(
            f"The record '{run_file}' is not read: {err}", out=out, reason="unreadable"
        ) from err
    if not isinstance(record, dict) or (
        record.get("kind"),
        record.get("spec_hash"),
    ) != (kind.name, given_hash):
        raise OutInUse(
            f"The output '{out}' holds the run of another spec or kind: resume "
            "it with the spec file the run was started with, not its planned "
            f"spec '{SPEC_FILE}', or give another output.",
            out=out,
            reason="other_run",
        )
    spec_file = out / SPEC_FILE
    if not spec_file.exists():
        return None
    return _validated(kind, spec_file.read_text(encoding="utf-8"))


def prepare_out(
    kind: Kind[Any], given_hash: str, out: Path, resume: bool
) -> Spec | None:
    """Create the output of a run and check that it may run into it.

    Args:
        kind: the kind of the run.
        given_hash: the hash of the given spec, see `spec_hash`.
        out: the output directory.
        resume: whether the run resumes a run in `out`.

    Returns:
        The planned spec of the run to resume, `None` for a new run.

    Raises:
        OutInUse: for an output the run may not run into, see `check_out_of`.
    """
    stored = check_out_of(kind, given_hash, out, resume)
    out.mkdir(parents=True, exist_ok=True)
    return stored


def resolved_workers(kind: Kind[Any], workers: int | None) -> int:
    """Get the number of processes a run uses at most.

    Args:
        kind: the kind of the run.
        workers: the given number, `None` for the CPUs of the quota of a kind
            which uses several cores and 1 for one which does not.

    Returns:
        The number of processes.
    """
    if workers is not None:
        return workers
    return available_cpus() if kind.uses_cores else 1


def _versions(spec: Spec) -> dict[str, str]:
    """Get the versions of the packages and of the distribution of the project.

    The distribution of the project is the one of the top level module of
    `spec.project`, found by its files or, for an editable install, whose
    files are no part of the distribution, by its name.
    """
    from sbmlsim.fit.provenance import (
        collect_versions,
        installed_versions,
        module_distributions,
    )

    versions = collect_versions()
    project = getattr(spec, "project", None)
    if isinstance(project, str) and project:
        versions |= {
            name: version
            for name, version in installed_versions(
                module_distributions(project.partition(":")[0])
            ).items()
            if name not in versions
        }
    return versions


def _git(spec: Spec, sources: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Get the git revision of the root of the project of a spec without sources."""
    project = getattr(spec, "project", None)
    if sources is not None or not isinstance(project, str) or not project:
        return None
    from sbmlsim.fit.provenance import git_info
    from sbmlsim.project import load_project

    try:
        root = load_project(project).root
    except (ImportError, ValueError) as err:
        logger.warning("The git revision of '%s' is not recorded: %s", project, err)
        return None
    info = git_info(root)
    return info.to_dict() if info is not None else None


def _now() -> str:
    """Get the time in UTC as ISO 8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def record_run(
    kind: Kind[Any],
    spec: Spec,
    given_hash: str,
    ctx: RunContext,
    body: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run the body of a run and record it in its output.

    Writes `run.json` (running) and `spec.json`, runs the body while the
    events of the process are recorded into `ctx.progress_path`, writes
    `summary.json` and `run.json` (done, or failed with the error, which is
    raised again).

    Args:
        kind: the kind of the run.
        spec: the planned spec.
        given_hash: the hash of the given spec, see `spec_hash`.
        ctx: the context of the run.
        body: runs the analysis and returns its summary.

    Returns:
        The summary.
    """
    record: dict[str, Any] = {
        "kind": kind.name,
        "spec_hash": given_hash,
        "versions": _versions(spec),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "sources": ctx.provenance,
        "git": _git(spec, ctx.provenance),
        "environment": recorded_environment(os.environ),
        "workers": resolved_workers(kind, ctx.workers),
        "started": _now(),
        "finished": None,
        "status": "running",
    }
    ctx.write_json(RUN_FILE, record)
    ctx.write_json(SPEC_FILE, spec)
    progress = ctx.progress_path
    progress.parent.mkdir(parents=True, exist_ok=True)
    try:
        with events.recording(progress):
            summary = body()
        ctx.write_json(SUMMARY_FILE, summary)
    except BaseException as err:
        record.update(
            finished=_now(), status="failed", error=f"{type(err).__name__}: {err}"
        )
        ctx.write_json(RUN_FILE, record)
        raise
    record.update(finished=_now(), status="done")
    ctx.write_json(RUN_FILE, record)
    return summary
