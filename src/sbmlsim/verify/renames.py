"""The expected renames and differences of a comparison of two dumps."""

from __future__ import annotations

import ast
import fnmatch
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from sbmlsim.verify.dump import Dump

__all__ = [
    "EXPECTED_PARTS",
    "ID_FIELDS",
    "RENAME_PARTS",
    "Expected",
    "MetadataRenames",
    "Renames",
    "apply_renames",
    "drop_figure_fields",
    "load_renames",
    "rename_ids",
    "rename_mapping",
]

#: the fields of the specification of a figure with the id of a task or a dataset
ID_FIELDS: tuple[str, ...] = ("task_id", "dset_id")

#: the parts of a dump whose differences can be expected, and their keys
EXPECTED_PARTS: tuple[str, ...] = (
    "results",
    "datasets",
    "mappings",
    "figures",
    "ranges",
    "builds",
)


class MetadataRenames(TypedDict, total=False):
    """The expected changes of the metadata of the fit mappings."""

    #: old field name -> new field name
    fields: dict[str, str]
    #: field (new name) -> old member name -> new member name
    members: dict[str, dict[str, str]]
    #: field -> member name of a field the old metadata did not have
    added: dict[str, str]


class Renames(TypedDict, total=False):
    """The expected renames and differences between two dumps."""

    metadata: MetadataRenames
    #: old text -> new text, in every key and string of the old dump
    strings: dict[str, str]
    #: old id -> new id, of a task, a dataset, a figure or a fit mapping
    ids: dict[str, str]
    #: every id of the old dump becomes its SId (`sbmlsim.experiment.to_sid`)
    sids: bool
    #: the keys of the specification of a figure which are left out of both
    #: dumps, at the level of the figure: a field of a figure which `compare`
    #: cannot map
    figure_fields: list[str]
    #: part of the dump -> patterns of the keys whose differences are expected
    #: (`*` and `?` are wildcards, a bracket is literal, e.g. `E/task_po/[A]`):
    #: `experiment/task/variable` (results; `experiment/task/time` also names
    #: the output times which only the new dump has),
    #: `experiment/dataset/column` (datasets), `experiment/figure` (figures),
    #: `experiment/mapping` (mappings, ranges), `experiment/task` (builds)
    expected: dict[str, list[str]]


#: the parts of a renames file
RENAME_PARTS: tuple[str, ...] = (
    "metadata",
    "strings",
    "ids",
    "sids",
    "figure_fields",
    "expected",
)


def load_renames(path: str | Path) -> Renames:
    """Load the expected renames from a json file.

    Raises:
        ValueError: for a part which is not known.
    """
    renames = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(renames, dict):
        raise ValueError(f"the renames {path} must be a json object")
    unknown = sorted(set(renames) - set(RENAME_PARTS)) + sorted(
        set(renames.get("metadata", {})) - {"fields", "members", "added"}
    )
    if unknown:
        raise ValueError(f"unknown parts of the renames {path}: {unknown}")
    _check_renames(path, renames)
    return cast("Renames", renames)


def _is_strings(value: object) -> bool:
    """Check for a json object of strings."""
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def _is_texts(value: object) -> bool:
    """Check for a json list of strings."""
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _check_renames(path: str | Path, renames: dict[str, object]) -> None:
    """Check the types of the parts of the renames.

    Raises:
        ValueError: naming the part which has the wrong type.
    """
    if "strings" in renames and not _is_strings(renames["strings"]):
        raise ValueError(
            f"the part strings of the renames {path} must map text to text"
        )
    if "ids" in renames and not _is_strings(renames["ids"]):
        raise ValueError(f"the part ids of the renames {path} must map text to text")
    if "sids" in renames and not isinstance(renames["sids"], bool):
        raise ValueError(f"the part sids of the renames {path} must be true or false")
    if "figure_fields" in renames and not _is_texts(renames["figure_fields"]):
        raise ValueError(
            f"the part figure_fields of the renames {path} must be a list of texts"
        )
    if "expected" in renames:
        expected = renames["expected"]
        if (
            not isinstance(expected, dict)
            or set(expected) - set(EXPECTED_PARTS)
            or not all(_is_texts(v) for v in expected.values())
        ):
            raise ValueError(
                f"the part expected of the renames {path} must map parts of "
                f"{list(EXPECTED_PARTS)} to lists of patterns"
            )
    if "metadata" not in renames:
        return
    metadata = renames["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"the part metadata of the renames {path} must be an object")
    for part in ("fields", "added"):
        if part in metadata and not _is_strings(metadata[part]):
            raise ValueError(
                f"the part {part} of the metadata of the renames {path} "
                "must map text to text"
            )
    members = metadata.get("members", {})
    if not isinstance(members, dict) or not all(
        _is_strings(v) for v in members.values()
    ):
        raise ValueError(
            f"the part members of the metadata of the renames {path} must map "
            "a field to an object of text to text"
        )


def rename_mapping(description: str, renames: MetadataRenames) -> str:
    """Apply the expected renames to the metadata of a mapping description."""
    head, metadata = description.rsplit("|", 1)
    fields = renames.get("fields", {})
    items = {fields.get(k, k): v for k, v in ast.literal_eval(metadata)}
    for name, members in renames.get("members", {}).items():
        if name in items:
            items[name] = members.get(items[name], items[name])
    for name, member in renames.get("added", {}).items():
        items.setdefault(name, member)
    return f"{head}|{sorted(items.items())}"


def _rename_strings(value: object, strings: Mapping[str, str]) -> object:
    """Replace texts in every key and string of a value, arrays untouched.

    Raises:
        ValueError: if two keys of a dict have the same new key.
    """
    if isinstance(value, str):
        for old, new in strings.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        renamed: dict[object, object] = {}
        for k, v in value.items():
            key = _rename_strings(k, strings)
            if key in renamed:
                raise ValueError(
                    f"Two keys of the old dump have the same new key {key!r} "
                    "after the renames of strings."
                )
            renamed[key] = _rename_strings(v, strings)
        return renamed
    if isinstance(value, list):
        return [_rename_strings(v, strings) for v in value]
    if isinstance(value, tuple):
        return tuple(_rename_strings(v, strings) for v in value)
    return value


def _rename_description(description: str, rename: Callable[[str], str]) -> str:
    """Rename the key, the datasets and the tasks of a mapping description.

    Raises:
        ValueError: if the head of the description is not its 16 fields, i.e. a
            key has a '|'.
    """
    head, metadata = description.rsplit("|", 1)
    fields = head.split("|")
    if len(fields) != 16:
        raise ValueError(
            f"The mapping description cannot be renamed, a key has a '|': {description}"
        )
    for k in (0, 1, 2, 8, 9):
        fields[k] = rename(fields[k])
    return "|".join([*fields, metadata])


def _rename_figure(spec: str, rename: Callable[[str], str]) -> str:
    """Rename the id of a figure and the ids of its tasks and datasets in its specification."""

    def walk(value: object, top: bool) -> object:
        if isinstance(value, dict):
            return {
                k: rename(v)
                if isinstance(v, str) and (k in ID_FIELDS or (top and k == "sid"))
                else walk(v, False)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [walk(v, False) for v in value]
        return value

    return json.dumps(walk(json.loads(spec), True), sort_keys=True)


def _unique[K, V](renamed: dict[K, V], before: Mapping[K, V], part: str) -> dict[K, V]:
    """Check that no two keys of a part got the same new key.

    Raises:
        ValueError: if two keys have the same new id.
    """
    if len(renamed) != len(before):
        raise ValueError(f"Two ids of the {part} of the old dump have the same new id.")
    return renamed


def rename_ids(dump: Dump, rename: Callable[[str], str]) -> Dump:
    """Rename the ids of the tasks, datasets, figures and fit mappings of a dump.

    Returns:
        A new dump; the given one is not changed.

    Raises:
        ValueError: if two ids of a part have the same new id, or a mapping
            description cannot be renamed.
    """
    result = cast("Dump", dict(dump))
    result["results"] = _unique(
        {(s, rename(t), v): x for (s, t, v), x in dump["results"].items()},
        dump["results"],
        "results",
    )
    result["datasets"] = _unique(
        {(s, rename(d), c): x for (s, d, c), x in dump["datasets"].items()},
        dump["datasets"],
        "datasets",
    )
    result["mappings"] = {
        sid: [_rename_description(d, rename) for d in descriptions]
        for sid, descriptions in dump["mappings"].items()
    }
    result["ranges"] = {
        sid: _unique({rename(k): v for k, v in items.items()}, items, "ranges")
        for sid, items in dump["ranges"].items()
    }
    result["figures"] = {
        sid: _unique(
            {rename(f): _rename_figure(spec, rename) for f, spec in figures.items()},
            figures,
            "figures",
        )
        for sid, figures in dump["figures"].items()
    }
    if "builds" in dump:
        result["builds"] = {
            sid: _unique({rename(t): v for t, v in tasks.items()}, tasks, "builds")
            for sid, tasks in dump["builds"].items()
        }
    return result


def drop_figure_fields(dump: Dump, fields: Collection[str]) -> Dump:
    """Leave out keys of the specifications of the figures, of the figure only.

    A key is left out at the level of the figure, not inside its plots, curves
    or axes: a plot has a `width` and a `height` of its own.

    Returns:
        A new dump; the given one is not changed.
    """
    names = set(fields)

    def without(spec: str) -> str:
        figure = json.loads(spec)
        if isinstance(figure, dict):
            figure = {k: v for k, v in figure.items() if k not in names}
        return json.dumps(figure, sort_keys=True)

    result = cast("Dump", dict(dump))
    result["figures"] = {
        sid: {f: without(spec) for f, spec in figures.items()}
        for sid, figures in dump["figures"].items()
    }
    return result


def _renamer(renames: Renames) -> Callable[[str], str]:
    """Get the rename of an id: `ids`, then its SId with `sids`."""
    from sbmlsim.experiment import to_sid

    ids = renames.get("ids", {})
    sids = renames.get("sids", False)

    def rename(text: str) -> str:
        text = ids.get(text, text)
        return to_sid(text) if sids else text

    return rename


def apply_renames(dump: Dump, renames: Renames) -> Dump:
    """Apply the expected renames to a dump, the old one of a comparison.

    The strings first, then the ids (`ids`, `sids`), then the metadata.

    Returns:
        A new dump; the given one is not changed.

    Raises:
        ValueError: if two keys or ids get the same new key or id, or a mapping
            description cannot be renamed.
    """
    result = cast("Dump", dict(dump))
    if strings := renames.get("strings"):
        # the texts of the parts, not the names of the parts
        result = cast(
            "Dump", {part: _rename_strings(v, strings) for part, v in result.items()}
        )
    if renames.get("ids") or renames.get("sids"):
        result = rename_ids(result, _renamer(renames))
    if metadata := renames.get("metadata"):
        result["mappings"] = {
            sid: [rename_mapping(d, metadata) for d in descriptions]
            for sid, descriptions in result["mappings"].items()
        }
    # a dump stores the mappings sorted, and a rename can change the order
    result["mappings"] = {
        sid: sorted(descriptions) for sid, descriptions in result["mappings"].items()
    }
    return result


class Expected:
    """The expected differences of a renames file and which of them occurred."""

    def __init__(self, patterns: Mapping[str, Sequence[str]]) -> None:
        """Create the expected differences.

        Args:
            patterns: the patterns of the keys per part of the dump.
        """
        self.patterns = {part: list(p) for part, p in patterns.items()}
        self.used: set[tuple[str, str]] = set()

    def match(self, part: str, key: str | tuple[str, ...]) -> bool:
        """Check whether the difference of a key is expected, recording the patterns.

        `*` and `?` of a pattern are the wildcards of `fnmatch`; a bracket is
        literal, so that `E/task/[A]` is the variable `[A]`.
        """
        text = "/".join(key) if isinstance(key, tuple) else key
        found = False
        for pattern in self.patterns.get(part, []):
            if fnmatch.fnmatchcase(text, pattern.replace("[", "[[]")):
                self.used.add((part, pattern))
                found = True
        return found

    def unused(self) -> list[tuple[str, str]]:
        """Get the patterns which matched no difference."""
        return [
            (part, pattern)
            for part, patterns in self.patterns.items()
            for pattern in patterns
            if (part, pattern) not in self.used
        ]
