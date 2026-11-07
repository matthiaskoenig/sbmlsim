"""Compare two dumps, `verify compare`, and list the data outside of the simulation."""

from __future__ import annotations

import argparse
import json
from typing import Literal

import numpy as np

from sbmlsim.verify.common import UsageError, report
from sbmlsim.verify.dump import Dump, load
from sbmlsim.verify.match import (
    equal,
    point_matches,
    relative_differences,
    task_columns,
)
from sbmlsim.verify.renames import (
    Expected,
    Renames,
    apply_renames,
    drop_figure_fields,
    load_renames,
)

__all__ = [
    "compare",
    "diff_lines",
    "json_paths",
    "ranges",
]


def _flat(
    dump: Dump, part: Literal["datasets", "mappings", "figures", "ranges", "builds"]
) -> dict[str | tuple[str, ...], object]:
    """Get a part of a dump by the keys a comparison reports.

    A fit mapping is keyed by its experiment and its key, the first field of its
    description.

    Raises:
        ValueError: if two fit mappings of an experiment have one key (a key with
            a '|').
    """
    if part == "mappings":
        mappings: dict[str | tuple[str, ...], object] = {}
        for sid, descriptions in dump["mappings"].items():
            for description in descriptions:
                key = (sid, description.split("|", 1)[0])
                if key in mappings:
                    raise ValueError(
                        f"Two fit mappings of {sid} have the key {key[1]}, a key "
                        "has a '|'."
                    )
                mappings[key] = description
        return mappings
    if part in ("figures", "ranges", "builds"):
        return {
            (sid, key): value
            for sid, items in dump.get(part, {}).items()
            for key, value in items.items()
        }
    return dict(dump[part].items())


def _listed(times: np.ndarray, count: int = 5) -> str:
    """List the first times of a vector."""
    listed = ", ".join(f"{t:g}" for t in times[:count])
    return listed + (", ..." if times.size > count else "")


def _detail(part: str, a: object, b: object) -> list[str]:
    """Describe the difference of a changed key."""
    if part == "mappings":
        return diff_lines([a] if a else [], [b] if b else [])
    if part == "figures":
        return json_paths(_json(a), _json(b))[:5]
    return [f"{a!r} -> {b!r}"]


def compare(args: argparse.Namespace) -> int:
    """Compare two dumps; the exit code is 1 if they differ beyond the threshold.

    An expected difference (`expected` of the renames) is printed and not
    counted; an expected difference which does not occur is counted. A row at a
    time which only one dump has is compared with the other dump interpolated
    at its time when the time lies between two rows of the other dump (the
    steps of the integrator of another run, `matched_rows`), and an array
    whose resolution there (the spread of the interpolants,
    `ArrayDifference.resolution`) is coarser than the threshold is unresolved
    on this grid, which counts as a difference unless `expected` names it; a
    row outside of the times of the other dump, or a change the other dump
    does not have, is a difference: one of the new dump is expected with the
    pattern of the variable `time` of its task (`experiment/task/time`, a
    changed output grid), one of the old dump always counts.

    Raises:
        UsageError: for renames which cannot be read or applied.
    """
    old, new = load(args.old), load(args.new)
    renames: Renames = {}
    if args.renames:
        try:
            renames = load_renames(args.renames)
            old = apply_renames(old, renames)
        except ValueError as err:
            raise UsageError(str(err)) from err
    if fields := renames.get("figure_fields"):
        old, new = drop_figure_fields(old, fields), drop_figure_fields(new, fields)
    try:
        parts = {
            part: (_flat(old, part), _flat(new, part))
            for part in ("datasets", "mappings", "figures", "ranges")
        }
    except ValueError as err:
        raise UsageError(str(err)) from err
    # a dump without the variants of its tasks has no builds
    builds = [label for label, d in (("old", old), ("new", new)) if "builds" in d]
    if len(builds) == 2:
        parts["builds"] = (_flat(old, "builds"), _flat(new, "builds"))
    expected = Expected(renames.get("expected", {}))
    grids = [dump_.get("grid") for dump_ in (old, new)]
    if any(g is not None for g in grids):
        report(
            "grid: "
            + ", ".join(
                f"{label} {'the steps of the integrator' if g is None else g}"
                for label, g in zip(("old", "new"), grids, strict=True)
            )
        )
    for label, dump_ in (("old", old), ("new", new)):
        if versions := dump_.get("versions"):
            report(
                f"{label}: "
                + ", ".join(f"{k} {v}" for k, v in sorted(versions.items()))
            )
    names: set[str] = set(args.experiments)

    def selected(sid: str) -> bool:
        return not names or sid in names

    differing = 0
    for label, dump_ in (("old", old), ("new", new)):
        for sid in sorted(filter(selected, dump_["errors"])):
            report(f"error in {label}: {sid}")
            differing += 1
    only_old = [s for s in old["order"] if s not in new["order"] and selected(s)]
    only_new = [s for s in new["order"] if s not in old["order"] and selected(s)]
    for sid in only_old:
        report(f"only in old: {sid}")
    for sid in only_new:
        report(f"only in new: {sid}")
    differing += len(only_old) + len(only_new)

    # simulations
    keys_old = {k for k in old["results"] if selected(k[0])}
    keys_new = {k for k in new["results"] if selected(k[0])}
    one_side = sorted(keys_old ^ keys_new)
    unexpected_side = [k for k in one_side if not expected.match("results", k)]
    for key in one_side[: args.top]:
        mark = "" if key in unexpected_side else "expected: "
        report(
            f"{mark}array only in {'old' if key in keys_old else 'new'}: {'/'.join(key)}"
        )
    compared = relative_differences(old, new, keys_old & keys_new)
    rows = sorted(((c.difference, key) for key, c in compared.items()), reverse=True)
    per_experiment: dict[str, float] = {}
    for difference, key in rows:
        per_experiment[key[0]] = max(per_experiment.get(key[0], 0.0), difference)
    above = [row for row in rows if row[0] > args.threshold]
    unexpected = [row for row in above if not expected.match("results", row[1])]
    # an array within the threshold whose interpolated rows cannot resolve it is
    # no pass: its difference may hide in the range of the interpolants
    unresolved = sorted(
        (
            (c.resolution, key)
            for key, c in compared.items()
            if c.difference <= args.threshold and c.resolution > args.threshold
        ),
        reverse=True,
    )
    unexpected_unresolved = [
        row for row in unresolved if not expected.match("results", row[1])
    ]
    report(
        f"simulations: {len(rows)} arrays compared, {len(above)} differ by more "
        f"than {args.threshold:g} ({len(above) - len(unexpected)} expected), "
        f"{len(unresolved)} unresolved on this grid "
        f"({len(unresolved) - len(unexpected_unresolved)} expected), "
        f"{len(one_side)} in one dump only"
    )
    for sid, difference in sorted(per_experiment.items(), key=lambda t: -t[1]):
        if difference > 0:
            report(f"  {difference:.3e} {sid}")
    if rows and rows[0][0] > 0:
        report("largest differences:")
        for difference, key in rows[: args.top]:
            if difference > 0:
                report(f"  {difference:.3e} {'/'.join(key)}")
    if unresolved:
        report(
            "unresolved on this grid: the resolution of the interpolation of the "
            f"steps of the other dump is coarser than {args.threshold:g}, a "
            "difference below it is not seen:"
        )
        for resolution, key in unresolved[: args.top]:
            report(
                f"  {resolution:.3e} resolution {'/'.join(key)} (difference "
                f"{compared[key].difference:.3e})"
            )
    differing += len(unexpected) + len(unexpected_side) + len(unexpected_unresolved)

    # the rows at a change and at the times of one dump only
    ignored_old = ignored_new = unmatched = unmatched_new = expected_new = 0
    interpolated_old = interpolated_new = 0
    changes: dict[str, int] = {}
    other_times: list[str] = []
    for sid, task in sorted({(k[0], k[1]) for k in keys_old & keys_new}):
        times = task_columns(old, new, sid, task)
        matches = [] if times is None else point_matches(times.old, times.new)
        for match in (m.match for m in matches):
            ignored_old += match.ignored_old
            ignored_new += match.ignored_new
            interpolated_old += match.interpolated_old.size
            interpolated_new += match.interpolated_new.size
            unmatched += match.unmatched
            unmatched_new += match.unmatched_new
            if match.ignored_new:
                changes[sid] = changes.get(sid, 0) + match.ignored_new
            if match.unmatched:
                other_times.append(
                    f"  times of {sid}/{task}: {match.unmatched} rows only in old, "
                    f"at {_listed(match.times_old)}"
                )
            if match.unmatched_new:
                # a changed output grid of a task is named by its variable `time`
                named = expected.match("results", (sid, task, "time"))
                expected_new += match.unmatched_new if named else 0
                other_times.append(
                    f"  times of {sid}/{task}: {match.unmatched_new} rows only in new, "
                    f"at {_listed(match.times_new)}{' (expected)' if named else ''}"
                )
    report(
        f"rows at a change: {ignored_new} rows of the new dump and {ignored_old} of "
        f"the old dump left out (the states before a change), {unmatched} rows of "
        f"the old dump without a time in the new dump, {unmatched_new} rows of the "
        f"new dump without a time in the old dump ({expected_new} expected)"
    )
    report(
        f"rows at other times: {interpolated_old} rows of the old dump and "
        f"{interpolated_new} of the new dump compared with the other dump "
        "interpolated at their times (the range of its linear, monotone cubic "
        "and cubic spline interpolants)"
    )
    for sid, count in sorted(changes.items()):
        report(f"changes: {sid} {count} rows")
    for line in other_times[: args.top]:
        report(line)
    # a time which only one dump has is a difference; one of the new dump can be
    # expected, the old dump is the reference
    differing += unmatched + unmatched_new - expected_new

    # everything else exactly
    for part, (flat_old, flat_new) in parts.items():
        a = {k: v for k, v in flat_old.items() if selected(_sid(k))}
        b = {k: v for k, v in flat_new.items() if selected(_sid(k))}
        changed = sorted(
            (k for k in a.keys() | b.keys() if not equal(a.get(k), b.get(k))), key=str
        )
        unexpected_keys = [k for k in changed if not expected.match(part, k)]
        report(
            f"{part}: {len(a)} old, {len(b)} new, {len(changed)} differ "
            f"({len(changed) - len(unexpected_keys)} expected)"
        )
        for k in changed[: args.top]:
            mark = "" if k in unexpected_keys else " (expected)"
            report(f"  {'/'.join(k) if isinstance(k, tuple) else k}{mark}")
            for line in _detail(part, a.get(k), b.get(k))[: args.top]:
                report(f"    {line}")
        differing += len(unexpected_keys)

    if len(builds) == 1:
        report(f"builds: in the {builds[0]} dump only, not compared")
    for part, pattern in expected.unused():
        report(f"expected difference which does not occur: {part}: {pattern}")
        differing += 1
    return 1 if differing else 0


def _sid(key: str | tuple[str, ...]) -> str:
    """Get the experiment of a key of a dump."""
    return key[0] if isinstance(key, tuple) else key


def diff_lines(a: object, b: object) -> list[str]:
    """Describe the difference of two lists or dicts line by line.

    The values of a dict are compared exactly, json values (the specification of
    a figure) by the paths of their elements which differ.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        lines: list[str] = []
        for k in sorted(a.keys() | b.keys(), key=str):
            if k not in a or k not in b:
                lines.append(f"{k}: only {'old' if k in a else 'new'}")
            elif not equal(a[k], b[k]):
                paths = json_paths(_json(a[k]), _json(b[k]))
                lines.append(f"{k}: {', '.join(paths[:5]) or 'changed'}")
        return lines
    x = list(a) if isinstance(a, list) else []
    y = list(b) if isinstance(b, list) else []
    return [f"- {v}" for v in x if v not in y] + [f"+ {v}" for v in y if v not in x]


def _json(value: object) -> object:
    """Parse a json string, any other value is returned as it is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def json_paths(a: object, b: object, path: str = "") -> list[str]:
    """Get the paths of the elements in which two json values differ.

    A key which only one object has is named with its value, also a `null` one
    (a field which one version of sbmlsim added to a figure).
    """
    if isinstance(a, dict) and isinstance(b, dict):
        paths: list[str] = []
        for k in sorted(a.keys() | b.keys()):
            if k in a and k in b:
                paths += json_paths(a[k], b[k], f"{path}.{k}")
            else:
                side, value = ("old", a[k]) if k in a else ("new", b[k])
                where = f"{path}.{k}".lstrip(".")
                paths.append(f"{where} only in {side}: {value!r}")
        return paths
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            p
            for k, (x, y) in enumerate(zip(a, b, strict=True))
            for p in json_paths(x, y, f"{path}[{k}]")
        ]
    if a != b:
        return [f"{path.lstrip('.')} {a!r} -> {b!r}"]
    return []


# --- ranges ---------------------------------------------------------------------


def ranges(args: argparse.Namespace) -> int:
    """List the fit mappings whose data lie outside of the simulated range."""
    data = load(args.dump)
    outside = 0
    for sid, mappings in sorted(data["ranges"].items()):
        for key, value in sorted(mappings.items()):
            if isinstance(value, str):
                report(f"{sid} {key}: {value}")
                outside += 1
                continue
            dmin, dmax, smin, smax, unit = value
            tolerance = 1e-9 * max(1.0, abs(smin), abs(smax))
            ok = dmin >= smin - tolerance and dmax <= smax + tolerance
            if not ok or args.verbose:
                status = "ok" if ok else "OUTSIDE"
                report(
                    f"{status} {sid} {key}: data [{dmin:.4g}, {dmax:.4g}], "
                    f"simulation [{smin:.4g}, {smax:.4g}] {unit}"
                )
            outside += not ok
    report(f"{outside} fit mappings with data outside of the simulation")
    return 1 if outside else 0


# --- fits -----------------------------------------------------------------------
