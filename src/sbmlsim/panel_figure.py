"""The panels of the readouts sidecar of a figure.

A model package which draws its model (pkpdsim writes `<sid>_body.svg` and
`<sid>_body.readouts.json` beside it) gets the panel declarations of its app
from the sidecar instead of writing them by hand: one panel per part of the
drawing which has readouts, and the `AppFigure` which maps the key path of
every part to its panel. The ids of the readouts are taken as the sidecar
gives them and are not checked against a compiled model.
"""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sbmlsim.panel import (
    MAX_CURVES,
    AppFigure,
    ContractError,
    Panel,
    PanelCurve,
    PanelSimulation,
)

CONTRACT = "figurereadouts/1"
"""The contract of the readouts sidecar this module reads."""


def panel_id(key: str, prefix: str = "") -> str:
    """Give the id of the panel of a part of the drawing.

    Args:
        key: the key path of the part, e.g. `li/sub:gli`.
        prefix: put before the id.

    Returns:
        The key path without `sub:` and with every non-word character replaced
        by `_`, e.g. `li_gli`, behind the prefix.
    """
    return prefix + re.sub(r"\W", "_", key.replace("sub:", ""))


def _read(readouts: Path) -> dict[str, Any]:
    """Read the sidecar and check its contract.

    Raises:
        ContractError: for a payload which is no object, of another contract or
            with no `keys` mapping.
    """
    payload = json.loads(readouts.read_text(encoding="utf-8"))
    contract = payload.get("contract") if isinstance(payload, dict) else None
    if contract != CONTRACT:
        raise ContractError(
            f"The readouts '{readouts}' are of the contract {contract!r}, "
            f"not {CONTRACT!r}."
        )
    keys = payload.get("keys")
    if not isinstance(keys, dict):
        raise ContractError(f"The readouts '{readouts}' carry no 'keys' mapping.")
    return keys


def _entries(key: str, group: Any, readouts: Path) -> tuple[str, list[dict[str, str]]]:
    """Check one group of the sidecar and give its label and readouts.

    Raises:
        ContractError: naming the key, for a malformed group, a readout without
            text `id`, `label` or `kind`, or two readouts of one id.
    """
    where = f"The key '{key}' of '{readouts}'"
    if not isinstance(group, dict):
        raise ContractError(f"{where} is not an object.")
    label = group.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ContractError(f"{where} has no label.")
    entries = group.get("readouts", [])
    if not isinstance(entries, list):
        raise ContractError(f"{where} has no list of readouts.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ContractError(f"{where} has a readout which is not an object.")
        for field in ("id", "label", "kind"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{where} has a readout without '{field}'.")
        if not isinstance(entry.get("unit", ""), str):
            raise ContractError(f"{where} has a readout with a unit which is no text.")
    ids = [entry["id"] for entry in entries]
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if twice:
        raise ContractError(f"{where} has the readouts {twice} twice.")
    return label, entries


def panels_from_figure(
    svg: Path,
    readouts: Path,
    *,
    simulation: PanelSimulation,
    title: str,
    credit: str = "",
    prefix: str = "",
) -> tuple[dict[str, Panel], AppFigure]:
    """Turn the readouts sidecar of a figure into panels and the app figure.

    The curves of a panel share one value axis, so a key whose readouts have
    several units opens one panel per unit, in the order the units first
    appear. Such a panel id is the one of the key with the kind of its curves
    appended (`li_urine_amount`), and the index of the panel after it when the
    kinds do not tell the panels of the key apart.

    Args:
        svg: the drawing, read as text for the inline figure.
        readouts: the sidecar beside it, `<svg stem>.readouts.json`.
        simulation: the simulation every panel runs.
        title: the title of the figure.
        credit: the source of the drawing.
        prefix: put before every panel id.

    Returns:
        The panels by id (the title the label of the key, the curves its
        readouts in order, the `y_label` the kinds of the curves, joined by
        " / " when they differ), and the figure whose `panels` map every key to
        its panel ids, none for a decoration.

    Raises:
        ContractError: for a sidecar of another contract or a malformed key,
            a key with more than `MAX_CURVES` readouts in one unit, or two keys
            of one panel id.
        OSError: for a file which cannot be read.
        json.JSONDecodeError: for a sidecar which is no JSON.
    """
    keys = _read(readouts)
    panels: dict[str, Panel] = {}
    selection: dict[str, tuple[str, ...]] = {}
    for key, group in keys.items():
        label, entries = _entries(key, group, readouts)
        by_unit: dict[str | None, list[dict[str, str]]] = {}
        for entry in entries:
            by_unit.setdefault(entry.get("unit") or None, []).append(entry)
        parts = list(by_unit.items())
        base = panel_id(key, prefix)
        kinds = [list(dict.fromkeys(e["kind"] for e in part)) for _, part in parts]
        suffixes = ["_".join(k) for k in kinds]
        if len(set(suffixes)) < len(suffixes):
            suffixes = [f"{s}_{n}" for n, s in enumerate(suffixes, 1)]
        ids: list[str] = []
        for (_, part), kind, suffix in zip(parts, kinds, suffixes, strict=True):
            pid = base if len(parts) == 1 else f"{base}_{suffix}"
            if pid in panels:
                raise ContractError(
                    f"The keys of '{readouts}' give the panel '{pid}' twice."
                )
            if len(part) > MAX_CURVES:
                raise ContractError(
                    f"The key '{key}' of '{readouts}' has {len(part)} readouts "
                    f"of one unit; a panel draws at most {MAX_CURVES} curves."
                )
            try:
                panels[pid] = Panel(
                    id=pid,
                    title=label,
                    y_label=" / ".join(kind),
                    curves=tuple(
                        PanelCurve(
                            id=e["id"], label=e["label"], unit=e.get("unit") or None
                        )
                        for e in part
                    ),
                    simulation=simulation,
                )
            except ValidationError as error:
                raise ContractError(
                    f"The key '{key}' of '{readouts}' is no panel: {error}"
                ) from error
            ids.append(pid)
        selection[key] = tuple(ids)
    figure = AppFigure(
        svg=svg.read_text(encoding="utf-8"),
        title=title,
        credit=credit,
        panels=selection,
    )
    return panels, figure
