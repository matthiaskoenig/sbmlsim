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
        ContractError: for a payload of another contract or of no keys.
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

    Args:
        svg: the drawing, read as text for the inline figure.
        readouts: the sidecar beside it, `<svg stem>.readouts.json`.
        simulation: the simulation every panel runs.
        title: the title of the figure.
        credit: the source of the drawing.
        prefix: put before every panel id.

    Returns:
        The panels by id, one per key with readouts (the title the label of the
        key, the curves its readouts in order, the `y_label` their kind when
        they share one), and the figure whose `panels` map every key to its
        panel ids, none for a decoration.

    Raises:
        ContractError: for a sidecar of another contract, a key with more than
            `MAX_CURVES` readouts, or two keys of one panel id.
    """
    keys = _read(readouts)
    panels: dict[str, Panel] = {}
    selection: dict[str, tuple[str, ...]] = {}
    for key, group in keys.items():
        entries = group.get("readouts", [])
        if not entries:
            selection[key] = ()
            continue
        if len(entries) > MAX_CURVES:
            raise ContractError(
                f"The key '{key}' of '{readouts}' has {len(entries)} readouts; "
                f"a panel draws at most {MAX_CURVES} curves."
            )
        pid = panel_id(key, prefix)
        if pid in panels:
            raise ContractError(
                f"The keys of '{readouts}' give the panel '{pid}' twice."
            )
        kinds = {entry["kind"] for entry in entries}
        panels[pid] = Panel(
            id=pid,
            title=group["label"],
            y_label=kinds.pop() if len(kinds) == 1 else "",
            curves=tuple(
                PanelCurve(id=e["id"], label=e["label"], unit=e.get("unit") or None)
                for e in entries
            ),
            simulation=simulation,
        )
        selection[key] = (pid,)
    figure = AppFigure(
        svg=svg.read_text(encoding="utf-8"),
        title=title,
        credit=credit,
        panels=selection,
    )
    return panels, figure
