"""The panels of the readouts sidecar of a figure.

A model package which draws its model (pkpdsim writes `<sid>_body.svg` and
`<sid>_body.readouts.json` beside it) gets the panel declarations of its app
from the sidecar instead of writing them by hand: one panel per part of the
drawing which has readouts, and the `AppFigure` which maps the key path of
every part to its panel. `panels_from_figure` takes the ids of the readouts as
the sidecar gives them, since pkpdsim never compiles a model; `restrict_to_model`
checks them against the compiled one. A caller which wants the readouts of a
part split otherwise than by their unit, e.g. the substances of one compartment
in one plot, passes a `Grouping`.
"""

import json
import logging
import re
from collections.abc import Callable, Mapping
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

logger = logging.getLogger(__name__)

CONTRACT = "figurereadouts/1"
"""The contract of the readouts sidecar this module reads."""

Grouping = Callable[[str, Mapping[str, str]], tuple[str, str]]
"""The group of a readout within its key and the label of its curve."""


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


def _group_of(
    key: str, entry: dict[str, str], group: Grouping, readouts: Path
) -> tuple[str, str]:
    """Ask a grouping for the group of a readout and the label of its curve.

    Raises:
        ContractError: naming the key and the readout, for a grouping which
            raises or answers with something which is no pair of two texts.
    """
    where = (
        f"The grouping of the readout '{entry['id']}' of the key '{key}' "
        f"of '{readouts}'"
    )
    try:
        answer = group(key, entry)
    except Exception as error:
        raise ContractError(f"{where} raised {error!r}.") from error
    if (
        not isinstance(answer, tuple)
        or len(answer) != 2
        or not all(isinstance(text, str) for text in answer)
    ):
        raise ContractError(f"{where} is no pair of two texts: {answer!r}.")
    return answer


def panels_from_figure(
    svg: Path,
    readouts: Path,
    *,
    simulation: PanelSimulation,
    title: str,
    credit: str = "",
    prefix: str = "",
    group: Grouping | None = None,
) -> tuple[dict[str, Panel], AppFigure]:
    """Turn the readouts sidecar of a figure into panels and the app figure.

    The curves of a panel share one value axis, so a key whose readouts have
    several units opens one panel per unit, in the order the units first
    appear. `group` splits a key further: it answers for every readout with
    the group of that readout within its key and the label of its curve, so
    the substances of one compartment of the drawing share a panel (pkpdsim
    labels a readout `"<substance>, <Organ> <compartment>"`, so a grouping of
    the text behind the first `", "` gives a panel per compartment). A part of
    a key is one unit and one group, never a group across units.

    The id of such a panel is the one of the key with its group appended
    (`li_plasma`) where the groups of the parts of the key are distinct texts,
    and otherwise with the kind of its curves (`li_urine_amount`), and the
    index of the panel after it when the kinds do not tell the panels of the
    key apart either; the title of a panel names its group exactly where its
    id does. A key of one part keeps the id and the title of the key.

    Args:
        svg: the drawing, read as text for the inline figure.
        readouts: the sidecar beside it, `<svg stem>.readouts.json`.
        simulation: the simulation every panel runs.
        title: the title of the figure.
        credit: the source of the drawing.
        prefix: put before every panel id.
        group: the grouping of the readouts of a key, see `Grouping`; without
            it the readouts of one unit are one group and the label of the
            sidecar is the label of the curve.

    Returns:
        The panels by id (the title the label of the key and its group, the
        curves its readouts in order, the `y_label` the kinds of the curves,
        joined by " / " when they differ), and the figure whose `panels` map
        every key to its panel ids, none for a decoration.

    Raises:
        ContractError: for a sidecar of another contract or a malformed key,
            a key with more than `MAX_CURVES` readouts in one unit and group,
            or two keys of one panel id; naming the key and the readout, for a
            grouping which raises or answers with no pair of texts; naming the
            drawing and the reason, for a blank drawing or title or any other
            figure the app refuses.
        OSError: for a file which cannot be read.
        json.JSONDecodeError: for a sidecar which is no JSON.
    """
    keys = _read(readouts)
    panels: dict[str, Panel] = {}
    selection: dict[str, tuple[str, ...]] = {}
    for key, declaration in keys.items():
        label, entries = _entries(key, declaration, readouts)
        by_part: dict[tuple[str | None, str], list[tuple[dict[str, str], str]]] = {}
        for entry in entries:
            name, curve = (
                ("", entry["label"])
                if group is None
                else _group_of(key, entry, group, readouts)
            )
            by_part.setdefault((entry.get("unit") or None, name), []).append(
                (entry, curve)
            )
        parts = list(by_part.items())
        base = panel_id(key, prefix)
        kinds = [list(dict.fromkeys(e["kind"] for e, _ in part)) for _, part in parts]
        groups = [re.sub(r"\W", "_", name) for (_, name), _ in parts]
        named = all(groups) and len(set(groups)) == len(groups)
        suffixes = groups if named else ["_".join(k) for k in kinds]
        if len(set(suffixes)) < len(suffixes):
            suffixes = [f"{s}_{n}" for n, s in enumerate(suffixes, 1)]
        ids: list[str] = []
        for ((_, name), part), kind, suffix in zip(parts, kinds, suffixes, strict=True):
            pid = base if len(parts) == 1 else f"{base}_{suffix}"
            if pid in panels:
                raise ContractError(
                    f"The keys of '{readouts}' give the panel '{pid}' twice."
                )
            if len(part) > MAX_CURVES:
                where = f"of the group '{name}'" if name else "of one unit"
                raise ContractError(
                    f"The key '{key}' of '{readouts}' has {len(part)} readouts "
                    f"{where}; a panel draws at most {MAX_CURVES} curves."
                )
            try:
                panels[pid] = Panel(
                    id=pid,
                    title=label if len(parts) == 1 or not named else f"{label}, {name}",
                    y_label=" / ".join(kind),
                    curves=tuple(
                        PanelCurve(id=e["id"], label=c, unit=e.get("unit") or None)
                        for e, c in part
                    ),
                    simulation=simulation,
                )
            except ValidationError as error:
                raise ContractError(
                    f"The key '{key}' of '{readouts}' is no panel: {error}"
                ) from error
            ids.append(pid)
        selection[key] = tuple(ids)
    try:
        figure = AppFigure(
            svg=svg.read_text(encoding="utf-8"),
            title=title,
            credit=credit,
            panels=selection,
        )
    except ValidationError as error:
        raise ContractError(
            f"The figure of the drawing '{svg}' with the readouts '{readouts}' "
            f"is no figure: {error}"
        ) from error
    return panels, figure


def restrict_to_model(
    panels: dict[str, Panel],
    figure: AppFigure,
    has_selection: Callable[[str], bool],
) -> tuple[dict[str, Panel], AppFigure, tuple[str, ...]]:
    """Drop the readouts the compiled model does not have.

    The ids of a sidecar are the ones of the id scheme of pkpdsim, which never
    compiles a model, so a sidecar may name an id the model does not have. Such
    a curve is dropped and reported in a warning; a panel left without a curve
    is dropped and leaves the selection of its key, so a key whose ids are all
    missing becomes a decoration which selects nothing. The panels and the
    figure are validated again.

    The caller must build the panels of the pages of its app from the returned
    panels and register the returned figure: a page which still lists a
    dropped panel is refused by `sbmlsim.serve.runner.ModelRegistry.of`, which
    finds no such panel in the model.

    Args:
        panels: the panels by id, see `panels_from_figure`.
        figure: the figure whose `panels` select them.
        has_selection: whether the compiled model can select an id, e.g.
            `SBMLModel.has_selection`.

    Returns:
        The panels which keep a curve, the figure which selects only them, and
        the dropped ids in the order they appear, each once.

    Raises:
        ContractError: naming the dropped ids, if the model lacks readouts and
            not one panel of the figure keeps a curve.
    """
    kept: dict[str, Panel] = {}
    dropped: list[str] = []
    for pid, panel in panels.items():
        present = tuple(c for c in panel.curves if has_selection(c.id))
        dropped.extend(c.id for c in panel.curves if not has_selection(c.id))
        if present:
            kept[pid] = (
                panel
                if len(present) == len(panel.curves)
                else Panel.model_validate({**dict(panel), "curves": present})
            )
    dropped = list(dict.fromkeys(dropped))
    lost = {
        key: [c.id for pid in ids if pid in panels for c in panels[pid].curves]
        for key, ids in figure.panels.items()
    }
    lost = {
        key: [i for i in ids if i in dropped]
        for key, ids in lost.items()
        if any(i in dropped for i in ids)
    }
    if dropped:
        if not kept:
            raise ContractError(
                f"The model has none of the readouts of the figure "
                f"'{figure.title}'; missing {dropped}."
            )
        logger.warning(
            "The model does not have the readouts %s of the figure '%s'; they "
            "are left out, so the parts %s lost readouts and %s select nothing.",
            dropped,
            figure.title,
            sorted(lost),
            sorted(
                key
                for key in lost
                if not any(pid in kept for pid in figure.panels[key])
            ),
        )
    selection = {
        key: tuple(pid for pid in ids if pid in kept)
        for key, ids in figure.panels.items()
    }
    restricted = AppFigure.model_validate({**dict(figure), "panels": selection})
    return kept, restricted, tuple(dropped)
