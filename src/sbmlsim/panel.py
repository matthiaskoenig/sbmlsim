"""The interactive panel of a model: its knobs, curves, study data and simulation.

A model declares its interactive surface in an annotation of its SBML: the
knobs a reader turns, the curves which are drawn, the study data under them and
the simulation they run. This module is the declaration itself, frozen and
picklable data, and the versioned wire contract `panel/1` which a renderer
sees: `panel_payload` gives the JSON of a panel and `panel_from_payload` reads
it back, each the inverse of the other for every field.

The module is pure data: it imports pydantic and the standard library only, no
simulator, no libsbml and no project, so `sbmlsim.project` imports it for the
field `Project.panels`.

The hard limits `MAX_END`, `MAX_POINTS`, `MAX_KNOBS`, `MAX_GRID` and
`MAX_CURVES` bound what a panel declares and what a request of a grid over its
knobs may ask for. Every payload carries them, so a renderer knows the limits
of the endpoint it talks to without a contract of its own.
"""

from collections.abc import Mapping
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CONTRACT",
    "MAX_CURVES",
    "MAX_END",
    "MAX_GRID",
    "MAX_KNOBS",
    "MAX_POINTS",
    "Choice",
    "ContractError",
    "Knob",
    "Panel",
    "PanelCurve",
    "PanelData",
    "PanelSimulation",
    "panel_from_payload",
    "panel_payload",
]

#: the version of the wire contract of a panel, in every payload
CONTRACT: Final = "panel/1"

#: the largest end of the simulation of a panel, in the time unit of its model
MAX_END: Final = 10000.0
#: the most output points of the simulation of a panel
MAX_POINTS: Final = 2000
#: the most knobs a request may grid over
MAX_KNOBS: Final = 4
#: the largest product of the sizes of the grid of a request
MAX_GRID: Final = 64
#: the most curves of a panel
MAX_CURVES: Final = 16

#: a panel and its parts forbid unknown fields and do not change
_CONFIG = ConfigDict(extra="forbid", frozen=True)


class ContractError(ValueError):
    """A payload which is not of the contract `CONTRACT` of this package."""


class Choice(BaseModel):
    """One choice of a knob of the kind `choice`, e.g. a dose of a tablet.

    Attributes:
        label: the choice as it is shown, e.g. `"12.5 mg"`.
        value: the value the knob sets, in its unit.
    """

    model_config = _CONFIG

    label: str
    value: float


class Knob(BaseModel):
    """A knob of a panel: an entity of the model a reader sets.

    A knob of the kind `range` has the bounds `min` and `max` and a grid of
    `steps` points over them, one of the kind `choice` its `choices`; see
    `values`.

    Attributes:
        id: the id of the knob, the key of its value in a request and the id
            of its axis in a grid.
        label: the knob as it is shown, e.g. `"renal function"`.
        target: the entity of the model the knob sets, a selection of the
            simulation, e.g. `"PODOSE_hctz"`.
        unit: the unit of the values, the unit of the target without.
        kind: `range` for a knob of bounds, `choice` for one of choices.
        min: the smallest value of a range knob.
        max: the largest value of a range knob.
        steps: the points of the grid of a range knob, at most `MAX_GRID`,
            see `values`.
        choices: the choices of a choice knob, in their order.
        default: the value the knob starts at, inside the bounds of a range
            knob and one of the values of a choice knob.
    """

    model_config = _CONFIG

    id: str
    label: str
    target: str
    unit: str | None = None
    kind: Literal["range", "choice"] = "range"
    min: float | None = None
    max: float | None = None
    steps: int = Field(default=5, ge=2, le=MAX_GRID)
    choices: tuple[Choice, ...] = ()
    default: float

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        """Check the knob against its kind.

        Raises:
            ValueError: for a range knob without its bounds, with bounds which
                are no interval or with a default outside them, and for a
                choice knob without choices or with a default which is no
                choice.
        """
        if self.kind == "range":
            if self.min is None or self.max is None:
                raise ValueError(
                    f"The range knob '{self.id}' needs the bounds 'min' and 'max'."
                )
            if self.min >= self.max:
                raise ValueError(
                    f"The range knob '{self.id}' needs min < max, not "
                    f"min={self.min} and max={self.max}."
                )
            if not self.min <= self.default <= self.max:
                raise ValueError(
                    f"The default {self.default} of the knob '{self.id}' is outside "
                    f"its bounds [{self.min}, {self.max}]."
                )
        elif not self.choices:
            raise ValueError(f"The choice knob '{self.id}' needs its choices.")
        elif self.default not in [choice.value for choice in self.choices]:
            raise ValueError(
                f"The default {self.default} of the knob '{self.id}' is none of its "
                f"choices {[choice.value for choice in self.choices]}."
            )
        return self

    def values(self, n: int | None = None) -> tuple[float, ...]:
        """The values of the knob which a grid over it runs.

        Args:
            n: the number of points of a range knob, its `steps` without; the
                values of a choice knob are its choices whatever `n` is.

        Returns:
            The points from `min` to `max`, both included, for a range knob,
            the values of the choices for a choice knob.

        Raises:
            ValueError: for fewer than one point, and for a range knob without
                bounds, which `_check_kind` refuses.
        """
        if self.kind == "choice":
            return tuple(choice.value for choice in self.choices)
        count = self.steps if n is None else n
        if count < 1:
            raise ValueError(
                f"A grid over the knob '{self.id}' needs a point, not {count}."
            )
        low, high = self.min, self.max
        if low is None or high is None:  # a range knob has both, see `_check_kind`
            raise ValueError(f"The range knob '{self.id}' has no bounds.")
        if count == 1:
            return (low,)
        # the weighted form keeps a decimal grid exact, where the affine one
        # does not: `0.2 + 2 * 0.2` is 0.6000000000000001, `(0.2 * 2 + 1.0 * 2)
        # / 4` is 0.6, and the ends are the bounds themselves
        last = count - 1
        inner = [(low * (last - i) + high * i) / last for i in range(1, last)]
        return (low, *inner, high)


class PanelCurve(BaseModel):
    """A curve of a panel: one output of the simulation over the time.

    Attributes:
        id: the id of the curve, a selection of the model, e.g. `"[Cve_hctz]"`,
            and the key of its values in a result.
        label: the curve as it is shown, its entry in the legend.
        unit: the unit of the values, the unit of the selection without.
        scale: the scale of the axis of the curve, `linear` or `log`.
    """

    model_config = _CONFIG

    id: str
    label: str
    unit: str | None = None
    scale: Literal["linear", "log"] = "linear"


class PanelData(BaseModel):
    """The study data of a panel, drawn under its curves.

    Attributes:
        id: the id of the series, the key of its values in a result.
        label: the series as it is shown, e.g. the study it comes from.
        dataset: the dataset of the series, a file of the data of the project.
        x: the column of the times.
        y: the column of the values.
        yerr: the column of the error of the values, none without.
        unit: the unit of the values, the one of the dataset without.
    """

    model_config = _CONFIG

    id: str
    label: str
    dataset: str
    x: str
    y: str
    yerr: str | None = None
    unit: str | None = None


class PanelSimulation(BaseModel):
    """The simulation a panel runs, the same for every point of its grid.

    Attributes:
        variant: the variant of the model, the default variant of the project
            without.
        end: the end of the simulation, in the time unit of the model; at most
            `MAX_END`.
        points: the output times of the simulation, at most `MAX_POINTS`.
        x_unit: the unit the time axis of a result is delivered in, e.g.
            `"hr"` for a model whose time unit is minutes; `None` delivers it
            in the time unit of the model. The `end` and the `points` are in
            the time unit of the model whatever it is.
        preinit: the values of the pre-initialization which are no knob, e.g.
            the body weight, by the entity of the model they set.
    """

    model_config = _CONFIG

    variant: str | None = None
    end: float = Field(gt=0.0, le=MAX_END)
    points: int = Field(default=200, ge=2, le=MAX_POINTS)
    x_unit: str | None = None
    preinit: Mapping[str, float] = {}


class Panel(BaseModel):
    """The interactive panel of a model, see the module.

    Attributes:
        id: the id of the panel, the one a request names.
        title: the title of the panel.
        subtitle: the question the panel answers, shown below the title.
        x_label: the label of the time axis, e.g. `"time (hr)"`.
        knobs: the knobs of the panel, in the order they are shown; they are
            the axes of the grid a request runs.
        curves: the curves of the panel, in the order they are drawn; at most
            `MAX_CURVES`.
        data: the study data of the panel, drawn under its curves.
        simulation: the simulation the panel runs.
    """

    model_config = _CONFIG

    id: str
    title: str
    subtitle: str = ""
    x_label: str = "time"
    knobs: tuple[Knob, ...] = ()
    curves: tuple[PanelCurve, ...] = Field(max_length=MAX_CURVES)
    data: tuple[PanelData, ...] = ()
    simulation: PanelSimulation

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        """Check that the knobs, the curves and the data have distinct ids.

        Raises:
            ValueError: naming the ids which are declared twice.
        """
        parts: tuple[tuple[str, list[str]], ...] = (
            ("knobs", [knob.id for knob in self.knobs]),
            ("curves", [curve.id for curve in self.curves]),
            ("data", [data.id for data in self.data]),
        )
        for kind, ids in parts:
            twice = sorted({id_ for id_ in ids if ids.count(id_) > 1})
            if twice:
                raise ValueError(
                    f"The {kind} of the panel '{self.id}' declare {twice} twice; "
                    "an id is the key of its values in a result."
                )
        return self


def panel_payload(panel: Panel) -> dict[str, Any]:
    """Get the `panel/1` payload of a panel, the shape a renderer sees.

    Args:
        panel: the panel.

    Returns:
        The payload: its `contract`, the texts of the `panel`, its knobs as the
        `axes` of the grid, its `curves`, its `data`, its `simulation` and the
        `limits` of a request. Plain JSON values only, no object of this
        package, see `panel_from_payload`.
    """
    return {
        "contract": CONTRACT,
        "panel": {
            "id": panel.id,
            "title": panel.title,
            "subtitle": panel.subtitle,
            "x_label": panel.x_label,
        },
        "axes": [knob.model_dump(mode="json") for knob in panel.knobs],
        "curves": [curve.model_dump(mode="json") for curve in panel.curves],
        "data": [data.model_dump(mode="json") for data in panel.data],
        "simulation": panel.simulation.model_dump(mode="json"),
        "limits": {
            "end": MAX_END,
            "points": MAX_POINTS,
            "knobs": MAX_KNOBS,
            "grid": MAX_GRID,
            "curves": MAX_CURVES,
        },
    }


def panel_from_payload(payload: Mapping[str, Any]) -> Panel:
    """Read the panel of a `panel/1` payload, the inverse of `panel_payload`.

    The `limits` of a payload are the ones of the package which wrote it and
    are not read back.

    Args:
        payload: the payload.

    Returns:
        The panel.

    Raises:
        ContractError: for a payload of another contract version.
        pydantic.ValidationError: for a payload which declares no panel,
            naming the field which is wrong or unknown.
    """
    contract = payload.get("contract")
    if contract != CONTRACT:
        raise ContractError(
            f"The payload is of the contract {contract!r} and not of {CONTRACT!r}."
        )
    return Panel.model_validate(
        {
            **dict(payload.get("panel") or {}),
            "knobs": list(payload.get("axes") or ()),
            "curves": list(payload.get("curves") or ()),
            "data": list(payload.get("data") or ()),
            "simulation": dict(payload.get("simulation") or {}),
        }
    )
