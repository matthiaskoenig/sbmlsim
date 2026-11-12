"""The bodies of the HTTP API: what a request may ask for and what it answers.

A request of a simulation is a `SimulateRequest`: the value of every knob a
reader set and the points of every knob a grid runs over. The limits of
`sbmlsim.panel` are the bounds of its fields and of its validator, so a request
over them is refused where it is read, before a handler runs and before a model
is loaded: at most `MAX_KNOBS` axes, at least one and at most `MAX_GRID` points
per axis and at most `MAX_GRID` points in all. A knob id the panel does not
declare is refused by `sbmlsim.serve.runner.PanelRunner.run`, which knows the
panel, and is a 422 as well.

The answer is a `SimulateResponse`, the `PanelResult` of that run with its
`spec_hash`, which identifies the resolved request, its `Provenance` and its
`Status`. Every response body carries the `contract` of the payloads
(`sbmlsim.panel.CONTRACT`), so a renderer reads one version of them.

JSON has no `NaN`: a point which did not integrate is `null` in the curves and
its flat index is in `Status.failed`, which is where a renderer reads it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sbmlsim.panel import CONTRACT, MAX_GRID, MAX_KNOBS

if TYPE_CHECKING:
    from sbmlsim.serve.runner import GridAxis, PanelResult

__all__ = [
    "GridAxisPayload",
    "HealthResponse",
    "ModelPayload",
    "ModelsResponse",
    "PanelPayload",
    "PanelsResponse",
    "Provenance",
    "SimulateRequest",
    "SimulateResponse",
    "Status",
]

#: a body of the API forbids unknown fields and does not change
_CONFIG = ConfigDict(extra="forbid", frozen=True)

#: the points of one axis of a grid, see `SimulateRequest`
type _Points = Annotated[int, Field(ge=1, le=MAX_GRID)]


class SimulateRequest(BaseModel):
    """The body of a simulation of a panel, see the module.

    Attributes:
        values: the value of a knob by its id, in the unit of the knob; a knob
            which a request leaves out runs at its `default`.
        grid: the points of a knob of the grid by its id, at most `MAX_KNOBS`
            knobs and `MAX_GRID` points in all; a knob of the grid takes no
            value.
    """

    model_config = _CONFIG

    values: dict[str, float] = Field(default_factory=dict)
    grid: dict[str, _Points] = Field(default_factory=dict, max_length=MAX_KNOBS)

    @model_validator(mode="after")
    def _check_grid(self) -> Self:
        """Check the points of the grid against `MAX_GRID`.

        Raises:
            ValueError: naming the points, for a grid of more than `MAX_GRID`
                points, which is the work of one request.
        """
        points = math.prod(self.grid.values())
        if points > MAX_GRID:
            raise ValueError(
                f"The grid of {points} points is over the limit of {MAX_GRID} "
                f"points of a request."
            )
        return self


class GridAxisPayload(BaseModel):
    """An axis of the grid a run covered, the payload of a `GridAxis`.

    Attributes:
        id: the id of the knob of the axis.
        label: the knob as it is shown.
        unit: the unit of the values, the one the knob declares or else the
            unit of its target in the model.
        values: the values of the knob the run covered, in their order.
    """

    model_config = _CONFIG

    id: str
    label: str
    unit: str | None
    values: list[float]

    @classmethod
    def of(cls, axis: GridAxis) -> GridAxisPayload:
        """Get the payload of an axis of a run.

        Args:
            axis: the axis.

        Returns:
            The payload.
        """
        return cls(
            id=axis.id,
            label=axis.label,
            unit=axis.unit,
            values=list(axis.values),
        )


class Provenance(BaseModel):
    """Where the curves of a response come from.

    Attributes:
        model: the id of the model, the one the request named.
        panel: the id of the panel.
        model_sha256: the sha256 of the SBML file which was simulated.
        variant: the variant of the model, the default variant without.
        engine: the engine which integrated, e.g. `"roadrunner"`.
        sbmlsim: the version of sbmlsim which ran the simulation.
        seconds: the seconds the run took in the service.
        created: when the run finished, an ISO 8601 timestamp in UTC.
    """

    # `model_sha256` is a field and no namespace of pydantic
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model: str
    panel: str
    model_sha256: str
    variant: str | None
    engine: str
    sbmlsim: str
    seconds: float
    created: str


class Status(BaseModel):
    """The outcome of a run of a panel, see `sbmlsim.serve.runner.PanelResult`.

    Attributes:
        ok: whether every point of the grid integrated.
        points: the points of the grid, one for a run without a grid.
        failed: the flat indices in C order of the points which did not
            integrate, whose values are `null`.
    """

    model_config = _CONFIG

    ok: bool
    points: int
    failed: list[int]


class SimulateResponse(BaseModel):
    """The answer of a simulation of a panel, see the module.

    Attributes:
        contract: the version of the payloads, `sbmlsim.panel.CONTRACT`.
        spec_hash: the sha256 of the canonical JSON of the resolved request,
            which identifies what was simulated.
        x: the output times, in the unit `units["x"]` names.
        dims: the axes of the grid, empty for a run of one point.
        curves: the values of every declared curve, a row per point of the
            grid in C order; a value which is not finite is `null`.
        units: the unit of every curve and, under `"x"`, of the x.
        provenance: where the curves come from.
        status: the outcome of the run.
    """

    model_config = _CONFIG

    contract: str = CONTRACT
    spec_hash: str
    x: list[float]
    dims: list[GridAxisPayload]
    curves: dict[str, list[list[float | None]]]
    units: dict[str, str]
    provenance: Provenance
    status: Status

    @classmethod
    def of(
        cls, result: PanelResult, *, spec_hash: str, provenance: Provenance
    ) -> SimulateResponse:
        """Get the payload of the result of a run.

        Args:
            result: the result of the run.
            spec_hash: the hash of the resolved request.
            provenance: where the curves come from.

        Returns:
            The response, the values which are not finite as `null`.
        """
        points = math.prod(len(axis.values) for axis in result.dims)
        return cls(
            spec_hash=spec_hash,
            x=list(result.x),
            dims=[GridAxisPayload.of(axis) for axis in result.dims],
            curves={
                name: [_finite(row) for row in rows]
                for name, rows in result.curves.items()
            },
            units=dict(result.units),
            provenance=provenance,
            status=Status(
                ok=not result.failed,
                points=points,
                failed=list(result.failed),
            ),
        )


class ModelPayload(BaseModel):
    """A model the service serves.

    Attributes:
        id: the id of the model, the one a request names.
        title: the model as it is shown.
    """

    model_config = _CONFIG

    id: str
    title: str


class ModelsResponse(BaseModel):
    """The models the service serves, in the order they are served.

    Attributes:
        contract: the version of the payloads, `sbmlsim.panel.CONTRACT`.
        models: the models.
    """

    model_config = _CONFIG

    contract: str = CONTRACT
    models: list[ModelPayload]


class PanelPayload(BaseModel):
    """A panel of a model as it is listed, without its knobs and curves.

    Attributes:
        id: the id of the panel, the one a request names.
        title: the title of the panel.
        subtitle: the question the panel answers.
    """

    model_config = _CONFIG

    id: str
    title: str
    subtitle: str


class PanelsResponse(BaseModel):
    """The panels a model declares, in the order it declares them.

    Attributes:
        contract: the version of the payloads, `sbmlsim.panel.CONTRACT`.
        model: the id of the model.
        panels: the panels, see `PanelPayload`; the declaration of one is
            `sbmlsim.panel.panel_payload`.
    """

    model_config = _CONFIG

    contract: str = CONTRACT
    model: str
    panels: list[PanelPayload]


class HealthResponse(BaseModel):
    """The answer of the health check of the service.

    Attributes:
        status: `"ok"` for a service which is up.
        sbmlsim: the version of sbmlsim which runs the simulations.
        models: the number of models the service serves.
    """

    model_config = _CONFIG

    status: str
    sbmlsim: str
    models: int


def _finite(values: list[float]) -> list[float | None]:
    """Get the values of a row with the ones which are not finite as `None`.

    JSON has no `NaN` and no infinity: a point which did not integrate is
    `NaN` in a `PanelResult` and `null` in a response, as the run layer writes
    them.

    Args:
        values: the values of one row of a curve.

    Returns:
        The values, a value which is not finite as `None`.
    """
    return [value if math.isfinite(value) else None for value in values]
