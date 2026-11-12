"""The bodies of the HTTP API: what a request may ask for and what it answers.

A request of a simulation is a `SimulateRequest`: the value of every knob a
reader set and the points of every knob a grid runs over. The limits of
`sbmlsim.panel` are the bounds of its fields and of its validator, so a request
over them is refused where it is read, before a handler runs and before a model
is loaded: at most `MAX_KNOBS` values and `MAX_KNOBS` axes, at least one and at
most `MAX_GRID` points per axis and at most `MAX_GRID` points in all. A knob id
the panel does not declare is refused by
`sbmlsim.serve.runner.PanelRunner.run`, which knows the panel, and is a 422 as
well.

The answer is a `SimulateResponse`, the `PanelResult` of that run with its
`spec_hash`, which identifies the resolved request, its `Provenance` and its
`Status`. Every response body carries the `contract` of the payloads
(`sbmlsim.panel.CONTRACT`), so a renderer reads one version of them.

JSON has no `NaN`: a point which did not integrate is `null` in the curves and
its flat index is in `Status.failed`, which is where a renderer reads it.

The study data behind the curves is a `PanelDataResponse` of the `PanelSeries`
of `sbmlsim.serve.data`, which carries the contract `DATA_CONTRACT` of its own:
the series are no part of the declaration of a panel and a renderer which draws
them reads their version.

The app of a model is an `AppResponse` of the `sbmlsim.panel.App` its project
declares, the contract `sbmlsim.panel.APP_CONTRACT`: the pages of its panels,
which a shell places. Its shape is the one `sbmlsim.panel.app_payload` writes,
which it reads back field by field (`AppResponse.of`), so the payload is the
one source of the contract and this body is its schema.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sbmlsim.panel import APP_CONTRACT, CONTRACT, MAX_GRID, MAX_KNOBS, app_payload

if TYPE_CHECKING:
    from sbmlsim.panel import App
    from sbmlsim.serve.data import PanelSeries
    from sbmlsim.serve.runner import GridAxis, PanelResult

__all__ = [
    "DATA_CONTRACT",
    "AppPagePayload",
    "AppResponse",
    "AppVariantPayload",
    "GridAxisPayload",
    "HealthResponse",
    "ModelPayload",
    "ModelsResponse",
    "PanelDataResponse",
    "PanelPayload",
    "PanelsResponse",
    "Provenance",
    "SeriesPayload",
    "SimulateRequest",
    "SimulateResponse",
    "Status",
]

#: the version of the wire contract of the study data of a panel, in the body
#: of every answer of it, see the module
DATA_CONTRACT: Final = "paneldata/1"

#: a body of the API forbids unknown fields and does not change
_CONFIG = ConfigDict(extra="forbid", frozen=True)

#: the value of one knob, see `SimulateRequest`; `json.loads` reads `NaN` and
#: `Infinity`, which are no value of a knob and are refused at the edge
type _Value = Annotated[float, Field(allow_inf_nan=False)]

#: the points of one axis of a grid, see `SimulateRequest`
type _Points = Annotated[int, Field(ge=1, le=MAX_GRID)]


class SimulateRequest(BaseModel):
    """The body of a simulation of a panel, see the module.

    Attributes:
        values: the value of a knob by its id, in the unit of the knob, at
            most `MAX_KNOBS` of them, which is the knobs of a request, and
            every one a number; a knob which a request leaves out runs at its
            `default`.
        grid: the points of a knob of the grid by its id, at most `MAX_KNOBS`
            knobs and `MAX_GRID` points in all; a knob of the grid takes no
            value.
    """

    model_config = _CONFIG

    values: dict[str, _Value] = Field(default_factory=dict, max_length=MAX_KNOBS)
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
        units: the unit of every curve and, under `"x"`, of the x, `null`
            for values which carry none. They correspond to the `unit` and the
            `x_unit` of the series of `paneldata/1` of the same panel, which
            are `null` for no unit as well, so the two sides of an axis state
            one unit.
        provenance: where the curves come from.
        status: the outcome of the run.
    """

    model_config = _CONFIG

    contract: str = CONTRACT
    spec_hash: str
    x: list[float]
    dims: list[GridAxisPayload]
    curves: dict[str, list[list[float | None]]]
    units: dict[str, str | None]
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


class SeriesPayload(BaseModel):
    """One series of the study data of a panel, the payload of a `PanelSeries`.

    Attributes:
        id: the id of the series, the one of its declaration, which names the
            curve it belongs under (`sbmlsim.panel.PanelData.curve`).
        label: the series as it is shown, e.g. the study it comes from.
        x: the times of the points, in the unit `x_unit` names, which is the
            unit of the x of a simulation of the same panel.
        y: the values of the points, in the unit `unit` names.
        yerr: the error of every value, `null` for a point without one, and
            `null` for a series without a column of the error.
        x_unit: the unit of the times, `null` for times without one, i.e. a
            model whose time has no unit and a dataset whose time column has
            none either.
        unit: the unit of the values, the one of the curve the series belongs
            under, which is the unit a simulation of the panel delivers that
            curve in; `null` for a curve which carries no unit and for a
            series which is delivered in the unit of its dataset and whose
            column has none.
        count: the number of subjects behind every point, `null` for a dataset
            which does not carry them.
    """

    model_config = _CONFIG

    id: str
    label: str
    x: list[float]
    y: list[float]
    yerr: list[float | None] | None
    x_unit: str | None
    unit: str | None
    count: list[int] | None

    @classmethod
    def of(cls, series: PanelSeries) -> SeriesPayload:
        """Get the payload of a series of the study data.

        Args:
            series: the series.

        Returns:
            The payload; a series carries no value which is not finite, see
            `sbmlsim.serve.data`.
        """
        return cls(
            id=series.id,
            label=series.label,
            x=list(series.x),
            y=list(series.y),
            yerr=None if series.yerr is None else list(series.yerr),
            x_unit=series.x_unit,
            unit=series.unit,
            count=None if series.count is None else list(series.count),
        )


class PanelDataResponse(BaseModel):
    """The study data behind the curves of a panel, see the module.

    Attributes:
        contract: the version of the payloads of the data, `DATA_CONTRACT`.
        series: the series the panel declares, in its order; none for a panel
            without study data.
    """

    model_config = _CONFIG

    contract: str = DATA_CONTRACT
    series: list[SeriesPayload]


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


class AppPagePayload(BaseModel):
    """One page of the app of a model, see `AppResponse`.

    Attributes:
        id: the id of the page, the one a link of the app names.
        title: the page as it is shown, its entry in the navigation.
        panels: the ids of the panels of the page, in the order they are
            shown; every one a panel the model declares, which a shell asks
            for with `GET /api/models/{model}/panels/{panel}`.
    """

    model_config = _CONFIG

    id: str
    title: str
    panels: list[str]


class AppVariantPayload(BaseModel):
    """One variant of the model of an app, see `AppResponse`.

    Attributes:
        id: the id of the variant, a variant of the model.
        label: the variant as it is shown.
    """

    model_config = _CONFIG

    id: str
    label: str


class AppResponse(BaseModel):
    """The app of a model: the pages of its panels, see the module.

    Attributes:
        contract: the version of the payloads of the app,
            `sbmlsim.panel.APP_CONTRACT`.
        title: the title of the app, the model as it is shown.
        subtitle: what the model is, shown below the title.
        pages: the pages of the app, in the order they are shown.
        variants: the variants of the model a reader switches between, in the
            order they are shown; none for an app of the default variant
            alone.
        citation: the citation of the model, shown in the provenance footer.
        doi: the DOI of the model, shown in that footer.
        licenses: the licenses of the model and its data, shown in that
            footer.
    """

    model_config = _CONFIG

    contract: str = APP_CONTRACT
    title: str
    subtitle: str
    pages: list[AppPagePayload]
    variants: list[AppVariantPayload]
    citation: str
    doi: str
    licenses: list[str]

    @classmethod
    def of(cls, app: App) -> AppResponse:
        """Get the payload of the app of a model.

        The body is read from `sbmlsim.panel.app_payload`, the one place the
        contract of the app is written, and this model is its schema: a field
        which the two do not share is refused here (`extra="forbid"`), so the
        document of the API cannot drift from the payload.

        Args:
            app: the app the project of the model declares.

        Returns:
            The response.
        """
        return cls.model_validate(app_payload(app))


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
