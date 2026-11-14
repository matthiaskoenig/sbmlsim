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
one source of the contract and this body is its schema. Next to it the body
carries what the service derives from the panels of each page and is no part of
the declaration: the knobs the panels of a page share and the ones a scenario
of the page sets (`shared`, see `sbmlsim.serve.runner.ModelEntry.shared`),
which a shell shows one control of,
and the title and the subtitle of every panel of a page (`heads`), which a
shell reserves the head of each card for before the panel has read its own
declaration, so a card does not move when it does.

The simulation experiments of a model are not awaited but submitted
(`sbmlsim.serve.jobs`): an `ExperimentsRequest` names the groups and the
experiment classes which run, at most `MAX_EXPERIMENTS` of them, and the
answer is the `JobResponse` of the job which was created, which a caller
follows by its id until it is `done` or `failed` and which carries the figures
of a finished one.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sbmlsim.panel import APP_CONTRACT, CONTRACT, MAX_GRID, MAX_KNOBS, app_payload
from sbmlsim.serve.jobs import MAX_EXPERIMENTS, JobStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sbmlsim.panel import App, Panel
    from sbmlsim.serve.data import PanelSeries
    from sbmlsim.serve.jobs import Job
    from sbmlsim.serve.runner import GridAxis, PanelResult, SharedKnob

__all__ = [
    "DATA_CONTRACT",
    "AppPagePayload",
    "AppResponse",
    "AppVariantPayload",
    "ExperimentsRequest",
    "GridAxisPayload",
    "HealthResponse",
    "JobResponse",
    "ModelPayload",
    "ModelsResponse",
    "PanelDataResponse",
    "PanelPayload",
    "PanelsResponse",
    "Provenance",
    "ScenarioPayload",
    "SeriesPayload",
    "SharedKnobPayload",
    "SharedPanelPayload",
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
        variant: the variant of the model to simulate (`Project.variants`),
            which an app switches between; the one the panel declares
            (`PanelSimulation.variant`) without. The knobs, the curves and the
            study data of a panel do not depend on it, only the SBML which is
            integrated does.
    """

    model_config = _CONFIG

    values: dict[str, _Value] = Field(default_factory=dict, max_length=MAX_KNOBS)
    grid: dict[str, _Points] = Field(default_factory=dict, max_length=MAX_KNOBS)
    variant: str | None = None

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
        variant: the name of the SBML which was simulated, the one the
            request asked for, else the one the panel declares, else the
            first variant of the project, which is its default; it is
            resolved, so a response of a simulation always names one.
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


class ScenarioPayload(BaseModel):
    """A named state of the knobs of a page, see `AppPagePayload`.

    Attributes:
        id: the id of the scenario, distinct within its page.
        label: the scenario as it is shown.
        values: the value of every knob the scenario sets, by the id of the
            knob, in the unit of the knob; every one a knob of `shared` of
            its page, which a shell sets all at once.
    """

    model_config = _CONFIG

    id: str
    label: str
    values: dict[str, float]


class AppPagePayload(BaseModel):
    """One page of the app of a model, see `AppResponse`.

    Attributes:
        id: the id of the page, the one a link of the app names.
        title: the page as it is shown, its entry in the navigation.
        panels: the ids of the panels of the page, in the order they are
            shown; every one a panel the model declares, which a shell asks
            for with `GET /api/models/{model}/panels/{panel}`.
        scenarios: the named states of the knobs of the page a reader picks
            from, in the order they are shown; none for a page without.
    """

    model_config = _CONFIG

    id: str
    title: str
    panels: list[str]
    scenarios: list[ScenarioPayload]


class AppVariantPayload(BaseModel):
    """One variant of the model of an app, see `AppResponse`.

    Attributes:
        id: the id of the variant, a variant of the model.
        label: the variant as it is shown.
    """

    model_config = _CONFIG

    id: str
    label: str


class SharedPanelPayload(BaseModel):
    """A panel a shared knob of a page sets, see `SharedKnobPayload`.

    Attributes:
        id: the id of the panel, one of the panels of the page.
        title: the title of the panel, which names it to a reader of the
            control which sets it.
    """

    model_config = _CONFIG

    id: str
    title: str


class SharedKnobPayload(BaseModel):
    """A knob the shell of a page owns, see `AppResponse`.

    It is a knob more than one panel of the page declares, or one a scenario
    of the page sets, which the shell owns even when one panel declares it,
    since it sets it, see `sbmlsim.serve.runner.ModelEntry.shared`.

    Attributes:
        knob: its declaration, an item of the `axes` of the declaration of
            every panel which declares it (`sbmlsim.panel.panel_payload`),
            which those panels declare alike.
        panels: the panels of the page which declare it, in the order the
            page shows them, at least one.
    """

    model_config = _CONFIG

    knob: dict[str, Any]
    panels: list[SharedPanelPayload] = Field(min_length=1)

    @classmethod
    def of(cls, shared: SharedKnob) -> SharedKnobPayload:
        """Get the payload of a shared knob.

        Args:
            shared: the knob and the panels which declare it.

        Returns:
            The payload.
        """
        return cls(
            knob=shared.knob.model_dump(mode="json"),
            panels=[
                SharedPanelPayload(id=panel.id, title=panel.title)
                for panel in shared.panels
            ],
        )


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
        shared: the knobs the shell of a page owns, by the id of the page,
            each in the order the first panel which declares one lists them:
            the knobs the panels of the page share and the knobs a scenario
            of the page sets; every page has an entry, empty for a page whose
            panels share none and which has no scenario. They are derived
            from the panels by the service and are no part of the declaration
            of the app.
        heads: the panels of a page with their titles and subtitles, by the id
            of the page, in the order the page shows them; every page has an
            entry. They are the heads of the declarations of the panels, which
            a shell lays the head of each card out with before the panel has
            read its declaration, and are no part of the declaration of the
            app either.
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
    shared: dict[str, list[SharedKnobPayload]]
    heads: dict[str, list[PanelPayload]]

    @classmethod
    def of(
        cls,
        app: App,
        shared: Mapping[str, Sequence[SharedKnob]],
        pages: Mapping[str, Sequence[Panel]],
    ) -> AppResponse:
        """Get the payload of the app of a model.

        The body is read from `sbmlsim.panel.app_payload`, the one place the
        contract of the app is written, and this model is its schema: a field
        which the two do not share is refused here (`extra="forbid"`), so the
        document of the API cannot drift from the payload. The shared knobs
        and the heads of the panels are added to it, which the payload of the
        declaration does not carry.

        Args:
            app: the app the project of the model declares.
            shared: the knobs the panels of each page share, by the id of the
                page, see `sbmlsim.serve.runner.ModelEntry.shared`.
            pages: the panels of each page, by the id of the page, see
                `sbmlsim.serve.runner.ModelEntry.pages`.

        Returns:
            The response.
        """
        knobs = {
            page.id: [SharedKnobPayload.of(one) for one in shared.get(page.id, ())]
            for page in app.pages
        }
        heads = {
            page.id: [
                PanelPayload(id=panel.id, title=panel.title, subtitle=panel.subtitle)
                for panel in pages.get(page.id, ())
            ]
            for page in app.pages
        }
        return cls.model_validate({**app_payload(app), "shared": knobs, "heads": heads})


class ExperimentsRequest(BaseModel):
    """The body of a submission of the experiments of a model, see the module.

    Attributes:
        experiments: the names of the groups and of the experiment classes
            which run, at least one and at most `MAX_EXPERIMENTS`; the group
            `all` is every experiment of the model. A name the model does not
            have is a 422 and submits nothing, see
            `sbmlsim.serve.jobs.check_experiments`.
    """

    model_config = _CONFIG

    experiments: list[str] = Field(min_length=1, max_length=MAX_EXPERIMENTS)


class JobResponse(BaseModel):
    """One run of the simulation experiments of a model, see the module.

    Attributes:
        contract: the version of the payloads, `sbmlsim.panel.CONTRACT`.
        id: the id of the job, which a caller follows it by.
        model: the id of the model whose experiments run.
        experiments: the names the submission gave, in its order.
        status: `queued`, `running`, `done` or `failed`.
        created: when the job was submitted, an ISO 8601 timestamp in UTC.
        message: why a job failed, empty for every other status; it names no
            path and carries no traceback.
        figures: the figures the job collected, as the paths a request of one
            of them names (`GET /api/jobs/{job_id}/figures/{path}`); a run
            collects them at its end, so a job which is not finished usually
            has none.
    """

    model_config = _CONFIG

    contract: str = CONTRACT
    id: str
    model: str
    experiments: list[str]
    status: JobStatus
    created: str
    message: str
    figures: list[str]

    @classmethod
    def of(cls, job: Job, figures: Sequence[str] = ()) -> JobResponse:
        """Get the payload of a job.

        Args:
            job: the record of the job as it is now.
            figures: the figures it collected, none for a job which is not
                finished.

        Returns:
            The response.
        """
        return cls(
            id=job.id,
            model=job.model,
            experiments=list(job.experiments),
            status=job.status,
            created=job.created,
            message=job.message,
            figures=list(figures),
        )


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
