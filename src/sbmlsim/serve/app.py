"""The HTTP API over the panels of the served models.

`create_app(registry)` is the ASGI application of a `ModelRegistry`: it serves
the declaration of the panels of its models (`sbmlsim.panel.panel_payload`,
the contract `panel/1`), the app of a model, the pages of these panels
(`sbmlsim.panel.app_payload`, the contract `app/1`), the study data behind
their curves (`sbmlsim.serve.data`, the contract `paneldata/1`) and runs their
simulations on one `PanelRunner` with warm models.

A request names a model id of the registry and a panel id of that model and
nothing else: no project reference, no module and no path reaches the service,
and an id which is not served is a 404 with the message of the registry. The
declaration of a panel may differ by the variant of its model, so the panel,
its study data, the app and a simulation take a variant (the query `variant` of
the routes which read, the field of a `SimulateRequest`), whose declaration
they are, the default variant without; a variant the model does not have is a
422. A request which is over the limits of `sbmlsim.panel` or which names a
knob the panel does not declare is a 422, a failure of the service is a 500 of
one fixed message, logged with its traceback: an error body carries no
traceback, no path and nothing of the configuration of the service.

The route of a simulation is a sync `def`, which the server runs in its thread
pool, and the runner holds the lock of its model while it simulates, so the
event loop keeps answering while a model integrates.

Every answer of a simulation is kept in the `ResponseCache` of the application
(`sbmlsim.serve.cache`) under its `spec_hash`, which identifies what was
simulated: the same request is answered with the stored body instead of being
simulated again, so a presentation of many viewers costs one simulation, and
`GET /api/sim/{spec_hash}` serves that body by its hash alone, as immutable
(`CACHE_CONTROL`). A hash which is not stored, whatever its shape, is a 404 of
one fixed message (`NOT_STORED`), so the route tells a caller nothing but
whether it holds the body it asks for.

The simulation experiments of a model are minutes and not seconds, so they are
submitted and not awaited: `POST /api/models/{model}/experiments` answers a
202 with the `Job` the `JobStore` of the application created
(`sbmlsim.serve.jobs`), `GET /api/jobs/{job_id}` is that job as it is now and
`GET /api/jobs/{job_id}/figures/{path}` serves a figure it collected, by a
path of the job and never a path of the host.

`static` serves a directory under the API, so the local mode delivers the
javascript component of the panels from the origin of the API, and
`allow_origins` lets the named origins talk to it from another one.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import uuid
import weakref
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import anyio.to_thread
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sbmlsim
from sbmlsim.hashing import canonical_json_sha256, file_sha256
from sbmlsim.panel import CONTRACT, Panel, panel_payload
from sbmlsim.serve.cache import ResponseCache
from sbmlsim.serve.data import panel_data
from sbmlsim.serve.jobs import JobStore, TooManyJobs, UnknownExperiment
from sbmlsim.serve.payload import (
    AppResponse,
    ExperimentsRequest,
    HealthResponse,
    JobResponse,
    ModelPayload,
    ModelsResponse,
    PanelDataResponse,
    PanelPayload,
    PanelsResponse,
    Provenance,
    SeriesPayload,
    SimulateRequest,
    SimulateResponse,
)
from sbmlsim.serve.runner import (
    ModelEntry,
    ModelRegistry,
    PanelRunner,
    UnknownKnob,
    UnknownModel,
    UnknownPanel,
    UnknownVariant,
    check_app_variant,
    check_knobs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BUSY",
    "CACHE_CONTROL",
    "FAILED",
    "JOBS_PREFIX",
    "NOT_STORED",
    "NO_APP",
    "NO_FIGURE",
    "NO_JOB",
    "create_app",
]

#: the body of a failure of the service, which carries nothing of it
FAILED: Final = "The request failed in the service."

#: the body of a model which declares no app, see `create_app.app_of`; it
#: names neither the model nor its project, as every error body of the service
NO_APP: Final = "This model declares no app."

#: the body of a hash which is not stored, see `create_app.stored`; a hash of
#: another shape is answered with it as well, so the route is no surface a
#: caller learns anything from but whether it holds the body it asks for
NOT_STORED: Final = "No response is stored under this hash."

#: the caching of a stored response, which is the one answer of its hash: a
#: year is what `max-age` takes at most in practice, `immutable` says that it
#: is never revalidated and `public` lets a proxy of a presentation keep it
CACHE_CONTROL: Final = "public, max-age=31536000, immutable"

#: the body of a job which the service does not hold, see `create_app.job_of`;
#: a job of another application and one of a service which was restarted are
#: answered with it as well, since the jobs are the ones of an application
NO_JOB: Final = "No job is stored under this id."

#: the body of a path which is no figure of a job, see `create_app.figure_of`;
#: a path which leaves the directory of the job is answered with it as well, so
#: the route tells a caller nothing of the files of the host
NO_FIGURE: Final = "This job has no such figure."

#: the body of a submission to a service which runs as many experiments as it
#: keeps in flight, see `create_app.submit_experiments`; it names neither the
#: jobs which run nor what the service is made of
BUSY: Final = "The service runs as many experiments as it can at once."

#: the prefix of the name of the directory of the jobs of an application which
#: was given no store of its own, which it removes when it shuts down, see
#: `create_app` and `_temporary_root`
JOBS_PREFIX: Final = "sbmlsim-jobs-"

#: the shape of a hash of the API, the sha256 of
#: `sbmlsim.hashing.canonical_json_sha256`, checked before anything is looked
#: up, see `create_app.stored`
HASH: Final = re.compile(r"[0-9a-f]{64}")

#: the title of the application, which its OpenAPI document carries
TITLE: Final = "sbmlsim panels"

#: the errors of an invalid body which are answered, see `create_app.invalid`;
#: a body of many values has an error per value, which would answer a large
#: request with a larger response
MAX_ERRORS: Final = 10


def create_app(
    registry: ModelRegistry,
    *,
    static: Path | None = None,
    allow_origins: Sequence[str] = (),
    jobs: JobStore | None = None,
) -> FastAPI:
    """Create the application which serves the panels of a registry.

    The models are the ones of the registry and are loaded when a panel of one
    of them is simulated first, by the one `PanelRunner` of the application,
    which keeps them warm.

    Args:
        registry: the models a request may name, see `ModelRegistry`.
        static: a directory served under the API, e.g. the javascript
            component of the panels and the pages which embed it; none is
            served without.
        allow_origins: the origins which may talk to the API from another
            origin (CORS); without one the API answers its own origin only.
        jobs: the store of the runs of the simulation experiments, see
            `sbmlsim.serve.jobs`; without one the application creates a store
            over a temporary directory of its own (`_temporary_root`, which is
            created with the first job) and removes that directory when it
            shuts down, while a store which is given it belongs to its caller
            and is neither closed nor removed.

    Returns:
        The application, whose store of the jobs is `app.state.jobs`.
    """
    store = jobs if jobs is not None else JobStore(_temporary_root())
    #: the store this application created itself, which it alone removes
    owned = None if jobs is not None else store
    if owned is not None:
        # the lifespan below is the orderly way out; this one covers a process
        # which ends another way than through the shutdown of its server,
        # where every finalizer runs, and an application which is collected
        # before that, e.g. one which never started a server at all
        weakref.finalize(owned, _remove_jobs, owned.root)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        """Remove the jobs of an application which created their store itself.

        The worker of the store is stopped first and the job which runs is
        waited for, so the directory is complete and nothing writes into it
        while it is removed. The wait is a thread of its own: it is the length
        of a run of simulation experiments, which must not block the event
        loop of the server while it shuts down.

        Args:
            application: the application, which serves until it shuts down.

        Yields:
            Nothing; what follows the `yield` is the shutdown.
        """
        try:
            yield
        finally:
            if owned is not None:
                await anyio.to_thread.run_sync(owned.close)
                _remove_jobs(owned.root)
                logger.debug("The jobs of the service were removed.")

    app = FastAPI(title=TITLE, version=sbmlsim.__version__, lifespan=lifespan)
    # the store of the jobs of this application, which its routes close over
    app.state.jobs = store
    runner = PanelRunner(registry)
    #: the answers of the simulations of this application alone, see
    #: `sbmlsim.serve.cache`
    cache = ResponseCache()
    #: the sha256 of every SBML file which was simulated, by its resolved path
    hashes: dict[Path, str] = {}

    def resolved(
        model_id: str, panel_id: str, variant: str | None = None
    ) -> tuple[ModelEntry, Panel, str]:
        """Get the model and the panel a request names, and the hash of its SBML.

        Args:
            model_id: the id of the model.
            panel_id: the id of the panel.
            variant: the variant a request asks for, whose declaration of the
                panel it is and whose SBML is simulated; without, the
                declaration of the default variant, which simulates the
                variant it declares.

        Returns:
            The entry, the panel of the variant and the sha256 of the SBML file
            which is simulated.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownPanel: for a panel the model does not declare, a 404.
            UnknownVariant: for a variant the project does not have.
            HTTPException: a 500 for a model whose panels or whose SBML cannot
                be read, which is a problem of the service and not of the
                request.
        """
        try:
            entry = registry.entry(model_id)
            panel = registry.panel(model_id, panel_id, variant)
            path = entry.project.variant_path(
                panel.simulation.variant if variant is None else variant
            ).resolve()
            sha = hashes.get(path)
            if sha is None:
                sha = file_sha256(path)
                hashes[path] = sha
            return entry, panel, sha
        except UnknownModel, UnknownPanel, UnknownVariant:
            raise
        except Exception:
            logger.exception(
                "The panel '%s' of the model '%s' cannot be read.", panel_id, model_id
            )
            raise HTTPException(status_code=500, detail=FAILED) from None

    @app.get("/healthz")
    async def healthz() -> HealthResponse:
        """Answer that the service is up, without loading a model."""
        return HealthResponse(
            status="ok",
            sbmlsim=sbmlsim.__version__,
            models=len(registry.ids()),
        )

    @app.get("/api/models")
    async def models() -> ModelsResponse:
        """List the models the service serves, in the order they are served."""
        return ModelsResponse(
            models=[
                ModelPayload(id=entry.id, title=entry.title)
                for entry in (registry.entry(model_id) for model_id in registry.ids())
            ]
        )

    @app.get("/api/models/{model}/app")
    def app_of(model: str, variant: str | None = None) -> AppResponse:
        """Get the app of a model, the pages of its panels.

        The app is the declaration of the project of the model
        (`sbmlsim.project.Project.app`), and the knobs the shell of each of
        its pages owns and the heads of its panels are derived from the
        panels of the model in a variant (`ModelEntry.shared`,
        `ModelEntry.pages`), which reads the SBML of the variant once: the
        registry of a service did so when it was built (`ModelRegistry.of`),
        and a registry built otherwise does it on the first request, which is
        why the route is a sync `def`. Its pages name the panels a shell then
        asks for, which the registry checked against the panels of every
        variant the app is served in.

        Args:
            model: the id of the model.
            variant: the variant whose panels the knobs and the heads are
                worked out of, one of `ModelEntry.app_variants`; the default
                variant without.

        Returns:
            The app, see `sbmlsim.panel.app_payload`.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownVariant: a 422 for a variant the project does not have or
                the app does not name.
            HTTPException: a 404 of `NO_APP` for a model which declares no
                app, which is a model whose panels are served on their own.
        """
        entry = registry.entry(model)
        declared = entry.project.app
        if declared is None:
            raise HTTPException(status_code=404, detail=NO_APP)
        check_app_variant(entry, variant)
        return AppResponse.of(declared, entry.shared(variant), entry.pages(variant))

    @app.get("/api/models/{model}/panels")
    def panels(model: str) -> PanelsResponse:
        """List the panels a model declares, in the order it declares them.

        The panels are the ones of the default variant of the model: the list
        names the panels a model has, while the declaration of one of them in
        another variant is served by `panel_of` with its `variant`. They are
        read from its SBML once, so the route is a sync `def` and reads it in
        the thread pool of the server.

        Args:
            model: the id of the model.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            HTTPException: a 500 for a model whose panels cannot be read.
        """
        try:
            declared = registry.entry(model).project.panels_of()
        except UnknownModel:
            raise
        except Exception:
            logger.exception("The panels of the model '%s' cannot be read.", model)
            raise HTTPException(status_code=500, detail=FAILED) from None
        return PanelsResponse(
            model=model,
            panels=[
                PanelPayload(id=panel.id, title=panel.title, subtitle=panel.subtitle)
                for panel in declared.values()
            ],
        )

    @app.get("/api/models/{model}/panels/{panel}")
    def panel_of(model: str, panel: str, variant: str | None = None) -> dict[str, Any]:
        """Get the declaration of a panel, the payload of the contract.

        The declaration and the hash of the model are read from the files of
        the model once, so the route is a sync `def` and reads them in the
        thread pool of the server, as `simulate` does.

        Args:
            model: the id of the model.
            panel: the id of the panel.
            variant: the variant whose declaration of the panel it is, see
                `sbmlsim.project.Project.panels_of`; the default variant
                without.

        Returns:
            The payload, see `sbmlsim.panel.panel_payload`.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownPanel: for a panel the variant does not declare, a 404.
            UnknownVariant: a 422 for a variant the project does not have.
        """
        _, declaration, _ = resolved(model, panel, variant)
        return panel_payload(declaration)

    @app.get("/api/models/{model}/panels/{panel}/data")
    def data_of(
        model: str, panel: str, variant: str | None = None
    ) -> PanelDataResponse:
        """Get the study data behind the curves of a panel.

        The data is read from the data of the project of the model, so the
        route is a sync `def` and reads the files in the thread pool of the
        server, as `panel_of` does.

        The series are delivered in the units of the panel
        (`PanelRunner.units`), which the model of the panel resolves: the
        times in the unit the x of a simulation of the same panel is
        delivered in and the values in the unit of the curve they belong
        under, so a reader draws the points on the axes of the curves. A panel
        which declares no study data is answered without its model, every
        other one with the load a run of the panel pays as well.

        Args:
            model: the id of the model.
            panel: the id of the panel.
            variant: the variant whose declaration of the panel the series are
                and whose model resolves their units, as `simulate` does with
                its `variant`; the default variant without.

        Returns:
            The series of the panel, see `sbmlsim.serve.data`; none for a
            panel which declares no study data.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownPanel: for a panel the variant does not declare, a 404.
            UnknownVariant: a 422 for a variant the project does not have.
            HTTPException: a 500 for study data which cannot be read, e.g. a
                dataset which is no file of the project or a column it has
                not, which is a problem of the declaration and not of the
                request; the message is the one of the service and carries
                neither the dataset nor a path.
        """
        entry, declaration, _ = resolved(model, panel, variant)
        if not declaration.data:
            return PanelDataResponse(series=[])
        try:
            units = runner.units(entry, declaration, variant)
            series = panel_data(entry, declaration, units)
        except Exception:
            logger.exception(
                "The study data of the panel '%s' of the model '%s' cannot be read.",
                panel,
                model,
            )
            raise HTTPException(status_code=500, detail=FAILED) from None
        return PanelDataResponse(series=[SeriesPayload.of(one) for one in series])

    @app.post("/api/models/{model}/panels/{panel}/simulate")
    def simulate(model: str, panel: str, request: SimulateRequest) -> SimulateResponse:
        """Run a panel at the values of its knobs or over a grid of them.

        The route is a sync `def`, which the server runs in its thread pool,
        see the module.

        Args:
            model: the id of the model.
            panel: the id of the panel.
            request: the values of the knobs and the grid, which is read
                against the limits of `sbmlsim.panel` before this runs.

        Returns:
            The curves of the run with their provenance and the hash of the
            resolved request; the stored answer of a request which was
            simulated already, as it is, see `sbmlsim.serve.cache`.

        The declaration which runs is the one of the variant of the request
        (`sbmlsim.project.Project.panels_of`), simulated on the SBML of that
        variant, and the one of the default variant without, simulated on the
        variant it declares.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownPanel: for a panel the variant does not declare, a 404.
            UnknownVariant: a 422 for a variant the project does not have,
                checked before the cache is read.
            HTTPException: a 422 for a knob id the panel does not declare,
                which is the error of a request a run is answered with and
                which is checked before the cache is read, and a 500 for every
                other failure of the run, whose message is the one of the
                service and not of the exception.
        """
        entry, declaration, sha = resolved(model, panel, request.variant)
        # the name of the SBML which runs, so a request which leaves the
        # variant out and one which names the default are one simulation
        variant = (
            request.variant
            or declaration.simulation.variant
            or next(iter(entry.project.variants))
        )
        try:
            check_knobs(declaration, request.values, request.grid)
            spec_hash = _spec_hash(entry, declaration, request, sha, variant)
            cached = cache.get(spec_hash)
            if cached is not None:
                return cached
            started = time.perf_counter()
            result = runner.run(
                model, declaration, request.values, request.grid, variant
            )
            seconds = time.perf_counter() - started
            settings = entry.project.simulator
            response = SimulateResponse.of(
                result,
                spec_hash=spec_hash,
                provenance=Provenance(
                    model=entry.id,
                    panel=declaration.id,
                    model_sha256=sha,
                    variant=variant,
                    engine=settings.engine.value,
                    sbmlsim=sbmlsim.__version__,
                    seconds=seconds,
                    created=datetime.now(UTC).isoformat(),
                ),
            )
        except UnknownKnob as error:
            # the ids which are no knobs of the panel, the one message of a
            # run which carries nothing of the service
            raise HTTPException(status_code=422, detail=str(error)) from None
        except Exception:
            logger.exception(
                "The panel '%s' of the model '%s' did not run.", panel, model
            )
            raise HTTPException(status_code=500, detail=FAILED) from None
        # after the try, so a run which failed stores nothing and the next
        # request of it is run again
        cache.put(response)
        return response

    @app.get("/api/sim/{spec_hash}")
    def stored(spec_hash: str, reply: Response) -> SimulateResponse:
        """Get the answer of a simulation by the hash of its request.

        The body of a hash is the one answer of that request, see
        `_spec_hash`, so it is served as immutable and a reader who follows a
        link of a hash fetches it once.

        The shape of the hash is checked before anything is looked up, and a
        hash the service does not hold is answered with the same fixed message
        whatever its shape, so the route carries nothing of the service and is
        no surface to probe it with.

        Args:
            spec_hash: the `spec_hash` of a response, 64 lower case
                hexadecimal digits.
            reply: the response of the server, whose headers the answer takes.

        Returns:
            The response as it was stored, its provenance included: the
            seconds and the time of the run which computed it.

        Raises:
            HTTPException: a 404 for a hash which is not stored.
        """
        if HASH.fullmatch(spec_hash) is None:
            raise HTTPException(status_code=404, detail=NOT_STORED)
        response = cache.get(spec_hash)
        if response is None:
            raise HTTPException(status_code=404, detail=NOT_STORED)
        reply.headers["cache-control"] = CACHE_CONTROL
        return response

    @app.post("/api/models/{model}/experiments", status_code=202)
    def submit_experiments(model: str, request: ExperimentsRequest) -> JobResponse:
        """Submit the simulation experiments of a model as a job.

        The experiments of a model are minutes and not seconds, so the route
        answers the job which was created and not its result: a caller follows
        it with `GET /api/jobs/{job_id}` and fetches its figures once it is
        `done`.

        The route is a sync `def`, which the server runs in its thread pool:
        the names of the experiments are checked against the ones of the model
        before the job is created, which imports them.

        Args:
            model: the id of the model.
            request: the names of the groups and experiment classes which run.

        Returns:
            The job as it was created, `queued`, with a 202.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            HTTPException: a 422 for a name the model does not have, which
                submits nothing, a 429 for a service which runs as many
                experiments as it keeps in flight, and a 500 for every other
                failure of the submission, e.g. a model without simulation
                experiments, whose message is the one of the service.
        """
        entry = registry.entry(model)
        try:
            job = store.submit(entry, request.experiments)
        except UnknownExperiment as error:
            # the names which are no experiments of the model, the one message
            # of a submission which carries nothing of the service
            raise HTTPException(status_code=422, detail=str(error)) from None
        except TooManyJobs:
            raise HTTPException(status_code=429, detail=BUSY) from None
        except Exception:
            logger.exception(
                "The experiments %s of the model '%s' were not submitted.",
                request.experiments,
                model,
            )
            raise HTTPException(status_code=500, detail=FAILED) from None
        return JobResponse.of(job)

    @app.get("/api/jobs/{job_id}")
    def job_of(job_id: str) -> JobResponse:
        """Get a job of the service as it is now, with its figures.

        The figures are the ones which are in the directory of the job when it
        is asked for, read from that directory, so the route is a sync `def`
        and reads it in the thread pool of the server. A run collects them at
        its end, so a job which is not finished usually has none, and a poll
        which lands in that window sees some of them on a `running` job.

        Args:
            job_id: the id of the job, the one its submission answered.

        Returns:
            The job with the figures its directory holds.

        Raises:
            HTTPException: a 404 of `NO_JOB` for an id the service does not
                hold, which is every id of another application as well, since
                the jobs belong to the one which ran them.
        """
        job = store.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=NO_JOB)
        return JobResponse.of(job, store.figures(job_id))

    @app.get("/api/jobs/{job_id}/figures/{path:path}")
    def figure_of(job_id: str, path: str) -> FileResponse:
        """Get one figure of a job, by a path of the job itself.

        The path is the one the job lists (`JobResponse.figures`) and is
        resolved against the directory of that job: a path which leaves it,
        over `..`, as an absolute path or through a symbolic link, is a 404 of
        the same message as a figure which does not exist, so no path of the
        host is reachable and the route tells a caller nothing about one.

        Args:
            job_id: the id of the job.
            path: the path of the figure relative to the directory of the job,
                e.g. `_figures/Study_Fig1.svg`.

        Returns:
            The file, whose media type is the one of its name.

        Raises:
            HTTPException: a 404 of `NO_FIGURE` for an id the service does not
                hold and for a path which is no file of its job.
        """
        file = store.figure(job_id, path)
        if file is None:
            raise HTTPException(status_code=404, detail=NO_FIGURE)
        return FileResponse(file)

    async def unprocessable(request: Request, exc: Exception) -> JSONResponse:
        """Answer a variant which the model does not have with a 422.

        Args:
            request: the request which named it.
            exc: the `UnknownVariant` of the registry.

        Returns:
            The body `{"detail": ...}` of the message of the exception, which
            names the model and its variants and nothing of the service.
        """
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    async def not_found(request: Request, exc: Exception) -> JSONResponse:
        """Answer a model or a panel which is not served with a 404.

        Args:
            request: the request which named it.
            exc: the `UnknownModel` or `UnknownPanel` of the registry.

        Returns:
            The body `{"detail": ...}` of the message of the exception, which
            names the id the request gave and nothing of the service.
        """
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    async def invalid(request: Request, exc: Exception) -> JSONResponse:
        """Answer a body which is not a `SimulateRequest` with a 422.

        The body of the answer is what is wrong and not what was sent: the
        `type`, the `loc` and the `msg` of at most `MAX_ERRORS` errors of
        pydantic, never its `input` and never its `ctx`, which FastAPI's own
        handler repeats, so the answer of a large body is small and carries
        nothing back.

        Args:
            request: the request which was read.
            exc: the `RequestValidationError` of the body.

        Returns:
            The body `{"detail": [...]}` of the errors.
        """
        errors = exc.errors() if isinstance(exc, RequestValidationError) else []
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "type": str(error.get("type", "")),
                        "loc": [str(part) for part in error.get("loc", ())],
                        "msg": str(error.get("msg", "")),
                    }
                    for error in errors[:MAX_ERRORS]
                ]
            },
        )

    app.add_exception_handler(UnknownModel, not_found)
    app.add_exception_handler(UnknownPanel, not_found)
    app.add_exception_handler(UnknownVariant, unprocessable)
    app.add_exception_handler(RequestValidationError, invalid)
    if allow_origins:
        # `allow_headers` is open because the service takes no credential:
        # `allow_credentials` stays `False`, so a browser sends no cookie and
        # no authorization header of another origin
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allow_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    if static is not None:
        # after the routes of the API, which a mount at the root would shadow
        app.mount("/", StaticFiles(directory=static, html=True), name="static")
    logger.info("The service serves the models %s.", list(registry.ids()))
    return app


def _temporary_root() -> Path:
    """Get the directory of the jobs of an application which was given no store.

    The directory is a name of its own under the temporary directory of the
    system and is not created here: the store creates its root with the first
    job it takes (and reads it by its owner alone), so an application which
    never runs one leaves nothing behind, which `tempfile.mkdtemp` would not.
    The name carries a `uuid4`, so two applications of one process never share
    their jobs and no other process guesses the name.

    Returns:
        The directory, which does not exist.
    """
    return Path(tempfile.gettempdir()) / f"{JOBS_PREFIX}{uuid.uuid4().hex}"


def _remove_jobs(root: Path) -> None:
    """Remove the directory of the jobs of an application which created it.

    It is the finalizer of such a store as well as the last step of the
    lifespan, so it tolerates a root which was never created, which is every
    application whose jobs nobody ran. It logs nothing: a finalizer which runs
    while the interpreter shuts down may run after the handlers of logging
    were closed.

    Args:
        root: the directory of the jobs, which may not exist.
    """
    shutil.rmtree(root, ignore_errors=True)


def _spec_hash(
    entry: ModelEntry,
    panel: Panel,
    request: SimulateRequest,
    model_sha256: str,
    variant: str,
) -> str:
    """Get the hash which identifies what a request simulated.

    The hash is the one of the resolved request: a knob which the request
    leaves out is its `default`, so two requests which run the same simulation
    have one hash, and the model is its sha256 and not its path, so the hash
    does not depend on where the service keeps its models.

    The settings of the simulator of the project are hashed whole and not by
    their engine alone: the tolerances, the optimization level and the
    settings of the integrator are the numbers of the curves as much as the
    engine is.

    The declaration of the panel is hashed whole (`panel_payload`) and not by
    its id: a panel which a project declares or overrides
    (`sbmlsim.project.Project.panels`) changes no SBML and so no
    `model_sha256`, and a changed declaration must not answer under the hash
    of the one before it, which a cache of the responses reads.

    Args:
        entry: the model of the request.
        panel: the panel of the request.
        request: the values of the knobs and the grid.
        model_sha256: the sha256 of the SBML file which is simulated.
        variant: the name of the variant which is simulated, the one the
            request names, else the one the panel declares, else the default
            variant of the project.

    Returns:
        The sha256 of the canonical JSON, see
        `sbmlsim.hashing.canonical_json_sha256`.
    """
    return canonical_json_sha256(
        {
            "contract": CONTRACT,
            "model": entry.id,
            "panel": panel.id,
            "declaration": panel_payload(panel),
            "variant": variant,
            "model_sha256": model_sha256,
            "values": {
                knob.id: float(request.values.get(knob.id, knob.default))
                for knob in panel.knobs
                if knob.id not in request.grid
            },
            "grid": {
                knob.id: int(request.grid[knob.id])
                for knob in panel.knobs
                if knob.id in request.grid
            },
            "simulator": entry.project.simulator.model_dump(mode="json"),
            "sbmlsim": sbmlsim.__version__,
        }
    )
