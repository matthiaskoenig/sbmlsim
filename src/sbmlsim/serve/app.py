"""The HTTP API over the panels of the served models.

`create_app(registry)` is the ASGI application of a `ModelRegistry`: it serves
the declaration of the panels of its models (`sbmlsim.panel.panel_payload`,
the contract `panel/1`) and runs their simulations on one `PanelRunner` with
warm models.

A request names a model id of the registry and a panel id of that model and
nothing else: no project reference, no module and no path reaches the service,
and an id which is not served is a 404 with the message of the registry. A
request which is over the limits of `sbmlsim.panel` or which names a knob the
panel does not declare is a 422, a failure of the service is a 500 of one
fixed message, logged with its traceback: an error body carries no traceback,
no path and nothing of the configuration of the service.

The route of a simulation is a sync `def`, which the server runs in its thread
pool, and the runner holds the lock of its model while it simulates, so the
event loop keeps answering while a model integrates.

`static` serves a directory under the API, so the local mode delivers the
javascript component of the panels from the origin of the API, and
`allow_origins` lets the named origins talk to it from another one.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import sbmlsim
from sbmlsim.hashing import canonical_json_sha256, file_sha256
from sbmlsim.panel import CONTRACT, Panel, panel_payload
from sbmlsim.serve.payload import (
    HealthResponse,
    ModelPayload,
    ModelsResponse,
    PanelPayload,
    PanelsResponse,
    Provenance,
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
)

logger = logging.getLogger(__name__)

__all__ = ["FAILED", "create_app"]

#: the body of a failure of the service, which carries nothing of it
FAILED: Final = "The request failed in the service."

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

    Returns:
        The application.
    """
    app = FastAPI(title=TITLE, version=sbmlsim.__version__)
    runner = PanelRunner(registry)
    #: the sha256 of every SBML file which was simulated, by its resolved path
    hashes: dict[Path, str] = {}

    def resolved(model_id: str, panel_id: str) -> tuple[ModelEntry, Panel, str]:
        """Get the model and the panel a request names, and the hash of its SBML.

        Args:
            model_id: the id of the model.
            panel_id: the id of the panel.

        Returns:
            The entry, the panel and the sha256 of the SBML file of its
            variant.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownPanel: for a panel the model does not declare, a 404.
            HTTPException: a 500 for a model whose panels or whose SBML cannot
                be read, which is a problem of the service and not of the
                request.
        """
        try:
            entry = registry.entry(model_id)
            panel = registry.panel(model_id, panel_id)
            path = entry.project.variant_path(panel.simulation.variant).resolve()
            sha = hashes.get(path)
            if sha is None:
                sha = file_sha256(path)
                hashes[path] = sha
            return entry, panel, sha
        except UnknownModel, UnknownPanel:
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

    @app.get("/api/models/{model}/panels")
    def panels(model: str) -> PanelsResponse:
        """List the panels a model declares, in the order it declares them.

        The panels of a model are read from its SBML once, so the route is a
        sync `def` and reads it in the thread pool of the server.

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
    def panel_of(model: str, panel: str) -> dict[str, Any]:
        """Get the declaration of a panel, the payload of the contract.

        The declaration and the hash of the model are read from the files of
        the model once, so the route is a sync `def` and reads them in the
        thread pool of the server, as `simulate` does.

        Args:
            model: the id of the model.
            panel: the id of the panel.

        Returns:
            The payload, see `sbmlsim.panel.panel_payload`.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownPanel: for a panel the model does not declare, a 404.
        """
        _, declaration, _ = resolved(model, panel)
        return panel_payload(declaration)

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
            resolved request.

        Raises:
            UnknownModel: for a model which is not served, a 404.
            UnknownPanel: for a panel the model does not declare, a 404.
            HTTPException: a 422 for a knob id the panel does not declare,
                which is the one error of a run a request is answered with,
                and a 500 for every other failure of the run, whose message
                is the one of the service and not of the exception.
        """
        entry, declaration, sha = resolved(model, panel)
        started = time.perf_counter()
        try:
            result = runner.run(model, declaration, request.values, request.grid)
            seconds = time.perf_counter() - started
            settings = entry.project.simulator
            return SimulateResponse.of(
                result,
                spec_hash=_spec_hash(entry, declaration, request, sha),
                provenance=Provenance(
                    model=entry.id,
                    panel=declaration.id,
                    model_sha256=sha,
                    variant=declaration.simulation.variant,
                    engine=settings.engine.value,
                    sbmlsim=sbmlsim.__version__,
                    seconds=seconds,
                    created=datetime.now(UTC).isoformat(),
                ),
            )
        except UnknownKnob as error:
            # the runner names the knob ids which are no knobs of the panel,
            # the one message of a run which carries nothing of the service
            raise HTTPException(status_code=422, detail=str(error)) from None
        except Exception:
            logger.exception(
                "The panel '%s' of the model '%s' did not run.", panel, model
            )
            raise HTTPException(status_code=500, detail=FAILED) from None

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


def _spec_hash(
    entry: ModelEntry,
    panel: Panel,
    request: SimulateRequest,
    model_sha256: str,
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
            "variant": panel.simulation.variant,
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
