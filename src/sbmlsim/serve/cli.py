"""The command line of the service of the panels, `python -m sbmlsim.serve`.

    python -m sbmlsim.serve --model ID=REFERENCE [--model ID=REFERENCE ...]
        [--host HOST] [--port PORT] [--static DIR]
        [--allow-origin ORIGIN ...] [--experiments] [--open]

A model is `ID=REFERENCE`, the id a request names and the project of the model
package it is served from, `module:attribute` of a `sbmlsim.project.Project`
(`parse_models`). The projects are loaded once, before the server starts
(`sbmlsim.serve.runner.ModelRegistry.of`), so a request reaches a model of this
command line and never a module, a reference or a path of the host.

The service binds `127.0.0.1` by default (`DEFAULT_HOST`), the loopback
address: the teaching mode is local, a lecture, a talk or a page of the
documentation on the machine which runs the command, and a service the network
reaches is a deliberate `--host 0.0.0.0` behind a proxy, see `docs/serve.md`.

`--static DIR` serves a directory under the API, so the host page of a panel
comes from the origin of the API: a page which is served this way needs no CORS
and no permission of the browser, while a page of another origin needs
`--allow-origin ORIGIN` for every origin it is served from. `--open` opens the
browser on the service once it is up and is off by default.

`--experiments` adds the routes which run the simulation experiments of a model
as jobs (`sbmlsim.serve.jobs`), which are off by default: one request of them
costs minutes of a core, so a public deployment leaves them off and a local
one which runs the experiments of its models turns them on.

The exit codes are the ones of `python -m sbmlsim.run`: 0 when the server
stopped, 2 for a bad command line, which argparse writes with the usage, and 4
for a model which is no project or whose app names something the model does not
declare and for an install without the extra `web`, whose message is on
stderr.

Neither uvicorn nor fastapi is imported when this module is imported: the
application is created and the server is started in the functions which need
them (`main`, `serve`), so the parser and `parse_models` run in an install
without the extra `web`.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "EXIT_CONFIGURATION",
    "EXIT_OK",
    "EXIT_USAGE",
    "main",
    "parse_models",
    "serve",
]

#: the server stopped
EXIT_OK: Final = 0
#: the command line is bad (argparse), as `python -m sbmlsim.run` has it
EXIT_USAGE: Final = 2
#: the service is configured wrongly, e.g. a model which is no project, as
#: `python -m sbmlsim.run` has it
EXIT_CONFIGURATION: Final = 4

#: the address the service binds without `--host`: the loopback address, so
#: the service of the teaching mode is reachable from the machine it runs on
#: and from nowhere else, see the module
DEFAULT_HOST: Final = "127.0.0.1"
#: the port the service binds without `--port`
DEFAULT_PORT: Final = 8000

#: the seconds `--open` waits for the server before it gives up opening the
#: browser, see `serve`
OPEN_TIMEOUT: Final = 30.0
#: the seconds between two looks at the server while `--open` waits for it
OPEN_INTERVAL: Final = 0.05

#: the form of a model of the command line, which a problem names
MODEL_FORM: Final = "id=module:attribute"


def _directory(value: str) -> Path:
    """Parse a directory of the command line.

    Args:
        value: the path.

    Returns:
        The path.

    Raises:
        argparse.ArgumentTypeError: for a path which is no directory, which
            the server would refuse when it mounts it.
    """
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"'{value}' is no directory")
    return path


def _port(value: str) -> int:
    """Parse a port of the command line.

    Args:
        value: the port.

    Returns:
        The port.

    Raises:
        argparse.ArgumentTypeError: for a value which is no port, i.e. no
            number of 1 to 65535.
    """
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is no port") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"'{value}' is no port of 1 to 65535")
    return port


def _parser() -> argparse.ArgumentParser:
    """Get the parser of the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m sbmlsim.serve",
        description="Serve the interactive panels of the models of their projects "
        "and run their simulations.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar=MODEL_FORM.upper(),
        help="a model of the service, the id a request names and the project it "
        f"is served from ({MODEL_FORM}); repeat it per model",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"the address the service binds, '{DEFAULT_HOST}' by default, which "
        "is reachable from this machine alone",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"the port the service binds, {DEFAULT_PORT} by default",
    )
    parser.add_argument(
        "--static",
        type=_directory,
        metavar="DIR",
        help="a directory served under the API, e.g. the slides or the pages "
        "which embed the panels, so that they are of its origin",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        metavar="ORIGIN",
        help="an origin which may talk to the API from another origin (CORS); "
        "repeat it per origin, none without",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="serve the routes which run the simulation experiments of a model "
        "as jobs; off by default, which is what a public deployment keeps",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the browser on the service once it is up",
    )
    return parser


def parse_models(values: Sequence[str]) -> dict[str, str]:
    """Parse the models of the command line, see the module.

    Args:
        values: the `--model` arguments, each `id=module:attribute`.

    Returns:
        The project reference of every model id, in the order of the command
        line.

    Raises:
        SystemExit: with the usage, for a value which is no `id=reference` and
            for an id which is given twice, which would serve one of the two
            projects and hide the other.
    """
    parser = _parser()
    models: dict[str, str] = {}
    for value in values:
        model_id, separator, reference = value.partition("=")
        if not separator or not model_id or not reference:
            parser.error(f"the model '{value}' is no '{MODEL_FORM}'")
        if model_id in models:
            parser.error(f"the model '{model_id}' is given twice")
        models[model_id] = reference
    return models


def serve(
    app: FastAPI,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = False,
) -> None:
    """Run an application on a server until it is stopped.

    uvicorn is imported here and not when the module is imported, see the
    module.

    Args:
        app: the application, see `sbmlsim.serve.app.create_app`.
        host: the address the server binds.
        port: the port the server binds.
        open_browser: open the browser on the service once the server is up;
            the browser is opened from a thread of its own, so a browser which
            is not opened leaves the server running.
    """
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_config=_logging_config())
    )
    url = f"http://{host}:{port}/"
    # flushed, so a caller which reads the address of a service it started
    # does not wait for the buffer of a stdout which is no terminal
    sys.stdout.write(f"{url}\n")
    sys.stdout.flush()
    if open_browser:
        _open_when_up(server, url)
    server.run()


def _logging_config() -> dict[str, Any]:
    """Get the logging of the server, its access log on stderr.

    uvicorn writes its access log on stdout and everything else on stderr;
    this command writes the address of the service on stdout and nothing else,
    so the access log is moved over to stderr, where the messages of the
    package are.

    Returns:
        The logging of uvicorn with that one change; its own one if it has
        another shape than the one of the version this was written against.
    """
    import uvicorn.config

    config: dict[str, Any] = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    try:
        config["handlers"]["access"]["stream"] = "ext://sys.stderr"
    except KeyError, TypeError:  # pragma: no cover - another uvicorn
        logger.debug("The access log of uvicorn stays where uvicorn writes it.")
    return config


class _Running(Protocol):
    """What `_open_when_up` reads of a server, which `uvicorn.Server` has."""

    @property
    def started(self) -> bool:
        """Whether the server is up and answers."""

    @property
    def should_exit(self) -> bool:
        """Whether the server is stopping."""


def _open_when_up(server: _Running, url: str) -> threading.Thread:
    """Open the browser on the service once the server is up.

    The browser is opened from a thread of its own, which waits for the server
    and gives up after `OPEN_TIMEOUT`: the server runs in the calling thread
    and a browser which cannot be opened never keeps it from running.

    Args:
        server: the server which is started next.
        url: the address of the service.

    Returns:
        The thread which waits for the server, a daemon, so a server which
        never starts does not keep the process alive.
    """

    def opened() -> None:
        """Wait for the server and open the browser on it."""
        deadline = time.monotonic() + OPEN_TIMEOUT
        while not server.started:
            if server.should_exit or time.monotonic() > deadline:
                logger.warning("The browser is not opened on '%s': no server.", url)
                return
            time.sleep(OPEN_INTERVAL)
        logger.info("The browser is opened on '%s'.", url)
        webbrowser.open(url)

    thread = threading.Thread(target=opened, name="sbmlsim-serve-open", daemon=True)
    thread.start()
    return thread


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line, see the module.

    Args:
        argv: the arguments without the program, `sys.argv[1:]` without.

    Returns:
        The exit code: `EXIT_OK` when the server stopped, `EXIT_USAGE` for a
        bad command line and `EXIT_CONFIGURATION` for a model which cannot be
        served.
    """
    try:
        args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
        models = parse_models(args.model)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE
    # after the command line is read, so a bad one is answered with its usage
    # in an install without the extra `web`
    try:
        from sbmlsim.serve.app import create_app
    except ImportError as err:
        sys.stderr.write(
            f"python -m sbmlsim.serve: error: the service needs the extra 'web', "
            f"`pip install sbmlsim[web]`: {err}\n"
        )
        return EXIT_CONFIGURATION
    from sbmlsim.serve.runner import ModelRegistry

    try:
        registry = ModelRegistry.of(models)
        app = create_app(
            registry,
            static=args.static,
            allow_origins=tuple(args.allow_origin or ()),
            experiments=args.experiments,
        )
    except (ImportError, OSError, ValueError) as err:
        # the models are named as well: the error of a reference which is no
        # module names the module and not the model it is served as
        sys.stderr.write(
            f"python -m sbmlsim.serve: error: the models {list(models)} are not "
            f"served: {err}\n"
        )
        return EXIT_CONFIGURATION
    serve(app, host=args.host, port=args.port, open_browser=args.open)
    return EXIT_OK
