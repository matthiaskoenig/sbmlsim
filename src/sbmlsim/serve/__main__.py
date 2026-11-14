"""The entry point of `python -m sbmlsim.serve`, see `sbmlsim.serve.cli`.

It logs on stderr, where the server logs its requests as well, so stdout
carries only the address of the service.

`MPLBACKEND` is set to `Agg` unless it is set, as `python -m sbmlsim.run` does
it, so the service draws without a display.
"""

import os
import sys


def _default_backend() -> None:
    """Set `MPLBACKEND` to `Agg` unless the environment names a backend.

    The jobs of the service render the figures of the simulation experiments of
    a model in a thread of the server (`sbmlsim.serve.jobs`), and an
    interactive backend releases the objects of its toolkit in whatever thread
    the garbage collector runs in, which aborts the process (#317). A backend
    the environment names is kept, so a run which wants another one says so.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    """Run the command line and exit with its code."""
    _default_backend()

    from rich.console import Console

    from sbmlsim.log import enable_rich_logging
    from sbmlsim.serve.cli import main as cli_main

    enable_rich_logging(console=Console(stderr=True))
    sys.exit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
