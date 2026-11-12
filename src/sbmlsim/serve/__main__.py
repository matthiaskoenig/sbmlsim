"""The entry point of `python -m sbmlsim.serve`, see `sbmlsim.serve.cli`.

It logs on stderr, where the server logs its requests as well, so stdout
carries only the address of the service.
"""

import sys


def main() -> None:
    """Run the command line and exit with its code."""
    from rich.console import Console

    from sbmlsim.log import enable_rich_logging
    from sbmlsim.serve.cli import main as cli_main

    enable_rich_logging(console=Console(stderr=True))
    sys.exit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
