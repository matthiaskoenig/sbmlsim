"""The entry point of `python -m sbmlsim.release`, see `sbmlsim.release.cli`.

It logs on stderr, so stdout carries only the path of a build and the line of
a publishing.
"""

import sys


def main() -> None:
    """Run the command line and exit with its code."""
    from rich.console import Console

    from sbmlsim.log import enable_rich_logging
    from sbmlsim.release.cli import main as cli_main

    enable_rich_logging(console=Console(stderr=True))
    sys.exit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
