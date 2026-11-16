"""Write the web app of an SBML model: `python -m sbmlsim.webapp MODEL.xml [-o DIR]`.

The app is `<DIR>/<stem>_app.html` (the directory of the model without `-o`):
the panels the model declares in its annotation, or the time courses of its
species, simulated in the browser, see `sbmlsim.webapp.assets.app_project`.
A project writes its app with its factory (`sbmlsim.project.Project.web_app`).
The exit code is 0 for a written app, 2 for a bad command line and 4 for a
model whose app cannot be written, as the run layer has them.
"""

import argparse
import logging
import sys
from pathlib import Path

#: the exit code of a model whose app cannot be written
EXIT_CONFIGURATION = 4


def main(argv: list[str] | None = None) -> int:
    """Run the command line, see the module."""
    parser = argparse.ArgumentParser(
        prog="python -m sbmlsim.webapp",
        description="Write the web app of an SBML model as one HTML file.",
    )
    parser.add_argument("sbml", type=Path, help="the SBML file of the model")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="the directory of the app, the one of the model without",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    from sbmlsim.webapp.assets import app_project
    from sbmlsim.webapp.page import write_app

    try:
        project = app_project(args.sbml)
        path = write_app(project, args.out or args.sbml.resolve().parent)
    except (OSError, ValueError) as err:
        logging.getLogger("sbmlsim.webapp").error("%s", err)
        return EXIT_CONFIGURATION
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
