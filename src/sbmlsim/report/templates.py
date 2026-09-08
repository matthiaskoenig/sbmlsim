"""Templates of the HTML reports.

The reports of the simulation experiments and of the parameter fitting share
their design: `report_base.html` carries the style, the search, the filters and
the sortable tables, the reports fill in its blocks. A report needs no network,
it is opened from a file, archived or sent as it is.
"""

from pathlib import Path

import jinja2

from sbmlsim import RESOURCES_DIR

#: directory of the jinja2 templates
TEMPLATE_DIR: Path = RESOURCES_DIR / "templates"

#: base template of the HTML reports, it provides the blocks and the style
BASE_TEMPLATE = "report_base.html"


def template_environment(template_dir: Path | None = None) -> jinja2.Environment:
    """Create the jinja2 environment of the reports.

    HTML is escaped, the markdown and latex templates are not touched.

    Args:
        template_dir: directory of the templates, the templates of the package
            by default.

    Returns:
        The environment the reports are rendered with.
    """
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir or TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
