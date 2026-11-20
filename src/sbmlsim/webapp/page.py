"""The web app of a model as one HTML file.

`write_app` writes `<name>_app.html` of a project: the panels and the app of
the model (`sbmlsim.panel`) as the bundle `panelbundle/1` of the service
(`sbmlsim.serve.export.build_bundle`, without the time of its export, so the
same model gives the same file), every model of the bundle compiled by
sbmlsim-core (`sbmlsim.webapp.compile`), the runtime of the engine
(`sbmlsim_core.runtime_wasm`), the inline build of the JavaScript host of
sbmlsim-core (`sbmlsim_core.web.script`, its panel endpoint of a bundle held
in memory and its pool of workers started from a `blob:` URL) and the
components of sbmlsim-web (`sbmlsim/resources/web/sbmlsim-web`, vendored by
`scripts/web_assets.py`). The page registers the endpoint `local:<model id>`
and shows the app of the model (`<sbmlsim-app>`), or every panel of its default
variant (`<sbmlsim-panel>`) for a project without an app.

Everything is embedded, the binary files as base64: the page loads nothing
from the network, needs no server and opens from `file://` as well. It carries
the notices of the libraries of the engine (`sbmlsim_core.THIRD_PARTY`, as
the plain text of `notices.plain_notices`) and the licenses of the components,
since it distributes them: the app shows them in its information view, which
takes them in its slot `licenses`, and a page of panels alone in its footer.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Mapping
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Final

import jinja2

import sbmlsim
from sbmlsim.atomic import write_text_atomic
from sbmlsim.project import Project
from sbmlsim.serve.export import MODELS_DIR, Bundle, build_bundle
from sbmlsim.serve.runner import ModelRegistry
from sbmlsim.webapp import APP_SUFFIX
from sbmlsim.webapp.compile import (
    DESCRIPTION_FILE,
    WASM_FILE,
    CompiledModel,
    compile_sbml,
)
from sbmlsim.webapp.notices import plain_notices

logger = logging.getLogger(__name__)

__all__ = [
    "APP_SUFFIX",
    "app_file",
    "build_app",
    "compile_bundle",
    "render_app",
    "web_components",
    "write_app",
]

#: the template of the page
_TEMPLATE: Final = "webapp.html"
#: the folder of the vendored components of sbmlsim-web in the resources
_WEB: Final = "web/sbmlsim-web"


def app_file(project: Project, directory: Path) -> Path:
    """Get the file of the app of a project in a directory, `<name>_app.html`."""
    return directory / f"{project.name}{APP_SUFFIX}"


@cache
def web_components() -> dict[str, Any]:
    """Get the vendored components of sbmlsim-web: script, license and version.

    Returns:
        `script` (the IIFE build), `license` (its text) and `version`.
    """
    folder = resources.files("sbmlsim.resources").joinpath(_WEB)
    record = json.loads(folder.joinpath("VERSION.json").read_text(encoding="utf-8"))
    return {
        "script": folder.joinpath("sbmlsim-web.iife.js").read_text(encoding="utf-8"),
        "license": folder.joinpath("LICENSE").read_text(encoding="utf-8"),
        "version": record["version"],
    }


def compile_bundle(bundle: Bundle) -> dict[str, CompiledModel]:
    """Compile every model of a bundle with sbmlsim-core, by its key."""
    return {key: compile_sbml(sbml) for key, sbml in sorted(bundle.sbml.items())}


def build_app(project: Project, model_id: str | None = None) -> Bundle:
    """Build the bundle of the app of a project, without the time of its export.

    Args:
        project: the project.
        model_id: the id of the model in the app and its endpoint, the name of
            the project without.

    Returns:
        The bundle, see `sbmlsim.serve.export.build_bundle`.

    Raises:
        ValueError: for an app or a panel the service refuses.
    """
    model_id = model_id or project.name
    registry = ModelRegistry.of_projects({model_id: project})
    return build_bundle(registry, model_id, stamped=False)


def _script(text: str) -> str:
    """Get a script which ends nowhere but at its own end tag inside a page."""
    return re.sub(r"</(script)", r"<\\/\1", text, flags=re.IGNORECASE).replace(
        "<!--", "<\\!--"
    )


def _json(value: Any) -> str:
    """Get JSON which is safe as the text of a script element."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _environment() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.PackageLoader("sbmlsim", "resources/templates"),
        autoescape=jinja2.select_autoescape(["html"]),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )


def render_app(
    bundle: Bundle, compiled: Mapping[str, CompiledModel], *, title: str
) -> str:
    """Render the page of the app of a bundle, see the module.

    Args:
        bundle: the bundle, see `build_app`.
        compiled: every model of the bundle compiled, by its key.
        title: the title of the page.

    Returns:
        The HTML of the page.

    Raises:
        ValueError: if a model of the bundle is not compiled.
    """
    import sbmlsim_core
    from sbmlsim_core import web

    missing = sorted(set(bundle.sbml) - set(compiled))
    if missing:
        raise ValueError(f"The models {missing} of the bundle are not compiled.")
    files = {}
    for key in sorted(bundle.sbml):
        files[f"{MODELS_DIR}/{key}/{WASM_FILE}"] = _base64(compiled[key].wasm)
        files[f"{MODELS_DIR}/{key}/{DESCRIPTION_FILE}"] = _base64(
            compiled[key].description.encode("utf-8")
        )
    runtime = sbmlsim_core.runtime_wasm()
    components = web_components()
    answers = bundle.content["answers"][""]
    return (
        _environment()
        .get_template(_TEMPLATE)
        .render(
            title=title,
            model=bundle.model,
            has_app=answers["app"] is not None,
            panels=list(answers["panels"]),
            versions={
                "sbmlsim": sbmlsim.__version__,
                "sbmlsim-core": sbmlsim_core.__version__,
                "sbmlsim-web": components["version"],
            },
            web_script=_script(components["script"]),
            core_script=_script(web.script()),
            bundle=_json(bundle.content),
            files=_json(files),
            engine=_base64(runtime["engine"]),
            mathlib=_base64(runtime["mathlib"]),
            third_party=plain_notices(sbmlsim_core.THIRD_PARTY),
            web_license=components["license"],
        )
    )


def write_app(project: Project, directory: Path, model_id: str | None = None) -> Path:
    """Write the app of a project into a directory, `<name>_app.html`.

    Args:
        project: the project.
        directory: the directory, created if it does not exist.
        model_id: the id of the model in the app, the name of the project
            without.

    Returns:
        The file of the app.

    Raises:
        ValueError: for an app or a panel the service refuses.
        WebAppError: for a model sbmlsim-core cannot compile.
    """
    bundle = build_app(project, model_id)
    title = project.app.title if project.app is not None else None
    page = render_app(
        bundle, compile_bundle(bundle), title=title or project.title or project.name
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = app_file(project, directory)
    write_text_atomic(path, page)
    logger.info("The app of the project '%s' is written: %s", project.name, path)
    return path
