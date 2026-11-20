"""The web app of a model as one HTML file."""

import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from importlib import resources
from pathlib import Path

import pytest

import sbmlsim
from sbmlsim.serve.export import build_bundle
from sbmlsim.serve.runner import ModelRegistry
from sbmlsim.webapp.page import (
    app_file,
    build_app,
    compile_bundle,
    web_components,
    write_app,
)

sbmlsim_core = pytest.importorskip("sbmlsim_core")

from examples.panels.hctz import PROJECT  # noqa: E402


class _Scripts(HTMLParser):
    """The scripts of a page with their attributes, and every URL it loads."""

    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self.loads: list[str] = []
        self._open: dict[str, str | None] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for name in ("src", "href"):
            if tag in ("script", "link", "img", "iframe") and values.get(name):
                self.loads.append(str(values[name]))
        if tag == "script":
            self._open = values
            self.scripts.append((values, ""))

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            attrs, text = self.scripts[-1]
            self.scripts[-1] = (attrs, text + data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._open = None


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_app(PROJECT, tmp_path_factory.mktemp("app"))


@pytest.fixture(scope="module")
def scripts(app: Path) -> _Scripts:
    parser = _Scripts()
    parser.feed(app.read_text(encoding="utf-8"))
    return parser


def _script(scripts: _Scripts, element_id: str) -> str:
    return next(
        text for attrs, text in scripts.scripts if attrs.get("id") == element_id
    )


def test_the_app_is_one_file_named_after_the_project(app: Path) -> None:
    assert app == app_file(PROJECT, app.parent)
    assert app.name == "hctz_app.html"
    assert [path.name for path in app.parent.iterdir()] == ["hctz_app.html"]


def test_the_page_loads_nothing(scripts: _Scripts) -> None:
    assert [url for url in scripts.loads if not url.startswith("data:")] == []


def test_the_page_embeds_the_bundle_of_the_service_without_a_time(
    scripts: _Scripts,
) -> None:
    registry = ModelRegistry.of_projects({"hctz": PROJECT})
    expected = build_bundle(registry, "hctz", stamped=False).content
    assert json.loads(_script(scripts, "sbmlsim-bundle")) == expected
    assert "created" not in expected


def test_the_page_embeds_every_model_of_the_bundle_compiled(scripts: _Scripts) -> None:
    bundle = json.loads(_script(scripts, "sbmlsim-bundle"))
    files = json.loads(_script(scripts, "sbmlsim-files"))
    assert len(bundle["models"]) > 1  # the page `patient` has derived targets
    for key in bundle["models"]:
        wasm = base64.b64decode(files[f"models/{key}/model.wasm"])
        description = json.loads(base64.b64decode(files[f"models/{key}/model.json"]))
        assert wasm[:4] == b"\0asm"
        assert description["contract"] == "model/2"


def test_the_page_embeds_the_runtime_and_the_scripts(scripts: _Scripts) -> None:
    runtime = sbmlsim_core.runtime_wasm()
    assert base64.b64decode(_script(scripts, "sbmlsim-engine")) == runtime["engine"]
    assert base64.b64decode(_script(scripts, "sbmlsim-mathlib")) == runtime["mathlib"]
    texts = [text for attrs, text in scripts.scripts if not attrs]
    assert "var sbmlsimCore" in texts[1]
    assert "registerEndpoint" in texts[2]
    assert '"local:hctz"' in texts[2]


def test_the_page_names_the_versions_and_carries_the_licenses(app: Path) -> None:
    text = app.read_text(encoding="utf-8")
    assert f"sbmlsim</a> {sbmlsim.__version__}" in text
    assert f"sbmlsim-core</a> {sbmlsim_core.__version__}" in text
    assert f"sbmlsim-web</a> {web_components()['version']}" in text
    assert "SUNDIALS" in text
    assert "<title>Hydrochlorothiazide</title>" in text
    assert '<sbmlsim-app endpoint="local:hctz" model="hctz">' in text


def test_the_page_of_an_app_is_as_wide_as_the_screen(app: Path) -> None:
    # the app draws the figure of its model beside its panels, which needs the room
    assert '<main class="app">' in app.read_text(encoding="utf-8")


def test_the_page_of_panels_keeps_the_reading_width() -> None:
    import jinja2

    environment = jinja2.Environment(
        loader=jinja2.PackageLoader("sbmlsim", "resources/templates"),
        autoescape=jinja2.select_autoescape(["html"]),
        undefined=jinja2.ChainableUndefined,
    )
    page = environment.get_template("webapp.html").render(
        has_app=False, panels=["p"], third_party=[]
    )
    assert "<main>" in page
    assert 'class="app"' not in page


def test_the_app_is_the_same_for_the_same_model(app: Path, tmp_path: Path) -> None:
    assert write_app(PROJECT, tmp_path).read_bytes() == app.read_bytes()


def test_no_script_ends_inside_an_embedded_script(scripts: _Scripts) -> None:
    # every script element is one of the template: the embedded ones did not end early
    assert len(scripts.scripts) == 7


def test_the_compiled_models_are_the_ones_of_the_bundle() -> None:
    bundle = build_app(PROJECT)
    compiled = compile_bundle(bundle)
    assert sorted(compiled) == sorted(bundle.sbml)


def test_the_vendored_components_are_the_ones_of_their_record() -> None:
    folder = resources.files("sbmlsim.resources").joinpath("web/sbmlsim-web")
    record = json.loads(folder.joinpath("VERSION.json").read_text(encoding="utf-8"))
    assert record["package"] == "sbmlsim-web"
    for name, digest in record["files"].items():
        assert hashlib.sha256(folder.joinpath(name).read_bytes()).hexdigest() == digest
    script = Path(__file__).parents[2] / "scripts" / "web_assets.py"
    assert re.search(
        rf'^VERSION = "{re.escape(record["version"])}"$',
        script.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
