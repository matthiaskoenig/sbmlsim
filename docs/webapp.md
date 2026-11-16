# Web app

Every model has a web app: one HTML file which shows the interactive panels of the model ([Interactive panels](serve.md)) and simulates them in the browser of its reader. The model is compiled to WebAssembly by [sbmlsim-core](https://matthiaskoenig.github.io/sbmlsim-core/), the panels are drawn by the components of [sbmlsim-web](https://matthiaskoenig.github.io/sbmlsim-web/), and the file embeds everything it runs: the panels and the app of the model, the compiled models, the engine and the JavaScript. It needs no server and no network and opens from disk (`file://`), from a repository or from GitHub Pages.

## With the factory of a project

`Project.create_models` writes the web assets of the model after its factory, into the directory the factory wrote the models into, from the SBML it just wrote, so they are always the ones of the models next to them:

- `wasm/<stem>/model.wasm` and `model.json`, every variant of the model compiled by sbmlsim-core (the layout of `sbmlsim-core compile`, which the JavaScript host of sbmlsim-core loads);
- `<name>_app.html`, the app of the model.

A rebuild (`create_models(out)`, the kind `factory` of the run layer) writes them into the rebuild, and the same models give the same files, so the comparison of a rebuild with the package holds them as well. A project switches them off with `web_app=False`. A model which declares no panel and no app gets the panels and the app of `Project.default_app`: by default the panels of `Project.default_panels`, the time courses of its species (`sbmlsim.webapp.assets.default_panel`), of the variables of its rules for a model without species, and no app; a framework which knows more of its models overrides it, e.g. pkpdsim with an app of pages of the doses and dosing protocols, the diseases and the scans of a pharmacokinetic model, each of small panels (`AppPage.size`) of the readouts its experiments compare with data, and the variants of the model a reader switches between. The panels of `default_app` are the ones of the default variant; a variant which shows them otherwise, e.g. the classes of a renal impairment at the healthy GFR of the build of the other sex, gets its own declaration of them from `Project.default_variant_panels(panels, variants)`, which a framework overrides as well (none by default).

The release of a model (`sbmlsim.release`) copies the app and the compiled variants into `models/` of the release repository and adds the page **App** to its documentation, the app in a frame and a link to the file, and a link to the README.

## Of an SBML file

```bash
python -m sbmlsim.webapp model.xml -o out    # writes out/model_app.html
```

The app shows the panels the model declares in its annotation (`sbmlsim.panel_sbml`), else the time courses of its species.

```py
from pathlib import Path

from sbmlsim.webapp.assets import app_project
from sbmlsim.webapp.page import write_app

write_app(app_project(Path("model.xml")), Path("out"))
```

## How the page works

The page is the bundle of the service, `panelbundle/1` ([Interactive panels](serve.md#the-static-bundle)), without the time of its export, with every model of the bundle compiled (`sbmlsim.webapp.compile`: a hierarchical model flattened, the uncertainties of `distrib` stripped, a model below level 3 converted to level 3 version 2; a math which draws from a distribution of `distrib` is refused). The inline build of the JavaScript host of sbmlsim-core (`sbmlsim_core.web.script()`) answers the routes of the service inside the page from the embedded bundle, with the simulations on a pool of Web Workers started from the embedded worker, and the app of sbmlsim-web talks to it as the endpoint `local:<name>`. The runtime is the one of the installed sbmlsim-core (`sbmlsim_core.runtime_wasm()`), so the compiled models and the engine are of one release; the page names the versions of sbmlsim, sbmlsim-core and sbmlsim-web and carries the notices of the libraries of the engine and the licenses of the components.

The components of sbmlsim-web are vendored into `sbmlsim/resources/web/sbmlsim-web` at a pinned version, checked against the integrity of the registry of npm (`scripts/web_assets.py`).
