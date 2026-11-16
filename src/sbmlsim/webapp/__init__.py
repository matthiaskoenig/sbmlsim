"""The web app of a model: one HTML file which simulates it in the browser.

See `sbmlsim.webapp.compile` (the SBML compiled to WebAssembly with
sbmlsim-core), `sbmlsim.webapp.page` (the page) and `sbmlsim.webapp.assets`
(what a factory writes). This module holds the names of the files only, so
the release reads them without importing the rest.
"""

from typing import Final

__all__ = ["APP_SUFFIX", "WASM_DIR"]

#: the suffix of the file of the app of a project, `<name>_app.html`
APP_SUFFIX: Final = "_app.html"
#: the directory of the compiled variants in the directory of the factory
WASM_DIR: Final = "wasm"
