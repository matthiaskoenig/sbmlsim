"""The code blocks of the documentation run.

Every `python` block of a page runs, in the order of the page and in one
namespace, so a block uses what the blocks before it define, as a reader does.
"""

import os
import re
from pathlib import Path

import pytest

#: the directory of the documentation
DOCS_DIR = Path(__file__).parents[2] / "docs"

#: the pages whose code is run
PAGES = [
    "index.md",
    "simulation.md",
    "scans.md",
    "models.md",
    "units.md",
    "data.md",
    "observables.md",
]

#: a fenced block of python
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("page", PAGES)
def test_the_code_of_a_page_runs(page: str, tmp_path: Path) -> None:
    """The blocks of a page run without an error in a working directory."""
    blocks = BLOCK.findall((DOCS_DIR / page).read_text(encoding="utf-8"))
    assert blocks, f"'{page}' has no python blocks"
    namespace: dict[str, object] = {}
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        for k, block in enumerate(blocks):
            try:
                exec(compile(block, f"{page}[{k}]", "exec"), namespace)
            except Exception as err:
                raise AssertionError(f"block {k} of '{page}' fails:\n{block}") from err
    finally:
        os.chdir(cwd)
