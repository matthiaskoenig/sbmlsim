"""The code blocks of the documentation run and their outputs are in the pages.

Every `python` block of a page runs, in the order of the page and in one
namespace, so a block uses what the blocks before it define, as a reader does.
`scripts/docs_outputs.py` writes what a block prints and draws below it; the
test fails when a page is out of step with its code. The text of an output is
not compared, it differs in the last digits between platforms.
"""

from pathlib import Path

import pytest

from scripts.docs_outputs import DOCS_DIR, PAGES, parse, run_page

#: the advice of a failure
REGENERATE = "run `uv run python scripts/docs_outputs.py {page}`"


@pytest.mark.parametrize("page", PAGES)
def test_the_code_of_a_page_runs_and_its_outputs_are_shown(
    page: str, tmp_path: Path
) -> None:
    """The blocks of a page run, and every output and figure of them is shown."""
    _, blocks = parse((DOCS_DIR / page).read_text(encoding="utf-8"))
    assert blocks, f"'{page}' has no python blocks"
    results = run_page(page, tmp_path)
    advice = REGENERATE.format(page=page)
    for k, (block, result) in enumerate(zip(blocks, results, strict=True)):
        where = f"block {k + 1} of '{page}'"
        assert bool(result.stdout) == (block.output is not None), (
            f"{where} prints {'something' if result.stdout else 'nothing'} but "
            f"{'has no' if result.stdout else 'has an'} output, {advice}"
        )
        assert len(block.images) == len(result.figures), (
            f"{where} draws {len(result.figures)} figures but shows "
            f"{len(block.images)}, {advice}"
        )
        for image in block.images:
            assert (DOCS_DIR / image).is_file(), f"{where}: '{image}' is missing"


def test_every_page_with_code_is_run() -> None:
    """A page with python blocks is one of the pages whose code is run."""
    pages = {
        str(path.relative_to(DOCS_DIR))
        for path in DOCS_DIR.glob("*.md")
        if parse(path.read_text(encoding="utf-8"))[1]
    }
    assert pages <= set(PAGES), f"pages whose code is not run: {pages - set(PAGES)}"
