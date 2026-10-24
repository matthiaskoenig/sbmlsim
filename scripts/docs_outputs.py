"""Write the outputs of the code blocks of the documentation into its pages.

The `python` blocks of a page, with or without a title, run in the order of
the page and in one namespace, as a reader runs them, as the main script. What a block prints is written below it as a
block titled `Output`, and the matplotlib figures it creates are saved as PNG
images in `docs/images/outputs/` and shown below the output. A block which
prints nothing and draws nothing has nothing below it.

```bash
uv run python scripts/docs_outputs.py                 # every page
uv run python scripts/docs_outputs.py tutorial.md     # one page
```

The outputs are committed with the pages. `tests/docs/test_docs_code.py` runs
the blocks of every page as well and fails when a page is out of step with its
code: a block which prints without an output, an output of a block which
prints nothing or a figure without its image. It does not compare the text,
which differs in the last digits between platforms; run this script after a
change of a page or of what its code prints.

A figure is a figure of pyplot which is open after the block, or a
`matplotlib.figure.Figure` which a name of the namespace holds and which no
block before showed. The working directory of the blocks is a temporary
directory, so the files a page writes do not end up in the repository.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

REPO_DIR: Path = Path(__file__).parent.parent
DOCS_DIR: Path = REPO_DIR / "docs"
#: the images of the figures, relative to the pages
IMAGES: str = "images/outputs"
OUTPUTS_DIR: Path = DOCS_DIR / IMAGES

#: the pages whose code is run, in the order of the documentation
PAGES: list[str] = [
    "index.md",
    "installation.md",
    "tutorial.md",
    "models.md",
    "simulation.md",
    "units.md",
    "experiments.md",
    "data.md",
    "plotting.md",
    "scans.md",
    "sampling.md",
    "uncertainty.md",
    "observables.md",
    "sensitivity.md",
    "fitting.md",
]

#: the fence of a python block and of its output
PYTHON_FENCE = "```python"
OUTPUT_FENCE = '```text title="Output"'
FENCE = "```"
#: an image of a figure of a block
IMAGE = re.compile(r"^!\[Output figure\]\(" + re.escape(IMAGES) + r"/[\w.-]+\.png\)$")

#: an output longer than this is a hint that the block prints too much
MAX_OUTPUT_LINES = 40

#: the settings of a saved figure
DPI = 100


@dataclass
class Block:
    """A python block of a page and the output below it.

    Attributes:
        code: the code of the block.
        start: the index of the line of its opening fence.
        end: the index after its closing fence.
        region_end: the index after its output and its images, `end` if it
            has none.
        output: the text of its output block, `None` without one.
        images: the paths of its images relative to the page.
    """

    code: str
    start: int
    end: int
    region_end: int
    output: str | None = None
    images: list[str] = field(default_factory=list)


@dataclass
class BlockResult:
    """What a block printed and drew.

    Attributes:
        stdout: the text it printed, without trailing whitespace.
        figures: the figures it created, `matplotlib.figure.Figure`.
    """

    stdout: str
    figures: list[Figure]


def parse(text: str) -> tuple[list[str], list[Block]]:
    """Get the lines of a page and its python blocks with their outputs.

    Args:
        text: the markdown of the page.

    Returns:
        The lines of the page and its python blocks in their order.

    Raises:
        ValueError: if a block is not closed.
    """
    lines = text.split("\n")
    blocks: list[Block] = []
    k = 0
    while k < len(lines):
        if lines[k] != PYTHON_FENCE and not lines[k].startswith(PYTHON_FENCE + " "):
            k += 1
            continue
        start = k
        k += 1
        while k < len(lines) and lines[k] != FENCE:
            k += 1
        if k == len(lines):
            raise ValueError(f"The python block in line {start + 1} is not closed.")
        end = k + 1
        block = Block(
            code="\n".join(lines[start + 1 : k]) + "\n",
            start=start,
            end=end,
            region_end=end,
        )
        k = _skip_blank(lines, end)
        if k < len(lines) and lines[k] == OUTPUT_FENCE:
            first = k + 1
            k = first
            while k < len(lines) and lines[k] != FENCE:
                k += 1
            if k == len(lines):
                raise ValueError(f"The output in line {first} is not closed.")
            block.output = "\n".join(lines[first:k])
            k += 1
            block.region_end = k
            k = _skip_blank(lines, k)
        while k < len(lines) and IMAGE.match(lines[k]):
            block.images.append(lines[k][len("![Output figure](") : -1])
            k += 1
            block.region_end = k
            k = _skip_blank(lines, k)
        blocks.append(block)
        k = block.region_end
    return lines, blocks


def _skip_blank(lines: list[str], k: int) -> int:
    """Get the index of the first line from `k` which is not blank."""
    while k < len(lines) and not lines[k].strip():
        k += 1
    return k


def _figures(namespace: dict[str, object]) -> list[Figure]:
    """Get the open figures of pyplot and the figures the namespace holds."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figures = [plt.figure(num) for num in plt.get_fignums()]
    for value in namespace.values():
        if isinstance(value, Figure) and all(value is not f for f in figures):
            figures.append(value)
    return figures


def _clean(stdout: str, workdir: Path) -> str:
    """Get the printed text without the working directory and blank edges."""
    text = stdout.replace(str(workdir.resolve()), ".").replace(str(workdir), ".")
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        # a progress bar of rich on a console which is no terminal prints
        # empty lines, one blank line separates parts of an output
        if not line and (not cleaned or not cleaned[-1]):
            continue
        cleaned.append(line)
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned)


def run_page(page: str, workdir: Path) -> list[BlockResult]:
    """Run the python blocks of a page in one namespace.

    Args:
        page: the page, relative to `docs/`.
        workdir: the working directory of the blocks.

    Returns:
        What every block printed and drew, in the order of the blocks.

    Raises:
        AssertionError: if a block fails, with the block in the message.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    _, blocks = parse((DOCS_DIR / page).read_text(encoding="utf-8"))
    # a block runs as a script does, its guard `if __name__ == "__main__":` too
    namespace: dict[str, object] = {"__name__": "__main__"}
    shown: list[Figure] = []
    results: list[BlockResult] = []
    cwd = Path.cwd()
    # a page may configure logging, e.g. `log.enable_rich_logging()`, whose
    # handler writes to the console: the next page starts without it
    loggers = [logging.getLogger(), logging.getLogger("sbmlsim")]
    states = [(lg.handlers[:], lg.level) for lg in loggers]
    os.chdir(workdir)
    try:
        with pd.option_context("display.width", 100, "display.max_columns", 20):
            for k, block in enumerate(blocks):
                stdout = io.StringIO()
                try:
                    with contextlib.redirect_stdout(stdout):
                        exec(compile(block.code, f"{page}[{k}]", "exec"), namespace)
                except Exception as err:
                    raise AssertionError(
                        f"block {k} of '{page}' fails:\n{block.code}"
                    ) from err
                figures = [
                    f for f in _figures(namespace) if all(f is not s for s in shown)
                ]
                shown.extend(figures)
                results.append(BlockResult(_clean(stdout.getvalue(), workdir), figures))
                plt.close("all")
    finally:
        os.chdir(cwd)
        for lg, (handlers, level) in zip(loggers, states, strict=True):
            lg.handlers[:] = handlers
            lg.setLevel(level)
    return results


def _image_name(page: str, k: int, i: int) -> str:
    """Get the file name of the `i`-th figure of the `k`-th block of a page."""
    stem = Path(page).stem
    suffix = "" if i == 0 else f"-{i + 1}"
    return f"{stem}-{k + 1:02d}{suffix}.png"


def _region(result: BlockResult, images: list[str]) -> list[str]:
    """Get the lines of the output and the images of a block."""
    lines: list[str] = []
    if result.stdout:
        lines += ["", OUTPUT_FENCE, *result.stdout.split("\n"), FENCE]
    for image in images:
        lines += ["", f"![Output figure]({image})"]
    return lines


def write_page(page: str, workdir: Path) -> bool:
    """Run the blocks of a page and write their outputs and figures.

    Args:
        page: the page, relative to `docs/`.
        workdir: the working directory of the blocks.

    Returns:
        Whether the page changed.
    """
    path = DOCS_DIR / page
    text = path.read_text(encoding="utf-8")
    lines, blocks = parse(text)
    results = run_page(page, workdir)

    stem = Path(page).stem
    for old in OUTPUTS_DIR.glob(f"{stem}-*.png"):
        old.unlink()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    new: list[str] = []
    position = 0
    for k, (block, result) in enumerate(zip(blocks, results, strict=True)):
        new += lines[position : block.end]
        images = []
        for i, figure in enumerate(result.figures):
            name = _image_name(page, k, i)
            figure.savefig(
                OUTPUTS_DIR / name,
                dpi=DPI,
                bbox_inches="tight",
                facecolor="white",
                metadata={"Software": None},
            )
            images.append(f"{IMAGES}/{name}")
        new += _region(result, images)
        position = block.region_end
        n_lines = result.stdout.count("\n") + 1
        if result.stdout and n_lines > MAX_OUTPUT_LINES:
            print(f"{page}: block {k + 1} prints {n_lines} lines", file=sys.stderr)
    new += lines[position:]
    updated = "\n".join(new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    """Write the outputs of the pages given on the command line, or of all."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pages", nargs="*", help="pages relative to docs/")
    args = parser.parse_args()
    pages = args.pages or PAGES
    unknown = [p for p in pages if p not in PAGES]
    if unknown:
        parser.error(f"not pages of the documentation with code: {unknown}")
    for page in pages:
        with tempfile.TemporaryDirectory() as tmp:
            changed = write_page(page, Path(tmp))
        print(f"{page}: {'updated' if changed else 'unchanged'}")


if __name__ == "__main__":
    main()
