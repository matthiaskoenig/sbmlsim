"""The figures of the reports: saved with their spec and inlined into a page.

A figure of a report is a `Chart` of the renderer of sbmlsim-core, saved as
its image (`svg` or `png`) and its spec (`json`), see `save_figure`. A page
shows the SVG inline with its spec in a script next to it, see
`inline_figure`: `<figure data-figure="<key>">` holds the SVG and
`<script type="application/json" data-figure-spec="<key>">` the spec, so a
page can read what a figure draws.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sbmlsim.plot.chart import Chart

__all__ = ["inline_figure", "inline_svg", "save_figure", "script_json"]

#: the formats a figure of a report is saved in
IMAGE_FORMATS: tuple[str, ...] = ("svg", "png")

_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>\s*", re.IGNORECASE)
_DOCTYPE = re.compile(r"<!DOCTYPE[^>]*>\s*", re.IGNORECASE)
#: the namespaces of an SVG file, which an HTML page gives inline SVG itself
_NAMESPACES = re.compile(
    r'\s+xmlns(?::xlink)?="http://www\.w3\.org/(?:2000/svg|1999/xlink)"'
)


def inline_svg(text: str) -> str:
    """Get the text of an SVG file as markup of an HTML page.

    Args:
        text: the SVG file.

    Returns:
        The SVG without the XML declaration and the DOCTYPE, which an HTML
        page must not carry inside its body, and without the namespaces of
        its root, which the HTML parser gives an inline SVG, so a page which
        needs no network names no URL.
    """
    text = _DOCTYPE.sub("", _XML_DECLARATION.sub("", text, count=1), count=1).strip()
    root = text.find(">")
    return _NAMESPACES.sub("", text[:root]) + text[root:]


def script_json(text: str) -> str:
    r"""Get JSON text which can sit in a script element of an HTML page.

    Args:
        text: the JSON text.

    Returns:
        The text with `</` as `<\/` (a JSON parser reads it back unchanged),
        so `</script>` and `<!--` in a label cannot end the script.
    """
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")


def save_figure(chart: Chart, path: Path, png_dpi: float | None = None) -> list[Path]:
    """Save a figure as its image and its spec.

    Args:
        chart: the figure.
        path: the image, its suffix `.svg` or `.png`; the spec is written
            next to it with the suffix `.json`.
        png_dpi: the resolution of a PNG: of the image of a `.png` path, the
            default of the chart without, and of a PNG copy of an SVG, which
            is written only with it, e.g. for LaTeX, which includes no SVG.

    Returns:
        The files written, the image first.

    Raises:
        ValueError: for a path of another suffix.
    """
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in IMAGE_FORMATS:
        raise ValueError(
            f"A figure of a report is saved as {' or '.join(IMAGE_FORMATS)}, "
            f"not as '{path.suffix}': '{path}'."
        )
    written = [chart.save(path, dpi=png_dpi), chart.save(path.with_suffix(".json"))]
    if suffix != "png" and png_dpi is not None:
        written.append(chart.save(path.with_suffix(".png"), dpi=png_dpi))
    return written


def inline_figure(path: Path) -> dict[str, Any]:
    """Get the SVG and the spec of a saved figure for a page.

    Args:
        path: the image of the figure, see `save_figure`.

    Returns:
        `svg`, the markup of the SVG, and `spec_text`, the spec for a script
        element, for an SVG with its spec next to it; empty otherwise, e.g. for
        a PNG, which a page shows as an image.
    """
    spec = path.with_suffix(".json")
    if path.suffix.lower() != ".svg" or not path.is_file() or not spec.is_file():
        return {}
    return {
        "svg": inline_svg(path.read_text(encoding="utf-8")),
        "spec_text": script_json(spec.read_text(encoding="utf-8")),
    }
