"""The formats of the files of a figure, without importing anything of sbmlsim.

A figure of a run is written as the image the renderer of sbmlsim-core draws,
SVG or PNG, and beside it as the spec of its chart, JSON, which the report
reads. `sbmlsim.experiment` writes them and `sbmlsim.project` checks the
formats of a project, whose module stays light (it imports no experiment).
"""

from collections.abc import Sequence

__all__ = [
    "DEFAULT_FIGURE_FORMATS",
    "FIGURE_FORMATS",
    "HTML_GONE",
    "SPEC_FORMAT",
    "figure_formats_of",
]

#: the formats of the files of a figure: the image the renderer of
#: sbmlsim-core draws as SVG or PNG, and the spec of the chart as JSON
FIGURE_FORMATS: tuple[str, ...] = ("svg", "png", "json")

#: the formats of the figures of a run which names none
DEFAULT_FIGURE_FORMATS: tuple[str, ...] = ("svg",)

#: the format of the spec of a figure, which is written beside every image,
#: since the report reads it
SPEC_FORMAT = "json"

#: the message of a run which asks for the interactive pages of plotly
HTML_GONE = (
    "the format html is gone with plotly; the report shows the svg and becomes "
    "interactive with the renderer"
)


def figure_formats_of(figure_formats: Sequence[str] | None) -> list[str]:
    """Get the formats of the files a run writes of every figure.

    Args:
        figure_formats: the formats asked for, `DEFAULT_FIGURE_FORMATS`
            without; any case.

    Returns:
        The formats in lower case and in their order without duplicates,
        followed by `SPEC_FORMAT`, which is written beside every image.

    Raises:
        TypeError: if the formats are a single string.
        ValueError: for `html`, with `HTML_GONE`, and for a format which is
            none of `FIGURE_FORMATS`.
    """
    if isinstance(figure_formats, str):
        raise TypeError(
            f"The figure formats must be a sequence of strings, not "
            f"{figure_formats!r}; use ({figure_formats!r},)."
        )
    asked = DEFAULT_FIGURE_FORMATS if figure_formats is None else figure_formats
    formats = list(dict.fromkeys(f.lower() for f in asked))
    if "html" in formats:
        raise ValueError(HTML_GONE)
    unknown = [f for f in formats if f not in FIGURE_FORMATS]
    if unknown:
        raise ValueError(
            f"The figure formats {unknown} are none of {list(FIGURE_FORMATS)}."
        )
    if SPEC_FORMAT not in formats:
        formats.append(SPEC_FORMAT)
    return formats
