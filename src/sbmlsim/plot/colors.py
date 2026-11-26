"""Normalize the colours of a style to `#rrggbb`, without matplotlib.

The colours of a curve, band or area are given as matplotlib accepts them: a
CSS name (`dimgrey`), a base colour of one letter (`k`, `r`, ...), a tableau
name (`tab:red`), an xkcd name (`xkcd:sky blue`), a colour of the default cycle
(`C0` to `C9`), hex (`#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`), a grey level as
a text of a number from 0 to 1 (`"0.4"`), `none` or a tuple of floats from 0
to 1. The spec of `figure/1` takes `#rrggbb` only, so its
writer (the chart builder of Task 3) calls `normalize_color` for the colour and
`color_alpha` for an opacity which the colour carries, and `points` uses it for
the shades of a curve.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from sbmlsim.plot._xkcd_colors import XKCD

#: the base colours of one letter of matplotlib
BASE = {
    "b": "#0000ff",
    "g": "#008000",
    "r": "#ff0000",
    "c": "#00bfbf",
    "m": "#bf00bf",
    "y": "#bfbf00",
    "k": "#000000",
    "w": "#ffffff",
}

#: the CSS4 (X11) colour names
CSS4 = {
    "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "azure": "#f0ffff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "black": "#000000",
    "blanchedalmond": "#ffebcd",
    "blue": "#0000ff",
    "blueviolet": "#8a2be2",
    "brown": "#a52a2a",
    "burlywood": "#deb887",
    "cadetblue": "#5f9ea0",
    "chartreuse": "#7fff00",
    "chocolate": "#d2691e",
    "coral": "#ff7f50",
    "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc",
    "crimson": "#dc143c",
    "cyan": "#00ffff",
    "darkblue": "#00008b",
    "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9",
    "darkgreen": "#006400",
    "darkgrey": "#a9a9a9",
    "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00",
    "darkorchid": "#9932cc",
    "darkred": "#8b0000",
    "darksalmon": "#e9967a",
    "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f",
    "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3",
    "deeppink": "#ff1493",
    "deepskyblue": "#00bfff",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "dodgerblue": "#1e90ff",
    "firebrick": "#b22222",
    "floralwhite": "#fffaf0",
    "forestgreen": "#228b22",
    "fuchsia": "#ff00ff",
    "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff",
    "gold": "#ffd700",
    "goldenrod": "#daa520",
    "gray": "#808080",
    "green": "#008000",
    "greenyellow": "#adff2f",
    "grey": "#808080",
    "honeydew": "#f0fff0",
    "hotpink": "#ff69b4",
    "indianred": "#cd5c5c",
    "indigo": "#4b0082",
    "ivory": "#fffff0",
    "khaki": "#f0e68c",
    "lavender": "#e6e6fa",
    "lavenderblush": "#fff0f5",
    "lawngreen": "#7cfc00",
    "lemonchiffon": "#fffacd",
    "lightblue": "#add8e6",
    "lightcoral": "#f08080",
    "lightcyan": "#e0ffff",
    "lightgoldenrodyellow": "#fafad2",
    "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90",
    "lightgrey": "#d3d3d3",
    "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa",
    "lightslategray": "#778899",
    "lightslategrey": "#778899",
    "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0",
    "lime": "#00ff00",
    "limegreen": "#32cd32",
    "linen": "#faf0e6",
    "magenta": "#ff00ff",
    "maroon": "#800000",
    "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585",
    "midnightblue": "#191970",
    "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1",
    "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead",
    "navy": "#000080",
    "oldlace": "#fdf5e6",
    "olive": "#808000",
    "olivedrab": "#6b8e23",
    "orange": "#ffa500",
    "orangered": "#ff4500",
    "orchid": "#da70d6",
    "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98",
    "paleturquoise": "#afeeee",
    "palevioletred": "#db7093",
    "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9",
    "peru": "#cd853f",
    "pink": "#ffc0cb",
    "plum": "#dda0dd",
    "powderblue": "#b0e0e6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "red": "#ff0000",
    "rosybrown": "#bc8f8f",
    "royalblue": "#4169e1",
    "saddlebrown": "#8b4513",
    "salmon": "#fa8072",
    "sandybrown": "#f4a460",
    "seagreen": "#2e8b57",
    "seashell": "#fff5ee",
    "sienna": "#a0522d",
    "silver": "#c0c0c0",
    "skyblue": "#87ceeb",
    "slateblue": "#6a5acd",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#fffafa",
    "springgreen": "#00ff7f",
    "steelblue": "#4682b4",
    "tan": "#d2b48c",
    "teal": "#008080",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "turquoise": "#40e0d0",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "white": "#ffffff",
    "whitesmoke": "#f5f5f5",
    "yellow": "#ffff00",
    "yellowgreen": "#9acd32",
}

#: the tableau colours by their name without the prefix `tab:`
TABLEAU = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

#: the names of the tables as matplotlib takes them: a tableau name of `gray`
#: also spelt `grey` (`tab:grey`), an xkcd name of `grey` also spelt `gray`,
#: which wins over a name of `gray` of the table (the names of
#: `matplotlib.colors.get_named_colors_mapping`); the CSS names list both
_TABLEAU_NAMES = TABLEAU | {
    name.replace("gray", "grey"): value
    for name, value in TABLEAU.items()
    if "gray" in name
}
_XKCD_NAMES = XKCD | {
    name.replace("grey", "gray"): value
    for name, value in XKCD.items()
    if "grey" in name
}

#: the colours `C0` to `C9` of the default colour cycle of matplotlib
CYCLE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _expand(digits: str, color: str) -> str:
    """Expand `rgb[a]` to `rrggbb[aa]` and check the digits."""
    if len(digits) in (3, 4):
        digits = "".join(c * 2 for c in digits)
    if len(digits) not in (6, 8) or any(c not in "0123456789abcdef" for c in digits):
        raise ValueError(f"unknown colour {color!r}")
    return digits


def _grey(text: str) -> str | None:
    """Get a grey level, a text of a number from 0 to 1, as `#rrggbb`, else `None`."""
    try:
        level = float(text)
    except ValueError:
        return None
    if not (math.isfinite(level) and 0.0 <= level <= 1.0):
        return None
    return "#" + f"{round(level * 255):02x}" * 3


def _parse(color: str | Sequence[float]) -> tuple[str | None, float]:
    """Parse a colour into `(#rrggbb or None, alpha)`."""
    if isinstance(color, str):
        text = color.strip().lower()
        if text == "none":
            return None, 1.0
        if text.startswith("tab:") and text[4:] in _TABLEAU_NAMES:
            text = _TABLEAU_NAMES[text[4:]]
        elif len(text) == 2 and text[0] == "c" and text[1].isdigit():
            text = CYCLE[int(text[1])]
        elif text.startswith("xkcd:") and text[5:] in _XKCD_NAMES:
            text = _XKCD_NAMES[text[5:]]
        elif text in BASE:
            text = BASE[text]
        elif text in CSS4:
            text = CSS4[text]
        if not text.startswith("#"):
            grey = _grey(text)
            if grey is None:
                raise ValueError(f"unknown colour {color!r}")
            text = grey
        digits = _expand(text[1:], color)
        alpha = int(digits[6:], 16) / 255.0 if len(digits) == 8 else 1.0
        return "#" + digits[:6], alpha
    values = [float(v) for v in color]
    if len(values) not in (3, 4) or any(not 0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"unknown colour {color!r}")
    alpha = values[3] if len(values) == 4 else 1.0
    return "#" + "".join(f"{round(v * 255):02x}" for v in values[:3]), alpha


def normalize_color(color: str | Sequence[float] | None) -> str | None:
    """Get a colour as `#rrggbb`, `None` for no colour (`None` or `none`).

    Raises:
        ValueError: for a colour which is none of the forms of the module docstring.
    """
    if color is None:
        return None
    return _parse(color)[0]


def color_alpha(color: str | Sequence[float] | None) -> tuple[str | None, float]:
    """Get a colour as `#rrggbb` and its opacity from 0 to 1 (1 without alpha)."""
    if color is None:
        return None, 1.0
    return _parse(color)
