"""The width of a text as the renderer of sbmlsim-core measures it.

The renderer lays out every text with the advances of Inter Regular, its
table of the characters U+0020 to U+007E in font units of 2048 per em
(`crates/plot/src/text.rs` of sbmlsim-core) and of the symbols of the
labels of sbmlsim (`_SYMBOLS`); another character takes the advance of `0`,
as the renderer does for one outside of its table, which is close to the
average of the others. A writer which places a text itself, e.g. a box of
metrics, measures it here the same way.
"""

from __future__ import annotations

from typing import Final

__all__ = ["TEXT_SIZE", "text_width"]

#: the font size in pixels of a text element of the renderer, the size of
#: the ticks and the legends by default
TEXT_SIZE: Final = 11.0

#: font units per em of the advances
_UNITS_PER_EM: Final = 2048.0

#: the advances of U+0020 to U+007E of Inter Regular in font units
_ASCII: Final = (
    576, 589, 954, 1297, 1314, 2011, 1319, 614, 747, 747, 1026, 1355, 590, 942, 590, 738,
    1292, 833, 1249, 1265, 1323, 1215, 1270, 1159, 1267, 1270, 590, 618, 1355, 1355, 1355, 1047,
    1978, 1413, 1340, 1496, 1478, 1231, 1209, 1528, 1522, 550, 1169, 1376, 1158, 1850, 1543, 1566,
    1308, 1566, 1318, 1314, 1322, 1524, 1413, 2018, 1397, 1390, 1288, 747, 738, 747, 965, 934,
    661, 1150, 1254, 1170, 1254, 1194, 758, 1256, 1211, 496, 496, 1124, 496, 1794, 1210, 1228,
    1254, 1254, 771, 1081, 670, 1211, 1151, 1676, 1118, 1151, 1131, 873, 681, 873, 1355,
)  # fmt: skip

#: the advances of the symbols of the labels of sbmlsim in font units, from
#: the table of the other characters of the renderer
_SYMBOLS: Final = {
    "²": 905, "³": 913, "±": 1355, "×": 1355, "÷": 1355, "·": 590, "√": 1289,
    "µ": 1200, "°": 933, "–": 1024, "…": 1770, "≤": 1355, "≥": 1355,
    "≈": 1355, "∞": 1917, "α": 1360, "β": 1220, "γ": 1151, "Δ": 1395,
    "σ": 1278,
}  # fmt: skip

#: the advance of another character, the one of `0`
_OTHER: Final = _ASCII[ord("0") - 0x20]


def _advance(character: str) -> int:
    """Get the advance of a character in font units."""
    code = ord(character)
    if 0x20 <= code < 0x7F:
        return _ASCII[code - 0x20]
    return _SYMBOLS.get(character, _OTHER)


def text_width(text: str, size: float = TEXT_SIZE) -> float:
    """Get the width of a text in pixels, the width of its widest line.

    Args:
        text: the text, its lines split at newlines.
        size: the font size in pixels.

    Returns:
        The width the renderer measures, exact for ASCII and close to it for
        the other characters.
    """
    units = max(sum(_advance(c) for c in line) for line in text.split("\n"))
    return units * size / _UNITS_PER_EM
