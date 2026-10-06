"""Test the utility functions."""

from collections.abc import Iterable
from pathlib import Path

import pytest

from sbmlsim.utils import paths_text


@pytest.mark.parametrize(
    "paths, text",
    [
        (None, "none"),
        ("data", "data"),
        (Path("data"), "data"),
        ([Path("a"), Path("b")], "a\nb"),
        ((p for p in ["a", Path("b")]), "a\nb"),
        ([], ""),
    ],
)
def test_paths_text(paths: str | Path | Iterable[str | Path] | None, text: str) -> None:
    """A path is its text, several paths are one per line."""
    assert paths_text(paths) == text
