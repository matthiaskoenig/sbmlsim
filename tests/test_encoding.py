"""Every text file is opened, read and written with an explicit encoding.

Without one python 3.14 uses the encoding of the locale, which is ASCII under
the locale `C` without its coercion to UTF-8 (a macOS runner of GitHub), and a
report with a `µ` or a `·` is then neither read nor written. Python 3.15 reads
and writes UTF-8 by default (PEP 686).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCES = ["src", "tests", "scripts", "examples", "conftest.py"]
# the modules and classes whose `open` opens no text file of a path
NOT_A_PATH = {"os", "webbrowser", "Image", "tarfile", "zipfile", "gzip", "h5py"}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in SOURCES:
        path = ROOT / name
        files.extend([path] if path.is_file() else sorted(path.rglob("*.py")))
    return files


def _is_binary(call: ast.Call, mode_index: int) -> bool:
    """Tell whether a call of `open` opens its file in binary mode."""
    modes = [*call.args[mode_index : mode_index + 1]]
    modes += [k.value for k in call.keywords if k.arg == "mode"]
    return any(isinstance(m, ast.Constant) and "b" in str(m.value) for m in modes)


def _opens_text(call: ast.Call) -> bool:
    """Tell whether a call opens, reads or writes a text file."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "open" and not _is_binary(call, 1)
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in {"read_text", "write_text"}:
        return True
    if func.attr == "open":
        module = isinstance(func.value, ast.Name) and func.value.id in NOT_A_PATH
        return not module and not _is_binary(call, 0)
    return False


def test_text_files_have_an_encoding() -> None:
    """No call opens, reads or writes a text file without an encoding."""
    missing = [
        f"{path.relative_to(ROOT)}:{call.lineno}"
        for path in _python_files()
        for call in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(call, ast.Call)
        and _opens_text(call)
        and not any(k.arg == "encoding" for k in call.keywords)
    ]
    assert missing == []
