"""Tests for the simulation engines."""

import importlib.util
import subprocess
import sys

import pytest

from sbmlsim.simulator import Engine, import_copasi, resolve_engine


def test_resolve_engine() -> None:
    """An engine is resolved from its name or itself."""
    assert resolve_engine("copasi") is Engine.COPASI
    assert resolve_engine(Engine.ROADRUNNER) is Engine.ROADRUNNER
    with pytest.raises(ValueError, match="roadrunner, copasi"):
        resolve_engine("amici")


def test_import_copasi_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing COPASI names the extra which installs it."""
    monkeypatch.setitem(sys.modules, "COPASI", None)
    with pytest.raises(ImportError, match=r"sbmlsim\[copasi\]"):
        import_copasi()


def test_sbmlsim_does_not_import_copasi() -> None:
    """Importing the package does not import COPASI."""
    code = "import sys, sbmlsim, sbmlsim.simulator; assert 'COPASI' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_copasi_keeps_the_locale() -> None:
    """Importing COPASI sets the locale `C`, which `import_copasi` sets back.

    The locale `C` makes the preferred encoding of python ASCII, which every
    text file and subprocess of the process would read with afterwards.
    """
    # not imported here, which would set the locale of the test process
    if importlib.util.find_spec("COPASI") is None:
        pytest.skip("COPASI is not installed")
    code = (
        "import locale\n"
        "before = locale.setlocale(locale.LC_ALL)\n"
        "encoding = locale.getpreferredencoding(False)\n"
        "from sbmlsim.engine import import_copasi\n"
        "import_copasi()\n"
        "assert locale.setlocale(locale.LC_ALL) == before\n"
        "assert locale.getpreferredencoding(False) == encoding\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=60)
