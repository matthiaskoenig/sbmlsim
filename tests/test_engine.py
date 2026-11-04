"""Tests for the simulation engines."""

import importlib.util
import subprocess
import sys

import pytest

from sbmlsim.simulator import Engine, import_amici, import_copasi, resolve_engine


def test_resolve_engine() -> None:
    """An engine is resolved from its name or itself."""
    assert resolve_engine("copasi") is Engine.COPASI
    assert resolve_engine(Engine.ROADRUNNER) is Engine.ROADRUNNER
    assert resolve_engine("amici") is Engine.AMICI
    with pytest.raises(ValueError, match="roadrunner, copasi, amici"):
        resolve_engine("cvode")


def test_import_copasi_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing COPASI names the extra which installs it."""
    monkeypatch.setitem(sys.modules, "COPASI", None)
    with pytest.raises(ImportError, match=r"sbmlsim\[copasi\]"):
        import_copasi()


def test_sbmlsim_does_not_import_copasi() -> None:
    """Importing the package does not import COPASI."""
    code = "import sys, sbmlsim, sbmlsim.simulator; assert 'COPASI' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=60)


def test_import_copasi_keeps_the_locale() -> None:
    """Importing COPASI sets the locale `C`, `import_copasi` sets its encoding back.

    The locale `C` makes the preferred encoding of python ASCII, which every
    text file and subprocess of the process would read with afterwards; the
    numbers of COPASI keep the locale `C`.
    """
    # not imported here, which would set the locale of the test process
    if importlib.util.find_spec("COPASI") is None:
        pytest.skip("COPASI is not installed")
    code = (
        "import locale\n"
        "before = locale.setlocale(locale.LC_CTYPE)\n"
        "encoding = locale.getpreferredencoding(False)\n"
        "from sbmlsim.engine import import_copasi\n"
        "import_copasi()\n"
        "assert locale.setlocale(locale.LC_CTYPE) == before\n"
        "assert locale.setlocale(locale.LC_NUMERIC) == 'C'\n"
        "assert locale.getpreferredencoding(False) == encoding\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=60)


def test_import_amici_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing AMICI names the extra which installs it and its build requirements."""
    monkeypatch.setitem(sys.modules, "amici", None)
    with pytest.raises(ImportError, match=r"sbmlsim\[amici\]") as info:
        import_amici()
    # the build requirements from the system, CMake and SWIG come from PyPI
    assert "libopenblas-dev libhdf5-dev" in str(info.value)
    assert "swig" not in str(info.value)


def test_sbmlsim_does_not_import_amici() -> None:
    """Importing the package does not import AMICI."""
    code = "import sys, sbmlsim, sbmlsim.simulator; assert 'amici' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=60)


def test_import_amici_keeps_the_locale() -> None:
    """`import_amici` leaves the locale of the process as it was."""
    if importlib.util.find_spec("amici") is None:
        pytest.skip("AMICI is not installed")
    code = (
        "import locale\n"
        "before = {c: locale.setlocale(c) for c in (locale.LC_CTYPE, locale.LC_NUMERIC)}\n"
        "encoding = locale.getpreferredencoding(False)\n"
        "from sbmlsim.engine import import_amici\n"
        "import_amici()\n"
        "assert locale.setlocale(locale.LC_CTYPE) == before[locale.LC_CTYPE]\n"
        "assert locale.setlocale(locale.LC_NUMERIC) == before[locale.LC_NUMERIC]\n"
        "assert locale.getpreferredencoding(False) == encoding\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=60)


def test_an_amici_simulator_without_amici_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulator imports the engine when it is created, not at the first run."""
    from sbmlsim.simulator import Simulator

    monkeypatch.setitem(sys.modules, "amici", None)
    with pytest.raises(ImportError, match=r"sbmlsim\[amici\]"):
        Simulator(engine="amici")


def test_the_runs_of_the_other_engines_do_not_import_amici() -> None:
    """A run of roadrunner, and of COPASI where it is installed, needs no AMICI."""
    engines = ["roadrunner"]
    if importlib.util.find_spec("COPASI") is not None:
        engines.append("copasi")
    code = (
        "import sys\n"
        "from sbmlsim.resources import DEMO_SBML\n"
        "from sbmlsim.simulation import Scan, Simulation, Dimension\n"
        "from sbmlsim.simulator import Simulator\n"
        "scan = Scan(Simulation(end=10), [Dimension('d', values={'Km_A': [1, 2]})])\n"
        f"for engine in {engines!r}:\n"
        "    Simulator(n_workers=1, engine=engine).run(DEMO_SBML, scan)\n"
        "assert 'amici' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=120)
