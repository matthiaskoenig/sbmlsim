"""Tests for the simulation engines."""

import importlib.util
import subprocess
import sys

import pytest

from sbmlsim.simulator import (
    Engine,
    import_amici,
    import_copasi,
    import_core,
    resolve_engine,
)

#: the seconds a process of python which imports sbmlsim may take: about 1.5 s
#: on an idle machine, more than 60 s next to other work (a load of 450 on 20
#: logical CPUs); the tests check what a process imports, not how fast
TIMEOUT = 300


def test_resolve_engine() -> None:
    """An engine is resolved from its name or itself."""
    assert resolve_engine("copasi") is Engine.COPASI
    assert resolve_engine(Engine.ROADRUNNER) is Engine.ROADRUNNER
    assert resolve_engine("amici") is Engine.AMICI
    assert resolve_engine("core") is Engine.CORE
    with pytest.raises(ValueError, match="roadrunner, copasi, amici, core"):
        resolve_engine("cvode")


def test_import_copasi_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing COPASI names the extra which installs it."""
    monkeypatch.setitem(sys.modules, "COPASI", None)
    with pytest.raises(ImportError, match=r"sbmlsim\[copasi\]"):
        import_copasi()


def test_sbmlsim_does_not_import_copasi() -> None:
    """Importing the package does not import COPASI."""
    code = "import sys, sbmlsim, sbmlsim.simulator; assert 'COPASI' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=TIMEOUT)


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
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=TIMEOUT)


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
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=TIMEOUT)


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
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=TIMEOUT)


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
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=TIMEOUT)


def test_import_core_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing bindings name the package to install, which is a dependency."""
    monkeypatch.setitem(sys.modules, "sbmlsim_core", None)
    with pytest.raises(ImportError, match=r"pip install sbmlsim-core") as info:
        import_core()
    assert "dependency of sbmlsim" in str(info.value)


def test_a_core_simulator_without_the_bindings_names_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulator imports the engine when it is created, not at the first run."""
    from sbmlsim.simulator import Simulator

    monkeypatch.setitem(sys.modules, "sbmlsim_core", None)
    with pytest.raises(ImportError, match=r"pip install sbmlsim-core"):
        Simulator(engine="core")


def test_sbmlsim_does_not_import_core() -> None:
    """Importing the package does not import the bindings of sbmlsim-core."""
    code = "import sys, sbmlsim, sbmlsim.simulator; assert 'sbmlsim_core' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=TIMEOUT)


def test_the_runs_of_the_other_engines_do_not_import_core() -> None:
    """A run of roadrunner, and of COPASI where it is installed, needs no sbmlsim-core."""
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
        "assert 'sbmlsim_core' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=TIMEOUT)


def test_every_engine_is_a_key_of_the_mappings_keyed_by_engine() -> None:
    """Every engine is a key of every mapping which is keyed by an engine.

    The provenance of a fit, the battery of the credibility and the tests of
    the engines each key a dictionary literal by the engine, which ty does not
    check for exhaustiveness, and the validator of the battery reads only
    `_ENGINE_LABELS`: a fifth engine added to that one alone passes every
    check and raises a `KeyError` in `_ENGINE_EXTRAS` and in `_installed`.
    This test fails at the first mapping which misses an engine instead.
    """
    from sbmlsim.credibility.battery import (
        _ENGINE_EXTRAS,
        _ENGINE_LABELS,
        _installed,
    )
    from sbmlsim.fit.provenance import ENGINE_PACKAGES
    from tests.engines import (
        _MODULES,
        INTEGRATORS,
        OTHER_ENGINES,
        PACKAGES,
        model_class,
    )

    engines = set(Engine)
    # the engines besides roadrunner, which is no second engine of the battery
    # and has no package and no module of its own in the tests
    others = engines - {Engine.ROADRUNNER}
    assert set(OTHER_ENGINES) == others
    assert set(ENGINE_PACKAGES) == engines
    assert set(INTEGRATORS) == engines
    assert set(PACKAGES) == others
    assert set(_MODULES) == others
    # the mappings of the battery are keyed by the name of the engine
    assert {Engine(name) for name in _ENGINE_LABELS} == others
    assert {Engine(name) for name in _ENGINE_EXTRAS} == others
    for engine in others:
        # the third mapping of the battery is a local of `_installed`, which
        # answers for every engine it holds and raises a `KeyError` otherwise
        _installed(engine.value)
        # `model_class` answers with the class of the engine and not with the
        # one of the branch it falls through to
        assert model_class(engine).engine is engine
