"""The engines besides roadrunner, which the tests of the engines compare with it.

A module `test_*_engines.py` runs its tests with every engine of
`OTHER_ENGINES` through the fixture `other_engine` of `tests.conftest`; a
test of an engine whose extra is not installed is skipped. The tests of one
engine only are in the modules of its name, e.g. `test_simulator_amici.py`.
"""

import contextlib
from pathlib import Path
from typing import Any

import pytest

from sbmlsim.engine import Engine, import_copasi
from sbmlsim.model.model_sbml import SBMLModel

#: the engines besides roadrunner, by their names
OTHER_ENGINES: list[Engine] = [Engine.COPASI, Engine.AMICI]

#: the module of the extra of an engine
_MODULES: dict[Engine, str] = {Engine.COPASI: "COPASI", Engine.AMICI: "amici"}

#: the integrator of an engine, as the methods of a fit report name it
INTEGRATORS: dict[Engine, str] = {
    Engine.ROADRUNNER: "CVODE",
    Engine.COPASI: "LSODA",
    Engine.AMICI: "CVODES",
}

#: the package of an engine whose version a fit records
PACKAGES: dict[Engine, str] = {Engine.COPASI: "python-copasi", Engine.AMICI: "amici"}


def require(engine: Engine) -> None:
    """Skip a test of an engine whose extra is not installed."""
    if engine is Engine.COPASI:
        with contextlib.suppress(ImportError):
            # COPASI imported with the locale of the process kept, see
            # `import_copasi`
            import_copasi()
    pytest.importorskip(_MODULES[engine])


def model_class(engine: Engine) -> type[SBMLModel]:
    """Get the class of the models of an engine besides roadrunner."""
    if engine is Engine.COPASI:
        from sbmlsim.model.model_copasi import CopasiSBMLModel

        return CopasiSBMLModel
    from sbmlsim.model.model_amici import AmiciSBMLModel

    return AmiciSBMLModel


def relative_tolerance(model: SBMLModel) -> float:
    """Get the relative tolerance a model of an engine integrates with."""
    _, settings = model.integrator_state()
    return float(dict(settings)["relative_tolerance"])


def another_cache(monkeypatch: pytest.MonkeyPatch, path: Path) -> Path:
    """Point `SBMLSIM_AMICI_CACHE` to an empty directory and fail a compilation of AMICI.

    What a worker sees whose environment differs from the one its parent
    loaded the models with: a model loaded from the directory of the parent,
    see `sbmlsim.model.model_amici.using_cache`, compiles nothing.

    Returns:
        The empty directory, which must stay empty.
    """
    from sbmlsim.model import model_amici

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("AMICI compiled a model")

    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(model_amici.CACHE_ENV, str(path))
    monkeypatch.setattr(model_amici, "_compile", refuse)
    return path
