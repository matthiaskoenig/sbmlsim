"""The engines which simulate a model."""

import contextlib
import importlib
import locale
import sys
from collections.abc import Iterable
from enum import StrEnum
from types import ModuleType


class Engine(StrEnum):
    """The engine which integrates a model."""

    ROADRUNNER = "roadrunner"
    COPASI = "copasi"
    AMICI = "amici"


def resolve_engine(engine: Engine | str) -> Engine:
    """Resolve an engine from its name.

    :param engine: an engine or its name
    :return: the engine
    :raises ValueError: for a name which is no engine
    """
    try:
        return Engine(engine)
    except ValueError:
        names = ", ".join(e.value for e in Engine)
        raise ValueError(
            f"Unknown engine '{engine}', the engines are: {names}"
        ) from None


def _import_optional(name: str, message: str) -> ModuleType:
    """Import an optional module, setting the encoding of the locale back.

    :param name: the name of the module
    :param message: the message of the `ImportError` when the module is missing
    :return: the module
    :raises ImportError: when the module is not installed
    """
    module = sys.modules.get(name)
    if module is not None:
        return module
    before = locale.setlocale(locale.LC_CTYPE)
    try:
        return importlib.import_module(name)
    except ImportError as err:
        raise ImportError(message) from err
    finally:
        locale.setlocale(locale.LC_CTYPE, before)


def import_copasi() -> ModuleType:
    """Import COPASI, the optional extra `copasi`.

    Importing COPASI sets the locale of the process to `C`, which makes the
    preferred encoding of python ASCII, so every text file and subprocess of
    the process which is read without an explicit encoding fails on a
    character which is not ASCII. The category `LC_CTYPE` of the locale, the
    encoding, is set back to the one before the import; the numbers keep the
    locale `C` of COPASI (`LC_NUMERIC`).

    :return: the module `COPASI`
    :raises ImportError: when python-copasi is not installed
    """
    return _import_optional(
        "COPASI",
        "The engine 'copasi' needs python-copasi, install it with "
        "`pip install sbmlsim[copasi]`.",
    )


def import_amici() -> ModuleType:
    """Import AMICI, the optional extra `amici`.

    AMICI is built from source on installation: a C++ compiler, a BLAS
    library and the headers of HDF5 come from the system, CMake and SWIG from
    PyPI; it needs `ninja`, which the extra installs, to compile a model.

    :return: the module `amici`
    :raises ImportError: when AMICI is not installed
    """
    return _import_optional(
        "amici",
        "The engine 'amici' needs AMICI, install it with "
        "`pip install sbmlsim[amici]`. AMICI is built from source and needs "
        "a C++ compiler, a BLAS library and the headers of HDF5 (e.g. "
        "`apt install build-essential libopenblas-dev libhdf5-dev`).",
    )


def amici_cache(engine: Engine, models: Iterable[object]) -> str | None:
    """Get the directory of the compiled models of AMICI the workers of a pool load from.

    The directory the first model of AMICI of `models` was loaded from, else
    the one of the process, see `sbmlsim.model.model_amici.cache_root`, so a
    worker does not compile the models again into the directory of its own
    environment, see `sbmlsim.model.model_amici.using_cache`.

    :param engine: the engine of the models
    :param models: the models of the parent
    :return: the directory, `None` for another engine
    """
    if engine is not Engine.AMICI:
        return None
    # the module of the model of AMICI does not import AMICI
    from sbmlsim.model.model_amici import AmiciSBMLModel, cache_root

    for model in models:
        if isinstance(model, AmiciSBMLModel):
            return str(model.cache_root)
    return str(cache_root())


def using_amici_cache(root: str | None) -> contextlib.AbstractContextManager[None]:
    """Load the models of AMICI from the directory of compiled models of a parent.

    :param root: the directory, see `amici_cache`; `None` changes nothing
    :return: the context in which the models are loaded from it
    """
    if root is None:
        return contextlib.nullcontext()
    from sbmlsim.model.model_amici import using_cache

    return using_cache(root)
