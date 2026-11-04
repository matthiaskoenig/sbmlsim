"""The engines which simulate a model."""

import importlib
import locale
import sys
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
    module = sys.modules.get("COPASI")
    if module is not None:
        return module
    before = locale.setlocale(locale.LC_CTYPE)
    try:
        return importlib.import_module("COPASI")
    except ImportError as err:
        raise ImportError(
            "The engine 'copasi' needs python-copasi, install it with "
            "`pip install sbmlsim[copasi]`."
        ) from err
    finally:
        locale.setlocale(locale.LC_CTYPE, before)


def import_amici() -> ModuleType:
    """Import AMICI, the optional extra `amici`.

    AMICI is built from source on installation (a C++ compiler, CMake, SWIG,
    a BLAS and HDF5) and needs `ninja` on the path to compile a model.

    :return: the module `amici`
    :raises ImportError: when AMICI is not installed
    """
    module = sys.modules.get("amici")
    if module is not None:
        return module
    before = locale.setlocale(locale.LC_CTYPE)
    try:
        return importlib.import_module("amici")
    except ImportError as err:
        raise ImportError(
            "The engine 'amici' needs AMICI, install it with "
            "`pip install sbmlsim[amici]`. AMICI is built from source and needs "
            "a C++ compiler, CMake, SWIG, a BLAS library and HDF5 (e.g. "
            "`apt install build-essential cmake swig libopenblas-dev "
            "libhdf5-dev`)."
        ) from err
    finally:
        locale.setlocale(locale.LC_CTYPE, before)
