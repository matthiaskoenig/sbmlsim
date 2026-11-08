"""The engines which simulate a model."""

import contextlib
import importlib
import locale
import sys
from collections.abc import Generator, Iterable
from enum import StrEnum
from types import ModuleType


class UnsupportedModelError(ValueError, NotImplementedError):
    """An engine does not simulate a model or a plan as roadrunner does.

    The feature checks of COPASI and AMICI raise it before the engine loads,
    compiles or runs what it would simulate otherwise, e.g. an algebraic rule
    or a delayed event; it is a limitation of the engine, not an error of the
    model. It is a `ValueError` (the checks of a model) and a
    `NotImplementedError` (the checks of a plan) alike.
    """


class Engine(StrEnum):
    """The engine which integrates a model."""

    ROADRUNNER = "roadrunner"
    COPASI = "copasi"
    AMICI = "amici"


class JitOptimization(StrEnum):
    """The optimization level LLJIT compiles the code of a model of roadrunner with (#331).

    Defined here, with the engines, so a module which names a level, e.g. a
    project, does not import roadrunner; `sbmlsim.model.model_roadrunner`
    resolves it, see `resolve_jit_optimization`.

    The levels of LLVM, `-O0` to `-O3`, which roadrunner calls `NONE`,
    `LESS`, `DEFAULT` and `AGGRESSIVE` (`LoadSBMLOptions`). A level trades
    the time of the compilation against the speed of the simulation, measured
    with libroadrunner 2.11 on the dulaglutide model of pkdb_models: `O0`
    compiles in 1.1 s, `O1` to `O3` in 3.9 to 4.2 s and MCJIT, roadrunner's
    default JIT, in 6.7 s, while a serial scan of 300 doses takes 4.5 s at
    `O0` and 3.9 s at `O2` and with MCJIT (the midazolam model 0.94 s and
    0.75 s). `O0` gives the results of MCJIT bitwise, the other levels differ
    from them in the last bits, so a fit can take another path.

    `O0` is the default (`DEFAULT_JIT_OPTIMIZATION`): it compiles several
    times faster and its code needs less memory (the experiments and fit
    mappings of the creatinine model of pkdb_models 3.8 GB against 4.9 GB at
    `O2`), and the higher levels simulate little faster, so they pay only for
    a run which simulates a model long after compiling it once; e.g. the run
    of all experiments of dulaglutide takes about 30 s at `O0` against 34 s
    at `O2`.
    """

    O0 = "O0"
    O1 = "O1"
    O2 = "O2"
    O3 = "O3"


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
    with kept_encoding():
        try:
            return importlib.import_module(name)
        except ImportError as err:
            raise ImportError(message) from err


@contextlib.contextmanager
def kept_encoding() -> Generator[None]:
    """Set the encoding of the locale back to the one before, after calls of COPASI.

    COPASI sets the locale of the process to `C` when it is imported and when
    it compiles some models, e.g. one whose conversion factor has an initial
    assignment, which makes the preferred encoding of python ASCII, so every
    text file and subprocess of the process which is read without an
    explicit encoding fails on a character which is not ASCII. The category
    `LC_CTYPE` of the locale, the encoding, is set back to the one before;
    the numbers keep the locale `C` of COPASI (`LC_NUMERIC`).

    :return: a context manager
    """
    before = locale.setlocale(locale.LC_CTYPE)
    try:
        yield
    finally:
        if locale.setlocale(locale.LC_CTYPE) != before:
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
