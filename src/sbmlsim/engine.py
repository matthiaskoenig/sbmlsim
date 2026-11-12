"""The engines which simulate a model."""

import contextlib
import importlib
import locale
import os
import sys
from collections.abc import Generator, Iterable
from enum import StrEnum
from importlib import metadata
from types import ModuleType
from typing import Final

#: the package index of the builds of the `develop` branch of libroadrunner
#: (matthiaskoenig/roadrunner-wheels), which every install of the package
#: names: the builds are not on PyPI
ROADRUNNER_INDEX: Final[str] = (
    "https://matthiaskoenig.github.io/roadrunner-wheels/simple/"
)

#: the page which describes the install, named by the report of a libroadrunner
#: which does not satisfy the requirement
INSTALLATION_DOCS: Final[str] = "https://matthiaskoenig.github.io/sbmlsim/installation/"

#: the environment variable which skips the check of the installed
#: libroadrunner, for an install the metadata of which does not describe it,
#: e.g. a libroadrunner built from source
SKIP_ROADRUNNER_CHECK: Final[str] = "SBMLSIM_SKIP_ROADRUNNER_CHECK"


#: the distribution of the simulator, the name of its requirement
ROADRUNNER_DISTRIBUTION: Final[str] = "libroadrunner"


def installed_roadrunner() -> str | None:
    """Get the version of the installed libroadrunner.

    The version of the distribution `libroadrunner`, read from the metadata of
    the environment, which does not import roadrunner.

    :return: the version, `None` when libroadrunner is not installed
    """
    try:
        return metadata.version(ROADRUNNER_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def roadrunner_requirement() -> str | None:
    """Get the requirement of libroadrunner of the package.

    The requirement of the metadata of the installed sbmlsim, e.g.
    `libroadrunner>=2.11.1.dev655`, so the requirement of `pyproject.toml` is
    the only place the build is named.

    :return: the requirement, `None` when the metadata of sbmlsim or the
        requirement is missing, e.g. in a checkout which is not installed
    """
    try:
        requires = metadata.requires("sbmlsim") or ()
    except metadata.PackageNotFoundError:
        return None
    for requirement in requires:
        # the requirement of a dependency of an extra carries a marker and is
        # not the requirement of the package; the name is read from the text
        # and not with `packaging.requirements`, which imports `packaging.tags`
        # and costs 30 ms of every import of the package, while
        # `packaging.version` costs 2 ms
        if ";" in requirement:
            continue
        text = requirement.strip()
        rest = text.removeprefix(ROADRUNNER_DISTRIBUTION)
        # the name of another distribution continues after the one of the
        # simulator, e.g. `libroadrunner-plugins`
        if rest != text and (not rest or not (rest[0].isalnum() or rest[0] in "-_.")):
            return text
    return None


def roadrunner_problem(installed: str | None, requirement: str) -> str | None:
    """Report an installed libroadrunner which does not satisfy the requirement.

    The simulations of the package rely on fixes of libroadrunner which are
    not released yet (roadrunner #1381, #1382, #1383, #1384, #1420), so it
    needs a build of the `develop` branch of libroadrunner, which the index
    `ROADRUNNER_INDEX` serves and PyPI does not have. The release of PyPI
    loads the models and simulates them, with other rows at the events and
    the changes of a simulation, i.e. it gives wrong results instead of an
    error, which is why the install is checked.

    :param installed: the version of the installed libroadrunner, `None` when
        it is not installed
    :param requirement: the requirement of the package, e.g.
        `libroadrunner>=2.11.1.dev655`
    :return: the message which names the install, `None` when the installed
        version satisfies the requirement
    """
    if installed is not None and satisfies_requirement(installed, requirement):
        return None
    what = (
        "libroadrunner is not installed"
        if installed is None
        else f"libroadrunner {installed} is installed"
    )
    return (
        f"{what}, sbmlsim needs {requirement}: a build of the `develop` branch "
        "of libroadrunner, which is not on PyPI, since sbmlsim relies on fixes "
        "of libroadrunner which are not released yet.\n"
        "Install the build from the index of the builds first and then "
        "sbmlsim, whose requirement the installed build satisfies:\n"
        f'    pip install --no-deps --index-url {ROADRUNNER_INDEX} "{requirement}"\n'
        "    pip install sbmlsim\n"
        f"{INSTALLATION_DOCS} describes the install with uv and in a project "
        f"which locks the build. Set {SKIP_ROADRUNNER_CHECK}=1 to skip this "
        "check, e.g. for a libroadrunner built from source."
    )


def satisfies_requirement(installed: str, requirement: str) -> bool:
    """Compare an installed version with the requirement of the package.

    The floor of a requirement of the form `libroadrunner>=<version>` is
    compared with `packaging.version`, which costs 2 ms of the import of the
    package; a requirement of another form is parsed with
    `packaging.requirements`, which imports `packaging.tags` and costs 30 ms.
    A development version satisfies a floor which is one (`>=2.11.1.dev655`
    is satisfied by `2.11.1.dev655`, by a later build and by the release
    `2.11.1`), which is why the comparison is not a comparison of the texts.

    :param installed: the installed version
    :param requirement: the requirement, e.g. `libroadrunner>=2.11.1.dev655`
    :return: whether the installed version satisfies the requirement
    """
    from packaging.version import InvalidVersion, Version

    floor = requirement.removeprefix(ROADRUNNER_DISTRIBUTION).strip()
    if floor.startswith(">=") and "," not in floor:
        try:
            return Version(installed) >= Version(floor.removeprefix(">=").strip())
        except InvalidVersion:
            pass
    from packaging.requirements import Requirement

    return Requirement(requirement).specifier.contains(installed, prereleases=True)


def check_roadrunner() -> None:
    """Check the installed libroadrunner against the requirement of the package.

    Called by `import sbmlsim`, where it costs the metadata of two
    distributions and imports no roadrunner. The environment variable
    `SKIP_ROADRUNNER_CHECK` skips it, and a checkout which is not installed,
    whose requirement is unknown, is not checked.

    :raises ImportError: when libroadrunner is missing or does not satisfy the
        requirement, with the commands which install the build
    """
    if os.environ.get(SKIP_ROADRUNNER_CHECK):
        return
    requirement = roadrunner_requirement()
    if requirement is None:
        return
    problem = roadrunner_problem(installed_roadrunner(), requirement)
    if problem is not None:
        raise ImportError(problem)


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

    `O2` is the default (`DEFAULT_JIT_OPTIMIZATION`), it simulates as fast as
    MCJIT. `O0` pays for a run which compiles more than it simulates, e.g.
    the initialization of experiments: the run of all experiments of
    dulaglutide takes about 30 s at `O0` against 34 s at `O2`.
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
