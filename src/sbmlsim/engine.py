"""The engines which simulate a model."""

import importlib
from enum import StrEnum
from types import ModuleType


class Engine(StrEnum):
    """The engine which integrates a model."""

    ROADRUNNER = "roadrunner"
    COPASI = "copasi"


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

    :return: the module `COPASI`
    :raises ImportError: when python-copasi is not installed
    """
    try:
        return importlib.import_module("COPASI")
    except ImportError as err:
        raise ImportError(
            "The engine 'copasi' needs python-copasi, install it with "
            "`pip install sbmlsim[copasi]`."
        ) from err
