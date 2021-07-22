from abc import ABC
from typing import Optional


class BaseObject(ABC):
    """Base class for SED-ML bases
    
    FIXME: support annotations and notes
    """

    def __init__(self, sid: Optional[str], name: Optional[str]):
        self.sid: Optional[str] = sid
        self.name: Optional[str] = name


class BaseObjectSIdRequired(BaseObject):
    """Base class for SED-ML bases with required sid."""

    def __init__(self, sid: str, name: Optional[str]):
        super(BaseObjectSIdRequired).__init__(sid=sid, name=name)


class Target:
    """XPath or other target to model element."""
    def __init__(self, target: str):
        self.target = target

class Symbol:
    """Symbol class"""
    def __init__(self, symbol: str):
        self.symbol = symbol