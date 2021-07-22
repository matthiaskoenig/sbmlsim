from abc import ABC
from typing import Optional


class BaseObject(ABC):
    """Base class for SED-ML bases"""

    def __init__(self, sid: Optional[str], name: Optional[str]):
        self.sid: Optional[str] = sid
        self.name: Optional[str] = name


class BaseObjectSIdRequired(BaseObject):
    """Base class for SED-ML bases with required sid."""

    def __init__(self, sid: str, name: Optional[str]):
        super(BaseObjectSIdRequired).__init__(sid=sid, name=name)
