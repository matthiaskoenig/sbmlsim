from abc import ABC


class BaseObject(ABC):
    """Base class for SED-ML bases"""

    def __init__(self, sid: str, name: str):
        self.sid = sid
        self.name = name
