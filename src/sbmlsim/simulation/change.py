from sbmlsim.simulation.base import BaseObject
from enum import Enum, unique, auto


@unique
class Target:
    def __init__(self, target: str):
        self.target = target



class Change(BaseObject):
    """Change class.

    A model might need to undergo pre-processing before simulation.
    Those pre-processing steps are specified in the listOfChanges via the Change class on Model.
    Changes can be of the following types:
    - Changes based on mathematical calculations (ComputeChange)
    - Changes on attributes of the model (ChangeAttribute)
    - For XML-encoded models, changes on any XML snippet of the model’s XML representation
      (AddXML, ChangeXML, RemoveXML)
    """

    def __init__(self, target: Target, sid=None, name=None):
        """Construct Calculation."""
        super(Change, self).__init__(sid=sid, name=name)
        self.target = target


class ComputeChange(Calculation):
    """ComputeChange class.

    The ComputeChange class permits to change
    the numerical value of any single element or attribute of a Model
    addressable by a target, based on a calculation."""

    def __init__(self, sid: str, target: TargetType):
        """Construct Calculation."""
        super(Change, self).__init__(sid=sid, name=name, target)
        
