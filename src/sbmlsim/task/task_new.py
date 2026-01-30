"""Task module.

FIXME: add discard flag on subtask; handle via outputEndTime
TODO: use pydantic whereever possible

1. The values of all ranges are calculated before the execution of the repeated task

The order of activities within each iteration of a RepeatedTask is as follows:
- The entire model state for any involved Model is reset if specified by the resetModel
  attribute
- Any changes to the model or models specified by SetValue objects in thelistOfChanges
  are applied to each Model.

Then, for each SubTask child of the RepeatedTask, in the order specified by its order attribute:
- Any AlgorithmParameter children of the associated Simulation are applied
  (with the possible exception of the seed; see Section 2.2.7.2).
- Any SetValue children of the SubTask are applied to the relevant Model.
- The referenced Task of the SubTask is executed.
"""

from dataclasses import dataclass

from sbmlsim.simulation import Dimension


@dataclass
class Change:
    """Change."""

    model: str
    target: str
    symbol: str

    variables: list  # current values
    parameters: list
    math: str
    range: str  # this is precalculated


@dataclass
class RepeatedTask:
    """RepeatedTask."""

    range: str  # dimension id
    ranges: [Dimension]
    reset_model: bool
    concatenate: bool


@dataclass
class Task:
    """Task."""

    model: str
    simulation: str


# Fields on Timecourse
#         self.selections = deepcopy(selections)
#         self.reset = reset
#         self.time_offset = time_offset


@dataclass
class SubTask:
    """Subtask."""

    model: str
    simulation: str
    changes: list[str]
    model_changes: list[str]
    model_manipulations: list[str]
    order: int
    discard: bool = False
