"""
Definition of timecourse simulations and timecourse definitions.
"""
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np

from sbmlsim.serialization import ObjectJSONEncoder
from sbmlsim.simulation import AbstractSim, Dimension
from sbmlsim.units import Units


logger = logging.getLogger(__name__)


class Timecourse(ObjectJSONEncoder):
    """Simulation definition.

    Definition of all information necessary to run a single timecourse simulation.

    A single simulation consists of multiple changes which are applied,
    all simulations are performed and collected.

    Changesets and selections are deepcopied for persistance

    """

    def __init__(
        self,
        start: float,
        end: float,
        steps: int,
        changes: dict = None,
        model_changes: dict = None,
    ):
        """Create a time course definition for simulation.

        :param start: start time
        :param end: end time
        :param steps: simulation steps
        :param changes: parameter and initial condition changes
        :param model_changes: changes to model structure
        """
        # Create empty changes and model changes for serialization
        if changes is None:
            changes = {}
        if model_changes is None:
            model_changes = {}

        self.start = start
        self.end = end
        self.steps = steps
        self.changes = deepcopy(changes)
        self.model_changes = deepcopy(model_changes)

    def __repr__(self):
        return f"Timecourse([{self.start},{self.end}])"

    def to_dict(self):
        """ Convert to dictionary. """
        d = dict()
        for key in self.__dict__:
            d[key] = self.__dict__[key]
        return d

    def add_change(self, sid: str, value: float):
        self.changes[sid] = value

    def remove_change(self, sid: str):
        del self.changes[sid]

    def add_model_change(self, sid: str, change):
        self.model_changes[sid] = change

    def remove_model_change(self, sid: str):
        del self.model_changes[sid]

    def normalize(self, udict, ureg):
        """ Normalize values to model units for all changes."""
        self.changes = Units.normalize_changes(self.changes, udict=udict, ureg=ureg)

    def strip_units(self):
        """Stripping units for parallel simulation.
        All changes must be normalized before stripping !.
        """
        self.changes = {k: v.magnitude for k, v in self.changes.items()}


class TimecourseSim(AbstractSim):
    """Timecourse simulation consisting of multiple concatenated timecourses.

    In case of a single timecourse, only the single timecourse is executed.
    """

    def __init__(
        self,
        timecourses: List[Timecourse],
        selections: list = None,
        reset: bool = True,
        time_offset: float = 0.0,
    ):
        """
        :param timecourses:
        :param selections:
        :param reset: resetToOrigin at beginning of simulation
        :param time_offset: time shift of simulation
        """
        if isinstance(timecourses, Timecourse):
            timecourses = [timecourses]

        self.timecourses = []
        for tc in timecourses:
            if isinstance(tc, dict):
                # construct from dict
                self.timecourses.append(Timecourse(**tc))
            else:
                self.timecourses.append(tc)
        if len(self.timecourses) == 0:
            logger.error("No timecourses in simulation")

        self.selections = deepcopy(selections)
        self.reset = reset
        self.time_offset = time_offset

        self.time = self._time()

    def __repr__(self):
        return f"TimecourseSim({[tc for tc in self.timecourses]})"

    def _time(self):
        """Calculates the time vector complete simulation."""
        t_offset = self.time_offset
        time_vecs = []
        for tc in self.timecourses:
            time_vecs.append(np.linspace(tc.start, tc.end, num=tc.steps + 1) + t_offset)
            t_offset += tc.end
        return np.concatenate(time_vecs)

    def dimensions(self) -> List[Dimension]:
        return [Dimension(dimension="time", index=self.time)]

    def normalize(self, udict, ureg):
        """Normalize timecourse simulation."""
        for tc in self.timecourses:
            tc.normalize(udict=udict, ureg=ureg)

    def strip_units(self):
        """Strip units from simulation."""
        for tc in self.timecourses:
            tc.strip_units()

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            "type": self.__class__.__name__,
            "selections": self.selections,
            "reset": self.reset,
            "time_offset": self.time_offset,
            "timecourses": [tc.to_dict() for tc in self.timecourses],
        }
        return d

    def to_json(self, path=None):
        """Convert definition to JSON for exchange.

        :param path: path for file, if None JSON str is returned
        :return:
        """
        if path is None:
            return json.dumps(self, cls=ObjectJSONEncoder, indent=2)
        else:
            with open(path, "w") as f_json:
                json.dump(self, fp=f_json, cls=ObjectJSONEncoder, indent=2)

    @staticmethod
    def from_json(json_info: Tuple[str, Path]) -> "TimecourseSim":
        """Load TimecourseSim from Path or str

        :param json_info:
        :return:
        """
        if isinstance(json_info, Path):
            with open(json_info, "r") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)
        if "type" in d:
            d.pop("type")  # serialized property
        return TimecourseSim(**d)

    def __str__(self):
        lines = ["-" * 40, f"{self.__class__.__name__}", "-" * 40, self.to_json()]
        return "\n".join(lines)
