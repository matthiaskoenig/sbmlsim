"""Definition of timecourses and timecourse simulations."""
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

    Changesets and selections are deep copied for persistence.

    """

    def __init__(
        self,
        start: float,
        end: float,
        steps: int,
        changes: dict = None,
        model_changes: dict = None,
        model_manipulations: dict = None,
        discard: bool = False,
    ):
        """Create a time course definition for simulation.

        Discarded simulations do not add time shifts, i.e. a pre-simulation
        does not increase the time of simulation.

        :param start: start time
        :param end: end time
        :param steps: simulation steps
        :param changes: parameter and initial condition changes
        :param model_changes: model parameter and initial condition changes
        :param model_manipulations: model structure changes
        :param discard: discards simulation from results (e.g. pre-simulations)
        """
        # Create empty changes and model changes for serialization
        if changes is None:
            changes = {}
        if model_changes is None:
            model_changes = {}
        if model_manipulations is None:
            model_manipulations = {}

        self.start = start
        self.end = end
        self.steps = steps
        self.changes = deepcopy(changes)
        self.model_changes = deepcopy(model_changes)
        self.model_manipulations = deepcopy(model_manipulations)
        self.discard = discard

    def __repr__(self) -> str:
        """Get representation."""
        return f"Timecourse([{self.start}:{self.end}])"

    def to_dict(self):
        """Convert to dictionary."""
        d = dict()
        for key in self.__dict__:
            d[key] = self.__dict__[key]
        return d

    def add_change(self, sid: str, value: float) -> None:
        """Add change."""
        self.changes[sid] = value

    def remove_change(self, sid: str) -> None:
        """Remove change for given id."""
        del self.changes[sid]

    def add_model_change(self, sid: str, change) -> None:
        """Add model change."""
        self.model_changes[sid] = change

    def add_model_changes(self, model_changes: Dict[str, str]) -> None:
        """Add model changes."""
        self.model_changes.update(model_changes)

    def remove_model_change(self, sid: str) -> None:
        """Remove model change for id."""
        del self.model_changes[sid]

    def normalize(self, udict, ureg):
        """Normalize values to model units for all changes."""
        self.model_changes = Units.normalize_changes(
            self.model_changes, udict=udict, ureg=ureg
        )
        self.changes = Units.normalize_changes(self.changes, udict=udict, ureg=ureg)

    def strip_units(self):
        """Strip units for parallel simulation.

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
        selections: Optional[List[str]] = None,
        reset: bool = True,
        time_offset: float = 0.0,
    ):
        """Initialize timecourse sim.

        :param timecourses:
        :param selections:
        :param reset: complete reset of model
        :param time_offset: time shift of simulation
        """
        if isinstance(timecourses, Timecourse):
            timecourses = [timecourses]

        self.timecourses = []
        for tc in timecourses:
            if not tc:
                # remove empty elements (allows for cleaner syntax)
                continue

            if isinstance(tc, dict):
                # construct from dict
                tc = Timecourse(**tc)

            # make a copy to ensure independence of instances
            self.timecourses.append(deepcopy(tc))

        if len(self.timecourses) == 0:
            logger.error("No timecourses in simulation")
        else:
            for k, tc in enumerate(self.timecourses):
                if k > 0 and tc.model_changes:
                    logger.error(
                        f"'model_changes' only allowed on first timecourse: {tc}"
                    )

        self.selections = deepcopy(selections)
        self.reset = reset
        self.time_offset = time_offset

        self.time = self._time()

    def __repr__(self) -> str:
        """Get representation."""
        return f"TimecourseSim({[tc for tc in self.timecourses]})"

    def _time(self) -> np.ndarray:
        """Calculate the time vector complete simulation."""
        t_offset = self.time_offset
        time_vecs = []
        for tc in self.timecourses:
            time_vecs.append(np.linspace(tc.start, tc.end, num=tc.steps + 1) + t_offset)
            t_offset += tc.end
        res: np.ndarray = np.concatenate(time_vecs)
        return res

    def dimensions(self) -> List[Dimension]:
        """Get dimensions."""
        return [Dimension(dimension="time", index=self.time)]

    def add_model_changes(self, model_changes: Dict) -> None:
        """Add model changes to given simulation."""
        if self.timecourses:
            tc = self.timecourses[0]  # type: Timecourse
            tc.add_model_changes(model_changes)

    def normalize(self, udict, ureg):
        """Normalize timecourse simulation."""
        for tc in self.timecourses:
            tc.normalize(udict=udict, ureg=ureg)

    def strip_units(self):
        """Strip units from simulation."""
        for tc in self.timecourses:
            tc.strip_units()

    def to_dict(self):
        """Convert to dictionary."""
        d = {
            "type": self.__class__.__name__,
            "selections": self.selections,
            "reset": self.reset,
            "time_offset": self.time_offset,
            "timecourses": [tc.to_dict() for tc in self.timecourses],
        }
        return d

    def to_json(self, path: Path = None) -> str:
        """Convert definition to JSON."""
        if path is None:
            return json.dumps(self, cls=ObjectJSONEncoder, indent=2)
        else:
            with open(path, "w") as f_json:
                json.dump(self, fp=f_json, cls=ObjectJSONEncoder, indent=2)

    @staticmethod
    def from_json(json_info: Union[str, Path]) -> "TimecourseSim":
        """Load from JSON."""
        if isinstance(json_info, Path):
            with open(json_info, "r") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)
        if "type" in d:
            d.pop("type")  # serialized property
        return TimecourseSim(**d)

    def __str__(self) -> str:
        """Get string representation."""
        lines = ["-" * 40, f"{self.__class__.__name__}", "-" * 40, self.to_json()]
        return "\n".join(lines)
