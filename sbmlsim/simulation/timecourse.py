"""
Definition of timecourse simulations and timecourse definitions.
"""
from pathlib import Path
import json
from typing import List, Tuple
from copy import deepcopy
import logging
from pint.errors import DimensionalityError

from sbmlsim.serialization import ObjectJSONEncoder, JSONEncoder
from sbmlsim.parametrization import ChangeSet
from sbmlsim.simulation.simulation import AbstractSim

logger = logging.getLogger(__name__)


class Timecourse(JSONEncoder):
    """ Simulation definition.

    Definition of all information necessary to run a single timecourse simulation.

    A single simulation consists of multiple changes which are applied,
    all simulations are performed and collected.

    Changesets and selections are deepcopied for persistance

    """
    def __init__(self, start: float, end: float, steps: int,
                 changes: dict = None, model_changes: dict = None,
                 normalized = False):
        """ Create a time course definition for simulation.

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

        self.normalized = normalized
        self.start = start
        self.end = end
        self.steps = steps
        self.changes = deepcopy(changes)
        self.model_changes = deepcopy(model_changes)

    def default(self, o):
        """json encoder"""
        # FIXME: unified handling of encoding via single encoder
        return o.__dict__

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
        Q_ = ureg.Quantity

        changes_normed = {}
        for key, item in self.changes.items():
            if hasattr(item, "units"):
                # perform unit conversion
                try:
                    # convert to model units
                    item = item.to(udict[key])
                except DimensionalityError as err:
                    logger.error(f"DimensionalityError "
                                 f"'{key} = {item}'. {err}")
                    raise err
                except KeyError as err:
                    logger.error(f"KeyError: '{key}' does not exist in unit dictionary of model.")
                    raise err
            else:
                item = Q_(item, udict[key])
                logger.warning(f"No units provided, assuming model units: "
                               f"{key} = {item}")
            changes_normed[key] = item

        self.changes = changes_normed
        self.normalized = True


class TimecourseSim(AbstractSim):
    """ Timecourse simulation consisting of multiple concatenated timecourses.

    In case of a single timecourse, only the single timecourse is executed.
    """
    def __init__(self, timecourses: List[Timecourse],
                 selections: list = None, reset: bool = True,
                 time_offset: float = 0.0):
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

        self.selections = deepcopy(selections)
        self.reset = reset
        self.time_offset = time_offset

    def normalize(self, udict, ureg):
        for tc in self.timecourses:
            tc.normalize(udict=udict, ureg=ureg)

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            'type': self.__class__.__name__,
            'selections': self.selections,
            'reset': self.reset,
            'time_offset': self.time_offset,
            'timecourses': [tc.to_dict() for tc in self.timecourses]
        }
        return d

    def to_json(self, path=None):
        """ Convert definition to JSON for exchange.

        :param path: path for file, if None JSON str is returned
        :return:
        """
        if path is None:
            return json.dumps(self, cls=ObjectJSONEncoder, indent=2)
        else:
            with open(path, "w") as f_json:
                json.dump(self, fp=f_json, cls=ObjectJSONEncoder, indent=2)

    @staticmethod
    def from_json(json_info: Tuple[str, Path]) -> 'TimecourseSim':
        """ Load TimecourseSim from Path or str

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
        lines = [
            "-" * 40,
            f"{self.__class__.__name__}",
            "-" * 40,
            self.to_json()
        ]
        return "\n".join(lines)







if __name__ == "__main__":

    tcsim = TimecourseSim([
        Timecourse(0, 100, steps=101),
        Timecourse(0, 100, steps=101, changes={'k': 100, 'p': 200}),
        Timecourse(0, 50, steps=51, changes={'k': 100, 'p': 50}),
    ])
    import numpy as np
    tcsims = ensemble(tcsim, changeset=ChangeSet.scan_changeset("k2", np.linspace(0, 4, num=5)))
    for tcs in tcsims:
        print(tcs)



