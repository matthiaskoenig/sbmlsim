"""
Definition of timecourse simulations and timecourse definitions.
"""
import json
from typing import List
from copy import deepcopy
from json import JSONEncoder
import logging

from sbmlsim.parametrization import ChangeSet


#
# TODO: test json serialization (reading and writing)


class Timecourse(JSONEncoder):
    """ Simulation definition.

    Definition of all information necessary to run a single timecourse simulation.

    A single simulation consists of multiple changes which are applied,
    all simulations are performed and collected.

    Changesets and selections are deepcopied for persistance

    """
    def __init__(self, start: float, end: float, steps: int,
                 changes: dict = None, model_changes: dict = None):
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

        self.start = start
        self.end = end
        self.steps = steps
        self.changes = deepcopy(changes)
        self.model_changes = deepcopy(model_changes)

    def default(self, o):
        """json encoder"""
        return o.__dict__

    def add_change(self, sid: str, value: float):
        self.changes[sid] = value

    def remove_change(self, sid: str):
        del self.changes[sid]

    def add_model_change(self, sid: str, change):
        self.model_changes[sid] = change

    def remove_model_change(self, sid: str):
        del self.model_changes[sid]


class ObjectJSONEncoder(JSONEncoder):
    def default(self, o):
        """json encoder"""
        return o.__dict__


class TimecourseSim(object):
    """ Timecourse simulation consisting of multiple concatenated timecourses.

    In case of a single timecourse, only the single timecourse is executed.
    """
    def __init__(self, timecourses: List[Timecourse],
                 selections: list = None, reset: bool = True):
        """
        :param timecourses:
        :param selections:
        :param reset: resetToOrigin at beginning of simulation
        """
        if isinstance(timecourses, Timecourse):
            timecourses = [timecourses]

        self.timecourses = timecourses
        self.selections = deepcopy(selections)
        self.reset = reset

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
    def from_json(json_str):
        d = json.loads(json_str)
        return TimecourseSim(**d)

    def __str__(self):
        lines = [
            "-" * 40,
            f"{self.__class__.__name__}",
            "-" * 40,
            self.to_json()
        ]
        return "\n".join(lines)


def ensemble(sim: TimecourseSim, changeset: ChangeSet) -> List[TimecourseSim]:
    """ Creates an ensemble of timecourse by mixin the changeset in.

    :return: List[TimecourseSimulation]
    """
    sims = []
    for changes in changeset:
        # FIXME: not sure if this is doing the correct thing or custom implementation of copy and deepcopy needed
        sim_new = deepcopy(sim)
        # changes are mixed in the first timecourse
        tc = sim_new.timecourses[0]
        for key, value in changes.items():
            tc.add_change(key, value)
        sims.append(sim_new)

    return sims


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




