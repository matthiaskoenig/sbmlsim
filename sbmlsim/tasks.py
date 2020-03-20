from sbmlsim.timecourse import TimecourseScan, TimecourseSim


class Task(object):

    # FIXME: just reference the simulation id
    def __init__(self, model: str, simulation: str):
        self.model_id = model
        self.simulation_id = simulation

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            "model": self.model_id,
            "simulation": self.simulation_id,
        }
        return d
