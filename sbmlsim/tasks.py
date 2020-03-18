from sbmlsim.timecourse import TimecourseScan, TimecourseSim


class Task(object):
    def __init__(self, model: str, simulation):
        self.model = model
        self.simulation = simulation  # TimecourseSim or TimecourseScan

