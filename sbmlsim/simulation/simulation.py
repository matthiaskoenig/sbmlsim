"""
Abstract base simulation.
"""


class AbstractSim(object):
    def normalize(self, udict, ureg):
        pass

    def to_dict(self):
        """ Convert to dictionary. """
        d = {
            'type': self.__class__.__name__,
        }
        return d
