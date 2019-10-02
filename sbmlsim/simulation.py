"""
Main SBML simulator with roadrunner.


"""
import pandas as pd
import roadrunner
from collections import namedtuple

# TODO: parameter sensitivity
# TODO: initial condition sensitivity
# TODO: simulation experiments


import logging
import json



class TimecourseDefinition(object):
    """ Timecourse definition.

    """
    def __init__(self, tstart, tend, steps=None, points=None, init_changeset=None, sim_changesets=None,
                 selections=None, repeats=1):
        """

        :param tstart:
        :param tend:
        :param init_changes:
        :param sim_changes:
        :param selections:
        :param repeats:
        """
        self.tstart = tstart
        self.tend = tend
        self.steps = None
        self.points = None
        self.init_changeset = init_changeset
        self.sim_changesets = sim_changesets
        self.repeats = repeats


    def check(self, r):
        """ Check definition with model.

        :param r:
        :return:
        """
        pass



TimecourseResult = namedtuple("Result", ['base', 'mean', 'std', 'min', 'max'])



def timecourse(r, tc_def):
    """ Timecourse simulations based on timecourse_definition.

    :param r: Roadrunner model instance
    :param timecourse_def: TimeCourseDefinition
    :return:
    """

    # set custom selections
    model_selections = r.timeCourseSelections
    if tc_def.selections is not None:
        r.timeCourseSelections = tc_def.selections

    # reset all
    reset_all(r)


    # general changes
    for key, value in changes.items():
        r[key] = value
    s = r.simulate(start=0, end=tend, steps=steps)
    s_base = pd.DataFrame(s, columns=s.colnames)

    if yfun:
        # conversion functio
        yfun(s_base)

    if parameters is None:
        return s_base
    else:
        # baseline
        Np = 2 * len(parameters)
        (Nt, Ns) = s_base.shape
        shape = (Nt, Ns, Np)

        # empty array for storage
        s_data = np.empty(shape) * np.nan

        # all parameter changes
        idx = 0
        for pid in parameters.keys():
            for change in [1.0 + sensitivity, 1.0 - sensitivity]:
                resetAll(r)
                reset_doses(r)
                # dosing
                if dosing:
                    set_dosing(r, dosing, bodyweight=bodyweight)
                # general changes
                for key, value in changes.items():
                    r[key] = value
                # parameter changes
                value = r[pid]
                new_value = value * change
                r[pid] = new_value

                s = r.simulate(start=0, end=tend, steps=steps)
                if yfun:
                    # conversion function
                    s = pd.DataFrame(s, columns=s.colnames)
                    yfun(s)
                    s_data[:, :, idx] = s
                else:
                    s_data[:, :, idx] = s
                idx += 1

        s_mean = pd.DataFrame(np.mean(s_data, axis=2), columns=s_base.columns)
        s_std = pd.DataFrame(np.std(s_data, axis=2), columns=s_base.columns)
        s_min = pd.DataFrame(np.min(s_data, axis=2), columns=s_base.columns)
        s_max = pd.DataFrame(np.max(s_data, axis=2), columns=s_base.columns)

        # reset selections
        model_selections = r.timeCourseSelections
        if tc_def.selections is not None:
            r.timeCourseSelections = model_selections

        # return {'base': s_base, 'mean': s_mean, 'std': s_std, 'min': s_min, 'max': s_max}
        return Result(base=s_base, mean=s_mean, std=s_std, min=s_min, max=s_max)

def simulate(r, start=None, end=None, steps=None, points=None, **kwargs):
    """ Simple simulation.

    :param r:
    :param start:
    :param end:
    :param steps:
    :param points:
    :param kwargs:
    :return:
    """
    s = r.simulate(start=start, end=end, steps=steps, points=points, **kwargs)
    return pd.DataFrame(s, columns=s.colnames)


def reset_all(r):
    """ Reset all model variables to CURRENT init(X) values.

    This resets all variables, S1, S2 etc to the CURRENT init(X) values. It also resets all
    parameters back to the values they had when the model was first loaded.
    """
    r.reset(roadrunner.SelectionRecord.TIME |
            roadrunner.SelectionRecord.RATE |
            roadrunner.SelectionRecord.FLOATING |
            roadrunner.SelectionRecord.GLOBAL_PARAMETER)

if __name__ == "__main__":
    import os
    import sbmlsim
    from sbmlsim.tests.settings import DATA_PATH
    path = os.path.join(DATA_PATH, 'models', 'repressilator.xml')
    init

    tc_def = TimecourseDefinition(tstart=0, tend=100, steps=100, init_changeset=)
    r = sbmlsim.load_model()
